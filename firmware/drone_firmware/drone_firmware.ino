/*
 * ESP32 Micro Quadcopter Flight Controller
 * -----------------------------------------
 * Hardware: ESP32-S3 Mini / ESP32-C3 SuperMini + MPU6050 + 4× coreless brushed
 * motors
 *
 * Architecture:
 *   - Inner attitude stabilization loop runs onboard at ~500Hz (gyro PID)
 *   - Outer position loop runs on laptop (webcam ArUco → PID → WiFi UDP
 * commands)
 *   - ESP32 creates its own WiFi AP; laptop connects and sends UDP packets
 *
 * UDP Command Packet (8 bytes, little-endian):
 *   int16_t roll     (-100 to +100)   desired roll angle mapping
 *   int16_t pitch    (-100 to +100)   desired pitch angle mapping
 *   int16_t yaw      (-100 to +100)   desired yaw rate
 *   int16_t throttle (0 to 1000)      base motor power
 *
 * Motor Layout (X-configuration, viewed from above):
 *
 *        FRONT
 *     M1(CCW)  M2(CW)
 *        \    /
 *         \  /
 *          \/
 *          /\
 *         /  \
 *        /    \
 *     M4(CW)  M3(CCW)
 *        BACK
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// ============================================================================
// PIN DEFINITIONS — Adjust for your specific ESP32 board
// ============================================================================
// For ESP32-S3 Mini / SuperMini (recommended for weight):
#define MOTOR_FL_PIN 4 // Front-Left  (M1, CCW)
#define MOTOR_FR_PIN 5 // Front-Right (M2, CW)
#define MOTOR_BR_PIN 6 // Back-Right  (M3, CCW)
#define MOTOR_BL_PIN 7 // Back-Left   (M4, CW)

#define I2C_SDA_PIN 8 // MPU6050 SDA
#define I2C_SCL_PIN 9 // MPU6050 SCL

#define VBAT_PIN 1 // Battery voltage via voltage divider (ADC)

// If using standard ESP32 DevKit, use these instead:
// #define MOTOR_FL_PIN  25
// #define MOTOR_FR_PIN  26
// #define MOTOR_BR_PIN  27
// #define MOTOR_BL_PIN  14
// #define I2C_SDA_PIN   21
// #define I2C_SCL_PIN   22
// #define VBAT_PIN      34

// LED for status (most boards have one)
#define LED_PIN 2

// ============================================================================
// WIFI CONFIGURATION
// ============================================================================
const char *AP_SSID = "DroneESP32";
const char *AP_PASS = "drone1234";
const int UDP_PORT = 4210;

WiFiUDP udp;

// ============================================================================
// MPU6050 REGISTERS
// ============================================================================
#define MPU6050_ADDR 0x68
#define MPU6050_PWR_MGMT_1 0x6B
#define MPU6050_ACCEL_XOUT 0x3B
#define MPU6050_GYRO_XOUT 0x43
#define MPU6050_CONFIG 0x1A
#define MPU6050_GYRO_CONFIG 0x1B

// ============================================================================
// TUNING PARAMETERS
// ============================================================================

// Inner-loop PID gains (attitude stabilization)
// START WITH THESE LOW and increase kp slowly until oscillation, then back off
// 30%
float roll_kp = 1.3, roll_ki = 0.0, roll_kd = 0.4;
float pitch_kp = 1.3, pitch_ki = 0.0, pitch_kd = 0.4;
float yaw_kp = 2.5, yaw_ki = 0.0, yaw_kd = 0.0;

// Complementary filter coefficient (0.96 = trust gyro 96%, accel 4%)
const float COMP_FILTER_ALPHA = 0.96f;

// Max tilt angle the outer loop can request (degrees)
const float MAX_ANGLE_CMD = 25.0f;

// Motor output limits
const int MOTOR_MIN = 0;
const int MOTOR_MAX = 255; // 8-bit PWM resolution
const int MOTOR_IDLE = 60; // minimum spin when armed (keeps motors responsive)

// Failsafe
const unsigned long FAILSAFE_TIMEOUT_MS =
    500;                        // cut motors if no command in 500ms
const float VBAT_CUTOFF = 3.3f; // auto-land below this voltage

// ============================================================================
// STATE VARIABLES
// ============================================================================

// IMU calibration offsets (computed at startup)
float gyro_offset_x = 0, gyro_offset_y = 0, gyro_offset_z = 0;
float accel_offset_x = 0, accel_offset_y = 0, accel_offset_z = 0;

// Filtered angles (degrees)
float angle_roll = 0.0f;
float angle_pitch = 0.0f;
float angle_yaw = 0.0f; // gyro-only (no magnetometer)

// PID state
float roll_integral = 0, roll_prev_error = 0;
float pitch_integral = 0, pitch_prev_error = 0;
float yaw_prev_error = 0;

// Command from laptop (via UDP)
struct Command {
  int16_t roll;     // -100 to +100
  int16_t pitch;    // -100 to +100
  int16_t yaw;      // -100 to +100
  int16_t throttle; // 0 to 1000
};
Command cmd = {0, 0, 0, 0};

bool armed = false;
unsigned long last_cmd_time = 0;
unsigned long loop_timer = 0;

// Motor outputs
int motor_fl = 0, motor_fr = 0, motor_br = 0, motor_bl = 0;

// Battery
float vbat = 4.2f;

// ============================================================================
// PWM SETUP (ESP32 LEDC)
// ============================================================================
// ESP32 LEDC channels for motor PWM
const int PWM_FREQ = 20000; // 20kHz — above audible range, good for brushed
const int PWM_RES = 8;      // 8-bit = 0-255

void setupMotors() {
  // Attach LEDC channels to pins
  ledcAttach(MOTOR_FL_PIN, PWM_FREQ, PWM_RES);
  ledcAttach(MOTOR_FR_PIN, PWM_FREQ, PWM_RES);
  ledcAttach(MOTOR_BR_PIN, PWM_FREQ, PWM_RES);
  ledcAttach(MOTOR_BL_PIN, PWM_FREQ, PWM_RES);

  // All motors off
  ledcWrite(MOTOR_FL_PIN, 0);
  ledcWrite(MOTOR_FR_PIN, 0);
  ledcWrite(MOTOR_BR_PIN, 0);
  ledcWrite(MOTOR_BL_PIN, 0);
}

void setMotors(int fl, int fr, int br, int bl) {
  motor_fl = constrain(fl, MOTOR_MIN, MOTOR_MAX);
  motor_fr = constrain(fr, MOTOR_MIN, MOTOR_MAX);
  motor_br = constrain(br, MOTOR_MIN, MOTOR_MAX);
  motor_bl = constrain(bl, MOTOR_MIN, MOTOR_MAX);

  ledcWrite(MOTOR_FL_PIN, motor_fl);
  ledcWrite(MOTOR_FR_PIN, motor_fr);
  ledcWrite(MOTOR_BR_PIN, motor_br);
  ledcWrite(MOTOR_BL_PIN, motor_bl);
}

void motorsOff() { setMotors(0, 0, 0, 0); }

// ============================================================================
// MPU6050 SETUP & READING
// ============================================================================

void setupMPU6050() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000); // 400kHz I2C fast mode

  // Wake up MPU6050
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_PWR_MGMT_1);
  Wire.write(0x00); // Wake up, use internal 8MHz oscillator
  Wire.endTransmission();
  delay(10);

  // Set Digital Low Pass Filter (DLPF) — reduces vibration noise
  // Config 3 = 44Hz accel bandwidth, 42Hz gyro bandwidth
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_CONFIG);
  Wire.write(0x03);
  Wire.endTransmission();

  // Gyro range: ±500°/s (good balance of resolution vs range for micro drone)
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_GYRO_CONFIG);
  Wire.write(0x08); // FS_SEL=1 → ±500°/s → 65.5 LSB/°/s
  Wire.endTransmission();

  delay(100);
}

// Raw IMU data
int16_t accel_raw[3], gyro_raw[3];

void readMPU6050() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_ACCEL_XOUT);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 14, true); // 6 accel + 2 temp + 6 gyro

  accel_raw[0] = (Wire.read() << 8) | Wire.read(); // X
  accel_raw[1] = (Wire.read() << 8) | Wire.read(); // Y
  accel_raw[2] = (Wire.read() << 8) | Wire.read(); // Z
  Wire.read();
  Wire.read();                                    // discard temperature
  gyro_raw[0] = (Wire.read() << 8) | Wire.read(); // X (roll rate)
  gyro_raw[1] = (Wire.read() << 8) | Wire.read(); // Y (pitch rate)
  gyro_raw[2] = (Wire.read() << 8) | Wire.read(); // Z (yaw rate)
}

void calibrateIMU() {
  Serial.println(
      "IMU CALIBRATION — KEEP DRONE PERFECTLY STILL ON FLAT SURFACE");
  Serial.println("Calibrating in 2 seconds...");
  delay(2000);

  const int SAMPLES = 2000;
  float gx_sum = 0, gy_sum = 0, gz_sum = 0;
  float ax_sum = 0, ay_sum = 0, az_sum = 0;

  for (int i = 0; i < SAMPLES; i++) {
    readMPU6050();
    gx_sum += gyro_raw[0];
    gy_sum += gyro_raw[1];
    gz_sum += gyro_raw[2];
    ax_sum += accel_raw[0];
    ay_sum += accel_raw[1];
    az_sum += accel_raw[2];
    delayMicroseconds(500);
  }

  gyro_offset_x = gx_sum / SAMPLES;
  gyro_offset_y = gy_sum / SAMPLES;
  gyro_offset_z = gz_sum / SAMPLES;
  accel_offset_x = ax_sum / SAMPLES;
  accel_offset_y = ay_sum / SAMPLES;
  // Z accel offset: subtract 1g (16384 LSB at ±2g range)
  accel_offset_z = (az_sum / SAMPLES) - 16384.0f;

  Serial.printf("Gyro offsets:  %.1f  %.1f  %.1f\n", gyro_offset_x,
                gyro_offset_y, gyro_offset_z);
  Serial.printf("Accel offsets: %.1f  %.1f  %.1f\n", accel_offset_x,
                accel_offset_y, accel_offset_z);
  Serial.println("Calibration DONE ✓");
}

// ============================================================================
// COMPLEMENTARY FILTER — Fuse gyro + accelerometer for stable angle
// ============================================================================

void updateAngles(float dt) {
  // Apply calibration offsets
  float gx = (gyro_raw[0] - gyro_offset_x) / 65.5f; // °/s (±500°/s range)
  float gy = (gyro_raw[1] - gyro_offset_y) / 65.5f;
  float gz = (gyro_raw[2] - gyro_offset_z) / 65.5f;

  float ax = accel_raw[0] - accel_offset_x;
  float ay = accel_raw[1] - accel_offset_y;
  float az = accel_raw[2] - accel_offset_z;

  // Accelerometer-based angle (only valid when not accelerating)
  float accel_roll = atan2(ay, az) * 57.2958f; // rad to deg
  float accel_pitch = atan2(-ax, sqrt(ay * ay + az * az)) * 57.2958f;

  // Complementary filter: trust gyro short-term, accel long-term
  angle_roll = COMP_FILTER_ALPHA * (angle_roll + gx * dt) +
               (1.0f - COMP_FILTER_ALPHA) * accel_roll;
  angle_pitch = COMP_FILTER_ALPHA * (angle_pitch + gy * dt) +
                (1.0f - COMP_FILTER_ALPHA) * accel_pitch;

  // Yaw: gyro-only integration (no magnetometer to correct drift)
  angle_yaw += gz * dt;
}

// ============================================================================
// PID CONTROLLER (Inner Attitude Loop)
// ============================================================================

float pidCompute(float setpoint, float measured, float kp, float ki, float kd,
                 float &integral, float &prev_error, float dt,
                 float integral_limit = 50.0f) {
  float error = setpoint - measured;

  integral += error * dt;
  integral = constrain(integral, -integral_limit, integral_limit);

  float derivative = (dt > 0) ? (error - prev_error) / dt : 0.0f;
  prev_error = error;

  return kp * error + ki * integral + kd * derivative;
}

// ============================================================================
// MOTOR MIXING (X-Configuration)
// ============================================================================
//
// Motor layout (top view):
//    M1(CCW) ╲    ╱ M2(CW)
//              ╲╱
//              ╱╲
//    M4(CW)  ╱    ╲ M3(CCW)
//
// Positive roll  = tilt RIGHT  → M1,M4 speed up; M2,M3 slow down
// Positive pitch = tilt FORWARD → M1,M2 speed up; M3,M4 slow down
// Positive yaw   = rotate CW   → CW motors (M2,M4) speed up; CCW (M1,M3) slow
// down
//

void mixMotors(float throttle, float roll_out, float pitch_out, float yaw_out) {
  if (!armed || throttle < 10) {
    motorsOff();
    return;
  }

  int m1 = (int)(throttle + pitch_out + roll_out - yaw_out); // FL (CCW)
  int m2 = (int)(throttle + pitch_out - roll_out + yaw_out); // FR (CW)
  int m3 = (int)(throttle - pitch_out - roll_out - yaw_out); // BR (CCW)
  int m4 = (int)(throttle - pitch_out + roll_out + yaw_out); // BL (CW)

  // Keep motors spinning when armed (prevents dead-band stalls)
  if (throttle > MOTOR_IDLE) {
    m1 = max(m1, MOTOR_IDLE);
    m2 = max(m2, MOTOR_IDLE);
    m3 = max(m3, MOTOR_IDLE);
    m4 = max(m4, MOTOR_IDLE);
  }

  setMotors(m1, m2, m3, m4);
}

// ============================================================================
// WIFI & UDP SETUP
// ============================================================================

void setupWiFi() {
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.printf("WiFi AP started: %s\n", AP_SSID);
  Serial.printf("IP: %s\n", WiFi.softAPIP().toString().c_str());

  udp.begin(UDP_PORT);
  Serial.printf("UDP listening on port %d\n", UDP_PORT);
}

// Special command values for arm/disarm
const int16_t ARM_MAGIC = 999;
const int16_t DISARM_MAGIC = -999;

void receiveUDP() {
  int packetSize = udp.parsePacket();
  if (packetSize >= (int)sizeof(Command)) {
    Command newCmd;
    udp.read((uint8_t *)&newCmd, sizeof(Command));
    last_cmd_time = millis();

    // Check for arm/disarm commands
    if (newCmd.throttle == ARM_MAGIC && newCmd.roll == ARM_MAGIC) {
      armed = true;
      // Reset PID state on arm
      roll_integral = pitch_integral = 0;
      roll_prev_error = pitch_prev_error = yaw_prev_error = 0;
      Serial.println(">>> ARMED <<<");
      return;
    }
    if (newCmd.throttle == DISARM_MAGIC || newCmd.roll == DISARM_MAGIC) {
      armed = false;
      motorsOff();
      Serial.println(">>> DISARMED <<<");
      return;
    }

    cmd = newCmd;
  }
}

// ============================================================================
// BATTERY MONITORING
// ============================================================================

float readBattery() {
  // Assumes a voltage divider: VBAT ──[10kΩ]──┬──[10kΩ]── GND
  //                                            └── ADC pin
  // So ADC reads half of VBAT.
  int raw = analogRead(VBAT_PIN);
  float voltage = (raw / 4095.0f) * 3.3f * 2.0f; // ×2 for voltage divider
  return voltage;
}

// ============================================================================
// TELEMETRY (Serial debug output, throttled)
// ============================================================================
unsigned long last_telemetry = 0;

void sendTelemetry() {
  if (millis() - last_telemetry < 200)
    return; // 5Hz telemetry
  last_telemetry = millis();

  Serial.printf("R:%6.1f P:%6.1f Y:%6.1f | cmd[%4d %4d %4d %4d] | M[%3d %3d "
                "%3d %3d] | %.2fV %s\n",
                angle_roll, angle_pitch, angle_yaw, cmd.roll, cmd.pitch,
                cmd.yaw, cmd.throttle, motor_fl, motor_fr, motor_br, motor_bl,
                vbat, armed ? "ARMED" : "SAFE");
}

// ============================================================================
// STATUS LED
// ============================================================================
unsigned long last_blink = 0;
bool led_state = false;

void updateLED() {
  unsigned long now = millis();
  int interval = armed ? 100 : 500; // fast blink when armed, slow when disarmed
  if (now - last_blink > (unsigned long)interval) {
    led_state = !led_state;
    digitalWrite(LED_PIN, led_state);
    last_blink = now;
  }
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("=================================");
  Serial.println("  ESP32 Micro Drone Controller");
  Serial.println("=================================");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  // 1. Initialize motors (all off)
  setupMotors();
  Serial.println("[OK] Motors initialized");

  // 2. Initialize MPU6050
  setupMPU6050();
  Serial.println("[OK] MPU6050 initialized");

  // 3. Calibrate IMU (drone must be still!)
  calibrateIMU();

  // 4. Start WiFi AP
  setupWiFi();

  Serial.println("\n>>> READY — Waiting for laptop connection <<<");
  Serial.println(">>> Send ARM command to enable motors <<<\n");

  loop_timer = micros();
}

// ============================================================================
// MAIN LOOP (~500Hz inner attitude stabilization)
// ============================================================================

void loop() {
  // ---- Receive any incoming UDP commands ----
  receiveUDP();

  // ---- Read IMU ----
  readMPU6050();

  // ---- Compute dt ----
  unsigned long now_us = micros();
  float dt = (now_us - loop_timer) / 1000000.0f;
  loop_timer = now_us;

  // Sanity clamp dt (e.g., after Serial prints cause a long frame)
  if (dt > 0.02f)
    dt = 0.02f;
  if (dt < 0.0001f)
    dt = 0.001f;

  // ---- Update angle estimates ----
  updateAngles(dt);

  // ---- Failsafe checks ----
  // 1. Lost link: no UDP packets for FAILSAFE_TIMEOUT_MS
  if (armed && (millis() - last_cmd_time > FAILSAFE_TIMEOUT_MS)) {
    Serial.println("!!! FAILSAFE: LOST LINK — DISARMING !!!");
    armed = false;
    motorsOff();
    cmd = {0, 0, 0, 0};
  }

  // 2. Low battery
  static unsigned long last_vbat_check = 0;
  if (millis() - last_vbat_check > 1000) {
    vbat = readBattery();
    last_vbat_check = millis();
    if (vbat < VBAT_CUTOFF && vbat > 1.0f) { // > 1.0 to ignore disconnected ADC
      Serial.printf("!!! LOW BATTERY: %.2fV — DISARMING !!!\n", vbat);
      armed = false;
      motorsOff();
    }
  }

  // 3. Extreme angle (flipped over)
  if (armed && (fabs(angle_roll) > 60.0f || fabs(angle_pitch) > 60.0f)) {
    Serial.println("!!! FAILSAFE: EXTREME ANGLE — DISARMING !!!");
    armed = false;
    motorsOff();
    cmd = {0, 0, 0, 0};
  }

  // ---- Compute inner-loop PID ----
  if (armed && cmd.throttle > 10) {
    // Convert laptop command (-100..+100) to desired angle
    // (-MAX_ANGLE..+MAX_ANGLE)
    float desired_roll = (cmd.roll / 100.0f) * MAX_ANGLE_CMD;
    float desired_pitch = (cmd.pitch / 100.0f) * MAX_ANGLE_CMD;
    float desired_yaw_rate = (cmd.yaw / 100.0f) * 200.0f; // °/s

    // Gyro rates for D-term (smoother than angle derivative)
    float gyro_x = (gyro_raw[0] - gyro_offset_x) / 65.5f;
    float gyro_y = (gyro_raw[1] - gyro_offset_y) / 65.5f;
    float gyro_z = (gyro_raw[2] - gyro_offset_z) / 65.5f;

    // PID on angle error, but use gyro rate for derivative (angle-rate PID)
    float roll_error = desired_roll - angle_roll;
    float pitch_error = desired_pitch - angle_pitch;
    float yaw_error = desired_yaw_rate - gyro_z;

    float roll_out = pidCompute(desired_roll, angle_roll, roll_kp, roll_ki,
                                roll_kd, roll_integral, roll_prev_error, dt);
    float pitch_out =
        pidCompute(desired_pitch, angle_pitch, pitch_kp, pitch_ki, pitch_kd,
                   pitch_integral, pitch_prev_error, dt);
    float yaw_out = yaw_kp * yaw_error; // P-only for yaw rate is usually enough

    // Scale throttle command (0-1000) to motor range
    float throttle_scaled =
        (cmd.throttle / 1000.0f) * (MOTOR_MAX - MOTOR_IDLE) + MOTOR_IDLE;

    // Mix and apply
    mixMotors(throttle_scaled, roll_out, pitch_out, yaw_out);
  } else {
    motorsOff();
  }

  // ---- Telemetry & LED ----
  sendTelemetry();
  updateLED();

  // ---- Loop rate control ----
  // Target: 2000µs per loop = 500Hz
  // If we're faster, busy-wait to maintain consistent dt
  while (micros() - now_us < 2000)
    ;
}
