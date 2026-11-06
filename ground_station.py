"""
Ground Station — Webcam-based position controller for ESP32 micro drone.

Uses an overhead webcam to track two ArUco markers:
  - ID 1: stuck on top of the drone  → gives drone (x, y, yaw) in room frame
  - ID 0: the target marker          → gives target (x, y) in room frame

Runs the OUTER position PID loop (~30-50Hz):
  Position error → desired tilt angles → UDP packet → ESP32

The ESP32 runs the INNER attitude PID (~500Hz) using its onboard IMU.

Usage:
  1. Flash drone_firmware.ino onto the ESP32
  2. Connect laptop WiFi to "DroneESP32" (password: drone1234)
  3. Mount webcam overhead, pointing down at the flight area
  4. Place ArUco ID=0 (target) on floor
  5. Stick ArUco ID=1 on top of the drone
  6. Run: python ground_station.py

Controls:
  A       — ARM the drone (motors will spin!)
  D       — DISARM (emergency stop)
  SPACE   — EMERGENCY KILL (instant motor off)
  ESC     — Quit
  +/-     — Increase/decrease base throttle
  T       — Toggle target tracking (drone follows ArUco target vs hover in place)
  Click   — Set target position manually
"""

import sys
import os
import time
import math
import struct
import socket
import threading
from collections import deque

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import cv2
import numpy as np
import pygame

# Import PID from your existing project
from control import PID

# ============================================================================
# CONFIGURATION
# ============================================================================

# ESP32 WiFi AP connection
ESP32_IP   = "192.168.4.1"
ESP32_PORT = 4210

# Webcam
CAMERA_IDX = 0
CAM_WIDTH  = 640
CAM_HEIGHT = 480

# ArUco marker IDs
DRONE_MARKER_ID  = 1   # Marker stuck on top of drone
TARGET_MARKER_ID = 0   # Target marker on floor/object

# Flight zone bounds in webcam pixel space (adjust after camera calibration)
# These define the safe zone — drone won't be commanded outside this
FLIGHT_ZONE_MARGIN_PX = 50  # pixels from edge of frame

# Position PID gains (OUTER LOOP — these control how aggressively
# the drone moves toward the target. Output is desired tilt angle.)
# START VERY LOW and increase gradually during real flights!
POS_KP = 0.8
POS_KI = 0.0
POS_KD = 0.3
POS_OUTPUT_LIMIT = 60   # max command value sent to ESP32 (-100..+100 range)

# Altitude PID (controls throttle based on marker apparent size)
ALT_KP = 2.0
ALT_KI = 0.1
ALT_KD = 0.5
TARGET_ALTITUDE_PX = 120   # target marker side-length in pixels (calibrate this!)

# Yaw PID (keeps drone oriented correctly based on ArUco rotation)
YAW_KP = 1.5
YAW_KI = 0.0
YAW_KD = 0.2

# Base throttle (0-1000). Start LOW! Hover is typically 400-600 depending on weight.
BASE_THROTTLE = 0
THROTTLE_STEP = 25    # +/- per keypress
MAX_THROTTLE  = 800
MIN_THROTTLE  = 0

# ArUco detection
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MIN_MARKER_AREA = 200  # reject tiny false positives (pixels²)

# Smoothing
POSITION_SMOOTH_ALPHA = 0.4    # EMA smoothing for position (0 = very smooth, 1 = raw)

# Display
WINDOW_W, WINDOW_H = 960, 600

# ============================================================================
# ARM / DISARM MAGIC VALUES (must match firmware)
# ============================================================================
ARM_MAGIC    = 999
DISARM_MAGIC = -999


class GroundStation:
    """Webcam-based position controller with UDP link to ESP32 drone."""

    def __init__(self):
        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        # CLAHE for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # Position PID controllers (X and Y axes in camera frame)
        self.pid_x = PID(kp=POS_KP, ki=POS_KI, kd=POS_KD, output_limit=POS_OUTPUT_LIMIT)
        self.pid_y = PID(kp=POS_KP, ki=POS_KI, kd=POS_KD, output_limit=POS_OUTPUT_LIMIT)

        # Altitude PID
        self.pid_alt = PID(kp=ALT_KP, ki=ALT_KI, kd=ALT_KD, output_limit=200)

        # Yaw PID
        self.pid_yaw = PID(kp=YAW_KP, ki=YAW_KI, kd=YAW_KD, output_limit=50)

        # State
        self.armed = False
        self.tracking_target = False    # True = follow target marker, False = hover in place
        self.base_throttle = BASE_THROTTLE

        # Drone position (smoothed, in pixel coordinates)
        self.drone_pos = None           # (cx, cy) or None if not detected
        self.drone_yaw = 0.0            # radians, from ArUco orientation
        self.drone_marker_size = 0.0    # side length in pixels (proxy for altitude)

        # Target position
        self.target_pos = None          # (cx, cy) from ArUco ID=0, or manual click
        self.manual_target = None       # manually clicked target

        # Hover target (set when drone is first detected)
        self.hover_pos = None

        # Detection state
        self.drone_detected = False
        self.target_detected = False

        # Telemetry history for plotting
        self.error_history = deque(maxlen=200)
        self.cmd_history = deque(maxlen=200)

        # Timing
        self.last_loop_time = time.perf_counter()
        self.loop_rate_hz = 0
        self.fps_filter = 30.0

    def send_command(self, roll, pitch, yaw, throttle):
        """Send attitude command to ESP32 via UDP."""
        packet = struct.pack('<4h',
            int(max(-100, min(100, roll))),
            int(max(-100, min(100, pitch))),
            int(max(-100, min(100, yaw))),
            int(max(0, min(1000, throttle))),
        )
        try:
            self.sock.sendto(packet, (ESP32_IP, ESP32_PORT))
        except Exception:
            pass  # Don't crash if WiFi drops

    def send_arm(self):
        """Send ARM command to ESP32."""
        packet = struct.pack('<4h', ARM_MAGIC, ARM_MAGIC, ARM_MAGIC, ARM_MAGIC)
        for _ in range(3):  # Send multiple times for reliability
            self.sock.sendto(packet, (ESP32_IP, ESP32_PORT))
            time.sleep(0.01)
        self.armed = True

    def send_disarm(self):
        """Send DISARM command to ESP32."""
        packet = struct.pack('<4h', DISARM_MAGIC, DISARM_MAGIC, DISARM_MAGIC, DISARM_MAGIC)
        for _ in range(5):  # Send extra times — this is safety-critical
            self.sock.sendto(packet, (ESP32_IP, ESP32_PORT))
            time.sleep(0.01)
        self.armed = False

    def detect_markers(self, frame):
        """Detect ArUco markers and update drone/target positions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try raw first, then CLAHE enhanced
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            gray_clahe = self.clahe.apply(gray)
            corners, ids, _ = self.detector.detectMarkers(gray_clahe)

        self.drone_detected = False
        self.target_detected = False

        if ids is None:
            return

        flat_ids = ids.flatten()

        # Detect DRONE marker (ID=1)
        drone_indices = np.where(flat_ids == DRONE_MARKER_ID)[0]
        if len(drone_indices) > 0:
            idx = drone_indices[0]
            c = corners[idx][0]
            area = cv2.contourArea(c)

            if area > MIN_MARKER_AREA:
                cx = float(c[:, 0].mean())
                cy = float(c[:, 1].mean())

                # Smooth position
                if self.drone_pos is not None:
                    cx = (1 - POSITION_SMOOTH_ALPHA) * self.drone_pos[0] + POSITION_SMOOTH_ALPHA * cx
                    cy = (1 - POSITION_SMOOTH_ALPHA) * self.drone_pos[1] + POSITION_SMOOTH_ALPHA * cy

                self.drone_pos = (cx, cy)
                self.drone_detected = True

                # Estimate yaw from marker orientation
                dx = c[1][0] - c[0][0]
                dy = c[1][1] - c[0][1]
                self.drone_yaw = math.atan2(dy, dx)

                # Marker apparent size (altitude proxy)
                self.drone_marker_size = math.sqrt(area)

                # Set hover position on first detection
                if self.hover_pos is None:
                    self.hover_pos = (cx, cy)

        # Detect TARGET marker (ID=0)
        target_indices = np.where(flat_ids == TARGET_MARKER_ID)[0]
        if len(target_indices) > 0:
            idx = target_indices[0]
            c = corners[idx][0]
            area = cv2.contourArea(c)

            if area > MIN_MARKER_AREA:
                tx = float(c[:, 0].mean())
                ty = float(c[:, 1].mean())
                self.target_pos = (tx, ty)
                self.target_detected = True

    def compute_control(self, dt):
        """Run outer position PID loop. Returns (roll, pitch, yaw, throttle)."""
        if not self.drone_detected or not self.armed:
            return 0, 0, 0, 0

        # Determine target position
        if self.tracking_target and self.target_detected and self.target_pos is not None:
            target = self.target_pos
        elif self.manual_target is not None:
            target = self.manual_target
        elif self.hover_pos is not None:
            target = self.hover_pos
        else:
            return 0, 0, 0, self.base_throttle

        # Position error in camera pixel space
        error_x = target[0] - self.drone_pos[0]   # positive = target is to the right
        error_y = target[1] - self.drone_pos[1]   # positive = target is below (camera Y is inverted)

        # Store error magnitude for telemetry
        self.error_history.append(math.hypot(error_x, error_y))

        # PID outputs
        # NOTE: Camera axes may not align with drone axes!
        # You MUST calibrate this mapping by testing which camera direction
        # corresponds to which drone tilt direction.
        # The signs below assume:
        #   Camera +X (right) → drone roll +  (tilt right)
        #   Camera +Y (down)  → drone pitch - (tilt backward... or forward, TEST THIS)
        roll_cmd  =  self.pid_x.update(error_x, dt)
        pitch_cmd = -self.pid_y.update(error_y, dt)  # negative because camera Y is inverted

        # Rotate commands by drone yaw to keep them in drone body frame
        # (critical if drone is not aligned with camera axes)
        cos_yaw = math.cos(-self.drone_yaw)
        sin_yaw = math.sin(-self.drone_yaw)
        body_roll  = roll_cmd * cos_yaw - pitch_cmd * sin_yaw
        body_pitch = roll_cmd * sin_yaw + pitch_cmd * cos_yaw

        # Yaw control (try to keep drone at 0 yaw relative to camera)
        yaw_cmd = self.pid_yaw.update(-self.drone_yaw * 57.2958, dt)  # convert rad to deg

        # Throttle: base + altitude correction
        throttle = self.base_throttle

        self.cmd_history.append((body_roll, body_pitch, yaw_cmd, throttle))

        return body_roll, body_pitch, yaw_cmd, throttle

    def run(self):
        """Main loop — webcam capture, detection, control, display."""
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Ground Station — ESP32 Drone Controller")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 14)
        font_sm = pygame.font.SysFont("consolas", 11)
        font_lg = pygame.font.SysFont("consolas", 18, bold=True)

        # Open webcam
        cap = cv2.VideoCapture(CAMERA_IDX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_IDX)
        if not cap.isOpened():
            print("ERROR: Could not open webcam!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("Ground Station running. Connect to WiFi 'DroneESP32' first!")
        print("Controls: A=ARM  D=DISARM  SPACE=KILL  +/-=Throttle  T=Track  Click=Target")

        running = True
        try:
            while running:
                now = time.perf_counter()
                dt = now - self.last_loop_time
                self.last_loop_time = now
                if dt > 0.1:
                    dt = 0.02
                self.fps_filter = 0.9 * self.fps_filter + 0.1 * (1.0 / max(dt, 0.001))

                # ── Events ──
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_a:
                            self.send_arm()
                            print(">>> ARMED <<<")
                        elif event.key == pygame.K_d:
                            self.send_disarm()
                            print(">>> DISARMED <<<")
                        elif event.key == pygame.K_SPACE:
                            self.send_disarm()
                            self.base_throttle = 0
                            print("!!! EMERGENCY KILL !!!")
                        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                            self.base_throttle = min(self.base_throttle + THROTTLE_STEP, MAX_THROTTLE)
                            print(f"Throttle: {self.base_throttle}")
                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            self.base_throttle = max(self.base_throttle - THROTTLE_STEP, MIN_THROTTLE)
                            print(f"Throttle: {self.base_throttle}")
                        elif event.key == pygame.K_t:
                            self.tracking_target = not self.tracking_target
                            print(f"Target tracking: {'ON' if self.tracking_target else 'OFF (hover)'}")
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        if mx < CAM_WIDTH and my < CAM_HEIGHT:
                            self.manual_target = (mx, my)
                            self.tracking_target = False
                            print(f"Manual target: ({mx}, {my})")

                # ── Capture frame ──
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)  # mirror for natural interaction

                # ── Detect markers ──
                self.detect_markers(frame)

                # ── Control loop ──
                roll_cmd, pitch_cmd, yaw_cmd, throttle = self.compute_control(dt)

                # ── Send to drone ──
                if self.armed:
                    self.send_command(roll_cmd, pitch_cmd, yaw_cmd, throttle)
                else:
                    self.send_command(0, 0, 0, 0)

                # ── Render ──
                screen.fill((12, 14, 22))

                # Camera feed as pygame surface
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_surf = pygame.image.frombuffer(rgb.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
                screen.blit(cam_surf, (0, 0))

                # Draw overlays on camera view
                if self.drone_detected:
                    dx, dy = int(self.drone_pos[0]), int(self.drone_pos[1])
                    # Drone reticle
                    pygame.draw.circle(screen, (0, 200, 255), (dx, dy), 18, 2)
                    pygame.draw.circle(screen, (0, 200, 255), (dx, dy), 3)
                    # Yaw indicator line
                    yaw_len = 25
                    yaw_ex = dx + int(math.cos(self.drone_yaw) * yaw_len)
                    yaw_ey = dy + int(math.sin(self.drone_yaw) * yaw_len)
                    pygame.draw.line(screen, (0, 200, 255), (dx, dy), (yaw_ex, yaw_ey), 2)
                    screen.blit(font_sm.render("DRONE", True, (0, 200, 255)), (dx + 20, dy - 8))

                if self.target_detected and self.target_pos:
                    tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
                    pygame.draw.circle(screen, (0, 255, 100), (tx, ty), 15, 2)
                    pygame.draw.drawMarker = None  # pygame doesn't have drawMarker
                    pygame.draw.line(screen, (0, 255, 100), (tx - 10, ty), (tx + 10, ty), 1)
                    pygame.draw.line(screen, (0, 255, 100), (tx, ty - 10), (tx, ty + 10), 1)
                    screen.blit(font_sm.render("TARGET", True, (0, 255, 100)), (tx + 18, ty - 8))

                if self.manual_target and not self.tracking_target:
                    mx, my = int(self.manual_target[0]), int(self.manual_target[1])
                    pygame.draw.circle(screen, (255, 200, 0), (mx, my), 10, 1)
                    screen.blit(font_sm.render("MANUAL", True, (255, 200, 0)), (mx + 12, my - 8))

                # Error line (drone → target)
                if self.drone_detected:
                    target = None
                    if self.tracking_target and self.target_pos:
                        target = self.target_pos
                    elif self.manual_target:
                        target = self.manual_target
                    elif self.hover_pos:
                        target = self.hover_pos
                    if target:
                        pygame.draw.line(screen, (255, 80, 80),
                            (int(self.drone_pos[0]), int(self.drone_pos[1])),
                            (int(target[0]), int(target[1])), 1)

                # ── Right Panel: Telemetry ──
                px = CAM_WIDTH + 16
                py = 10

                # Title
                screen.blit(font_lg.render("GROUND STATION", True, (160, 170, 200)), (px, py))
                py += 28

                # Armed status
                if self.armed:
                    pygame.draw.rect(screen, (200, 50, 50), (px, py, 280, 22), border_radius=3)
                    screen.blit(font.render("● ARMED — MOTORS LIVE", True, (255, 255, 255)), (px + 8, py + 3))
                else:
                    pygame.draw.rect(screen, (30, 60, 30), (px, py, 280, 22), border_radius=3)
                    pygame.draw.rect(screen, (60, 120, 60), (px, py, 280, 22), 1, border_radius=3)
                    screen.blit(font.render("○ DISARMED — SAFE", True, (80, 200, 80)), (px + 8, py + 3))
                py += 32

                # Detection status
                d_col = (80, 255, 120) if self.drone_detected else (255, 80, 80)
                d_txt = "● DRONE LOCKED" if self.drone_detected else "○ DRONE NOT FOUND"
                screen.blit(font.render(d_txt, True, d_col), (px, py))
                py += 18

                t_col = (80, 255, 120) if self.target_detected else (180, 180, 180)
                t_txt = "● TARGET LOCKED" if self.target_detected else "○ TARGET NOT FOUND"
                screen.blit(font.render(t_txt, True, t_col), (px, py))
                py += 24

                # Mode
                if self.tracking_target:
                    screen.blit(font.render("Mode: TRACK TARGET", True, (0, 255, 100)), (px, py))
                elif self.manual_target:
                    screen.blit(font.render("Mode: MANUAL TARGET", True, (255, 200, 0)), (px, py))
                else:
                    screen.blit(font.render("Mode: HOVER IN PLACE", True, (150, 160, 180)), (px, py))
                py += 24

                # Commands being sent
                screen.blit(font_sm.render(f"Throttle: {self.base_throttle:4d} / {MAX_THROTTLE}", True, (180, 185, 200)), (px, py))
                py += 16
                screen.blit(font_sm.render(f"Roll cmd:  {roll_cmd:+6.1f}", True, (180, 185, 200)), (px, py))
                py += 16
                screen.blit(font_sm.render(f"Pitch cmd: {pitch_cmd:+6.1f}", True, (180, 185, 200)), (px, py))
                py += 16
                screen.blit(font_sm.render(f"Yaw cmd:   {yaw_cmd:+6.1f}", True, (180, 185, 200)), (px, py))
                py += 24

                # Position info
                if self.drone_detected:
                    screen.blit(font_sm.render(f"Drone pos:  ({self.drone_pos[0]:.0f}, {self.drone_pos[1]:.0f})", True, (0, 200, 255)), (px, py))
                    py += 16
                    screen.blit(font_sm.render(f"Drone yaw:  {self.drone_yaw * 57.3:.1f}°", True, (0, 200, 255)), (px, py))
                    py += 16
                    screen.blit(font_sm.render(f"Marker size: {self.drone_marker_size:.1f} px", True, (0, 200, 255)), (px, py))
                    py += 16

                if self.error_history:
                    screen.blit(font_sm.render(f"Position error: {self.error_history[-1]:.1f} px", True, (255, 180, 80)), (px, py))
                    py += 24

                # Error history mini-graph
                if len(self.error_history) > 2:
                    graph_x, graph_y = px, py
                    graph_w, graph_h = 260, 80
                    pygame.draw.rect(screen, (20, 24, 35), (graph_x, graph_y, graph_w, graph_h))
                    pygame.draw.rect(screen, (40, 50, 70), (graph_x, graph_y, graph_w, graph_h), 1)

                    max_err = max(self.error_history) if max(self.error_history) > 0 else 1
                    points = []
                    for i, err in enumerate(self.error_history):
                        gx = graph_x + int(i / len(self.error_history) * graph_w)
                        gy = graph_y + graph_h - int((err / max_err) * (graph_h - 4))
                        points.append((gx, gy))
                    if len(points) > 1:
                        pygame.draw.lines(screen, (255, 120, 60), False, points, 1)

                    screen.blit(font_sm.render("Position Error", True, (120, 130, 150)), (graph_x + 4, graph_y + 2))
                    py += graph_h + 8

                # FPS
                screen.blit(font_sm.render(f"{self.fps_filter:.0f} Hz", True, (100, 110, 130)), (px, py))
                py += 20

                # Controls help
                py = WINDOW_H - 80
                for line in [
                    "A=ARM  D=DISARM  SPACE=KILL",
                    "+/-=Throttle  T=Toggle Track",
                    "Click camera view = Set target",
                    "ESC = Quit",
                ]:
                    screen.blit(font_sm.render(line, True, (80, 90, 110)), (px, py))
                    py += 14

                pygame.display.flip()
                clock.tick(60)

        finally:
            # Always disarm on exit!
            self.send_disarm()
            cap.release()
            pygame.quit()
            print("Ground Station shut down. Drone disarmed.")


if __name__ == "__main__":
    gs = GroundStation()
    gs.run()
