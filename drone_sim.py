"""drone_sim.py —
Leader and Follower drone kinematics for the swarm simulation."""
import math


class Leader:
    """Leader drone that flies either an elliptical orbit,
    or smoothly tracks a real-time target commanded by the webcam tracker."""

    def __init__(self, center=(520, 350), radius=140, speed=0.5, max_speed=190.0):
        self.center = center
        self.radius = radius
        self.speed = speed  # angular speed (rad/s)
        self.t = 0.0
        self.x, self.y = float(self.center[0]), float(self.center[1])
        self.vx = 0.0
        self.vy = 0.0
        self.max_speed = max_speed
        self.target_x, self.target_y = self.x, self.y
        self.mode = "webcam"  # "webcam" or "orbit"

        # Horizontal elliptical orbit: 2:1 width-to-height ratio.
        self.orbit_radius_x = float(radius)
        self.orbit_radius_y = float(radius) * 0.55

    def set_mode(self, new_mode):
        """Seamlessly transitions between webcam and orbit modes without coordinate teleportation."""
        if new_mode == self.mode:
            return
        self.mode = new_mode
        if self.mode == "orbit":
            # Re-center the ellipse around current position to prevent teleportation.
            self.center = (self.x, self.y)
            self.t = 0.0
        else:
            # Sync target coordinates with current position so leader doesn't jolt toward stale targets.
            self.target_x = self.x
            self.target_y = self.y

    def set_target(self, tx, ty):
        """Update commanded target position from webcam tracker."""
        self.target_x = float(tx)
        self.target_y = float(ty)

    def update_target(self, dt):
        """Smoothly steers the leader toward target_x, target_y with velocity and acceleration limits."""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)

        # Proportional pursuit with soft arrival deceleration.
        kp = 3.5
        desired_vx = dx * kp
        desired_vy = dy * kp
        speed = math.hypot(desired_vx, desired_vy)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            desired_vx *= scale
            desired_vy *= scale

        # Smooth acceleration ramping (first-order low pass).
        accel_alpha = min(1.0, dt * 8.0)
        self.vx = (1.0 - accel_alpha) * self.vx + accel_alpha * desired_vx
        self.vy = (1.0 - accel_alpha) * self.vy + accel_alpha * desired_vy

        self.x += self.vx * dt
        self.y += self.vy * dt

    def update(self, dt):
        if self.mode == "webcam":
            self.update_target(dt)
        else:
            self.t += self.speed * dt

            # Elliptical orbit (not a figure-eight).
            rx = self.orbit_radius_x
            ry = self.orbit_radius_y
            self.x = self.center[0] + rx * math.cos(self.t)
            self.y = self.center[1] + ry * math.sin(self.t)

            # Analytical velocities for correct direction and telemetry.
            self.vx = -rx * math.sin(self.t) * self.speed
            self.vy = ry * math.cos(self.t) * self.speed
            self.target_x, self.target_y = self.x, self.y

    @property
    def position(self):
        return (self.x, self.y)


class Follower:
    """Follower drone driven purely by velocity commands from the control
    loop. It has NO direct access to the leader's position — only to
    whatever the vision pipeline extracts from its camera frame."""

    def __init__(self, start_pos=(200, 300), max_speed=220.0):
        self.x, self.y = float(start_pos[0]), float(start_pos[1])
        self.max_speed = max_speed  # world units per second

    def update(self, vx, vy, dt):
        speed = math.hypot(vx, vy)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
        self.x += vx * dt
        self.y += vy * dt
        return vx, vy

    @property
    def position(self):
        return (self.x, self.y)
