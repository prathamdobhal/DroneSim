"""A simple PID controller with anti-windup, used one instance per axis."""


class PID:
    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=50.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._has_prev = False

    def reset(self):
        """Call this when the marker is lost, so stale integral/derivative
        terms don't cause a lurch when tracking resumes."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._has_prev = False

    def update(self, error, dt):
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

        derivative = 0.0
        if self._has_prev and dt > 0:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error
        self._has_prev = True

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        if self.output_limit is not None:
            output = max(-self.output_limit, min(self.output_limit, output))
        return output
