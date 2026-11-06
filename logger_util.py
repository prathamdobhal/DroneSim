"""CSV logging utilities for the drone swarm simulation.

ErrorLogger  — original single-follower logger (Phase 1, kept for compat).
SwarmLogger  — Phase 2: logs aggregate + per-drone stats for N followers.
"""
import csv
import time


class ErrorLogger:
    """Single-follower logger from Phase 1 — kept for backward compatibility."""

    def __init__(self, path="tracking_log.csv"):
        self.path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["t", "error_x", "error_y", "error_magnitude", "marker_visible"])
        self._start = time.time()

    def log(self, error_x, error_y, visible):
        t = time.time() - self._start
        mag = (error_x ** 2 + error_y ** 2) ** 0.5
        self._writer.writerow([
            f"{t:.3f}", f"{error_x:.2f}", f"{error_y:.2f}",
            f"{mag:.2f}", int(visible),
        ])

    def close(self):
        self._file.close()


class SwarmLogger:
    """
    Logs per-timestep swarm statistics to a CSV for N followers.

    Columns:
      t              — elapsed time (s)
      mean_error     — mean formation error magnitude across all followers (world units)
      max_error      — worst-case formation error magnitude
      n_visible      — number of followers currently locked onto the leader marker
      error_00 …     — individual error magnitude for each follower
    """

    def __init__(self, n_followers: int, path: str = "tracking_log.csv"):
        self.n = n_followers
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        per_drone_cols = [f"error_{i:02d}" for i in range(n_followers)]
        self._writer.writerow(
            ["t", "mean_error", "max_error", "n_visible"] + per_drone_cols
        )
        self._start = time.time()

    def log(self, errors: list, visibles: list):
        """
        Parameters
        ----------
        errors   : list of (error_x, error_y) tuples, one per follower
        visibles : list of bool, one per follower
        """
        t = time.time() - self._start
        mags = [(ex ** 2 + ey ** 2) ** 0.5 for ex, ey in errors]
        mean_e = sum(mags) / len(mags)
        max_e  = max(mags)
        n_vis  = sum(visibles)
        row = (
            [f"{t:.3f}", f"{mean_e:.2f}", f"{max_e:.2f}", n_vis]
            + [f"{m:.2f}" for m in mags]
        )
        self._writer.writerow(row)

    def close(self):
        self._file.close()
