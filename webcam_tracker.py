"""webcam_tracker.py-
Webcam feed capture and ArUco marker tracking for Phase 3.

Runs frame acquisition and marker detection in a dedicated background thread,
allowing the main Pygame simulation loop to run at smooth 60 FPS without blocking
on camera hardware I/O or the initial Windows driver startup delay.
"""
import threading
import time
import cv2
import numpy as np
import pygame


class WebcamTracker:
    """Non-blocking webcam stream with real-time ArUco marker tracking."""

    def __init__(self, camera_idx=0, target_id=0, world_bounds=((100, 860), (80, 620))):
        self.camera_idx = camera_idx
        self.target_id = target_id
        self.world_x_bounds = world_bounds[0]
        self.world_y_bounds = world_bounds[1]

        # Multi-dictionary detector setup
        # Primary: DICT_4X4_50 (matches the marker we generate)
        # Fallback: DICT_4X4_100 only — more dictionaries = more false positives
        self.dict_names = [
            "DICT_4X4_50",
            "DICT_4X4_100",
        ]

        def _make_params():
            p = cv2.aruco.DetectorParameters()
            # Wider adaptive threshold search for varying lighting / phone glare
            p.adaptiveThreshWinSizeMin = 3
            p.adaptiveThreshWinSizeMax = 53
            p.adaptiveThreshWinSizeStep = 4
            p.adaptiveThreshConstant = 7
            # Accept both small and large markers in the frame
            p.minMarkerPerimeterRate = 0.02
            p.maxMarkerPerimeterRate = 4.0
            # Moderate polygon approximation (default 0.05 — reject noisy quads)
            p.polygonalApproxAccuracyRate = 0.05
            p.minDistanceToBorder = 0
            # Middle ground: reject random face edges but tolerate phone-screen glare
            p.maxErroneousBitsInBorderRate = 0.4
            p.errorCorrectionRate = 0.7
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            p.cornerRefinementWinSize = 5
            p.cornerRefinementMaxIterations = 40
            return p

        self.detectors = []
        for name in self.dict_names:
            try:
                d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
                self.detectors.append((name, cv2.aruco.ArucoDetector(d, _make_params())))
            except Exception:
                pass

        # Primary detector for fast path
        self.detector = self.detectors[0][1] if self.detectors else None

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) for phone screen glare
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # Threading state
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Shared state
        self.cap = None
        self.camera_available = False
        self.is_tracking = False
        self.status_message = "CONNECTING TO WEBCAM (DEVICE 0)..."
        self.marker_corners = None
        self.pixel_pos = None     # (cx, cy) in webcam image
        self.world_target = (
            (world_bounds[0][0] + world_bounds[0][1]) / 2.0,
            (world_bounds[1][0] + world_bounds[1][1]) / 2.0,
        )
        self.last_detection_time = 0.0

        # Preview geometry
        self.preview_w = 320
        self.preview_h = 240
        self.world_w = 960
        self.world_h = 700

        # Ensure font subsystem is initialized for placeholder surfaces
        if not pygame.font.get_init():
            pygame.font.init()

        # Pre-build initial placeholder surfaces (high-contrast cyber theme so never plain black)
        self.preview_surface = self._make_preview_placeholder(
            "CONNECTING WEBCAM...", "Device 0 initializing"
        )
        self.world_surface = self._make_world_placeholder(
            "INITIALIZING LIVE WEBCAM FEED...",
            "Holding for camera exposure (usually < 1 sec)..."
        )

        # Multi-dict match state
        self.matched_dict = None
        self.matched_id = None

        # Minimum marker area (pixels²) — reject tiny false-positive quads
        self.min_marker_area = 250

        # World surface throttle — only regenerate every N frames to reduce lag
        self._frame_counter = 0
        self._world_surface_interval = 3

        # Hand tremor smoothing filter (0.0 to 1.0)
        self.smooth_alpha = 0.35

    def _make_preview_placeholder(self, text, subtext=""):
        surf = pygame.Surface((self.preview_w, self.preview_h))
        surf.fill((22, 26, 38))
        font = pygame.font.SysFont("consolas", 13, bold=True)
        font_sm = pygame.font.SysFont("consolas", 11)
        lbl = font.render(text, True, (0, 220, 255))
        surf.blit(lbl, (self.preview_w // 2 - lbl.get_width() // 2, self.preview_h // 2 - 14))
        if subtext:
            lbl2 = font_sm.render(subtext, True, (150, 165, 195))
            surf.blit(lbl2, (self.preview_w // 2 - lbl2.get_width() // 2, self.preview_h // 2 + 10))
        return surf

    def _make_world_placeholder(self, text, subtext=""):
        surf = pygame.Surface((self.world_w, self.world_h))
        surf.fill((18, 22, 34))
        # Draw tech grid
        for gx in range(0, self.world_w, 60):
            pygame.draw.line(surf, (26, 32, 48), (gx, 0), (gx, self.world_h), 1)
        for gy in range(0, self.world_h, 60):
            pygame.draw.line(surf, (26, 32, 48), (0, gy), (self.world_w, gy), 1)

        font = pygame.font.SysFont("consolas", 20, bold=True)
        font_sm = pygame.font.SysFont("consolas", 14)
        lbl = font.render(text, True, (0, 230, 160))
        surf.blit(lbl, (self.world_w // 2 - lbl.get_width() // 2, self.world_h // 2 - 20))
        if subtext:
            lbl2 = font_sm.render(subtext, True, (160, 175, 205))
            surf.blit(lbl2, (self.world_w // 2 - lbl2.get_width() // 2, self.world_h // 2 + 16))
        return surf

    def start(self):
        """Starts the background camera acquisition thread."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the thread and releases the camera."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _open_camera(self):
        """Attempts to open camera using fast DirectShow backend first, then fallback."""
        try:
            cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            else:
                cap.release()
        except Exception:
            pass

        try:
            cap = cv2.VideoCapture(self.camera_idx)
            if cap.isOpened():
                return cap
            else:
                cap.release()
        except Exception:
            pass

        return None

    def _capture_loop(self):
        """Background loop continuously capturing and processing frames."""
        self.cap = self._open_camera()

        if self.cap is None or not self.cap.isOpened():
            with self.lock:
                self.camera_available = False
                self.status_message = "WEBCAM NOT DETECTED (AUTO-FALLBACK)"
                self.preview_surface = self._make_preview_placeholder(
                    "WEBCAM NOT DETECTED", "Press 'W' for Auto-Orbit"
                )
                self.world_surface = self._make_world_placeholder(
                    "WEBCAM NOT DETECTED", "Running in Synthetic Auto-Orbit Mode"
                )
            return

        with self.lock:
            self.camera_available = True
            self.status_message = "WEBCAM CONNECTED - SEARCHING"

        prev_target = np.array(self.world_target, dtype=np.float32)

        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Mirror horizontally for natural mirror behavior
            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            h, w = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found = False
            target_cx, target_cy = None, None
            matched_dict = None
            matched_id = None
            all_rejected = []
            target_corners = None
            target_ids = None

            # 1. Try primary detector on raw gray
            if self.detectors:
                primary_name, primary_det = self.detectors[0]
                corners, ids, rejected = primary_det.detectMarkers(gray)
                if rejected is not None:
                    all_rejected.extend(rejected)
                if ids is not None and len(ids) > 0:
                    found = True
                    matched_dict = primary_name
                    target_corners = corners
                    target_ids = ids

            # 2. If not found, try CLAHE enhanced gray (handles phone screens & glare)
            if not found and self.detectors:
                gray_clahe = self.clahe.apply(gray)
                primary_name, primary_det = self.detectors[0]
                corners, ids, rejected = primary_det.detectMarkers(gray_clahe)
                if rejected is not None:
                    all_rejected.extend(rejected)
                if ids is not None and len(ids) > 0:
                    found = True
                    matched_dict = primary_name
                    target_corners = corners
                    target_ids = ids

            # 3. Fallback: Check other dictionaries (e.g. DICT_6X6_250, DICT_5X5_100, DICT_4X4_100)
            if not found:
                for d_name, d_det in self.detectors[1:]:
                    corners, ids, rejected = d_det.detectMarkers(gray)
                    if ids is not None and len(ids) > 0:
                        found = True
                        matched_dict = d_name
                        target_corners = corners
                        target_ids = ids
                        break

            if found:
                flat_ids = target_ids.flatten()
                target_idx = 0
                if self.target_id in flat_ids:
                    target_idx = np.where(flat_ids == self.target_id)[0][0]
                matched_id = int(flat_ids[target_idx])

                c = target_corners[target_idx][0]

                # Reject tiny false-positive quads (e.g. facial features)
                area = cv2.contourArea(c)
                if area < self.min_marker_area:
                    found = False

            if found:
                c = target_corners[target_idx][0]
                target_cx = float(c[:, 0].mean())
                target_cy = float(c[:, 1].mean())

                # Draw marker box & reticle
                pts = c.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 240, 100), 2)
                cv2.circle(frame, (int(target_cx), int(target_cy)), 5, (0, 160, 255), -1)
                cv2.drawMarker(frame, (int(target_cx), int(target_cy)), (255, 255, 255),
                               cv2.MARKER_CROSS, 16, 2)
                # Subtle indicator on clean background
                cv2.polylines(clean_frame, [pts], True, (0, 230, 140), 2)

                # Coordinate mapping calibrated to world bounds
                margin_x = (self.world_x_bounds[0] / float(self.world_w)) * w
                margin_y = (self.world_y_bounds[0] / float(self.world_h)) * h
                norm_x = np.clip((target_cx - margin_x) / max(1.0, (w - 2.0 * margin_x)), 0.0, 1.0)
                norm_y = np.clip((target_cy - margin_y) / max(1.0, (h - 2.0 * margin_y)), 0.0, 1.0)

                wx = self.world_x_bounds[0] + norm_x * (self.world_x_bounds[1] - self.world_x_bounds[0])
                wy = self.world_y_bounds[0] + norm_y * (self.world_y_bounds[1] - self.world_y_bounds[0])

                # Smooth hand movement
                curr_target = np.array([wx, wy], dtype=np.float32)
                prev_target = (1.0 - self.smooth_alpha) * prev_target + self.smooth_alpha * curr_target

                # Visual telemetry on preview
                cv2.putText(
                    frame, f"TRACKING: {matched_dict} ID {matched_id}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 240, 100), 2
                )
                cv2.putText(
                    frame, f"World: ({prev_target[0]:.0f}, {prev_target[1]:.0f})",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1
                )
            else:
                # If marker not decoded yet, highlight candidate quads (helpful visual feedback)
                candidate_seen = False
                if all_rejected:
                    # Draw up to 2 largest candidates with faint yellow lines
                    sorted_rej = sorted(all_rejected, key=lambda r: cv2.contourArea(r[0]), reverse=True)
                    for r in sorted_rej[:2]:
                        area = cv2.contourArea(r[0])
                        if area > 800:  # big enough to be a marker
                            pts = r[0].astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame, [pts], True, (60, 200, 255), 1)
                            candidate_seen = True

                if candidate_seen:
                    cv2.putText(
                        frame, "CANDIDATE SEEN - HOLD STEADY",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 200, 255), 2
                    )
                    cv2.putText(
                        frame, "Avoid glare / check quiet-zone margin",
                        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 240), 1
                    )
                else:
                    cv2.putText(
                        frame, "SEARCHING: SHOW ARUCO (ID: 0)",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (60, 140, 255), 2
                    )
                    cv2.putText(
                        frame, "Open 'aruco_marker_id0.png' on phone/screen",
                        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 175, 195), 1
                    )

            # 1. Resize to right-panel preview dimensions (320x240)
            small = cv2.resize(frame, (self.preview_w, self.preview_h))
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            # .copy() is critical: frombuffer() shares the bytes buffer, which
            # becomes invalid once the local goes out of scope on the next
            # iteration. copy() forces the Surface to own its pixel memory.
            surf_preview = pygame.image.frombuffer(
                rgb_small.tobytes(), (self.preview_w, self.preview_h), "RGB"
            ).copy()

            # 2. Resize clean frame to full world view dimensions (960x700) for AR background
            # Throttled to every Nth frame to reduce CPU/memory pressure
            self._frame_counter += 1
            if self._frame_counter % self._world_surface_interval == 0:
                world_img = cv2.resize(clean_frame, (self.world_w, self.world_h))
                rgb_world = cv2.cvtColor(world_img, cv2.COLOR_BGR2RGB)
                surf_world = pygame.image.frombuffer(
                    rgb_world.tobytes(), (self.world_w, self.world_h), "RGB"
                ).copy()
            else:
                surf_world = None

            with self.lock:
                self.camera_available = True
                self.is_tracking = found
                if found:
                    self.status_message = f"TRACKING: {matched_dict} ID {matched_id}"
                    self.pixel_pos = (target_cx, target_cy)
                    self.world_target = (float(prev_target[0]), float(prev_target[1]))
                    self.last_detection_time = time.time()
                else:
                    self.status_message = "CANDIDATE SEEN - HOLD STEADY" if candidate_seen else "SEARCHING FOR MARKER"
                    self.matched_dict = None
                    self.matched_id = None
                self.preview_surface = surf_preview
                if surf_world is not None:
                    self.world_surface = surf_world

        if self.cap:
            self.cap.release()
            self.cap = None

    def get_state(self):
        """Thread-safe query for the main simulation loop."""
        with self.lock:
            return {
                "camera_available": self.camera_available,
                "is_tracking": self.is_tracking,
                "status_message": self.status_message,
                "world_target": self.world_target,
                "pixel_pos": self.pixel_pos,
                "preview_surface": self.preview_surface,
                "world_surface": self.world_surface,
                "last_detection_time": self.last_detection_time,
                "matched_dict": self.matched_dict,
                "matched_id": self.matched_id,
            }
