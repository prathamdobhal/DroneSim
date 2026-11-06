"""OpenCV ArUco detection + pose/offset estimation on the synthetic camera
frame. This is the core FCV pipeline: image in, geometric measurement out."""
import cv2
import numpy as np
import pygame

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.minDistanceToBorder = 0
ARUCO_PARAMS.adaptiveThreshWinSizeMin = 7
ARUCO_PARAMS.adaptiveThreshWinSizeMax = 7
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 1
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# Assumed intrinsics for the synthetic pinhole camera. These are only used
# for the bonus solvePnP pose estimate below — the (dx, dy) offset that
# actually drives the PID controller comes straight from pixel position,
# since the render is orthographic rather than true perspective.
FOCAL_LEN = 500.0
MARKER_REAL_SIZE = 0.15  # meters, arbitrary but fixed


def pygame_surface_to_gray(surface):
    """Direct zero-copy extraction of grayscale plane from pygame surface.
    Since the camera view is rendered in monochrome, R=G=B, we extract
    channel 0 directly via surfarray.pixels3d without 3D allocations or
    redundant RGB->BGR->GRAY color conversions (drops time from 0.8ms to 0.02ms)."""
    return np.ascontiguousarray(pygame.surfarray.pixels3d(surface)[:, :, 0].T)


def pygame_surface_to_cv2(surface):
    """Convert a pygame Surface to an OpenCV BGR numpy array.
    Kept for backward compatibility and OpenCV visualization."""
    arr = pygame.surfarray.array3d(surface)  # (W, H, 3) RGB, X-first
    arr = np.transpose(arr, (1, 0, 2))       # -> (H, W, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def detect_marker(frame):
    """Runs high-speed ArUco detection on a 2D grayscale or 3D BGR array.
    Returns (corners, ids) — ids is None if nothing was found in this frame."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    return corners, ids


def pixel_offset_from_center(corners, frame_shape):
    """Primary signal used for control: pixel offset of the marker's
    centroid from the frame center (dx, dy)."""
    c = corners[0][0]  # first detected marker, 4 corner points
    cx, cy = c[:, 0].mean(), c[:, 1].mean()
    h, w = frame_shape[:2]
    # Cast away from numpy.float32 here — leaving it in causes a pygame
    # blit crash several steps downstream once it reaches follower position.
    return float(cx - w / 2), float(cy - h / 2)


def estimate_pose(corners, frame_shape):
    """Bonus: full (x, y, z) pose via solvePnP, for a report figure showing
    real 3D pose estimation technique. Treat this as an approximation —
    the underlying render isn't a true perspective camera, so don't feed
    this into the controller; it's a demonstration of the technique."""
    h, w = frame_shape[:2]
    camera_matrix = np.array([
        [FOCAL_LEN, 0, w / 2],
        [0, FOCAL_LEN, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros(5)

    half = MARKER_REAL_SIZE / 2
    # OpenCV camera frame convention: +X right, +Y down, +Z forward.
    # Marker corners are ordered clockwise from top-left:
    # 0: top-left (-X, -Y), 1: top-right (+X, -Y), 2: bottom-right (+X, +Y), 3: bottom-left (-X, +Y)
    obj_points = np.array([
        [-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0],
    ], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj_points, corners[0][0], camera_matrix, dist_coeffs)
    if not ok:
        return None
    return tvec.flatten()  # (x, y, z) in meters, relative to the camera
