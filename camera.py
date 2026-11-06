"""
Camera.py — Renders the follower's synthetic downward-facing camera view.

This is the ONLY window into the world the vision pipeline is allowed to
use. The world-to-pixel mapping here is intentionally simple (orthographic,
fixed scale) rather than a full perspective camera, which keeps the render
tractable in Pygame while still giving OpenCV a real image to detect a real
marker in — the CV pipeline doesn't get to cheat by reading world state.
"""
import pygame

CAM_W, CAM_H = 320, 240
WORLD_VIEW_SIZE = 900  # world units visible across horizontal axis (sized for full swarm formation)


def world_to_cam(world_pos, follower_pos):
    """Convert a world coordinate into camera-frame pixel coordinates,
    centered on the follower with per-axis scale so the aspect ratio is
    preserved correctly for both X and Y."""
    scale_x = CAM_W / WORLD_VIEW_SIZE
    # Y-axis world view size is proportional to the aspect ratio
    world_view_h = WORLD_VIEW_SIZE * (CAM_H / CAM_W)
    scale_y = CAM_H / world_view_h
    dx = (world_pos[0] - follower_pos[0]) * scale_x
    dy = (world_pos[1] - follower_pos[1]) * scale_y
    return int(round(CAM_W / 2 + dx)), int(round(CAM_H / 2 + dy))


def render_camera_view(follower_pos, leader_pos, marker_surface, target_surface=None):
    """Returns a pygame.Surface representing what the follower's camera sees.
    Reuses target_surface if provided to eliminate heap allocations per frame."""
    cam = target_surface if target_surface is not None else pygame.Surface((CAM_W, CAM_H))
    cam.fill((25, 25, 30))  # dark background = "ground" from above

    mx, my = world_to_cam(leader_pos, follower_pos)
    marker_rect = marker_surface.get_rect(center=(mx, my))

    # Only draw the marker if it actually falls within the camera frame —
    # this is what forces the "SEARCHING" state when the follower falls
    # too far behind.
    if cam.get_rect().colliderect(marker_rect):
        cam.blit(marker_surface, marker_rect)

    return cam
