"""
main.py — Simplified dashboard for the vision-based 1-leader / 14-follower swarm.

The simulation, synthetic camera, ArUco vision, PID control, parallel workers,
webcam tracker and CSV logger retain the same roles as the previous version.
This revision changes the presentation and adds clearer pause/reset controls.
"""

import sys
import math
import colorsys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import pygame

from drone_sim import Leader, Follower
from camera import render_camera_view, CAM_W, CAM_H, WORLD_VIEW_SIZE
from vision import detect_marker, pixel_offset_from_center, pygame_surface_to_gray
from control import PID
from logger_util import SwarmLogger
from webcam_tracker import WebcamTracker


# ---------------------------------------------------------------------------
# Window and layout
# ---------------------------------------------------------------------------

WINDOW_W, WINDOW_H = 1440, 820
WORLD_W = 1020
PANEL_X = 1040
PANEL_W = WINDOW_W - PANEL_X
FPS = 60

BG = (12, 17, 27)
PANEL_BG = (19, 27, 41)
CARD_BG = (27, 38, 55)
BORDER = (48, 63, 84)
TEXT = (225, 233, 245)
MUTED = (143, 158, 180)
GREEN = (64, 220, 145)
CYAN = (70, 195, 235)
ORANGE = (245, 174, 75)
RED = (245, 95, 105)

LEADER_GAP = 90
_DX, _DY = 46, 42

FORMATION_OFFSETS = [
    (-LEADER_GAP, -_DY),
    (-LEADER_GAP, 0),
    (-LEADER_GAP, _DY),
    (-LEADER_GAP - _DX, -1.5 * _DY),
    (-LEADER_GAP - _DX, -0.5 * _DY),
    (-LEADER_GAP - _DX, 0.5 * _DY),
    (-LEADER_GAP - _DX, 1.5 * _DY),
    (-LEADER_GAP - 2 * _DX, -1.5 * _DY),
    (-LEADER_GAP - 2 * _DX, -0.5 * _DY),
    (-LEADER_GAP - 2 * _DX, 0.5 * _DY),
    (-LEADER_GAP - 2 * _DX, 1.5 * _DY),
    (-LEADER_GAP - 3 * _DX, -_DY),
    (-LEADER_GAP - 3 * _DX, 0),
    (-LEADER_GAP - 3 * _DX, _DY),
]
N = len(FORMATION_OFFSETS)

COLORS = []
for i in range(N):
    r, g, b = colorsys.hsv_to_rgb(i / N, 0.78, 1.0)
    COLORS.append((int(r * 255), int(g * 255), int(b * 255)))


def make_marker_surface(size=40):
    """Create the synthetic ArUco marker used by each simulated camera."""
    img = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, size
    )
    bordered = cv2.copyMakeBorder(
        img, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255
    )
    rgb = cv2.cvtColor(bordered, cv2.COLOR_GRAY2RGB)
    surface = pygame.image.frombuffer(
        rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB"
    )
    return surface.convert()


def vision_worker(args):
    """Render and process one follower's synthetic downward-facing camera."""
    (
        idx, follower_pos, leader_pos, marker_surface, cam_surface,
        pid_x, pid_y, dt, desired_offset, previous_velocity
    ) = args

    cam_surface = render_camera_view(
        follower_pos, leader_pos, marker_surface, target_surface=cam_surface
    )
    gray = pygame_surface_to_gray(cam_surface)
    corners, ids = detect_marker(gray)
    visible = ids is not None and len(ids) > 0

    if visible:
        px_x, px_y = pixel_offset_from_center(corners, gray.shape)
        scale = CAM_W / WORLD_VIEW_SIZE
        wx, wy = px_x / scale, px_y / scale

        # The controller sees only the marker offset extracted from the image.
        ex = wx + desired_offset[0]
        ey = wy + desired_offset[1]
        vx = pid_x.update(ex, dt)
        vy = pid_y.update(ey, dt)
    else:
        pid_x.reset()
        pid_y.reset()
        vx = previous_velocity[0] * 0.92
        vy = previous_velocity[1] * 0.92

        # Ground-truth error is logged only for simulation evaluation.
        ex = (leader_pos[0] + desired_offset[0]) - follower_pos[0]
        ey = (leader_pos[1] + desired_offset[1]) - follower_pos[1]

    return idx, vx, vy, ex, ey, visible, cam_surface


def draw_text(surface, font, text, color, pos):
    surface.blit(font.render(str(text), True, color), pos)


def draw_card(surface, rect, title, value, accent, font, small_font):
    pygame.draw.rect(surface, CARD_BG, rect, border_radius=10)
    pygame.draw.rect(surface, BORDER, rect, 1, border_radius=10)
    draw_text(surface, small_font, title.upper(), MUTED, (rect.x + 12, rect.y + 9))
    draw_text(surface, font, value, accent, (rect.x + 12, rect.y + 27))


def draw_button(surface, rect, label, active, font):
    fill = (34, 62, 65) if active else CARD_BG
    edge = GREEN if active else BORDER
    pygame.draw.rect(surface, fill, rect, border_radius=7)
    pygame.draw.rect(surface, edge, rect, 1, border_radius=7)
    text = font.render(label, True, TEXT if active else MUTED)
    surface.blit(
        text,
        (rect.centerx - text.get_width() // 2,
         rect.centery - text.get_height() // 2)
    )


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Drone Swarm Control | Phase 3")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("consolas", 23, bold=True)
    font = pygame.font.SysFont("consolas", 16)
    font_small = pygame.font.SysFont("consolas", 13)
    font_tiny = pygame.font.SysFont("consolas", 11)

    tracker = WebcamTracker(
        world_bounds=((120, WORLD_W - 120), (100, WINDOW_H - 100))
    )
    tracker.start()

    leader = Leader(
        center=(WORLD_W // 2, WINDOW_H // 2),
        radius=140,
        speed=0.5
    )

    followers = [
        Follower(start_pos=(
            leader.position[0] + ox,
            leader.position[1] + oy
        ))
        for ox, oy in FORMATION_OFFSETS
    ]

    marker_surfaces = [make_marker_surface() for _ in range(N)]
    camera_surfaces = [pygame.Surface((CAM_W, CAM_H)) for _ in range(N)]

    pids = [
        (
            PID(kp=1.8, ki=0.05, kd=0.6, output_limit=followers[i].max_speed),
            PID(kp=1.8, ki=0.05, kd=0.6, output_limit=followers[i].max_speed)
        )
        for i in range(N)
    ]

    logger = SwarmLogger(N)
    errors = [(0.0, 0.0)] * N
    visible = [False] * N
    camera_frames = [None] * N
    velocities = [(0.0, 0.0)] * N

    selected = 0
    pip_mode = "follower"
    bg_mode = "dark"
    user_set_mode = False
    paused = False
    running = True

    avg_batch_ms = 0.0
    avg_drone_ms = 0.0

    # Clickable controls
    mode_rect = pygame.Rect(PANEL_X + 18, 66, 174, 34)
    pause_rect = pygame.Rect(PANEL_X + 202, 66, 174, 34)
    reset_rect = pygame.Rect(PANEL_X + 18, 108, 174, 32)
    bg_rect = pygame.Rect(PANEL_X + 202, 108, 174, 32)
    cam_rect = pygame.Rect(PANEL_X + 18, 244, 358, 28)

    # Camera preview is scaled to fit the sidebar.
    preview_w, preview_h = 320, 180
    preview_rect = pygame.Rect(PANEL_X + 27, 278, preview_w, preview_h)

    # Soft dark overlay for the world when webcam background is used.
    tint = pygame.Surface((WORLD_W, WINDOW_H), pygame.SRCALPHA)
    tint.fill((5, 10, 20, 95))

    def reset_swarm():
        leader.x = float(WORLD_W // 2)
        leader.y = float(WINDOW_H // 2)
        leader.vx = leader.vy = 0.0
        leader.center = (leader.x, leader.y)
        leader.t = 0.0
        leader.target_x, leader.target_y = leader.x, leader.y

        for i, (ox, oy) in enumerate(FORMATION_OFFSETS):
            followers[i].x = leader.x + ox
            followers[i].y = leader.y + oy
            velocities[i] = (0.0, 0.0)
            errors[i] = (0.0, 0.0)
            visible[i] = False
            camera_frames[i] = None
            pids[i][0].reset()
            pids[i][1].reset()

    try:
        with ThreadPoolExecutor(max_workers=N) as pool:
            while running:
                dt = min(clock.tick(FPS) / 1000.0, 0.05)

                # ----------------------------------------------------------
                # Input
                # ----------------------------------------------------------
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_w:
                            user_set_mode = True
                            leader.set_mode(
                                "orbit" if leader.mode == "webcam" else "webcam"
                            )
                        elif event.key == pygame.K_p:
                            paused = not paused
                        elif event.key == pygame.K_r:
                            reset_swarm()
                        elif event.key == pygame.K_b:
                            bg_mode = "webcam" if bg_mode == "dark" else "dark"
                        elif event.key == pygame.K_c:
                            pip_mode = "webcam" if pip_mode == "follower" else "follower"
                        elif event.key in (pygame.K_TAB, pygame.K_RIGHT):
                            selected = (selected + 1) % N
                            pip_mode = "follower"
                        elif event.key == pygame.K_LEFT:
                            selected = (selected - 1) % N
                            pip_mode = "follower"

                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos

                        if mode_rect.collidepoint(mx, my):
                            user_set_mode = True
                            leader.set_mode(
                                "orbit" if leader.mode == "webcam" else "webcam"
                            )
                        elif pause_rect.collidepoint(mx, my):
                            paused = not paused
                        elif reset_rect.collidepoint(mx, my):
                            reset_swarm()
                        elif bg_rect.collidepoint(mx, my):
                            bg_mode = "webcam" if bg_mode == "dark" else "dark"
                        elif cam_rect.collidepoint(mx, my):
                            pip_mode = "webcam" if pip_mode == "follower" else "follower"
                        elif (
                            PANEL_X + 16 <= mx <= WINDOW_W - 16
                            and 568 <= my < 568 + N * 15
                        ):
                            selected = min(N - 1, (my - 568) // 15)
                            pip_mode = "follower"
                        elif mx < WORLD_W:
                            clicked_drone = False
                            for i, f in enumerate(followers):
                                if (f.x - mx) ** 2 + (f.y - my) ** 2 <= 18 ** 2:
                                    selected = i
                                    pip_mode = "follower"
                                    clicked_drone = True
                                    break
                            if not clicked_drone and leader.mode == "webcam":
                                leader.set_target(mx, my)

                webcam_state = tracker.get_state()

                # ----------------------------------------------------------
                # Simulation update
                # ----------------------------------------------------------
                if not paused:
                    if leader.mode == "webcam":
                        if webcam_state["camera_available"] and webcam_state["is_tracking"]:
                            leader.set_target(*webcam_state["world_target"])
                        elif not webcam_state["camera_available"] and not user_set_mode:
                            leader.set_mode("orbit")

                    leader.update(dt)

                    start = time.perf_counter()
                    args = [
                        (
                            i, followers[i].position, leader.position,
                            marker_surfaces[i], camera_surfaces[i],
                            pids[i][0], pids[i][1], dt,
                            FORMATION_OFFSETS[i], velocities[i]
                        )
                        for i in range(N)
                    ]
                    results = list(pool.map(vision_worker, args))
                    batch_ms = (time.perf_counter() - start) * 1000.0
                    avg_batch_ms = avg_batch_ms * 0.9 + batch_ms * 0.1
                    avg_drone_ms = avg_drone_ms * 0.9 + (batch_ms / N) * 0.1

                    for idx, vx, vy, ex, ey, is_visible, cam in results:
                        # Keep the existing leader-clearance repulsion.
                        dx = followers[idx].x - leader.x
                        dy = followers[idx].y - leader.y
                        distance = math.hypot(dx, dy)
                        if 0.001 < distance < 45.0:
                            push = 160.0 * (1.0 - distance / 45.0)
                            vx += (dx / distance) * push
                            vy += (dy / distance) * push

                        velocities[idx] = followers[idx].update(vx, vy, dt)
                        errors[idx] = (ex, ey)
                        visible[idx] = is_visible
                        camera_frames[idx] = cam

                    logger.log(errors, visible)

                # ----------------------------------------------------------
                # Draw world
                # ----------------------------------------------------------
                screen.fill(BG)

                if (
                    bg_mode == "webcam"
                    and webcam_state.get("world_surface") is not None
                ):
                    screen.blit(webcam_state["world_surface"], (0, 0))
                    screen.blit(tint, (0, 0))
                else:
                    pygame.draw.rect(screen, BG, (0, 0, WORLD_W, WINDOW_H))

                # Subtle grid makes positions easier to read.
                for x in range(0, WORLD_W, 50):
                    pygame.draw.line(screen, (23, 32, 47), (x, 0), (x, WINDOW_H), 1)
                for y in range(0, WINDOW_H, 50):
                    pygame.draw.line(screen, (23, 32, 47), (0, y), (WORLD_W, y), 1)

                # World title and status
                draw_text(screen, font, "LIVE SWARM VIEW", TEXT, (24, 20))
                draw_text(
                    screen, font_small,
                    "LEADER + 14 FOLLOWERS  |  CLICK A DRONE TO INSPECT",
                    MUTED, (24, 47)
                )

                # Orbit guide
                if leader.mode == "orbit":
                    pygame.draw.ellipse(
                        screen, (38, 57, 77),
                        pygame.Rect(
                            leader.center[0] - leader.radius,
                            leader.center[1] - int(leader.radius * 0.55),
                            leader.radius * 2,
                            int(leader.radius * 1.10)
                        ), 1
                    )

                # Formation links, intentionally subtle.
                for i, f in enumerate(followers):
                    col = COLORS[i]
                    pygame.draw.line(
                        screen,
                        (col[0] // 9, col[1] // 9, col[2] // 9),
                        (int(f.x), int(f.y)),
                        (int(leader.x), int(leader.y)),
                        1
                    )

                # Followers
                for i, f in enumerate(followers):
                    col = COLORS[i]
                    x, y = int(f.x), int(f.y)
                    if visible[i]:
                        pygame.draw.circle(screen, col, (x, y), 15, 1)
                    pygame.draw.circle(screen, col, (x, y), 8)
                    if i == selected:
                        pygame.draw.circle(screen, TEXT, (x, y), 19, 1)
                    draw_text(
                        screen, font_tiny, f"F{i:02d}", TEXT,
                        (x - 10, y - 30)
                    )

                # Leader
                lx, ly = int(leader.x), int(leader.y)
                pygame.draw.circle(screen, (255, 255, 255), (lx, ly), 19, 2)
                pygame.draw.circle(screen, (245, 78, 88), (lx, ly), 12)
                draw_text(screen, font, "LEADER", TEXT, (lx - 31, ly - 43))

                # Webcam target reticle when in webcam mode.
                if leader.mode == "webcam":
                    tx, ty = int(leader.target_x), int(leader.target_y)
                    reticle_color = GREEN if webcam_state["is_tracking"] else ORANGE
                    pygame.draw.circle(screen, reticle_color, (tx, ty), 12, 1)
                    pygame.draw.line(screen, reticle_color, (tx - 17, ty), (tx + 17, ty), 1)
                    pygame.draw.line(screen, reticle_color, (tx, ty - 17), (tx, ty + 17), 1)

                # ----------------------------------------------------------
                # Draw sidebar
                # ----------------------------------------------------------
                pygame.draw.rect(
                    screen, PANEL_BG,
                    (PANEL_X, 0, PANEL_W, WINDOW_H)
                )
                pygame.draw.line(
                    screen, BORDER,
                    (PANEL_X, 0), (PANEL_X, WINDOW_H), 1
                )

                draw_text(
                    screen, font_title, "DRONE SWARM", TEXT,
                    (PANEL_X + 18, 18)
                )
                status_label = "PAUSED" if paused else "RUNNING"
                status_color = ORANGE if paused else GREEN
                draw_text(
                    screen, font_small, status_label, status_color,
                    (PANEL_X + 260, 27)
                )

                draw_button(
                    screen, mode_rect,
                    "MODE: " + ("AUTO ORBIT" if leader.mode == "orbit" else "WEBCAM"),
                    True, font_small
                )
                draw_button(
                    screen, pause_rect,
                    "RESUME" if paused else "PAUSE",
                    paused, font_small
                )
                draw_button(screen, reset_rect, "RESET SWARM", False, font_small)
                draw_button(
                    screen, bg_rect,
                    "BACKGROUND: " + ("CAMERA" if bg_mode == "webcam" else "DARK"),
                    False, font_tiny
                )

                n_locked = sum(visible)
                mean_error = sum(
                    math.hypot(ex, ey) for ex, ey in errors
                ) / N

                card_y = 158
                card_w = 112
                card_gap = 10
                draw_card(
                    screen, pygame.Rect(PANEL_X + 18, card_y, card_w, 62),
                    "DRONES", "15", CYAN, font, font_tiny
                )
                draw_card(
                    screen, pygame.Rect(PANEL_X + 18 + card_w + card_gap, card_y, card_w, 62),
                    "LOCKED", f"{n_locked}/14",
                    GREEN if n_locked == N else ORANGE, font, font_tiny
                )
                draw_card(
                    screen, pygame.Rect(PANEL_X + 18 + 2 * (card_w + card_gap), card_y, 112, 62),
                    "FPS", f"{clock.get_fps():.0f}", TEXT, font, font_tiny
                )

                # Camera preview
                draw_text(
                    screen, font, "CAMERA PREVIEW", TEXT,
                    (PANEL_X + 18, 222)
                )
                draw_button(
                    screen, cam_rect,
                    "SHOW: " + (
                        f"FOLLOWER F{selected:02d}" if pip_mode == "follower"
                        else "WEBCAM"
                    ),
                    True, font_small
                )

                if pip_mode == "webcam":
                    preview = webcam_state.get("preview_surface")
                    if preview is not None:
                        scaled = pygame.transform.smoothscale(
                            preview, (preview_w, preview_h)
                        )
                        screen.blit(scaled, preview_rect)
                    else:
                        pygame.draw.rect(screen, (10, 16, 25), preview_rect)
                        msg = "WEBCAM NOT AVAILABLE"
                        rendered = font_small.render(msg, True, ORANGE)
                        screen.blit(
                            rendered,
                            (preview_rect.centerx - rendered.get_width() // 2,
                             preview_rect.centery - rendered.get_height() // 2)
                        )
                else:
                    frame = camera_frames[selected]
                    if frame is not None:
                        scaled = pygame.transform.smoothscale(
                            frame, (preview_w, preview_h)
                        )
                        screen.blit(scaled, preview_rect)
                    else:
                        pygame.draw.rect(screen, (10, 16, 25), preview_rect)

                border_color = GREEN if (
                    pip_mode == "follower" and visible[selected]
                ) else BORDER
                pygame.draw.rect(screen, border_color, preview_rect, 2)

                if pip_mode == "webcam":
                    camera_status = webcam_state.get(
                        "status_message", "Webcam status unavailable"
                    )
                    camera_color = GREEN if webcam_state.get("is_tracking") else ORANGE
                else:
                    camera_status = (
                        f"F{selected:02d} MARKER LOCKED"
                        if visible[selected] else f"F{selected:02d} SEARCHING"
                    )
                    camera_color = GREEN if visible[selected] else RED

                draw_text(
                    screen, font_tiny, camera_status[:48], camera_color,
                    (PANEL_X + 20, preview_rect.bottom + 6)
                )

                # Compact performance strip
                draw_text(
                    screen, font_tiny,
                    f"VISION {avg_drone_ms:.2f} ms/drone  |  BATCH {avg_batch_ms:.1f} ms",
                    MUTED, (PANEL_X + 18, 474)
                )
                draw_text(
                    screen, font_tiny,
                    f"MEAN TRACKING ERROR  {mean_error:.1f} world units",
                    MUTED, (PANEL_X + 18, 492)
                )

                # Follower list: compact, readable, selectable.
                draw_text(
                    screen, font, "FOLLOWER STATUS", TEXT,
                    (PANEL_X + 18, 526)
                )
                draw_text(
                    screen, font_tiny,
                    "SELECT A ROW OR USE LEFT / RIGHT ARROWS",
                    MUTED, (PANEL_X + 18, 547)
                )

                list_y = 568
                row_h = 15
                for i in range(N):
                    y = list_y + i * row_h
                    row_rect = pygame.Rect(PANEL_X + 16, y, PANEL_W - 32, row_h - 1)
                    if i == selected:
                        pygame.draw.rect(screen, (39, 54, 74), row_rect, border_radius=4)

                    col = COLORS[i]
                    pygame.draw.circle(screen, col, (PANEL_X + 27, y + 8), 4)
                    draw_text(
                        screen, font_tiny, f"F{i:02d}", TEXT,
                        (PANEL_X + 40, y + 2)
                    )

                    magnitude = math.hypot(*errors[i])
                    state = "LOCK" if visible[i] else "SEARCH"
                    state_color = GREEN if visible[i] else RED
                    draw_text(
                        screen, font_tiny, f"{magnitude:6.1f} wu", MUTED,
                        (PANEL_X + 105, y + 2)
                    )
                    draw_text(
                        screen, font_tiny, state, state_color,
                        (PANEL_X + 245, y + 2)
                    )

                # Footer controls
                footer_y = WINDOW_H - 24
                pygame.draw.line(
                    screen, BORDER,
                    (PANEL_X + 16, footer_y - 8),
                    (WINDOW_W - 16, footer_y - 8), 1
                )
                draw_text(
                    screen, font_tiny,
                    "W MODE   P PAUSE   R RESET   C CAMERA   B BACKGROUND   ESC QUIT",
                    MUTED, (PANEL_X + 18, footer_y)
                )

                # World footer
                draw_text(
                    screen, font_tiny,
                    f"LEADER SPEED  {math.hypot(leader.vx, leader.vy):.1f} wu/s",
                    MUTED, (24, WINDOW_H - 28)
                )
                draw_text(
                    screen, font_tiny,
                    "SIMULATION VIEW — NOT PHYSICAL FLIGHT TELEMETRY",
                    MUTED, (300, WINDOW_H - 28)
                )

                pygame.display.flip()

    finally:
        tracker.stop()
        logger.close()
        pygame.quit()


if __name__ == "__main__":
    main()
