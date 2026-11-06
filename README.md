# Vision-Based Leader-Follower Drone Swarm — Phase 3

A 100%-software FCV simulation: **1 leader drone and 14 follower drones** flying in an organized swarm formation. Each follower drone independently tracks the leader drone using only its own simulated onboard downward-facing camera and an ArUco marker — no ground-truth position is ever handed to any follower's controller. The leader can either fly an autonomous figure-eight path or dynamically pursue a physical ArUco marker tracked in real time through a live webcam feed.

```
Physical Marker → Webcam Tracker → Leader Pursuit → Synthetic Camera Frame → ArUco Detection → PID → Follower Velocity
```

## Features in Phase 3

- **1 Leader + 14 Followers**: A 15-drone swarm holding a coordinated grid formation trailing behind the leader.
- **Physical Webcam Marker Tracking**: Real-time physical ArUco tracking via background webcam acquisition with multi-dictionary fallback (DICT_4X4_50, 4X4_100, 6X6_250, 5X5_100) and CLAHE lighting enhancement.
- **Automatic Fallback to Synthetic Orbit**: If no physical webcam is detected, the simulation automatically falls back to smooth autonomous figure-eight flight without stalling.
- **Live Augmented Reality (AR) Background**: Toggle the world view between a minimalist dark canvas and a live AR camera feed aligned with swarm coordinates.
- **True Multi-Agent Vision**: Each follower renders its own synthetic camera view and runs an independent ArUco detector + PID loop.
- **Parallel Vision Pipeline**: Computer vision runs concurrently using Python's `ThreadPoolExecutor(max_workers=14)`. Because OpenCV's ArUco backend releases the GIL during image processing, all 14 frame detections execute in parallel at ~60 FPS.
- **Rainbow Color Gradient**: Each follower is assigned a distinct hue along a rainbow spectrum (HSV gradient), with corresponding colored trails, telemetry entries, and PiP view borders.
- **Live PiP Camera Feed & Reticle**: Switch between any follower's onboard camera view and the live webcam zoom feed using `Tab`, `Left/Right Arrows`, or direct mouse click.
- **Real-Time Telemetry Dashboard**: Right-panel heads-up telemetry displaying lock status (`LOCK`/`LOST`), individual per-axis tracking errors (`dx`, `dy`), error magnitudes (`|err|`), and swarm cohesion rate.
- **Swarm CSV Logger & Analysis Plotting**: Telemetry logged to `tracking_log.csv` and visualized using `plot_results.py` to produce publication-ready error envelopes and swarm lock metrics.

## Setup

```bash
pip install -r requirements.txt
```

To display the physical ArUco marker on a phone or second screen, open `show_marker.html` or display `aruco_marker_id0.png`.

## Run the Simulation

```bash
python main.py
```

### Controls

| Action | Control |
|---|---|
| **Toggle Control Mode (Webcam / Auto-Orbit)** | `W` or click **MODE** badge |
| **Toggle World Background (AR Video / Dark)** | `B` or click **BG** badge |
| **Toggle PiP Panel (Follower Cam / Webcam Zoom)** | `C` or click PiP tabs |
| **Cycle Follower Camera (Next / Prev)** | `Tab` / `Right Arrow` / `Left Arrow` |
| **Select Follower Directly** | Click any drone in the world view or any telemetry table row |
| **Manual Target Positioning (Webcam Mode)** | Click anywhere in the world view |
| **Quit Simulation** | `ESC` or close window |

## Generate the Report Plot

After running the simulation (or exiting with `ESC`), run:

```bash
python plot_results.py
```

This generates `error_vs_time.png` containing:
1. **Top Panel**: Swarm tracking error envelope (Mean swarm error + Worst-case drone error + translucent per-drone rainbow curves).
2. **Bottom Panel**: Real-time marker lock count (number of drones maintaining visual lock out of 14).

## File Guide

| File | Role |
|---|---|
| `main.py` | Swarm simulation coordinator, ThreadPoolExecutor vision pipeline, UI and telemetry |
| `webcam_tracker.py` | Threaded webcam acquisition, multi-dictionary ArUco tracker, and AR frame generator |
| `drone_sim.py` | Kinematics models for `Leader` (pursuit & figure-eight) and `Follower` |
| `camera.py` | Renders per-follower synthetic orthographic camera views (`WORLD_VIEW_SIZE = 900`) |
| `vision.py` | OpenCV ArUco detector (`DICT_4X4_50`), pixel centroid extraction, bonus `solvePnP` |
| `control.py` | PID controller with integral clamping / anti-windup (independent per axis) |
| `logger_util.py` | `SwarmLogger` (multi-follower aggregated + individual CSV logger) & legacy `ErrorLogger` |
| `plot_results.py` | Multi-panel performance plot generator with swarm metrics and backward compatibility |
| `show_marker.html` | Browser display helper for presenting `aruco_marker_id0.png` to webcam |

## Tuning

- **PID Gains**: Set in `main.py` (`kp=1.8, ki=0.05, kd=0.6`). `kp` drives reaction speed; `kd` damps turning oscillations; `ki` handles steady-state tracking offsets.
- **Formation Geometry**: Defined by `FORMATION_OFFSETS` in `main.py` with `_DX = 46, _DY = 42`.
- **Field of View**: `WORLD_VIEW_SIZE` in `camera.py` (calibrated to 900 units) ensures all 14 follower slots maintain continuous visual lock during high-speed turns.
