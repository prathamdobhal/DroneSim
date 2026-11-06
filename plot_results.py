"""Plot results from tracking_log.csv for the drone swarm simulation.

Supports both:
- Phase 2 SwarmLogger (aggregate mean/max + per-drone rainbow curves + lock telemetry)
- Phase 1 ErrorLogger (single drone error magnitude + visibility)

Usage:
    python plot_results.py
Saves:
    error_vs_time.png
"""
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

CSV_PATH = "tracking_log.csv"
OUTPUT_PNG = "error_vs_time.png"

if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found. Run 'python main.py' first to generate log data.")
    exit(1)

with open(CSV_PATH, mode="r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader, None)

    if not header:
        print(f"Error: {CSV_PATH} is empty.")
        exit(1)

    rows = list(reader)

if not rows:
    print(f"Error: No data rows found in {CSV_PATH}.")
    exit(1)

# Check whether this is a Swarm log or single-drone log
if "mean_error" in header:
    # --- Phase 2: Swarm Data ---
    t_idx = header.index("t")
    mean_idx = header.index("mean_error")
    max_idx = header.index("max_error")
    n_vis_idx = header.index("n_visible")
    drone_cols = [c for c in header if c.startswith("error_")]
    drone_indices = [header.index(c) for c in drone_cols]
    n_followers = len(drone_cols)

    t = [float(r[t_idx]) for r in rows]
    mean_err = [float(r[mean_idx]) for r in rows]
    max_err = [float(r[max_idx]) for r in rows]
    n_vis = [int(r[n_vis_idx]) for r in rows]

    drone_errs = {c: [float(r[idx]) for r in rows] for c, idx in zip(drone_cols, drone_indices)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})

    # Palette for individual followers
    colors = cm.rainbow(np.linspace(0, 1, n_followers))

    # Plot individual drone errors
    for i, c in enumerate(drone_cols):
        ax1.plot(t, drone_errs[c], color=colors[i], alpha=0.35, linewidth=1.0,
                 label="Followers (F00–F13)" if i == 0 else "")

    # Plot mean and max
    ax1.plot(t, mean_err, color="#0077b6", linewidth=2.4, label=f"Swarm Mean Error (avg: {np.mean(mean_err):.1f} wu)")
    ax1.plot(t, max_err, color="#d90429", linewidth=1.6, linestyle="--", label=f"Worst-Case Error (peak: {np.max(max_err):.1f} wu)")
    ax1.fill_between(t, 0, mean_err, color="#0077b6", alpha=0.1)

    ax1.set_ylabel("Tracking Error (world units)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Vision-Based Drone Swarm Formation Tracking (1 Leader + {n_followers} Followers)", fontsize=13, fontweight="bold", pad=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Plot visibility / lock rate
    ax2.plot(t, n_vis, color="#2a9d8f", linewidth=2.0, label="Drones Locked")
    ax2.fill_between(t, 0, n_vis, color="#2a9d8f", alpha=0.25)
    ax2.axhline(y=n_followers, color="#264653", linestyle=":", linewidth=1.5, label=f"Full Swarm Target ({n_followers})")

    ax2.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Locked Drones", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.5, n_followers + 1.2)
    ticks = sorted(list(set(list(range(0, n_followers, max(1, n_followers // 4))) + [n_followers])))
    ax2.set_yticks(ticks)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved swarm report plot to {OUTPUT_PNG}")
    full_lock_pct = np.mean(np.array(n_vis) == n_followers) * 100
    avg_drone_lock_pct = np.mean(n_vis) / n_followers * 100
    print(f"Summary: {len(t)} timesteps, duration: {t[-1]:.1f}s, mean error: {np.mean(mean_err):.2f} wu, full-swarm lock: {full_lock_pct:.1f}%, mean follower lock: {avg_drone_lock_pct:.1f}%")

else:
    # --- Phase 1: Single Drone Data ---
    t = [float(r[header.index("t")]) for r in rows]
    mag = [float(r[header.index("error_magnitude")]) for r in rows]
    vis = [int(r[header.index("marker_visible")]) for r in rows]

    plt.figure(figsize=(9, 4.5))
    plt.plot(t, mag, color="#0077b6", linewidth=2.0, label=f"Tracking error (avg: {np.mean(mag):.1f} wu)")
    plt.fill_between(t, 0, mag, color="#0077b6", alpha=0.15)
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Error (world units)", fontsize=11)
    plt.title("Follower tracking error over time", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved single-drone report plot to {OUTPUT_PNG}")
