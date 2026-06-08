"""
debug_episode.py  —  visualize convert_episode.py outputs

Usage:
    python debug_episode.py episode_0.hdf5               # save to episode_0_debug.png
    python debug_episode.py episode_0.hdf5 --show        # open interactive window
    python debug_episode.py episode_0.hdf5 --frame 120   # also dump one body-skeleton frame
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

BODY_EDGES = [          # SMPL-H 24-joint skeleton edges
    (0, 1), (0, 2), (0, 3),           # pelvis → hips + spine1
    (1, 4), (2, 5),                   # hips → knees
    (4, 7), (5, 8),                   # knees → ankles
    (7, 10), (8, 11),                 # ankles → feet
    (3, 6), (6, 9),                   # spine chain
    (9, 12), (9, 13), (9, 14),        # chest → neck, l-collar, r-collar
    (12, 15),                         # neck → head
    (13, 16), (14, 17),               # collars → shoulders
    (16, 18), (17, 19),               # shoulders → elbows
    (18, 20), (19, 21),               # elbows → wrists
    (20, 22), (21, 23),               # wrists → hand tips
]

PLANNER_MODE_NAMES = {0: "idle", 1: "slowWalk", 2: "walk", 4: "squat", 22: "crouch"}
PLANNER_MODE_COLORS = {0: "gray", 1: "steelblue", 2: "royalblue", 4: "darkorange", 22: "purple"}


def load(f: h5py.File, key: str) -> np.ndarray:
    return f[key][:]


def time_axis(N: int, fps: float = 50.0) -> np.ndarray:
    return np.arange(N) / fps


# ─────────────────────────────────────────────────────────────────────────────
# Individual panel functions
# ─────────────────────────────────────────────────────────────────────────────

def panel_trajectory(ax, positions_xyz, rotation_xy):
    """Bird's-eye pelvis trajectory with heading arrows."""
    x, y = positions_xyz[:, 0], positions_xyz[:, 1]
    N = len(x)
    ax.plot(x, y, lw=1.2, color="steelblue", alpha=0.7)
    ax.scatter(x[0], y[0], s=60, color="green", zorder=5, label="start")
    ax.scatter(x[-1], y[-1], s=60, color="red", zorder=5, label="end")
    # Heading arrows every ~1 s (50 frames)
    step = max(1, N // 20)
    for i in range(0, N, step):
        hx, hy = rotation_xy[i, 0], rotation_xy[i, 1]
        ax.annotate("", xy=(x[i] + hx * 0.05, y[i] + hy * 0.05),
                    xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.0))
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Pelvis trajectory (bird's eye)")
    ax.legend(fontsize=7)


def panel_height(ax, t, delta_height, planner_height):
    ax.plot(t, delta_height, label="Δheight (human)", color="steelblue")
    ax2 = ax.twinx()
    ax2.plot(t, planner_height, label="planner height (m)", color="darkorange", lw=1.5)
    ax2.set_ylabel("planner height (m)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Δheight (human)"); ax.set_title("Height control")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)


def panel_planner_mode(ax, t, planner_mode):
    unique_modes = np.unique(planner_mode)
    for m in unique_modes:
        mask = planner_mode == m
        # draw horizontal color band
        ax.fill_between(t, 0, 1, where=mask,
                        color=PLANNER_MODE_COLORS.get(int(m), "gray"),
                        alpha=0.6, label=PLANNER_MODE_NAMES.get(int(m), str(m)),
                        transform=ax.get_xaxis_transform())
    ax.set_yticks([])
    ax.set_title("Planner mode")
    ax.legend(fontsize=7, loc="upper right")


def panel_nav_commands(ax, t, nav_cmd):
    ax.plot(t, nav_cmd[:, 0], label="vx (fwd)", color="steelblue")
    ax.plot(t, nav_cmd[:, 1], label="vy (lat)", color="darkorange")
    ax.plot(t, nav_cmd[:, 2], label="yaw_rate", color="green")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("cmd (m/s or rad/s)"); ax.set_title("Navigation commands (local frame)")
    ax.legend(fontsize=7)


def panel_3point_pos(ax, t, target_pos):
    labels = ["lw_x", "lw_y", "lw_z", "rw_x", "rw_y", "rw_z", "hd_x", "hd_y", "hd_z"]
    styles = [("-", "steelblue"), ("--", "steelblue"), (":", "steelblue"),
              ("-", "darkorange"), ("--", "darkorange"), (":", "darkorange"),
              ("-", "green"), ("--", "green"), (":", "green")]
    for i, (label, (ls, col)) in enumerate(zip(labels, styles)):
        ax.plot(t, target_pos[:, i], ls=ls, color=col, lw=1.0, label=label, alpha=0.8)
    ax.set_ylabel("m"); ax.set_title("3-point local positions [lw, rw, head]")
    ax.legend(fontsize=6, ncol=3)


def panel_hand_joints(ax, t, left_joints, right_joints):
    # Show mean across 6 joints (0=open, 1=closed)
    ax.plot(t, left_joints.mean(axis=1),  label="left (mean)", color="steelblue")
    ax.plot(t, right_joints.mean(axis=1), label="right (mean)", color="darkorange")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("joint value"); ax.set_title("Hand joints (0=open, 1=closed)")
    ax.legend(fontsize=7)


def panel_eef_pos(ax, t, eef):
    """EEF xyz positions in pelvis frame."""
    for i, (label, col) in enumerate(
        [("lx", "steelblue"), ("ly", "cornflowerblue"), ("lz", "lightblue"),
         ("rx", "darkorange"), ("ry", "orange"), ("rz", "moccasin")]
    ):
        ax.plot(t, eef[:, i], color=col, lw=0.9, label=label, alpha=0.8)
    ax.set_ylabel("m"); ax.set_title("EEF positions in pelvis frame")
    ax.legend(fontsize=6, ncol=3)


def panel_delta_eef(ax, t, delta_eef):
    ax.plot(t, np.linalg.norm(delta_eef[:, :3], axis=1),  label="|Δpos_L|", color="steelblue")
    ax.plot(t, np.linalg.norm(delta_eef[:, 6:9], axis=1), label="|Δpos_R|", color="darkorange")
    ax.plot(t, np.linalg.norm(delta_eef[:, 3:6], axis=1), label="|Δrot_L| (rad)", color="steelblue", ls="--")
    ax.plot(t, np.linalg.norm(delta_eef[:, 9:12], axis=1), label="|Δrot_R| (rad)", color="darkorange", ls="--")
    ax.set_ylabel("magnitude"); ax.set_title("Delta EEF magnitudes")
    ax.legend(fontsize=7)


def panel_cam_sync(ax, t, diff_ms):
    ax.plot(t, diff_ms, color="steelblue", lw=0.8)
    ax.axhline(16.7, color="red", ls="--", lw=0.8, label="1 cam frame (16.7ms)")
    ax.fill_between(t, 0, diff_ms, alpha=0.3, color="steelblue")
    ax.set_ylabel("ms"); ax.set_title("Camera sync error (timestamp diff)")
    ax.legend(fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Body skeleton frame
# ─────────────────────────────────────────────────────────────────────────────

def plot_skeleton_frame(h5_path: Path, frame_idx: int, out_path: Path | None, show: bool):
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    with h5py.File(h5_path) as f:
        if "debug/body_pose" not in f:
            print("  (no debug/body_pose in file, skipping skeleton)")
            return
        body_pose = f["debug/body_pose"][frame_idx]   # (24,7)
        pos_xyz   = f["debug/positions_xyz"][frame_idx]
        rot_xy    = f["debug/rotation_xy"][frame_idx]

    from scipy.spatial.transform import Rotation as R
    _WORLD_ROT = R.from_euler('z', -90, degrees=True) * R.from_euler('x', 90, degrees=True)
    _WORLD_ROT_MAT = _WORLD_ROT.as_matrix()

    pts_muj = (_WORLD_ROT_MAT @ body_pose[:, :3].T).T   # (24,3) in MuJoCo frame

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for a, b in BODY_EDGES:
        ax.plot([pts_muj[a, 0], pts_muj[b, 0]],
                [pts_muj[a, 1], pts_muj[b, 1]],
                [pts_muj[a, 2], pts_muj[b, 2]], "o-", color="steelblue", lw=1.5, ms=3)

    # Mark wrists and head
    for idx, label, col in [(20, "L wrist", "green"), (21, "R wrist", "darkorange"), (15, "head", "red")]:
        ax.scatter(*pts_muj[idx], s=60, color=col, zorder=5)
        ax.text(*pts_muj[idx], f"  {label}", fontsize=7)

    # Pelvis heading arrow
    hx, hy = rot_xy[0], rot_xy[1]
    pelvis = pts_muj[0]
    ax.quiver(pelvis[0], pelvis[1], pelvis[2], hx * 0.2, hy * 0.2, 0,
              color="darkorange", linewidth=2, label="heading")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Body skeleton — frame {frame_idx} (MuJoCo frame)")
    ax.legend(fontsize=8)

    skel_path = (out_path.parent / (out_path.stem + f"_skeleton_f{frame_idx}.png")) if out_path else None
    if skel_path:
        fig.savefig(skel_path, dpi=120, bbox_inches="tight")
        print(f"  skeleton → {skel_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main summary figure
# ─────────────────────────────────────────────────────────────────────────────

def make_summary(h5_path: Path, out_path: Path | None, show: bool):
    with h5py.File(h5_path) as f:
        N = len(f["local_timestamps_ns"])
        t = time_axis(N)

        pos_xyz     = load(f, "debug/positions_xyz")
        rot_xy      = load(f, "debug/rotation_xy")
        nav_cmd     = load(f, "debug/navigation_command")
        delta_h     = load(f, "debug/delta_height")
        plan_h      = load(f, "debug/planner_height")
        plan_mode   = load(f, "debug/planner_mode")
        target_pos  = load(f, "vr_3point_local_target")
        left_joints = load(f, "action.left_hand_joints")
        right_joints= load(f, "action.right_hand_joints")
        eef         = load(f, "action_eef")
        delta_eef   = load(f, "action_delta_eef")
        diff_ms     = load(f, "timestamp_diff_ms")

    fig = plt.figure(figsize=(20, 26))
    fig.suptitle(f"SONIC conversion debug — {h5_path.name}  ({N} frames @ 50 Hz)",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.32)

    # Row 0: trajectory (left), cam sync (right)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_sync = fig.add_subplot(gs[0, 1])
    panel_trajectory(ax_traj, pos_xyz, rot_xy)
    panel_cam_sync(ax_sync, t, diff_ms)

    # Row 1: height (left), planner mode (right)
    ax_h    = fig.add_subplot(gs[1, 0])
    ax_mode = fig.add_subplot(gs[1, 1])
    panel_height(ax_h, t, delta_h, plan_h)
    panel_planner_mode(ax_mode, t, plan_mode)
    ax_mode.set_xlabel("time (s)")

    # Row 2: nav commands (left), 3-point positions (right)
    ax_nav  = fig.add_subplot(gs[2, 0])
    ax_3pt  = fig.add_subplot(gs[2, 1])
    panel_nav_commands(ax_nav, t, nav_cmd)
    panel_3point_pos(ax_3pt, t, target_pos)

    # Row 3: hand joints (left), EEF pos (right)
    ax_hand = fig.add_subplot(gs[3, 0])
    ax_eef  = fig.add_subplot(gs[3, 1])
    panel_hand_joints(ax_hand, t, left_joints, right_joints)
    panel_eef_pos(ax_eef, t, eef)

    # Row 4: delta EEF (span both columns)
    ax_deef = fig.add_subplot(gs[4, :])
    panel_delta_eef(ax_deef, t, delta_eef)
    ax_deef.set_xlabel("time (s)")

    for ax in [ax_traj, ax_sync, ax_h, ax_nav, ax_3pt, ax_hand, ax_eef]:
        ax.set_xlabel("time (s)")

    # Print stats to console
    print(f"\n{'─'*60}")
    print(f"  File : {h5_path}")
    print(f"  Frames: {N}  ({N/50:.1f} s @ 50 Hz)")
    print(f"  Cam sync: mean={diff_ms.mean():.1f}ms  max={diff_ms.max():.1f}ms  "
          f">16.7ms: {(diff_ms > 16.7).sum()}")
    print(f"  Planner height: {plan_h.min():.3f} – {plan_h.max():.3f} m")
    unique_m, counts_m = np.unique(plan_mode, return_counts=True)
    for m, c in zip(unique_m, counts_m):
        print(f"    mode {m:2d} ({PLANNER_MODE_NAMES.get(int(m),'?'):10s}): "
              f"{c:4d} frames ({100*c/N:.1f}%)")
    print(f"  Hand L closed: {(left_joints.mean(axis=1) > 0.5).sum()} frames")
    print(f"  Hand R closed: {(right_joints.mean(axis=1) > 0.5).sum()} frames")
    print(f"  Target vel: mean={load_arr(diff_ms*0, t, nav_cmd):.3f} m/s"
          if False else "")  # placeholder — just use nav_cmd norms
    vels = np.linalg.norm(nav_cmd[:, :2], axis=1)
    print(f"  Ground speed: mean={vels.mean():.3f}  max={vels.max():.3f} m/s")
    print(f"{'─'*60}\n")

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  summary → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def load_arr(dummy, t, nav_cmd):  # unused helper placeholder
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Debug visualizer for convert_episode.py output")
    ap.add_argument("hdf5", type=Path, help="SONIC-converted episode HDF5")
    ap.add_argument("--show",  action="store_true", help="Open interactive matplotlib window")
    ap.add_argument("--frame", type=int, default=None,
                    help="Also render body skeleton at this frame index")
    ap.add_argument("--out",   type=Path, default=None,
                    help="Output PNG path (default: <hdf5_stem>_debug.png next to input)")
    args = ap.parse_args()

    if not args.hdf5.exists():
        sys.exit(f"File not found: {args.hdf5}")

    if not args.show:
        matplotlib.use("Agg")

    out_path = args.out or args.hdf5.parent / (args.hdf5.stem + "_debug.png")
    make_summary(args.hdf5, out_path if not args.show or args.out else None, args.show)

    if args.frame is not None:
        plot_skeleton_frame(args.hdf5, args.frame, out_path, args.show)


if __name__ == "__main__":
    main()
