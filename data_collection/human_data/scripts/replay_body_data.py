#!/usr/bin/env python3
"""
Replay body/hand motion trajectory from an episode HDF5 file in MeshCat.

Usage:
    python replay_body_data.py [--hdf5 PATH] [--speed 1.0] [--loop]

    --hdf5   Path to episode HDF5 file (default: data_collection/body_data/episode_0.hdf5)
    --speed  Playback speed multiplier (default: 1.0)
    --loop   Loop the animation (default: play once)

Examples:
    python replay_body_data.py
    python replay_body_data.py --hdf5 data_collection/body_data/episode_1.hdf5
    python replay_body_data.py --speed 0.5 --loop
"""

import argparse
import time

import h5py
import meshcat
import meshcat.geometry as g
import numpy as np


# ── Skeleton topology ──────────────────────────────────────────────────────────

def get_body_connections():
    connections = []
    connections.extend([(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)])   # spine + head
    connections.extend([(0, 1), (1, 4), (4, 7), (7, 10)])              # left leg
    connections.extend([(0, 2), (2, 5), (5, 8), (8, 11)])              # right leg
    connections.extend([(9, 13), (13, 16), (16, 18), (18, 20), (20, 22)])  # left arm
    connections.extend([(9, 14), (14, 17), (17, 19), (19, 21), (21, 23)])  # right arm
    return connections


def get_finger_connections():
    connections = []
    connections.extend([(1, 2), (2, 3), (3, 4), (4, 5)])
    connections.extend([(1, 6), (6, 7), (7, 8), (8, 9), (9, 10)])
    connections.extend([(1, 11), (11, 12), (12, 13), (13, 14), (14, 15)])
    connections.extend([(1, 16), (16, 17), (17, 18), (18, 19), (19, 20)])
    connections.extend([(1, 21), (21, 22), (22, 23), (23, 24), (24, 25)])
    return connections


# ── Coordinate transforms (same as collection script) ─────────────────────────

RX90 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
RZ90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
RZS180 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

T_body = RZ90 @ RX90
T_left = RZ90 @ RX90


def _transform_pos(pos, T, extra_T=None):
    """Apply 4x4 transform to a (3,) position vector."""
    mat = np.eye(4)
    mat[:3, 3] = pos
    if extra_T is not None:
        mat = T @ mat @ extra_T
    else:
        mat = T @ mat
    out = mat[:3, 3].copy()
    out[2] += 0.7
    out[0] += 0.55
    return out


def transform_body_positions(positions):
    """positions: (N, 3)"""
    out = np.zeros_like(positions)
    for i, p in enumerate(positions):
        out[i] = _transform_pos(p, T_body)
    return out


def transform_hand_positions(positions, is_right=False):
    """positions: (N, 3)"""
    extra = RZS180 if is_right else None
    out = np.zeros_like(positions)
    for i, p in enumerate(positions):
        out[i] = _transform_pos(p, T_left, extra)
    return out


def transform_single(pos):
    return _transform_pos(pos, T_body)


# ── MeshCat scene setup ────────────────────────────────────────────────────────

def _line(vis, name, color):
    pts = np.zeros((3, 2), dtype=np.float32)
    vis[name].set_object(g.Line(g.PointsGeometry(pts), g.LineBasicMaterial(color=color, linewidth=2)))


def setup_body(vis):
    for i in range(24):
        if i == 0:
            color = 0x00ff00
        elif i == 15:
            color = 0xff0000
        elif i <= 11:
            color = 0x0066ff + i * 0x000011
        elif i <= 15:
            color = 0xffaa00 + (i - 12) * 0x000011
        else:
            color = 0xff6600 + (i - 16) * 0x000011
        vis[f"body/j{i}"].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=color))
        vis[f"body/j{i}"].set_transform(np.eye(4))

    for idx, (a, b) in enumerate(get_body_connections()):
        _line(vis, f"body/l{idx}", 0xffffff)


def setup_hand(vis, prefix, color_set):
    base_colors, line_color = color_set
    for i in range(26):
        vis[f"{prefix}/k{i}"].set_object(g.Sphere(0.01), g.MeshLambertMaterial(color=base_colors[i % len(base_colors)]))
        vis[f"{prefix}/k{i}"].set_transform(np.eye(4))
    for idx, _ in enumerate(get_finger_connections()):
        _line(vis, f"{prefix}/l{idx}", line_color)


def setup_controllers(vis):
    vis["ctrl/left"].set_object(g.Sphere(0.03), g.MeshLambertMaterial(color=0x0066ff))
    vis["ctrl/left"].set_transform(np.eye(4))
    vis["ctrl/right"].set_object(g.Sphere(0.03), g.MeshLambertMaterial(color=0xff6600))
    vis["ctrl/right"].set_transform(np.eye(4))


# ── Per-frame update ───────────────────────────────────────────────────────────

def _set_pos(vis, name, pos):
    T = np.eye(4)
    T[:3, 3] = pos
    vis[name].set_transform(T)


def _set_line(vis, name, p0, p1, color):
    pts = np.array([p0, p1], dtype=np.float32).T
    vis[name].set_object(g.Line(g.PointsGeometry(pts), g.LineBasicMaterial(color=color, linewidth=2)))


def update_body(vis, body_pose_frame):
    raw_pos = body_pose_frame[:, :3]
    pos = transform_body_positions(raw_pos)
    for i in range(24):
        _set_pos(vis, f"body/j{i}", pos[i])
    for idx, (a, b) in enumerate(get_body_connections()):
        _set_line(vis, f"body/l{idx}", pos[a], pos[b], 0xffffff)


def update_hand(vis, prefix, hand_pose_frame, is_right=False, line_color=0xffffff):
    raw_pos = hand_pose_frame[:, :3]
    pos = transform_hand_positions(raw_pos, is_right=is_right)
    for i in range(26):
        _set_pos(vis, f"{prefix}/k{i}", pos[i])
    for idx, (a, b) in enumerate(get_finger_connections()):
        _set_line(vis, f"{prefix}/l{idx}", pos[a], pos[b], line_color)


def update_controllers(vis, left_pose, right_pose):
    if np.linalg.norm(left_pose[:3]) > 1e-6:
        _set_pos(vis, "ctrl/left", transform_single(left_pose[:3]))
    if np.linalg.norm(right_pose[:3]) > 1e-6:
        _set_pos(vis, "ctrl/right", transform_single(right_pose[:3]))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Replay body/hand motion from HDF5 in MeshCat")
    parser.add_argument("--hdf5", default="data_collection/body_data/episode_0.hdf5",
                        help="Path to episode HDF5 file")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the animation")
    args = parser.parse_args()

    print(f"Loading {args.hdf5} ...")
    with h5py.File(args.hdf5, "r") as f:
        body_pose = f["body_pose"][:]                       # (T, 24, 7)
        left_hand_pose = f["left_hand_pose"][:]             # (T, 26, 7)
        right_hand_pose = f["right_hand_pose"][:]           # (T, 26, 7)
        left_controller_pose = f["left_controller_pose"][:] # (T, 7)
        right_controller_pose = f["right_controller_pose"][:]
        left_hand_active = f["left_hand_active"][:]
        right_hand_active = f["right_hand_active"][:]
        timestamps_ns = f["local_timestamps_ns"][:]

    n_frames = len(timestamps_ns)
    duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    fps = n_frames / duration_s
    frame_dt = 1.0 / (fps * args.speed)

    print(f"  Frames : {n_frames}")
    print(f"  Duration: {duration_s:.2f}s  ({fps:.1f} FPS)")
    print(f"  Playback speed: {args.speed}x  → frame interval {frame_dt*1000:.1f}ms")

    print("\nInitializing MeshCat ...")
    vis = meshcat.Visualizer()
    vis.open()
    vis.delete()

    setup_body(vis)
    setup_hand(vis, "left_hand",
               ([0x0066ff, 0x0088ff, 0x00aaff, 0x00ccff, 0x00eeff, 0x00ffff], 0x66ccff))
    setup_hand(vis, "right_hand",
               ([0xff6600, 0xff8800, 0xffaa00, 0xffcc00, 0xffee00, 0xffff00], 0xff9966))
    setup_controllers(vis)

    print("\nOpen your browser at http://localhost:7000/static/ to view the replay.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            print(f"Playing episode ({n_frames} frames) ...")
            for i in range(n_frames):
                t0 = time.perf_counter()

                update_body(vis, body_pose[i])
                update_controllers(vis, left_controller_pose[i], right_controller_pose[i])

                if left_hand_active[i]:
                    update_hand(vis, "left_hand", left_hand_pose[i],
                                is_right=False, line_color=0x66ccff)
                if right_hand_active[i]:
                    update_hand(vis, "right_hand", right_hand_pose[i],
                                is_right=True, line_color=0xff9966)

                if i % 100 == 0:
                    pct = i / n_frames * 100
                    elapsed = i * frame_dt
                    print(f"\r  Frame {i+1}/{n_frames} ({pct:.0f}%)  t={elapsed:.1f}s", end="", flush=True)

                elapsed = time.perf_counter() - t0
                sleep_time = frame_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"\r  Frame {n_frames}/{n_frames} (100%) — done.              ")

            if not args.loop:
                break
            print("Looping ...\n")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
