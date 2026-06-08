"""
video_episode.py  —  render a debug video from a SONIC-converted episode HDF5

Layout per frame:
  ┌──────────────────────┬──────────────────────┐
  │                      │  skeleton top-down   │
  │   camera (left eye)  │  (XY, bird's eye)    │
  │                      ├──────────────────────┤
  │                      │  skeleton side view  │
  │                      │  (XZ, height)        │
  ├──────────────────────┴──────────────────────┤
  │   timeline strip: mode | height | hands     │
  └─────────────────────────────────────────────┘

Usage:
    python video_episode.py episode_0.hdf5
    python video_episode.py episode_0.hdf5 --out episode_0_debug.mp4 --fps 25
    python video_episode.py episode_0.hdf5 --fps 50   # full speed
"""

import argparse
import sys
from pathlib import Path
from io import BytesIO

import cv2
import h5py
import numpy as np
from PIL import Image

from scipy.spatial.transform import Rotation as R

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_WORLD_ROT_MAT = (R.from_euler('z', -90, degrees=True) * R.from_euler('x', 90, degrees=True)).as_matrix()

BODY_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5),
    (4, 7), (5, 8),
    (7, 10), (8, 11),
    (3, 6), (6, 9),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (20, 22), (21, 23),
]

SPECIAL_JOINTS = {
    15: ("head",    (255, 80,  80)),
    20: ("L wrist", (80,  200, 80)),
    21: ("R wrist", (255, 165, 0)),
}

PLANNER_MODE_NAMES  = {0: "idle", 1: "slowWalk", 2: "walk", 4: "squat", 22: "crouch"}
PLANNER_MODE_COLORS_BGR = {
    0:  (140, 140, 140),
    1:  (200, 160,  60),
    2:  (220, 120,  30),
    4:  ( 30, 140, 220),
    22: (180,  50, 160),
}

# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_jpeg(buf: bytes) -> np.ndarray:
    """JPEG bytes → BGR uint8 ndarray."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img if img is not None else np.zeros((480, 640, 3), np.uint8)


def resize_keep_aspect(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((target_h, target_w, 3), np.uint8)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton projection helpers
# ─────────────────────────────────────────────────────────────────────────────

class SkeletonView:
    """Projects body joints onto a canvas for one orthographic view."""

    def __init__(self, W: int, H: int, axis_x: int, axis_y: int, title: str,
                 margin: float = 0.5):
        self.W, self.H = W, H
        self.ax_x, self.ax_y = axis_x, axis_y
        self.title = title
        self.margin = margin
        self._scale = None
        self._cx = None
        self._cy = None

    def fit(self, all_pts: np.ndarray):
        """Pre-compute scale/offset from all frames so the view doesn't jump."""
        xs = all_pts[:, :, self.ax_x].ravel()
        ys = all_pts[:, :, self.ax_y].ravel()
        xmin, xmax = xs.min() - self.margin, xs.max() + self.margin
        ymin, ymax = ys.min() - self.margin, ys.max() + self.margin
        s = min(self.W / (xmax - xmin), self.H / (ymax - ymin)) * 0.88
        self._scale = s
        self._cx = (xmin + xmax) / 2
        self._cy = (ymin + ymax) / 2

    def _project(self, pts_3d: np.ndarray) -> np.ndarray:
        """(24,3) → (24,2) pixel coords on canvas."""
        px = (pts_3d[:, self.ax_x] - self._cx) * self._scale + self.W / 2
        py = self.H / 2 - (pts_3d[:, self.ax_y] - self._cy) * self._scale
        return np.stack([px, py], axis=1).astype(int)

    def render(self, pts_3d: np.ndarray,
               trail_pts: list[np.ndarray] | None = None) -> np.ndarray:
        """Return a BGR canvas for this frame."""
        canvas = np.ones((self.H, self.W, 3), np.uint8) * 30  # dark bg

        # Trajectory trail of pelvis
        if trail_pts and len(trail_pts) > 1:
            trail_px = self._project(np.array([p[0] for p in trail_pts]))
            for k in range(1, len(trail_px)):
                alpha = k / len(trail_px)
                col = (int(60 * alpha), int(120 * alpha), int(200 * alpha))
                cv2.line(canvas, tuple(trail_px[k-1]), tuple(trail_px[k]), col, 1)

        px = self._project(pts_3d)

        # Draw edges
        for a, b in BODY_EDGES:
            pa, pb = tuple(px[a]), tuple(px[b])
            if self._in_canvas(pa) or self._in_canvas(pb):
                cv2.line(canvas, pa, pb, (180, 180, 180), 1, cv2.LINE_AA)

        # Draw joints
        for idx in range(len(pts_3d)):
            if idx in SPECIAL_JOINTS:
                label, col_rgb = SPECIAL_JOINTS[idx]
                col_bgr = (col_rgb[2], col_rgb[1], col_rgb[0])
                cv2.circle(canvas, tuple(px[idx]), 5, col_bgr, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, tuple(px[idx]), 2, (200, 200, 200), -1)

        # Title
        cv2.putText(canvas, self.title, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas

    def _in_canvas(self, p):
        return 0 <= p[0] < self.W and 0 <= p[1] < self.H


# ─────────────────────────────────────────────────────────────────────────────
# Timeline strip
# ─────────────────────────────────────────────────────────────────────────────

def build_timeline_strip(N: int, W: int, H: int,
                         planner_mode: np.ndarray,
                         planner_height: np.ndarray,
                         left_joints: np.ndarray,
                         right_joints: np.ndarray,
                         diff_ms: np.ndarray) -> np.ndarray:
    """Pre-render the full timeline strip (W×H) once."""
    strip = np.zeros((H, W, 3), np.uint8)
    row_h = H // 4

    # Row 0: planner mode color bar
    for i in range(N):
        x0 = int(i / N * W)
        x1 = int((i + 1) / N * W)
        col = PLANNER_MODE_COLORS_BGR.get(int(planner_mode[i]), (100, 100, 100))
        strip[0:row_h, x0:x1] = col

    # Row 1: height bar (orange fill proportional to height)
    h_min, h_max = 0.55, 0.85
    for i in range(N):
        x = int(i / N * W)
        frac = (planner_height[i] - h_min) / (h_max - h_min)
        frac = float(np.clip(frac, 0, 1))
        y_fill = int(row_h * (1 - frac))
        strip[row_h + y_fill: 2*row_h, x] = (40, 140, 220)

    # Row 2: hand state (L=blue, R=orange)
    for i in range(N):
        x = int(i / N * W)
        l_closed = left_joints[i].mean() > 0.5
        r_closed = right_joints[i].mean() > 0.5
        mid = 2 * row_h + row_h // 2
        if l_closed:
            strip[2*row_h: mid, x] = (220, 100, 40)      # blue (BGR) for left
        if r_closed:
            strip[mid: 3*row_h, x] = (40, 140, 220)      # orange (BGR) for right

    # Row 3: cam sync (red where > 16.7ms)
    for i in range(N):
        x = int(i / N * W)
        frac = float(np.clip(diff_ms[i] / 33.0, 0, 1))
        col = (40, 40, int(40 + 200 * frac))
        strip[3*row_h: H, x] = col

    # Labels on right edge
    labels = [
        (row_h // 2,    "mode"),
        (row_h + row_h // 2, "height"),
        (2*row_h + row_h // 2, "hands"),
        (3*row_h + row_h // 2, "sync"),
    ]
    for y, text in labels:
        cv2.putText(strip, text, (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

    return strip


def draw_timeline_cursor(strip_base: np.ndarray, frame_idx: int,
                          N: int, W: int) -> np.ndarray:
    strip = strip_base.copy()
    x = int(frame_idx / N * W)
    cv2.line(strip, (x, 0), (x, strip.shape[0]), (255, 255, 255), 1)
    return strip


# ─────────────────────────────────────────────────────────────────────────────
# Status overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_status(canvas: np.ndarray, frame_idx: int, t: float,
                mode: int, height: float, diff_ms: float,
                l_closed: bool, r_closed: bool,
                target_vel: float):
    lines = [
        f"t={t:.2f}s  frame={frame_idx}",
        f"mode: {PLANNER_MODE_NAMES.get(mode, str(mode))}",
        f"height: {height:.3f} m",
        f"speed: {target_vel:.3f} m/s",
        f"cam sync: {diff_ms:.1f} ms",
        f"L hand: {'CLOSED' if l_closed else 'open '}   R hand: {'CLOSED' if r_closed else 'open '}",
    ]
    colors = [
        (220, 220, 220),
        PLANNER_MODE_COLORS_BGR.get(mode, (180, 180, 180)),
        (40, 200, 220),
        (200, 200, 200),
        (80, 80, 255) if diff_ms > 16.7 else (200, 200, 200),
        (220, 220, 220),
    ]
    y0 = 24
    for line, col in zip(lines, colors):
        cv2.putText(canvas, line, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
        y0 += 22


# ─────────────────────────────────────────────────────────────────────────────
# Main render loop
# ─────────────────────────────────────────────────────────────────────────────

def render_video(h5_path: Path, out_path: Path, fps: float, trail_len: int):
    with h5py.File(h5_path) as f:
        N = len(f["local_timestamps_ns"])
        body_pose   = f["debug/body_pose"][:]        # (N,24,7)  PICO frame
        pos_xyz     = f["debug/positions_xyz"][:]    # (N,3) MuJoCo
        plan_mode   = f["debug/planner_mode"][:]
        plan_height = f["debug/planner_height"][:]
        delta_height= f["debug/delta_height"][:]
        target_vel  = f["debug/target_vel"][:]
        diff_ms     = f["timestamp_diff_ms"][:]
        l_joints    = f["action.left_hand_joints"][:]
        r_joints    = f["action.right_hand_joints"][:]
        jpeg_l      = f["observation_image_left"]    # vlen dataset — read per frame

        # Pre-convert all body positions to MuJoCo frame
        pts_all = np.einsum("ij,nkj->nki", _WORLD_ROT_MAT, body_pose[:, :, :3])  # (N,24,3)

    # Layout dimensions
    CAM_W, CAM_H   = 640, 480
    SKEL_W, SKEL_H = 320, 240
    STRIP_H        = 80
    FRAME_W = CAM_W + SKEL_W
    FRAME_H = CAM_H + STRIP_H

    # Pre-fit skeleton views to all frames
    view_top  = SkeletonView(SKEL_W, SKEL_H, axis_x=0, axis_y=1, title="top-down (XY)")
    view_side = SkeletonView(SKEL_W, SKEL_H, axis_x=0, axis_y=2, title="side view (XZ)")
    view_top.fit(pts_all)
    view_side.fit(pts_all)

    # Pre-render timeline strip
    timeline_base = build_timeline_strip(
        N, FRAME_W, STRIP_H, plan_mode, plan_height, l_joints, r_joints, diff_ms
    )

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        sys.exit(f"Could not open video writer for {out_path}")

    print(f"  Rendering {N} frames → {out_path}  ({fps:.0f} fps)")

    with h5py.File(h5_path) as f:
        jpeg_l = f["observation_image_left"]

        for i in range(N):
            # ── Camera image ──
            cam = decode_jpeg(bytes(jpeg_l[i]))
            cam = resize_keep_aspect(cam, CAM_W, CAM_H)

            # ── Status overlay on camera ──
            l_closed = bool(l_joints[i].mean() > 0.5)
            r_closed = bool(r_joints[i].mean() > 0.5)
            draw_status(cam, i, i / 50.0, int(plan_mode[i]), float(plan_height[i]),
                        float(diff_ms[i]), l_closed, r_closed, float(target_vel[i]))

            # ── Skeleton views ──
            trail = [pts_all[max(0, i - trail_len):i + 1, :, :][k]
                     for k in range(min(trail_len, i + 1))]
            sk_top  = view_top.render(pts_all[i], trail)
            sk_side = view_side.render(pts_all[i], trail)

            # ── Assemble right column ──
            right_col = np.vstack([sk_top, sk_side])   # (CAM_H, SKEL_W, 3)

            # ── Timeline strip ──
            strip = draw_timeline_cursor(timeline_base, i, N, FRAME_W)

            # ── Final frame ──
            top_row   = np.hstack([cam, right_col])          # (CAM_H, FRAME_W, 3)
            frame_bgr = np.vstack([top_row, strip])          # (FRAME_H, FRAME_W, 3)

            writer.write(frame_bgr)

            if i % 100 == 0:
                print(f"    {i}/{N}  ({100*i/N:.0f}%)", end="\r", flush=True)

    writer.release()
    print(f"\n  Done → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Render debug video from SONIC episode HDF5")
    ap.add_argument("hdf5", type=Path, help="SONIC-converted episode HDF5")
    ap.add_argument("--out",       type=Path,  default=None,
                    help="Output MP4 path (default: <stem>_debug.mp4 next to input)")
    ap.add_argument("--fps",       type=float, default=25.0,
                    help="Output video FPS (default 25; data is always 50 Hz)")
    ap.add_argument("--trail",     type=int,   default=50,
                    help="Pelvis trail length in frames (default 50 = 1 s)")
    ap.add_argument("--batch-dir", type=Path,  default=None,
                    help="Process all .hdf5 files in this directory")
    args = ap.parse_args()

    if args.batch_dir:
        files = sorted(args.batch_dir.glob("*.hdf5"))
        if not files:
            sys.exit(f"No HDF5 files in {args.batch_dir}")
        for f in files:
            out = f.parent / (f.stem + "_debug.mp4")
            print(f"\n{'='*60}\n  {f.name}\n{'='*60}")
            render_video(f, out, args.fps, args.trail)
    else:
        if not args.hdf5.exists():
            sys.exit(f"File not found: {args.hdf5}")
        out = args.out or args.hdf5.parent / (args.hdf5.stem + "_debug.mp4")
        render_video(args.hdf5, out, args.fps, args.trail)


if __name__ == "__main__":
    main()
