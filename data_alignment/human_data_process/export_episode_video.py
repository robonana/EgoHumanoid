#!/usr/bin/env python3
"""
Export final episode HDF5 to MP4 with overlaid action data.

Layout (1280 x 760):
  Row 0 (1280 x 360): left camera | right camera (each 640x360)
  Row 1 (1280 x 200):
    - Top-down direction widget  (200x200)  — driven by teleop_navigate_command
    - Teleop bars + raw nav_command + delta_height (centre)
    - Hand status bars (right)
  Row 2 (1280 x 200): End-effector panel
    - XZ position scatter (200x200) showing both hands in base frame
    - Left hand: position bars (x,y,z) + delta bars (dx,dy,dz,drx,dry,drz)
    - Right hand: same

Usage:
    python export_episode_video.py [--hdf5 PATH] [--out PATH] [--fps FPS] [--speed SPEED]
"""

import argparse
import io
import math

import cv2
import h5py
import numpy as np
from PIL import Image

# ── colours (BGR for OpenCV) ──────────────────────────────────────────────────
BG          = (30, 30, 30)
WHITE       = (255, 255, 255)
GRAY        = (120, 120, 120)
GREEN       = (80, 200, 80)
RED         = (80, 80, 220)
YELLOW      = (60, 210, 210)
CYAN        = (200, 180, 60)
ORANGE      = (60, 140, 240)
PANEL_BG    = (20, 20, 20)

GREEN_DIM   = (40, 100, 40)
CYAN_DIM    = (100, 90, 30)
ORANGE_DIM  = (30, 70, 120)

# EEF-panel colours
LCOL        = (220, 160, 50)   # left  hand: golden-cyan (BGR)
RCOL        = (80,  80, 220)   # right hand: red (BGR)
LCOL_DIM    = (110, 80, 25)
RCOL_DIM    = (40,  40, 110)

W, H_NAV    = 1280, 560        # original top + nav-panel height
CAM_W, CAM_H = 640, 360
PANEL_H     = H_NAV - CAM_H   # 200
NAV_SZ      = PANEL_H         # 200×200 nav widget
EEF_H       = 200              # height of new EEF panel
H           = H_NAV + EEF_H   # 760 total


def decode_jpeg(buf) -> np.ndarray:
    img = Image.open(io.BytesIO(bytes(buf))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ── navigation widget ─────────────────────────────────────────────────────────

def draw_nav_widget(canvas, x, y, sz, vx, vy, yaw):
    cx, cy = x + sz // 2, y + sz // 2
    r = sz // 2 - 10
    cv2.circle(canvas, (cx, cy), r, (45, 45, 45), -1)
    cv2.circle(canvas, (cx, cy), r, GRAY, 1)
    cv2.line(canvas, (cx - r, cy), (cx + r, cy), (55, 55, 55), 1)
    cv2.line(canvas, (cx, cy - r), (cx, cy + r), (55, 55, 55), 1)
    cv2.putText(canvas, "fwd", (cx - 12, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA)

    speed = math.hypot(vx, vy)
    arrow_len = int(r * min(speed / 0.3, 1.0)) if speed > 1e-4 else 0
    if arrow_len > 4:
        angle_rad = math.atan2(-vy, vx)
        ax = cx + int(arrow_len * math.cos(math.pi / 2 - angle_rad))
        ay = cy - int(arrow_len * math.sin(math.pi / 2 - angle_rad))
        cv2.arrowedLine(canvas, (cx, cy), (ax, ay), GREEN, 2,
                        tipLength=0.25, line_type=cv2.LINE_AA)

    yaw_clamp = max(-math.pi, min(math.pi, yaw))
    if abs(yaw_clamp) > 0.05:
        arc_r = r - 18
        start_deg = -90
        sweep_deg = int(math.degrees(-yaw_clamp))
        d0, d1 = min(start_deg, start_deg + sweep_deg), max(start_deg, start_deg + sweep_deg)
        colour = ORANGE if yaw > 0 else CYAN
        cv2.ellipse(canvas, (cx, cy), (arc_r, arc_r), 0, d0, d1, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{yaw:+.2f}", (cx - 22, cy + arc_r + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{speed:.3f}m/s", (x + 4, y + sz - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)


# ── bar charts ────────────────────────────────────────────────────────────────

def draw_bar(canvas, x, y, bw, bh, value, vmin, vmax, label, colour,
             show_label=True):
    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (50, 50, 50), -1)
    cv2.rectangle(canvas, (x, y), (x + bw, y + bh), GRAY, 1)
    if vmin < 0 < vmax:
        mid = x + bw // 2
        frac = max(-1.0, min(1.0, value / max(abs(vmin), abs(vmax))))
        px = int(mid + frac * (bw // 2))
        if px > mid:
            cv2.rectangle(canvas, (mid, y + 2), (px, y + bh - 2), colour, -1)
        elif px < mid:
            cv2.rectangle(canvas, (px, y + 2), (mid, y + bh - 2), colour, -1)
        cv2.line(canvas, (mid, y), (mid, y + bh), GRAY, 1)
    else:
        frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin) if vmax > vmin else 0))
        px = x + int(frac * bw)
        cv2.rectangle(canvas, (x + 1, y + 2), (px, y + bh - 2), colour, -1)
    if show_label:
        cv2.putText(canvas, f"{label}: {value:+.3f}",
                    (x + 4, y + bh - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, WHITE, 1, cv2.LINE_AA)


# ── EEF XZ scatter view ───────────────────────────────────────────────────────

def draw_eef_scatter(canvas, ox, oy, sz,
                     lx, lz, rx, rz,
                     x_range=(-0.15, 0.80),
                     z_range=(-0.40, 0.40)):
    """Plot both hand positions in the X-Z plane (forward vs height) of base frame."""
    pad = 16
    # background
    cv2.rectangle(canvas, (ox, oy), (ox + sz, oy + sz), (35, 35, 35), -1)
    cv2.rectangle(canvas, (ox, oy), (ox + sz, oy + sz), GRAY, 1)

    def to_px(xv, zv):
        fx = (xv - x_range[0]) / (x_range[1] - x_range[0])
        fz = 1.0 - (zv - z_range[0]) / (z_range[1] - z_range[0])
        px = ox + pad + int(fx * (sz - 2 * pad))
        py = oy + pad + int(fz * (sz - 2 * pad))
        return (int(np.clip(px, ox, ox + sz)), int(np.clip(py, oy, oy + sz)))

    # grid lines at 0
    zero_px = to_px(0, 0)
    mid_x = to_px(0, 0)[0]
    mid_z = to_px(0, 0)[1]
    cv2.line(canvas, (ox + pad, mid_z), (ox + sz - pad, mid_z), (55, 55, 55), 1)
    cv2.line(canvas, (mid_x, oy + pad), (mid_x, oy + sz - pad), (55, 55, 55), 1)

    # axis labels
    cv2.putText(canvas, "x(fwd)", (ox + pad, oy + sz - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "z(up)", (ox + 2, oy + pad + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "EEF pos (base frame)",
                (ox + 4, oy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY, 1, cv2.LINE_AA)

    # hands
    lp = to_px(lx, lz)
    rp = to_px(rx, rz)
    cv2.circle(canvas, lp, 6, LCOL, -1, cv2.LINE_AA)
    cv2.circle(canvas, rp, 6, RCOL, -1, cv2.LINE_AA)
    cv2.putText(canvas, "L", (lp[0] + 7, lp[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, LCOL, 1, cv2.LINE_AA)
    cv2.putText(canvas, "R", (rp[0] + 7, rp[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, RCOL, 1, cv2.LINE_AA)


# ── main frame builder ────────────────────────────────────────────────────────

def build_frame(left_img, right_img,
                tvx, tvy, tyaw,
                nvx, nvy, nyaw,
                dh,
                hand_l, hand_r,
                eef,        # (14,) [lx,ly,lz,lqx,lqy,lqz,lqw, rx,ry,rz,...]
                deef,       # (12,) [ldx,ldy,ldz,lrx,lry,lrz, rdx,rdy,rdz,rrx,rry,rrz]
                frame_idx, n_frames, time_s) -> np.ndarray:

    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    # ── cameras ──
    canvas[0:CAM_H, 0:CAM_W]  = resize(left_img,  CAM_W, CAM_H)
    canvas[0:CAM_H, CAM_W:W]  = resize(right_img, CAM_W, CAM_H)
    cv2.line(canvas, (CAM_W, 0), (CAM_W, CAM_H), GRAY, 1)
    cv2.putText(canvas, "LEFT",  (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "RIGHT", (CAM_W + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

    # ── nav panel background ──
    py = CAM_H
    cv2.rectangle(canvas, (0, py), (W, H_NAV), PANEL_BG, -1)
    cv2.line(canvas, (0, py), (W, py), GRAY, 1)

    draw_nav_widget(canvas, 0, py, NAV_SZ, tvx, tvy, tyaw)
    cv2.line(canvas, (NAV_SZ, py), (NAV_SZ, H_NAV), GRAY, 1)

    # teleop + raw nav bars
    bx, bw = NAV_SZ + 10, 440
    bh, rbh, gap, inner = 22, 9, 7, 3
    by = py + 14
    cv2.putText(canvas, "Teleop Navigate Command  (bright=discretised  dim=raw continuous)",
                (bx, py + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, GRAY, 1, cv2.LINE_AA)

    groups = [
        (tvx, nvx, -0.1,  0.2, "vx  (m/s)", GREEN,  GREEN_DIM),
        (tvy, nvy, -0.2,  0.2, "vy  (m/s)", CYAN,   CYAN_DIM),
        (tyaw, nyaw, -0.3, 0.3, "yaw (r/s)", ORANGE, ORANGE_DIM),
    ]
    group_h = bh + inner + rbh
    for k, (tv, nv, vmin, vmax, label, mcol, dcol) in enumerate(groups):
        gy = by + k * (group_h + gap)
        draw_bar(canvas, bx, gy, bw, bh, tv, vmin, vmax, label, mcol)
        draw_bar(canvas, bx, gy + bh + inner, bw, rbh, nv, vmin, vmax, "", dcol, show_label=False)
        cv2.putText(canvas, f"raw {nv:+.3f}",
                    (bx + bw + 4, gy + bh + inner + rbh - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, dcol, 1, cv2.LINE_AA)

    dh_y = by + 3 * (group_h + gap)
    draw_bar(canvas, bx, dh_y, bw, bh, dh, -0.02, 0.02, "dHeight(m)", YELLOW)
    cv2.line(canvas, (bx + bw + 10, py), (bx + bw + 10, H_NAV), GRAY, 1)

    # hand status
    hx = bx + bw + 20
    hw = W - hx - 10
    hby = py + 30
    hbh = 50
    cv2.putText(canvas, "Hand Status",
                (hx, py + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1, cv2.LINE_AA)
    for hand_idx, (label, val, colour) in enumerate([
        ("Left  Hand", hand_l, CYAN),
        ("Right Hand", hand_r, RED),
    ]):
        by_h = hby + hand_idx * (hbh + 20)
        draw_bar(canvas, hx, by_h, hw, hbh, val, 0.0, 1.0, label, colour)
        status = "CLOSED" if val > 0.5 else "open"
        col = WHITE if val > 0.5 else GRAY
        cv2.putText(canvas, status, (hx + 4, by_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # bottom info strip (nav panel)
    info = (f"Frame {frame_idx + 1}/{n_frames}   t={time_s:.2f}s   "
            f"teleop vx={tvx:+.2f} vy={tvy:+.2f} yaw={tyaw:+.2f}   "
            f"raw vx={nvx:+.3f} vy={nvy:+.3f} yaw={nyaw:+.3f}   "
            f"dH={dh:+.4f}   hand L={'C' if hand_l > 0.5 else 'O'} R={'C' if hand_r > 0.5 else 'O'}")
    cv2.putText(canvas, info, (NAV_SZ + 10, H_NAV - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRAY, 1, cv2.LINE_AA)

    # ── EEF panel ────────────────────────────────────────────────────────────
    ey = H_NAV   # panel starts here
    cv2.rectangle(canvas, (0, ey), (W, H), (15, 15, 15), -1)
    cv2.line(canvas, (0, ey), (W, ey), GRAY, 1)

    # XZ scatter (left 210px)
    SCATTER_SZ = 195
    draw_eef_scatter(canvas, 2, ey + 2, SCATTER_SZ,
                     lx=eef[0], lz=eef[2],
                     rx=eef[7], rz=eef[9])
    cv2.line(canvas, (SCATTER_SZ + 4, ey), (SCATTER_SZ + 4, H), GRAY, 1)

    # Left & right hand bars  — position (x,y,z) + delta (dx,dy,dz,drx,dry,drz)
    HAND_BW  = 495    # bar width per hand section
    HAND_GAP = 10     # gap between scatter and left section, and between sections
    ebh   = 16        # bar height
    egap  = 4         # gap between bars
    emarg = 12        # top margin inside panel

    sections = [
        # (label, eef_slice_pos, deef_slice, colour, dim_colour, x_offset)
        ("Left  EEF  (base frame)", slice(0, 3), slice(0, 6),  LCOL, LCOL_DIM,
         SCATTER_SZ + HAND_GAP + 4),
        ("Right EEF  (base frame)", slice(7, 10), slice(6, 12), RCOL, RCOL_DIM,
         SCATTER_SZ + HAND_GAP + 4 + HAND_BW + HAND_GAP),
    ]

    for sec_label, pos_sl, delta_sl, col, col_dim, sx in sections:
        cv2.putText(canvas, sec_label, (sx, ey + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA)

        pos_labels  = ["x (fwd, m)", "y (left,m)", "z (up,  m)"]
        delta_labels = ["dx (m)", "dy (m)", "dz (m)", "drx(r)", "dry(r)", "drz(r)"]
        pos_vals   = eef[pos_sl]
        delta_vals = deef[delta_sl]

        by0 = ey + emarg + 8
        bw_half = (HAND_BW - 10) // 2   # position bars left, delta bars right

        # position bars (x,y,z)  — left half
        for k, (lbl, v) in enumerate(zip(pos_labels, pos_vals)):
            yy = by0 + k * (ebh + egap)
            draw_bar(canvas, sx, yy, bw_half, ebh, v, -0.5, 0.8, lbl, col)

        # delta bars (dx,dy,dz,drx,dry,drz) — right half
        dx2 = sx + bw_half + 6
        delta_ranges = [(-0.1, 0.1)] * 3 + [(-0.4, 0.4)] * 3
        for k, (lbl, v, (vmin, vmax)) in enumerate(zip(delta_labels, delta_vals, delta_ranges)):
            yy = by0 + k * (ebh + egap)
            draw_bar(canvas, dx2, yy, bw_half, ebh, v, vmin, vmax, lbl, col_dim)

        # vertical separator between left and right hand sections
        if sx == sections[0][5]:  # after left section
            sep_x = sx + HAND_BW + HAND_GAP // 2
            cv2.line(canvas, (sep_x, ey), (sep_x, H), GRAY, 1)

    # progress bar
    prog = int(W * (frame_idx + 1) / n_frames)
    cv2.rectangle(canvas, (0, H - 4), (prog, H), GREEN, -1)

    return canvas


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export final HDF5 episode to MP4")
    parser.add_argument("--hdf5",  default="data_collection/final/episode_0.hdf5")
    parser.add_argument("--out",   default=None)
    parser.add_argument("--fps",   type=float, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    out_path = args.out or args.hdf5.replace(".hdf5", "_viz.mp4")

    print(f"Loading {args.hdf5} ...")
    with h5py.File(args.hdf5, "r") as f:
        imgs_l  = f["observation_image_left"][:]
        imgs_r  = f["observation_image_right"][:]
        teleop  = f["teleop_navigate_command"][:]
        navcmd  = f["navigation_command"][:]
        dheight = f["delta_height"][:]
        hand    = f["hand_status"][:]
        eef_all = f["action_eef"][:]
        deef_all= f["action_delta_eef"][:]
        ts      = f["local_timestamps_ns"][:]

    n = len(ts)
    duration = (ts[-1] - ts[0]) / 1e9
    src_fps  = n / duration
    out_fps  = (args.fps or src_fps) * args.speed
    print(f"  {n} frames  |  {duration:.1f}s  |  {src_fps:.1f} FPS  →  {out_fps:.1f} FPS out")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (W, H))

    print(f"Writing {out_path} ...")
    for i in range(n):
        left  = decode_jpeg(imgs_l[i])
        right = decode_jpeg(imgs_r[i])
        t_s   = (ts[i] - ts[0]) / 1e9

        frame = build_frame(
            left, right,
            tvx=float(teleop[i, 0]),
            tvy=float(teleop[i, 1]),
            tyaw=float(teleop[i, 2]),
            nvx=float(navcmd[i, 0]),
            nvy=float(navcmd[i, 1]),
            nyaw=float(navcmd[i, 2]),
            dh=float(dheight[i]),
            hand_l=float(hand[i, 0]),
            hand_r=float(hand[i, 1]),
            eef=eef_all[i],
            deef=deef_all[i],
            frame_idx=i, n_frames=n, time_s=t_s,
        )
        writer.write(frame)

        if i % 50 == 0 or i == n - 1:
            print(f"\r  {i+1}/{n} ({100*(i+1)//n}%)", end="", flush=True)

    writer.release()
    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    main()
