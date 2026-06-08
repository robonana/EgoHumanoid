#!/usr/bin/env python3
"""
Convert EgoHumanoid raw body HDF5 + ZED SVO2 to SONIC-compatible HDF5 at 50 Hz.

Pipeline:
  1. Stride-2 decimate body data 100 Hz -> 50 Hz
  2. Read ZED SVO2 at 60 FPS, nearest-frame match to 50 Hz body timestamps
  3. Compute navigation commands (tangent pipeline) at 50 Hz
  4. Compute EEF in pelvis frame at 50 Hz
  5. Compute upper-body 3-point targets (head + wrists in pelvis heading frame)
  6. Compute 6D hand joint targets (binary open/close templates)
  7. Compute SONIC planner commands (movement_dir, facing_dir, target_vel, height, mode)

Usage:
  # single episode
  python convert_episode.py \\
    data_collection/body_data/0521_1/episode_0.hdf5 \\
    --svo data_collection/body_data/0521_1/episode_0.svo2 \\
    --out data_collection/sonic/episode_0.hdf5

  # batch (auto-finds svo2 next to hdf5)
  python convert_episode.py --batch data_collection/body_data/0521_1 \\
    --out-dir data_collection/sonic
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp

# ── Joint indices in body_pose ────────────────────────────────────────────────
PELVIS_IDX      = 0
LEFT_HIP_IDX    = 1
RIGHT_HIP_IDX   = 2
HEAD_IDX        = 15
LEFT_WRIST_IDX  = 20
RIGHT_WRIST_IDX = 21
LEFT_HAND_IDX   = 22
RIGHT_HAND_IDX  = 23

# Finger tip indices within hand_pose (26 joints)
FINGER_TIPS = [5, 10, 15, 20, 25]   # Thumb, Index, Middle, Ring, Little

# ── Coordinate transform: PICO world → MuJoCo-compatible ─────────────────────
# Rotate +90° around X, then -90° around Z  (same as process_human_eef_pipeline.py)
_WORLD_ROT     = R.from_euler('z', -90, degrees=True) * R.from_euler('x', 90, degrees=True)
_WORLD_ROT_MAT = _WORLD_ROT.as_matrix()   # (3, 3)

# ── G1 / SONIC planner constants (tune against SONIC docs) ───────────────────
G1_NOMINAL_ROOT_HEIGHT  = 0.793   # standing pelvis height (m)
HUMAN_TO_G1_HEIGHT_SCALE = 0.8   # human delta-height → G1 delta-height
HEIGHT_MIN               = 0.55
HEIGHT_MAX               = 0.85
SQUAT_HEIGHT_THRESH      = G1_NOMINAL_ROOT_HEIGHT - 0.15   # ≈ 0.643 m
CROUCH_HEIGHT_THRESH     = G1_NOMINAL_ROOT_HEIGHT - 0.08   # ≈ 0.713 m
MAX_SPEED                = 1.5    # m/s clip for target_vel

# ── 6-DoF hand joint templates  (open=0, closed=1; tune to Dex3 limits) ──────
HAND_OPEN_6D   = np.zeros(6, dtype=np.float32)
HAND_CLOSED_6D = np.ones(6,  dtype=np.float32)

# ── Savgol defaults ───────────────────────────────────────────────────────────
SG_WINDOW = 27    # ~0.54 s at 50 Hz (odd)
SG_POLY   = 2
SG_PASSES = 2


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SVO2 reader
# ═══════════════════════════════════════════════════════════════════════════════

def read_svo2(svo2_path: Path) -> tuple[np.ndarray, list[bytes]]:
    """
    Read all frames from ZED SVO2 file.

    Returns
    -------
    timestamps_ns : (M,) int64   — ZED image timestamps in nanoseconds
    jpeg_left     : list[bytes]  — JPEG-compressed left images (length M)
    jpeg_right    : list[bytes]  — JPEG-compressed right images (length M)
    """
    import pyzed.sl as sl

    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo2_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Cannot open SVO2 {svo2_path}: {err}")

    nb = zed.get_svo_number_of_frames()
    zed.set_svo_position(0)

    mat_l, mat_r = sl.Mat(), sl.Mat()
    timestamps_ns, jpeg_l, jpeg_r = [], [], []

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    frame = 0
    while frame < nb:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            break
        ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        timestamps_ns.append(ts)

        zed.retrieve_image(mat_l, sl.VIEW.LEFT,  sl.MEM.CPU)
        zed.retrieve_image(mat_r, sl.VIEW.RIGHT, sl.MEM.CPU)

        for mat, lst in ((mat_l, jpeg_l), (mat_r, jpeg_r)):
            arr = mat.get_data()
            if arr.ndim == 3 and arr.shape[2] == 4:   # BGRA → BGR
                arr = arr[:, :, :3]
            _, buf = cv2.imencode('.jpg', arr, encode_param)
            lst.append(bytes(buf))

        frame += 1

    zed.close()
    return np.array(timestamps_ns, dtype=np.int64), jpeg_l, jpeg_r


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Body data decimation
# ═══════════════════════════════════════════════════════════════════════════════

def decimate_body(h5_path: Path, stride: int = 2) -> dict:
    """Read raw body HDF5 and stride-decimate to target rate."""
    with h5py.File(str(h5_path), 'r') as f:
        data = {
            'body_pose':             f['body_pose'][::stride],           # (N,24,7)
            'left_hand_pose':        f['left_hand_pose'][::stride],      # (N,26,7)
            'right_hand_pose':       f['right_hand_pose'][::stride],     # (N,26,7)
            'left_hand_active':      f['left_hand_active'][::stride],    # (N,)
            'right_hand_active':     f['right_hand_active'][::stride],   # (N,)
            'local_timestamps_ns':   f['local_timestamps_ns'][::stride], # (N,)
            'dt_body':               float(f.attrs.get('collection_interval_s', 0.01)) * stride,
        }
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Camera sync
# ═══════════════════════════════════════════════════════════════════════════════

def sync_camera(
    body_ts_ns:  np.ndarray,
    cam_ts_ns:   np.ndarray,
    jpeg_l:      list[bytes],
    jpeg_r:      list[bytes],
) -> tuple[list[bytes], list[bytes], np.ndarray, np.ndarray]:
    """
    Nearest-frame match: for each body timestamp find the closest camera frame.

    Returns matched_jpeg_l, matched_jpeg_r, matched_cam_ts, diff_ms
    """
    idx = np.searchsorted(cam_ts_ns, body_ts_ns)
    idx = np.clip(idx, 0, len(cam_ts_ns) - 1)
    prev_idx = np.maximum(idx - 1, 0)

    curr_diff = np.abs(cam_ts_ns[idx]      - body_ts_ns)
    prev_diff = np.abs(cam_ts_ns[prev_idx] - body_ts_ns)
    best = np.where(curr_diff < prev_diff, idx, prev_idx)

    matched_l   = [jpeg_l[i] for i in best]
    matched_r   = [jpeg_r[i] for i in best]
    matched_ts  = cam_ts_ns[best]
    diff_ms     = np.abs(matched_ts - body_ts_ns).astype(np.float32) / 1e6

    return matched_l, matched_r, matched_ts, diff_ms


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Coordinate helpers
# ═══════════════════════════════════════════════════════════════════════════════

def apply_world_transform_pos(positions: np.ndarray) -> np.ndarray:
    """(N,3) PICO positions → (N,3) MuJoCo-compatible positions."""
    return (positions @ _WORLD_ROT_MAT.T)


def apply_world_transform_quat(quats_xyzw: np.ndarray) -> np.ndarray:
    """(N,4) xyzw quaternions in PICO frame → (N,4) xyzw in MuJoCo frame."""
    N = len(quats_xyzw)
    result = np.zeros_like(quats_xyzw)
    for i in range(N):
        q = R.from_quat(quats_xyzw[i])
        result[i] = (_WORLD_ROT * q).as_quat()
    return result


def _make_valid_savgol(n: int, window: int, poly: int) -> tuple[int, int]:
    wl = max(3, min(window, n if n % 2 == 1 else n - 1))
    if wl % 2 == 0:
        wl -= 1
    return max(3, wl), min(max(1, poly), wl - 1)


def smooth_pos(pos: np.ndarray, window: int, poly: int, passes: int) -> np.ndarray:
    if len(pos) < 3:
        return pos
    wl, po = _make_valid_savgol(len(pos), window, poly)
    out = pos.copy()
    for _ in range(passes):
        for c in range(out.shape[1]):
            out[:, c] = savgol_filter(out[:, c], wl, po, mode='nearest')
    return out


def smooth_quat_expmap(quats: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Smooth quaternion sequence in tangent space."""
    n = len(quats)
    q = quats.copy()
    for i in range(1, n):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]

    ref = R.from_quat(q[n // 2])
    rotvecs = np.zeros((n, 3))
    for i in range(n):
        rotvecs[i] = (ref.inv() * R.from_quat(q[i])).as_rotvec()

    wl, po = _make_valid_savgol(n, window, poly)
    for c in range(3):
        rotvecs[:, c] = savgol_filter(rotvecs[:, c], wl, po, mode='nearest')

    result = np.zeros((n, 4))
    for i in range(n):
        result[i] = (ref * R.from_rotvec(rotvecs[i])).as_quat()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Navigation commands (tangent pipeline, reused from process_navigation_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_xy(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _make_heading_continuous(rot_xy: np.ndarray) -> np.ndarray:
    r = _normalize_xy(rot_xy).copy()
    for i in range(1, len(r)):
        if np.dot(r[i - 1], r[i]) < 0.0:
            r[i] *= -1.0
    return r


def compute_navigation_commands(body_pose: np.ndarray, dt: float,
                                 baseline_sec: float = 10.0,
                                 sg_window: int = 11, sg_poly: int = 3,
                                 tangent_lag: int = 5) -> tuple:
    """
    Compute navigation commands from body_pose using the tangent pipeline.

    Returns
    -------
    positions_xyz : (N,3)  pelvis position in MuJoCo frame
    rotation_xy   : (N,2)  heading unit vector in MuJoCo XY plane
    nav_cmd       : (N,3)  [vx, vy, yaw_rate] in local heading frame (m/s, rad/s)
    """
    pelvis  = body_pose[:, PELVIS_IDX,   :3]
    hip_l   = body_pose[:, LEFT_HIP_IDX,  :3]
    hip_r   = body_pose[:, RIGHT_HIP_IDX, :3]
    midhip  = 0.5 * (hip_l + hip_r)

    root_world = apply_world_transform_pos(midhip)

    # Savgol smooth
    wl, po = _make_valid_savgol(len(root_world), sg_window, sg_poly)
    root_sm = root_world.copy()
    for c in range(3):
        root_sm[:, c] = savgol_filter(root_sm[:, c], wl, po, mode='nearest')

    # Baseline (low-frequency heading path)
    bw = max(3, round(baseline_sec / max(dt, 1e-9)))
    if bw % 2 == 0:
        bw += 1
    bwl, bpo = _make_valid_savgol(len(root_sm), bw, 3)
    baseline_xy = root_sm[:, :2].copy()
    for c in range(2):
        baseline_xy[:, c] = savgol_filter(baseline_xy[:, c], bwl, bpo, mode='nearest')

    # Tangent direction
    n = len(baseline_xy)
    lag = max(1, tangent_lag)
    rot_xy = np.zeros((n, 2))
    for i in range(n):
        i0, i1 = max(0, i - lag), min(n - 1, i + lag)
        diff = baseline_xy[i1] - baseline_xy[i0]
        norm = np.linalg.norm(diff)
        rot_xy[i] = diff / norm if norm > 1e-9 else (rot_xy[i - 1] if i > 0 else np.array([1.0, 0.0]))
    rot_xy = _make_heading_continuous(rot_xy)

    positions_xyz = np.zeros((n, 3))
    positions_xyz[:, :2] = baseline_xy
    positions_xyz[:, 2]  = root_sm[:, 2]

    # Navigation velocities (local frame)
    dp   = positions_xyz[1:, :2] - positions_xyz[:-1, :2]
    hx, hy = rot_xy[:-1, 0], rot_xy[:-1, 1]
    nx, ny = -hy, hx
    vx   = (dp[:, 0] * hx + dp[:, 1] * hy) / dt
    vy   = (dp[:, 0] * nx + dp[:, 1] * ny) / dt
    cross = rot_xy[:-1, 0] * rot_xy[1:, 1] - rot_xy[:-1, 1] * rot_xy[1:, 0]
    dot   = rot_xy[:-1, 0] * rot_xy[1:, 0] + rot_xy[:-1, 1] * rot_xy[1:, 1]
    wz   = np.arctan2(cross, dot) / dt

    nav_cmd = np.zeros((n, 3))
    nav_cmd[:-1, 0] = vx
    nav_cmd[:-1, 1] = vy
    nav_cmd[:-1, 2] = wz
    nav_cmd[-1]     = nav_cmd[-2]

    return positions_xyz, rot_xy, nav_cmd


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EEF in pelvis frame  (BASE_IDX = 0)
# ═══════════════════════════════════════════════════════════════════════════════

# Hand local-frame adjustments (from process_human_eef_pipeline.py)
_HAND_LOCAL_X180      = R.from_euler('x', 180, degrees=True)
_LEFT_HAND_LOCAL_Z180 = R.from_euler('z', 180, degrees=True)


def _pose7_to_matrix(p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3]  = p[:3]
    T[:3, :3] = R.from_quat(p[3:7]).as_matrix()
    return T


def _matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()])


def compute_eef_pelvis(body_pose: np.ndarray,
                        sg_window: int = SG_WINDOW,
                        sg_poly:   int = SG_POLY,
                        sg_passes: int = SG_PASSES) -> tuple:
    """
    Compute left/right EEF (HAND joints) in pelvis base frame at native rate.

    Returns left_eef, right_eef — each (N,7) [x,y,z,qx,qy,qz,qw]
    """
    N = len(body_pose)
    pelvis_pose = body_pose[:, PELVIS_IDX, :]

    def to_base(hand_raw):
        result = np.zeros((N, 7))
        for i in range(N):
            T_base = _pose7_to_matrix(pelvis_pose[i])
            T_hand = _pose7_to_matrix(hand_raw[i])
            result[i] = _matrix_to_pose7(np.linalg.inv(T_base) @ T_hand)
        return result

    left_eef  = to_base(body_pose[:, LEFT_HAND_IDX,  :])
    right_eef = to_base(body_pose[:, RIGHT_HAND_IDX, :])

    # Apply world coordinate transform
    for eef in (left_eef, right_eef):
        eef[:, :3] = apply_world_transform_pos(eef[:, :3])
        eef[:, 3:] = apply_world_transform_quat(eef[:, 3:])

    # Hand local-frame corrections
    for i in range(N):
        left_eef[i,  3:] = (_WORLD_ROT * R.from_quat(body_pose[i, LEFT_HAND_IDX,  3:]) *
                             _HAND_LOCAL_X180 * _LEFT_HAND_LOCAL_Z180).as_quat()
        right_eef[i, 3:] = (_WORLD_ROT * R.from_quat(body_pose[i, RIGHT_HAND_IDX, 3:]) *
                             _HAND_LOCAL_X180).as_quat()

    # Smooth
    if sg_window > 0:
        left_eef[:,  :3] = smooth_pos(left_eef[:,  :3], sg_window, sg_poly, sg_passes)
        right_eef[:, :3] = smooth_pos(right_eef[:, :3], sg_window, sg_poly, sg_passes)
        left_eef[:,  3:] = smooth_quat_expmap(left_eef[:,  3:], sg_window, sg_poly)
        right_eef[:, 3:] = smooth_quat_expmap(right_eef[:, 3:], sg_window, sg_poly)

    return left_eef, right_eef


def compute_delta_eef(left_eef: np.ndarray, right_eef: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame delta EEF. Returns (N,12) [l:dx,dy,dz,r,p,y, r:...]."""
    N = len(left_eef)
    delta = np.zeros((N, 12))
    for i in range(1, N):
        for j, eef in enumerate((left_eef, right_eef)):
            T_prev = _pose7_to_matrix(eef[i - 1])
            T_curr = _pose7_to_matrix(eef[i])
            dT = np.linalg.inv(T_prev) @ T_curr
            xyz = dT[:3, 3]
            rpy = R.from_matrix(dT[:3, :3]).as_euler('xyz')
            delta[i, j * 6: j * 6 + 6] = np.concatenate([xyz, rpy])
    return delta


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Upper-body 3-point targets (SONIC vr_3point_local_target)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_3point_targets(body_pose: np.ndarray,
                            positions_xyz: np.ndarray,
                            rotation_xy:   np.ndarray) -> tuple:
    """
    Express head and wrist poses in the pelvis heading frame.

    Pelvis heading frame:
      origin : pelvis position in MuJoCo frame
      X axis : heading direction (rotation_xy)
      Y axis : left normal of heading
      Z axis : up

    Returns
    -------
    vr_3point_local_target     : (N, 9)  [lw_xyz, rw_xyz, head_xyz]
    vr_3point_local_orn_target : (N, 12) [lw_wxyz, rw_wxyz, head_wxyz]
    """
    N = len(body_pose)
    target_pos = np.zeros((N, 9),  dtype=np.float32)
    target_orn = np.zeros((N, 12), dtype=np.float32)

    for i in range(N):
        # Pelvis heading frame
        pelvis_pos = positions_xyz[i]           # MuJoCo frame
        hx, hy    = rotation_xy[i, 0], rotation_xy[i, 1]
        # Rotation matrix: columns = [forward, left, up]
        R_heading = np.array([[hx, -hy, 0],
                               [hy,  hx, 0],
                               [0,   0,  1]], dtype=np.float64)

        for j, (joint_idx, col_pos, col_orn) in enumerate([
            (LEFT_WRIST_IDX,  0, 0),
            (RIGHT_WRIST_IDX, 3, 4),
            (HEAD_IDX,        6, 8),
        ]):
            # Position: PICO → MuJoCo → pelvis heading frame
            p_pico   = body_pose[i, joint_idx, :3]
            p_mujoco = _WORLD_ROT_MAT @ p_pico
            p_local  = R_heading.T @ (p_mujoco - pelvis_pos)
            target_pos[i, col_pos:col_pos + 3] = p_local.astype(np.float32)

            # Orientation: PICO → MuJoCo → pelvis heading frame → wxyz
            q_pico  = R.from_quat(body_pose[i, joint_idx, 3:7])
            q_muj   = _WORLD_ROT * q_pico
            q_local = R.from_matrix(R_heading.T) * q_muj
            qxyzw   = q_local.as_quat()
            # reorder xyzw → wxyz for SONIC convention
            target_orn[i, col_orn: col_orn + 4] = np.array(
                [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]], dtype=np.float32
            )

    return target_pos, target_orn


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Hand joints (6-DoF binary templates)
# ═══════════════════════════════════════════════════════════════════════════════

def _fingertip_avg_distance(hand_pose: np.ndarray) -> np.ndarray:
    """(N,26,7) → (N,) mean pairwise fingertip distance (open=large, closed=small)."""
    tips = hand_pose[:, FINGER_TIPS, :3]   # (N, 5, 3)
    N = len(tips)
    dists = []
    for a in range(5):
        for b in range(a + 1, 5):
            dists.append(np.linalg.norm(tips[:, a] - tips[:, b], axis=1))
    return np.mean(dists, axis=0)   # (N,)


def compute_hand_joints(left_hand_pose:    np.ndarray,
                         right_hand_pose:   np.ndarray,
                         left_hand_active:  np.ndarray,
                         right_hand_active: np.ndarray) -> tuple:
    """
    Compute 6-DoF hand joint targets using binary open/close templates.

    Thresholding:
      - inactive hand → open
      - active hand   → closed if fingertip distance < median, else open

    Returns left_joints (N,6), right_joints (N,6)  float32
    """
    N = len(left_hand_pose)
    left_joints  = np.zeros((N, 6), dtype=np.float32)
    right_joints = np.zeros((N, 6), dtype=np.float32)

    for joints, hand_pose, active in (
        (left_joints,  left_hand_pose,  left_hand_active),
        (right_joints, right_hand_pose, right_hand_active),
    ):
        dist = _fingertip_avg_distance(hand_pose)
        med  = np.median(dist[active == 1]) if np.any(active == 1) else np.median(dist)
        closed_mask = (active == 1) & (dist < med)
        joints[closed_mask]  = HAND_CLOSED_6D
        joints[~closed_mask] = HAND_OPEN_6D

    return left_joints, right_joints


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SONIC planner commands
# ═══════════════════════════════════════════════════════════════════════════════

def compute_planner_commands(nav_cmd:       np.ndarray,
                              positions_xyz: np.ndarray,
                              rotation_xy:   np.ndarray,
                              calibration_frames: int = 50) -> dict:
    """
    Build SONIC planner input fields from navigation commands.

    Returns dict with keys:
      movement_direction (N,3), facing_direction (N,3),
      target_vel (N,), height (N,), planner_mode (N,),
      delta_height (N,)
    """
    N = len(nav_cmd)
    vx, vy, wz = nav_cmd[:, 0], nav_cmd[:, 1], nav_cmd[:, 2]

    # movement_direction: rotate local [vx,vy] to world frame
    hx, hy = rotation_xy[:, 0], rotation_xy[:, 1]
    nx, ny = -hy, hx
    vx_w  = hx * vx + nx * vy
    vy_w  = hy * vx + ny * vy
    speed = np.sqrt(vx ** 2 + vy ** 2)

    movement_dir = np.zeros((N, 3), dtype=np.float32)
    moving       = speed > 0.05
    movement_dir[moving, 0] = (vx_w / np.maximum(speed, 1e-6))[moving].astype(np.float32)
    movement_dir[moving, 1] = (vy_w / np.maximum(speed, 1e-6))[moving].astype(np.float32)

    # facing_direction from heading
    facing_dir = np.zeros((N, 3), dtype=np.float32)
    facing_dir[:, 0] = rotation_xy[:, 0].astype(np.float32)
    facing_dir[:, 1] = rotation_xy[:, 1].astype(np.float32)

    # target_vel
    target_vel = np.clip(speed, 0.0, MAX_SPEED).astype(np.float32)

    # delta_height: deviation of pelvis Z from neutral standing height
    pelvis_z    = positions_xyz[:, 2]
    neutral_z   = np.mean(pelvis_z[:min(calibration_frames, N)])
    dh          = (pelvis_z - neutral_z).astype(np.float32)

    # absolute height for planner
    height = np.clip(
        G1_NOMINAL_ROOT_HEIGHT + HUMAN_TO_G1_HEIGHT_SCALE * dh,
        HEIGHT_MIN, HEIGHT_MAX
    ).astype(np.float32)

    # planner mode
    mode = np.zeros(N, dtype=np.int64)
    for i in range(N):
        s = float(speed[i])
        h = float(height[i])
        if s < 0.05 and abs(float(wz[i])) < 0.1:
            mode[i] = 0   # idle
        elif h < SQUAT_HEIGHT_THRESH:
            mode[i] = 4   # squat
        elif h < CROUCH_HEIGHT_THRESH:
            mode[i] = 4   # IDLE_SQUAT (mode 22 "crouch" does not exist in LocomotionMode)
        elif s < 0.8:
            mode[i] = 1   # slowWalk
        else:
            mode[i] = 2   # walk

    return {
        'movement_direction': movement_dir,
        'facing_direction':   facing_dir,
        'target_vel':         target_vel,
        'height':             height,
        'planner_mode':       mode,
        'delta_height':       dh,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SONIC Protocol v3 SMPL fields
# ═══════════════════════════════════════════════════════════════════════════════

# SMPL-24 parent indices (index = joint, value = parent joint; -1 = root)
_SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                  12, 13, 14, 16, 17, 18, 19, 20, 21]

# Matching pico_manager_thread_server.py / isaac_utils/rotations.py:
#   smpl_pose  : global_rots * Y180 → local (parent-relative) — stays in PICO frame, no _WORLD_ROT
#   body_quat  : X90 * (root_pico * Y180) * SMPL_BASE_ROT_INV
_Y180           = R.from_euler("y", 180, degrees=True)
_X90            = R.from_euler("x",  90, degrees=True)   # smpl_root_ytoz_up (Y-up → Z-up)
_SMPL_BASE_ROT_INV = R.from_quat([0.5, 0.5, 0.5, 0.5]).inv()  # remove_smpl_base_rot


def compute_smpl_v3(body_pose: np.ndarray) -> dict:
    """
    Compute SONIC Protocol v3 SMPL fields from decimated body_pose.

    body_pose : (T, 24, 7)  PICO world frame, [x, y, z, qx, qy, qz, qw]

    Returns
    -------
    smpl_joints : (T, 24, 3)  joint positions pelvis-relative in robot (Z-up) frame
    smpl_pose   : (T, 21, 3)  parent-relative axis-angle joints 1-21 in PICO+Y180 frame
    body_quat   : (T, 4)      pelvis wxyz after ytoz + SMPL base rot removal
    frame_index : (T,)        int32 frame counter
    joint_pos   : (T, 29)     G1 joint positions (zeros — fill later)
    joint_vel   : (T, 29)     G1 joint velocities (zeros — fill later)
    """
    T = len(body_pose)
    quat_xyzw = body_pose[:, :, 3:7]  # (T, 24, 4) xyzw

    # ── smpl_pose ──────────────────────────────────────────────────────────────
    # Post-multiply Y-180 in PICO frame (no coordinate frame conversion here).
    # This matches: global_rots = global_rots * R.from_euler("y", 180) in pico_manager.
    rots_pico  = R.from_quat(quat_xyzw.reshape(-1, 4))
    rots_y180  = (rots_pico * _Y180).as_quat().reshape(T, 24, 4)

    local_rotvec = np.zeros((T, 24, 3), dtype=np.float64)
    for j in range(24):
        p = _SMPL_PARENTS[j]
        if p == -1:
            local_rotvec[:, j] = R.from_quat(rots_y180[:, j]).as_rotvec()
        else:
            q_parent = R.from_quat(rots_y180[:, p])
            q_child  = R.from_quat(rots_y180[:, j])
            local_rotvec[:, j] = (q_parent.inv() * q_child).as_rotvec()

    smpl_pose = local_rotvec[:, 1:22, :].astype(np.float32)

    # ── body_quat ──────────────────────────────────────────────────────────────
    # smpl_root_ytoz_up : X90 * (root * Y180)   (pre-multiply, Y-up → Z-up)
    # remove_smpl_base_rot : post-multiply by conjugate of SMPL default orientation
    root_pico   = R.from_quat(quat_xyzw[:, 0, :])           # (T,) pelvis in PICO
    root_final  = _X90 * (root_pico * _Y180) * _SMPL_BASE_ROT_INV
    body_quat   = root_final.as_quat()[:, [3,0,1,2]].astype(np.float32)  # xyzw→wxyz

    # ── smpl_joints ────────────────────────────────────────────────────────────
    # Apply ONLY X90 (Y-up → Z-up) to positions — no Z-90 yaw.
    # This matches smpl_root_ytoz_up which only applies X90, not the full
    # _WORLD_ROT (Z-90 * X90).  Using _WORLD_ROT here would add an extra 90°
    # yaw that isn't present in body_quat, causing a systematic rightward bias.
    _X90_MAT = _X90.as_matrix()                               # (3, 3)
    pts_x90  = (body_pose[:, :, :3] @ _X90_MAT.T)            # (T, 24, 3)
    pts_rel  = pts_x90 - pts_x90[:, 0:1, :]                   # pelvis-relative
    pelvis_inv = root_final.inv()
    smpl_joints = np.stack(
        [pelvis_inv[t].apply(pts_rel[t]) for t in range(T)], axis=0
    ).astype(np.float32)

    return {
        'smpl_joints': smpl_joints,
        'smpl_pose':   smpl_pose,
        'body_quat':   body_quat,
        'frame_index': np.arange(T, dtype=np.int32),
        'joint_pos':   np.zeros((T, 29), dtype=np.float32),
        'joint_vel':   np.zeros((T, 29), dtype=np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. HDF5 writer
# ═══════════════════════════════════════════════════════════════════════════════

def _ds(f, name, data, **attrs):
    """Create compressed dataset and attach attrs."""
    ds = f.create_dataset(name, data=np.asarray(data), compression='gzip', compression_opts=4)
    for k, v in attrs.items():
        ds.attrs[k] = str(v) if isinstance(v, (list, tuple, np.ndarray)) else v
    return ds


def write_output(out_path: Path, body: dict, cam_l: list, cam_r: list,
                 matched_ts: np.ndarray, diff_ms: np.ndarray,
                 positions_xyz: np.ndarray, rotation_xy: np.ndarray,
                 nav_cmd: np.ndarray, left_eef: np.ndarray,
                 right_eef: np.ndarray, delta_eef: np.ndarray,
                 target_pos: np.ndarray, target_orn: np.ndarray,
                 left_joints: np.ndarray, right_joints: np.ndarray,
                 planner: dict, smpl_v3: dict, fps: float):

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dt = h5py.special_dtype(vlen=np.uint8)

    with h5py.File(str(out_path), 'w') as f:
        f.attrs['fps']           = fps
        f.attrs['dt_s']          = 1.0 / fps
        f.attrs['frame_count']   = len(body['local_timestamps_ns'])
        f.attrs['coordinate']    = 'mujoco_compatible'
        f.attrs['sonic_version'] = 3

        # timestamps
        _ds(f, 'local_timestamps_ns', body['local_timestamps_ns'],
            description='body data timestamps (ns)')
        _ds(f, 'camera_timestamp',   matched_ts,
            description='matched ZED frame timestamp (ns)')
        _ds(f, 'timestamp_diff_ms',  diff_ms,
            description='|body_ts - camera_ts| in ms')

        # camera images
        N = len(cam_l)
        ds_l = f.create_dataset('observation_image_left',  shape=(N,), dtype=dt)
        ds_r = f.create_dataset('observation_image_right', shape=(N,), dtype=dt)
        for i in range(N):
            ds_l[i] = np.frombuffer(cam_l[i], dtype=np.uint8)
            ds_r[i] = np.frombuffer(cam_r[i], dtype=np.uint8)

        # raw body signals (kept for debugging)
        _ds(f, 'debug/body_pose',           body['body_pose'],
            description='body_pose (N,24,7) at target fps, PICO world frame')
        _ds(f, 'debug/left_hand_pose',      body['left_hand_pose'])
        _ds(f, 'debug/right_hand_pose',     body['right_hand_pose'])
        _ds(f, 'debug/left_hand_active',    body['left_hand_active'])
        _ds(f, 'debug/right_hand_active',   body['right_hand_active'])

        # navigation
        _ds(f, 'debug/navigation_command',  nav_cmd,
            description='[vx, vy, yaw_rate] in local heading frame (m/s, rad/s)')
        _ds(f, 'debug/positions_xyz',       positions_xyz,
            description='pelvis position in MuJoCo frame (m)')
        _ds(f, 'debug/rotation_xy',         rotation_xy,
            description='heading unit vector in MuJoCo XY plane')

        # EEF
        action_eef = np.hstack([left_eef, right_eef]).astype(np.float32)
        _ds(f, 'action_eef',       action_eef,
            description='[left: x,y,z,qx,qy,qz,qw, right: x,y,z,qx,qy,qz,qw] in pelvis frame')
        _ds(f, 'action_delta_eef', delta_eef.astype(np.float32),
            description='[left: dx,dy,dz,r,p,y, right: ...] frame-to-frame delta')

        # SONIC upper-body targets
        _ds(f, 'vr_3point_local_target',     target_pos,
            description='[lw_xyz, rw_xyz, head_xyz] in pelvis heading frame (m)')
        _ds(f, 'vr_3point_local_orn_target', target_orn,
            description='[lw_wxyz, rw_wxyz, head_wxyz] in pelvis heading frame')

        # Hand joints
        _ds(f, 'action.left_hand_joints',  left_joints,
            description='6-DoF left hand joints (binary open/close template)')
        _ds(f, 'action.right_hand_joints', right_joints,
            description='6-DoF right hand joints (binary open/close template)')

        # SONIC planner commands — top-level for publisher + downstream consumers
        # Lower-body discrete commands (locomotion)
        _ds(f, 'planner/movement_direction', planner['movement_direction'].astype(np.float32),
            description='world-frame movement direction unit vector (3,) — lower-body locomotion')
        _ds(f, 'planner/facing_direction',   planner['facing_direction'].astype(np.float32),
            description='world-frame facing direction unit vector (3,) — lower-body locomotion')
        _ds(f, 'planner/target_vel',         planner['target_vel'].astype(np.float32),
            description='locomotion speed magnitude clipped to MAX_SPEED (m/s)')
        _ds(f, 'planner/height',             planner['height'].astype(np.float32),
            description='absolute G1 root height for SONIC planner (m) — height control')
        _ds(f, 'planner/mode',               planner['planner_mode'].astype(np.int32),
            description='SONIC planner mode: 0=idle,1=slowWalk,2=walk,4=squat,22=crouch')

        # Also keep under debug/ for backward compat with existing visualisation scripts
        _ds(f, 'debug/delta_height',       planner['delta_height'],
            description='pelvis Z deviation from neutral standing (m)')
        _ds(f, 'debug/movement_direction', planner['movement_direction'],
            description='world-frame movement direction unit vector (3D)')
        _ds(f, 'debug/facing_direction',   planner['facing_direction'],
            description='world-frame facing direction unit vector (3D)')
        _ds(f, 'debug/target_vel',         planner['target_vel'],
            description='speed magnitude clipped to MAX_SPEED (m/s)')
        _ds(f, 'debug/planner_height',     planner['height'],
            description='absolute G1 root height for SONIC planner (m)')
        _ds(f, 'debug/planner_mode',       planner['planner_mode'],
            description='SONIC planner mode: 0=idle,1=slowWalk,2=walk,4=squat,22=crouch')

        # hand_status kept for compatibility with existing visualisation
        hand_status = np.stack([
            (left_joints.sum(axis=1) > 0).astype(np.float32),
            (right_joints.sum(axis=1) > 0).astype(np.float32),
        ], axis=1)
        _ds(f, 'hand_status', hand_status,
            description='[left, right] binary open/close (1=closed)')

        # SONIC Protocol v3 SMPL fields
        _ds(f, 'smpl_joints', smpl_v3['smpl_joints'],
            description='(T,24,3) human joint positions in MuJoCo frame (m)')
        _ds(f, 'smpl_pose',   smpl_v3['smpl_pose'],
            description='(T,21,3) global axis-angle for joints 1-21 in MuJoCo frame (first pass; not parent-relative)')
        _ds(f, 'body_quat',   smpl_v3['body_quat'],
            description='(T,4) pelvis quaternion wxyz in MuJoCo frame')
        _ds(f, 'frame_index', smpl_v3['frame_index'],
            description='(T,) integer frame counter')
        _ds(f, 'joint_pos',   smpl_v3['joint_pos'],
            description='(T,29) G1 joint positions — zeros placeholder')
        _ds(f, 'joint_vel',   smpl_v3['joint_vel'],
            description='(T,29) G1 joint velocities — zeros placeholder')


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Top-level convert
# ═══════════════════════════════════════════════════════════════════════════════

def convert_episode(body_h5_path: Path, svo2_path: Path, out_path: Path,
                    stride: int = 2, fps: float = 50.0,
                    baseline_sec: float = 10.0, sg_window: int = SG_WINDOW,
                    sg_poly: int = SG_POLY, sg_passes: int = SG_PASSES,
                    calibration_frames: int = 50):
    print(f"\n{'='*60}")
    print(f"  {body_h5_path.name}  →  {out_path.name}")
    print(f"  SVO2: {svo2_path.name}")
    print(f"{'='*60}")

    # 1. Body data
    print("[1/7] Decimating body data ...")
    body = decimate_body(body_h5_path, stride=stride)
    N = len(body['body_pose'])
    dt = body['dt_body']
    print(f"      {N} frames @ {1/dt:.0f} Hz  (stride={stride})")

    # 2. Camera
    print("[2/7] Reading SVO2 ...")
    cam_ts, jpeg_l, jpeg_r = read_svo2(svo2_path)
    print(f"      {len(cam_ts)} camera frames @ ~{1e9/(cam_ts[-1]-cam_ts[0])*len(cam_ts):.0f} FPS")

    print("[2/7] Syncing camera to body timestamps ...")
    matched_l, matched_r, matched_ts, diff_ms = sync_camera(
        body['local_timestamps_ns'], cam_ts, jpeg_l, jpeg_r
    )
    print(f"      mean diff={diff_ms.mean():.1f} ms  max={diff_ms.max():.1f} ms")

    # 3. Navigation
    print("[3/7] Computing navigation commands ...")
    positions_xyz, rotation_xy, nav_cmd = compute_navigation_commands(
        body['body_pose'], dt, baseline_sec=baseline_sec,
        sg_window=11, sg_poly=3, tangent_lag=5
    )

    # 4. EEF
    print("[4/7] Computing EEF in pelvis frame ...")
    left_eef, right_eef = compute_eef_pelvis(
        body['body_pose'], sg_window=sg_window, sg_poly=sg_poly, sg_passes=sg_passes
    )
    delta_eef = compute_delta_eef(left_eef, right_eef)

    # 5. 3-point targets
    print("[5/7] Computing upper-body 3-point targets ...")
    target_pos, target_orn = compute_3point_targets(
        body['body_pose'], positions_xyz, rotation_xy
    )

    # 6. Hand joints
    print("[6/7] Computing hand joints ...")
    left_joints, right_joints = compute_hand_joints(
        body['left_hand_pose'], body['right_hand_pose'],
        body['left_hand_active'], body['right_hand_active']
    )

    # 7. Planner commands
    print("[7/7] Computing SONIC planner commands ...")
    planner = compute_planner_commands(
        nav_cmd, positions_xyz, rotation_xy,
        calibration_frames=calibration_frames
    )

    # 8. SONIC v3 SMPL fields
    print("[8/8] Computing SONIC v3 SMPL fields ...")
    smpl_v3 = compute_smpl_v3(body['body_pose'])

    # Write
    print(f"Writing → {out_path} ...")
    write_output(
        out_path, body, matched_l, matched_r, matched_ts, diff_ms,
        positions_xyz, rotation_xy, nav_cmd, left_eef, right_eef,
        delta_eef, target_pos, target_orn, left_joints, right_joints,
        planner, smpl_v3, fps
    )

    import os
    sz_mb = os.path.getsize(out_path) / 1e6
    print(f"✓ Done  ({N} frames, {sz_mb:.1f} MB)")
    return N


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Convert EgoHumanoid body HDF5 + ZED SVO2 to SONIC-compatible HDF5 at 50 Hz'
    )
    parser.add_argument('body_h5', nargs='?', help='Single body HDF5 path')
    parser.add_argument('--svo',   help='SVO2 path (default: same dir as body_h5, same stem)')
    parser.add_argument('--out',   help='Output HDF5 path (single-file mode)')
    parser.add_argument('--batch', help='Batch mode: directory containing episode_*.hdf5')
    parser.add_argument('--out-dir', help='Output directory (batch mode)')
    parser.add_argument('--stride', type=int, default=2,
                        help='Decimation stride (default: 2 → 50 Hz from 100 Hz)')
    parser.add_argument('--fps', type=float, default=50.0, help='Target fps label (default: 50)')
    parser.add_argument('--baseline-sec', type=float, default=10.0,
                        help='Tangent baseline smoothing window (s, default: 10)')
    parser.add_argument('--sg-window', type=int, default=SG_WINDOW)
    parser.add_argument('--sg-poly',   type=int, default=SG_POLY)
    parser.add_argument('--sg-passes', type=int, default=SG_PASSES)
    parser.add_argument('--calibration-frames', type=int, default=50,
                        help='Number of initial frames used as neutral standing reference (default: 50)')
    args = parser.parse_args()

    kwargs = dict(
        stride=args.stride, fps=args.fps, baseline_sec=args.baseline_sec,
        sg_window=args.sg_window, sg_poly=args.sg_poly, sg_passes=args.sg_passes,
        calibration_frames=args.calibration_frames,
    )

    if args.batch:
        batch_dir = Path(args.batch)
        out_dir   = Path(args.out_dir) if args.out_dir else batch_dir.parent / 'sonic'
        out_dir.mkdir(parents=True, exist_ok=True)

        episodes = sorted(batch_dir.glob('episode_*.hdf5'),
                          key=lambda p: int(p.stem.split('_')[-1]))
        if not episodes:
            raise FileNotFoundError(f'No episode_*.hdf5 found in {batch_dir}')

        total = 0
        for ep in episodes:
            svo  = ep.with_suffix('.svo2')
            if not svo.exists():
                print(f'[SKIP] {ep.name}: SVO2 not found')
                continue
            out = out_dir / ep.name
            total += convert_episode(ep, svo, out, **kwargs)
        print(f'\nBatch done: {total} frames total → {out_dir}')

    else:
        if not args.body_h5:
            parser.error('Provide body_h5 or --batch')
        body_h5 = Path(args.body_h5)
        svo     = Path(args.svo) if args.svo else body_h5.with_suffix('.svo2')
        out     = Path(args.out) if args.out else body_h5.parent.parent / 'sonic' / body_h5.name
        convert_episode(body_h5, svo, out, **kwargs)


if __name__ == '__main__':
    main()
