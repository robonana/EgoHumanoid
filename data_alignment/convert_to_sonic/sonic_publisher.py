"""
sonic_publisher.py  —  replay human motion into SONIC N1.7 (GEAR-SONIC) simulation

Wire format (NOT msgpack):
  Every message = topic_bytes + 1280-byte JSON header + raw little-endian binary
  Uses gear_sonic.utils.teleop.zmq.zmq_planner_sender:
    build_command_message()  →  topic b"command"  (start / stop signal)
    build_planner_message()  →  topic b"planner"  (VR 3-point + discrete commands)
    pack_pose_message()      →  topic b"pose"     (SMPL body tracking)

Three stream modes
------------------
  smpl (recommended)
    Protocol V3 — SMPL body data → N1.7 human body encoder (SMPL encoder mode).
    ZMQEndpointInterface auto-detects "v":3 in the header → uses SMPL encoder.
    Sends per message (W-frame window):
      smpl_joints   (W,24,3)  joint positions (pelvis-relative, robot frame)
      smpl_pose     (W,21,3)  parent-relative axis-angle joints 1-21
      body_quat_w   (W,4)     pelvis quaternion wxyz
      joint_pos     (W,29)    G1 joint positions, wrists retargeted from SMPL
      joint_vel     (W,29)    zeros
      frame_index   (W,)      int64 frame counters
      vr_position   (9,)      VR 3-point positions [lw_xyz, rw_xyz, neck_xyz]
      vr_orientation (12,)    VR 3-point orientations [lw_wxyz, rw_wxyz, neck_wxyz]
      + hand joints, trigger/grip, timestamps, heading_increment, toggles
    W = window size (num_frames_to_send, default 5)
    SONIC side: bash deploy.sh --input-type zmq sim

  vr3pt
    Sends SONIC kinematic planner inputs per frame (requires ZMQManager):
      Upper body : vr_3point_local_target (9,) + vr_3point_local_orn_target (12,)
      Lower body : movement_direction (3,), facing_direction (3,), target_vel, planner_height
      Planner mode: 0=IDLE, 1=SLOW_WALK, 2=WALK, 4=IDLE_SQUAT
      Hands      : left_hand_joints (6,), right_hand_joints (6,)
    SONIC side: bash deploy.sh --input-type zmq_manager sim

  v1
    Protocol V1 — direct G1 joint replay, bypasses SMPL encoder.
    ZMQEndpointInterface auto-detects "v":1 → encoder mode 0 (joint-based).
    Sends per frame:
      joint_pos   (1,29)  f32  — G1 joint angles (wrists retargeted from SMPL)
      joint_vel   (1,29)  f32  — zeros
      body_quat_w (1,4)   f32  — pelvis quaternion wxyz
      frame_index (1,)    i64  — frame counter
      catch_up    (1,)    u8   — always 0
    SONIC side: bash deploy.sh --input-type zmq --zmq-topic pose sim

Input-type mapping (--input-type in deploy.sh)
-----------------------------------------------
  zmq         → ZMQEndpointInterface: direct ZMQ "pose" subscriber, auto-starts,
                handles Protocol V1 (encoder mode 0) and SMPL v2/v3 (encoder mode 2)
  zmq_manager → ZMQManager: command + planner + pose topics, needs start command
  manager     → InterfaceManager: keyboard/gamepad/zmq *switcher*, starts in keyboard mode
                ** NOT a ZMQ subscriber — use zmq or zmq_manager instead **

Usage
-----
  # SMPL body stream → N1.7 human body encoder  (recommended)
  python sonic_publisher.py episode_2.hdf5 --stream smpl
  # SONIC: bash deploy.sh --input-type zmq sim

  # VR 3-point + kinematic planner
  python sonic_publisher.py episode_2.hdf5 --stream vr3pt
  # SONIC: bash deploy.sh --input-type zmq_manager sim

  # Protocol V1 direct joint replay (bypasses encoder)
  python sonic_publisher.py episode_2.hdf5 --stream v1
  # SONIC: bash deploy.sh --input-type zmq --zmq-topic pose sim

  # Loop forever
  python sonic_publisher.py episode_2.hdf5 --stream smpl --loop

  # Validate only (no ZMQ)
  python sonic_publisher.py episode_2.hdf5 --check-only

  # Dry-run timing test
  python sonic_publisher.py episode_2.hdf5 --dry-run --skip-check

GR00T-WBC path (default: ~/Projects/GR00T-WholeBodyControl):
  python sonic_publisher.py episode_2.hdf5 --groot-wbc-root /path/to/GR00T-WholeBodyControl
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

# ─────────────────────────────────────────────────────────────────────────────
# VR 3-point position frame and human→G1 retargeting
#
# SONIC expects vr_3pt_position in robot local frame: x=forward, y=left, z=up
# (pelvis-relative, same convention as C++ GatherVR3PointPosition non-buffered path).
#
# smpl_joints from convert_episode.py are computed identically to pico_manager's
# _process_3pt_pose: both apply the Unity→Robot Q-matrix then rotate by inverse
# root quaternion. Verified numerically: values match to 1e-7.
#
# Joint indices (SMPL 24-joint):  22 = L_Hand, 23 = R_Hand, 12 = Neck,
#                                  16 = L_Shoulder, 17 = R_Shoulder,
#                                  18 = L_Elbow, 19 = R_Elbow, 20 = L_Wrist, 21 = R_Wrist
#
# Retargeting strategy (human → Unitree G1):
#   G1 FK shoulder positions (pelvis-relative, zero pose) from g1_29dof.xml:
#     L shoulder: [-0.00072, +0.10022, +0.29178] m
#     R shoulder: [-0.00072, -0.10021, +0.29178] m
#   G1 arm length (upper arm + forearm bone sums from FK): 19.3 + 18.4 = 37.7 cm
#   Typical human arm (from SMPL data): ~25.7 + 29.7 = 55.4 cm  → scale ≈ 0.681
#   G1 head height above pelvis (VR_3POINT_OFFSETS default): 0.35 m
#   Typical human neck height: ~0.547 m → neck scale ≈ 0.640
#
#   For each frame: wrist_g1 = shoulder_g1 + arm_scale * (hand_smpl - shoulder_smpl)
#   arm_scale is computed per-episode from actual SMPL bone lengths so it adapts to actors.
# ─────────────────────────────────────────────────────────────────────────────

# G1 shoulder positions in pelvis-relative robot local frame (from FK at zero pose)
_G1_L_SHOULDER = np.array([-0.00072,  0.10022, 0.29178], dtype=np.float64)
_G1_R_SHOULDER = np.array([-0.00072, -0.10021, 0.29178], dtype=np.float64)
# G1 arm length: upper arm 19.3cm + forearm 18.4cm (FK bone sums from g1_29dof.xml)
_G1_ARM_LENGTH = 0.193 + 0.184   # metres
# G1 head reference height above pelvis (from C++ VR_3POINT_OFFSETS default)
_G1_HEAD_HEIGHT = 0.35            # metres


def _retarget_vr3pt(smpl_joints: np.ndarray) -> np.ndarray:
    """Retarget human VR 3-point positions to Unitree G1 proportions.

    smpl_joints : (T, 24, 3)  robot local frame, pelvis-relative
    Returns     : (T, 9)      [L_wrist, R_wrist, Neck]  in G1 pelvis frame
    """
    T = smpl_joints.shape[0]

    # --- arm scale: computed from this episode's SMPL bone lengths ---
    l_upper = np.linalg.norm(smpl_joints[:, 18] - smpl_joints[:, 16], axis=1).mean()
    l_fore  = np.linalg.norm(smpl_joints[:, 20] - smpl_joints[:, 18], axis=1).mean()
    human_arm = l_upper + l_fore
    arm_scale = _G1_ARM_LENGTH / human_arm

    # --- neck scale: proportional to G1 head height / human neck height ---
    human_neck_z = smpl_joints[:, 12, 2].mean()
    neck_scale   = _G1_HEAD_HEIGHT / human_neck_z

    print(f"  Retargeting: human arm={human_arm*100:.1f}cm → G1 arm={_G1_ARM_LENGTH*100:.1f}cm "
          f"(scale={arm_scale:.3f})")
    print(f"               human neck z={human_neck_z*100:.1f}cm → G1 head={_G1_HEAD_HEIGHT*100:.1f}cm "
          f"(scale={neck_scale:.3f})")

    # --- arm vectors from shoulder to hand (joint 22/23), scaled from G1 shoulder ---
    l_vec = smpl_joints[:, 22] - smpl_joints[:, 16]   # (T, 3)
    r_vec = smpl_joints[:, 23] - smpl_joints[:, 17]   # (T, 3)

    l_wrist = _G1_L_SHOULDER + arm_scale * l_vec      # (T, 3)
    r_wrist = _G1_R_SHOULDER + arm_scale * r_vec      # (T, 3)
    neck    = smpl_joints[:, 12] * neck_scale          # (T, 3)

    vr_pos = np.zeros((T, 9), dtype=np.float32)
    vr_pos[:, 0:3] = l_wrist
    vr_pos[:, 3:6] = r_wrist
    vr_pos[:, 6:9] = neck
    return vr_pos


# pico_manager Q matrix and OFFSETS — matches pico_manager_thread_server._process_3pt_pose
_PM_Q_MAT = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
_PM_OFFSETS = {
    0:  R.from_euler("xyz", [  0,   0,  -90], degrees=True),  # Root: yaw -90°
    22: R.from_euler("xyz", [ 90,   0,    0], degrees=True),  # L_Hand: roll +90°
    23: R.from_euler("xyz", [-90,   0,  180], degrees=True),  # R_Hand: roll -90°, yaw +180°
    12: R.from_euler("xyz", [  0,   0,  -90], degrees=True),  # Neck: yaw -90°
}


def _compute_vr3pt_orn(body_pose: np.ndarray) -> np.ndarray:
    """Compute VR 3-point wrist orientations matching pico_manager convention.

    body_pose : (T, 24, 7)  PICO frame [x,y,z, qx,qy,qz,qw]
    Returns   : (T, 12)     [lw_wxyz, rw_wxyz, neck_wxyz]  pelvis-relative robot frame

    Replicates _process_3pt_pose in pico_manager_thread_server.py:
      1. Conjugate by Q matrix (Unity → robot frame)
      2. Post-multiply per-joint OFFSET (frame alignment correction)
      3. Make relative to root (root also has yaw -90° OFFSET applied)
    """
    def _q_conjugate(q_xyzw):
        """Q @ rot @ Q.T for each quaternion in (T, 4) array."""
        mats = R.from_quat(q_xyzw).as_matrix()              # (T, 3, 3)
        mats = _PM_Q_MAT @ mats @ _PM_Q_MAT.T               # broadcast Q conjugation
        return R.from_matrix(mats)

    root_r = _q_conjugate(body_pose[:, 0, 3:7]) * _PM_OFFSETS[0]  # (T,)

    orn = np.zeros((len(body_pose), 12), dtype=np.float32)
    for joint_idx, col in [(22, 0), (23, 4), (12, 8)]:
        q_local = root_r.inv() * (_q_conjugate(body_pose[:, joint_idx, 3:7]) * _PM_OFFSETS[joint_idx])
        wxyz = q_local.as_quat()[:, [3, 0, 1, 2]]           # xyzw → wxyz
        orn[:, col:col + 4] = wxyz.astype(np.float32)
    return orn


# ─────────────────────────────────────────────────────────────────────────────
# LocomotionMode values (matches pico_manager_thread_server.LocomotionMode)
# Max valid value is INJURED_WALK = 19. Mode 22 does NOT exist.
# ─────────────────────────────────────────────────────────────────────────────
LOCO_IDLE       = 0
LOCO_SLOW_WALK  = 1
LOCO_WALK       = 2
LOCO_IDLE_SQUAT = 4

VALID_MODES = set(range(20))   # 0..19


def _sanitize_mode(m: int) -> int:
    if m in VALID_MODES:
        return m
    if m == 22:          # old "crouch" label
        return LOCO_IDLE_SQUAT
    return LOCO_IDLE


# ─────────────────────────────────────────────────────────────────────────────
# G1 wrist retargeting from SMPL elbow + wrist axis-angle
# Matches PoseStreamer logic in pico_manager_thread_server.py exactly
# ─────────────────────────────────────────────────────────────────────────────

# SMPL indices into smpl_pose (joints 1..21, so index 0 = joint 1)
_SMPL_L_ELBOW = 17   # joint 18
_SMPL_L_WRIST = 19   # joint 20
_SMPL_R_ELBOW = 18   # joint 19
_SMPL_R_WRIST = 20   # joint 21

# G1 joint_pos indices (29-dim)
_G1_L_WRIST_ROLL  = 23
_G1_L_WRIST_PITCH = 25
_G1_L_WRIST_YAW   = 27
_G1_R_WRIST_ROLL  = 24
_G1_R_WRIST_PITCH = 26
_G1_R_WRIST_YAW   = 28

_L_ELBOW_AXIS = np.array([0.0, 1.0, 0.0])
_R_ELBOW_AXIS = np.array([0.0, 1.0, 0.0])


def _decompose_rotation_aa_scipy(aa: np.ndarray, axis: np.ndarray):
    """
    Twist-swing decomposition of axis-angle `aa` (shape (1,3)) around `axis`.
    Returns (q_twist, q_swing) each shape (1,4) in xyzw format.
    Mirrors gear_sonic.trl.utils.rotation_conversion.decompose_rotation_aa.
    """
    rot = R.from_rotvec(aa[0])
    rotvec = rot.as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        return (R.identity().as_quat()[None], R.identity().as_quat()[None])

    rot_axis = rotvec / angle
    twist_angle = np.dot(rot_axis, axis) * angle
    twist = R.from_rotvec(twist_angle * axis)
    swing = twist.inv() * rot
    return (twist.as_quat()[None], swing.as_quat()[None])   # xyzw


def compute_joint_pos_from_smpl(smpl_pose: np.ndarray) -> np.ndarray:
    """
    Retarget SMPL elbow+wrist rotations to G1 3-DOF wrist joints.

    smpl_pose : (21, 3) axis-angle for SMPL joints 1-21 (parent-relative)
    Returns   : joint_pos (29,) float32, wrist indices filled; rest are 0
    """
    joint_pos = np.zeros(29, dtype=np.float32)
    bp = smpl_pose.reshape(1, 21, 3)   # (1, 21, 3)

    l_el_aa = bp[:, _SMPL_L_ELBOW]    # (1, 3)
    l_wr_aa = bp[:, _SMPL_L_WRIST]
    r_el_aa = bp[:, _SMPL_R_ELBOW]
    r_wr_aa = bp[:, _SMPL_R_WRIST]

    _, l_el_swing = _decompose_rotation_aa_scipy(l_el_aa, _L_ELBOW_AXIS)
    _, r_el_swing = _decompose_rotation_aa_scipy(r_el_aa, _R_ELBOW_AXIS)

    # swing quaternion: xyzw → matrix → euler XYZ (extrinsic)
    l_el_euler = R.from_quat(l_el_swing[0]).as_euler("XYZ")
    r_el_euler = R.from_quat(r_el_swing[0]).as_euler("XYZ")
    l_wr_euler = R.from_rotvec(l_wr_aa[0]).as_euler("XYZ")
    r_wr_euler = R.from_rotvec(r_wr_aa[0]).as_euler("XYZ")

    joint_pos[_G1_L_WRIST_ROLL]  =  l_el_euler[0] + l_wr_euler[0]
    joint_pos[_G1_L_WRIST_PITCH] = -l_wr_euler[1]
    joint_pos[_G1_L_WRIST_YAW]   =  l_el_euler[2] + l_wr_euler[2]

    joint_pos[_G1_R_WRIST_ROLL]  = -(r_el_euler[0] + r_wr_euler[0])
    joint_pos[_G1_R_WRIST_PITCH] = -r_wr_euler[1]
    joint_pos[_G1_R_WRIST_YAW]   =  r_el_euler[2] + r_wr_euler[2]

    return joint_pos


# ─────────────────────────────────────────────────────────────────────────────
# Continuous hand joint computation from raw 26-joint hand pose
#
# The HDF5 action.{left,right}_hand_joints fields are stored as binary 0/1
# (open/closed template) by convert_episode.py. The actual continuous finger
# curl is in debug/{left,right}_hand_pose (T, 26, 7) with per-joint xyz+quat
# in PICO world frame.
#
# G1 Dex3-1 hand has 3 fingers with 7 actuated joints total:
#   [0] thumb_0   (abduction,  L: ±1.047, R: ∓1.047)
#   [1] thumb_1   (flexion-1,  L/R same sign convention)
#   [2] thumb_2   (flexion-2,  L: 0→2.09, R: 0→-2.09)
#   [3] index_0   (MCP,        L: 0→-1.57, R: 0→+1.57)
#   [4] index_1   (PIP,        L: 0→-1.75, R: 0→+1.75)
#   [5] middle_0  (MCP,        L: 0→-1.57, R: 0→+1.57)
#   [6] middle_1  (PIP,        L: 0→-1.75, R: 0→+1.75)
# "Closed" values from test_zmq_manager.py:
#   left:  [0, 0, +1.75, -1.57, -1.75, -1.57, -1.75]
#   right: [0, 0, -1.75, +1.57, +1.75, +1.57, +1.75]  (mirror)
#
# Human 26-joint hand pose layout (joint indices):
#   0=wrist root, then 5 joints per finger in order:
#   Thumb(1-5), Index(6-10), Middle(11-15), Ring(16-20), Little(21-25)
#   Tip indices: Thumb=5, Index=10, Middle=15, Ring=20, Little=25
#
# Human→robot finger mapping:
#   Robot thumb  ← Human thumb (tip 5)
#   Robot index  ← Human index (tip 10)
#   Robot middle ← mean(Human middle tip 15, ring tip 20, little tip 25)
# ─────────────────────────────────────────────────────────────────────────────

# "Fully closed" joint angles for each side (7 values, radians)
_LEFT_HAND_CLOSED  = np.array([0.0, 0.0,  1.75, -1.57, -1.75, -1.57, -1.75], dtype=np.float64)
_RIGHT_HAND_CLOSED = np.array([0.0, 0.0, -1.75,  1.57,  1.75,  1.57,  1.75], dtype=np.float64)

# Human fingertip joint indices in the 26-joint hand pose
_TIP_THUMB  = 5
_TIP_INDEX  = 10
_TIP_MIDDLE = 15
_TIP_RING   = 20
_TIP_LITTLE = 25


def _finger_curl(hand_pose: np.ndarray, tip_idx: int, active: np.ndarray) -> np.ndarray:
    """
    Compute per-frame curl in [0, 1] for one finger.
    Curl = 1 − norm(tip-to-wrist distance).  Larger distance = more open.
    Normalization bounds are computed only over active (tracking-valid) frames.
    Inactive frames are set to 0 (open).
    """
    pos  = hand_pose[:, :, :3]                              # (T, 26, 3)
    dist = np.linalg.norm(pos[:, tip_idx] - pos[:, 0], axis=1)  # (T,)

    d_ref = dist[active] if active.any() else dist
    d_min, d_max = d_ref.min(), d_ref.max()

    if d_max - d_min < 1e-4:
        return np.zeros(len(dist), dtype=np.float32)

    curl = 1.0 - (dist - d_min) / (d_max - d_min)
    curl = np.clip(curl, 0.0, 1.0).astype(np.float32)
    curl[~active] = 0.0   # tracking-loss → open
    return curl


def _compute_hand_joints_7dof(hand_pose: np.ndarray, side: str,
                               hand_active: np.ndarray | None = None) -> np.ndarray:
    """
    Compute 7-DOF Dex3-1 joint angles from 26-joint human hand pose.

    hand_pose   : (T, 26, 7)  joint positions+quaternions in PICO world frame
    side        : "left" or "right"
    hand_active : (T,) int    1=tracking valid, 0=lost; inactive frames → open (0 rad)
    Returns     : (T, 7)  joint angles in radians
                  [thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1]
    """
    T      = hand_pose.shape[0]
    active = np.ones(T, dtype=bool) if hand_active is None else (hand_active == 1)

    ct = _finger_curl(hand_pose, _TIP_THUMB,  active)
    ci = _finger_curl(hand_pose, _TIP_INDEX,  active)
    # Robot middle finger covers human middle+ring+little
    cm = (_finger_curl(hand_pose, _TIP_MIDDLE, active)
        + _finger_curl(hand_pose, _TIP_RING,   active)
        + _finger_curl(hand_pose, _TIP_LITTLE, active)) / 3.0

    closed = _LEFT_HAND_CLOSED if side == "left" else _RIGHT_HAND_CLOSED
    out    = np.zeros((T, 7), dtype=np.float32)

    # thumb_0 (abduction): leave at 0 — no reliable abduction signal from distance
    # thumb_1:             leave at 0 (matches test_zmq_manager closed pose)
    out[:, 2] = (closed[2] * ct).astype(np.float32)   # thumb_2 flexion
    out[:, 3] = (closed[3] * ci).astype(np.float32)   # index_0 MCP
    out[:, 4] = (closed[4] * ci).astype(np.float32)   # index_1 PIP
    out[:, 5] = (closed[5] * cm).astype(np.float32)   # middle_0 MCP
    out[:, 6] = (closed[6] * cm).astype(np.float32)   # middle_1 PIP

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

_PLANNER_LEGACY = {
    "planner/movement_direction": "debug/movement_direction",
    "planner/facing_direction":   "debug/facing_direction",
    "planner/target_vel":         "debug/target_vel",
    "planner/height":             "debug/planner_height",
    "planner/mode":               "debug/planner_mode",
}


def _load(f: h5py.File, key: str) -> np.ndarray:
    if key in f:
        return f[key][:]
    fb = _PLANNER_LEGACY.get(key)
    if fb and fb in f:
        return f[fb][:]
    raise KeyError(f"Field not found: {key}  (also tried {fb})")


def load_episode(h5_path: Path) -> dict:
    keys = [
        # VR 3-point
        "vr_3point_local_target",
        "vr_3point_local_orn_target",
        "debug/body_pose",           # (T,24,7) PICO frame — needed for _vr3pt_orn_corrected
        "action.left_hand_joints",
        "action.right_hand_joints",
        # Planner discrete commands
        "planner/movement_direction",
        "planner/facing_direction",
        "planner/target_vel",
        "planner/height",
        "planner/mode",
        # SMPL
        "smpl_joints",           # (T,24,3) pelvis-relative robot frame
        "smpl_pose",             # (T,21,3) parent-relative axis-angle
        "body_quat",             # (T,4)    wxyz — stored as body_quat, sent as body_quat_w
        "frame_index",
        "local_timestamps_ns",
    ]
    _HAND_POSE_KEYS = [
        "debug/left_hand_pose",
        "debug/right_hand_pose",
        "debug/left_hand_active",
        "debug/right_hand_active",
    ]
    with h5py.File(h5_path, "r") as f:
        data = {}
        for k in keys:
            try:
                data[k] = _load(f, k)
            except KeyError as e:
                print(f"  WARNING: {e}")
        # Optionally load raw hand pose for continuous joint computation
        for k in _HAND_POSE_KEYS:
            if k in f:
                data[k] = f[k][:]
        data["_T"] = data["smpl_joints"].shape[0]

    # Upgrade binary hand joints → continuous 7-DOF joint angles (radians).
    # action.{left,right}_hand_joints are stored as binary 0/1 templates by
    # convert_episode.py; the actual per-finger motion is in debug/*_hand_pose.
    # SONIC expects 7 joint angles per hand (Dex3-1: thumb×3, index×2, middle×2).
    for side in ("left", "right"):
        k_joints = f"action.{side}_hand_joints"
        k_pose   = f"debug/{side}_hand_pose"
        k_active = f"debug/{side}_hand_active"
        if k_pose in data:
            active = data.get(k_active)
            joints_7dof = _compute_hand_joints_7dof(data[k_pose], side, active)
            data[k_joints] = joints_7dof
            print(f"  Hand joints [{side}]: → 7-DOF radians from {k_pose}")
            print(f"    joint range: "
                  + "  ".join(f"j{j}=[{joints_7dof[:,j].min():.2f},{joints_7dof[:,j].max():.2f}]"
                               for j in range(7)))
        # Free raw pose data — not needed downstream
        data.pop(k_pose,  None)
        data.pop(k_active, None)

    # Pre-compute wrist joint_pos for every frame
    T = data["_T"]
    data["_joint_pos"] = np.stack(
        [compute_joint_pos_from_smpl(data["smpl_pose"][i]) for i in range(T)], axis=0
    )   # (T, 29)

    # VR 3-point positions: retargeted from human proportions to G1 proportions.
    # smpl_joints are in robot local frame (x=fwd, y=left, z=up), same as pico_manager output.
    # _retarget_vr3pt anchors at the G1's FK shoulder and scales arm reach to G1 arm length.
    data["_vr3pt_pos_raw"]       = data["smpl_joints"][:, [22, 23, 12]].reshape(T, 9).astype(np.float32)
    data["_vr3pt_pos_corrected"] = _retarget_vr3pt(data["smpl_joints"])   # (T, 9)

    # VR 3-point orientations: recomputed via pico_manager's exact convention
    # (Q conjugation + per-joint OFFSETS + pelvis-relative), replacing the buggy
    # stored vr_3point_local_orn_target which was in the heading frame, not pelvis frame.
    if "debug/body_pose" in data:
        data["_vr3pt_orn_corrected"] = _compute_vr3pt_orn(data["debug/body_pose"])  # (T, 12)
        print("  VR 3-point orientations: recomputed from debug/body_pose (pico_manager convention)")
    else:
        data["_vr3pt_orn_corrected"] = data["vr_3point_local_orn_target"]
        print("  VR 3-point orientations: using stored (debug/body_pose not found)")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Compliance check
# ─────────────────────────────────────────────────────────────────────────────

def check_compliance(h5_path: Path) -> bool:
    print(f"\n{'═'*62}")
    print(f"  SONIC N1.7 compliance: {h5_path.name}")
    print(f"{'═'*62}")
    ok = True

    with h5py.File(h5_path, "r") as f:
        T = f["smpl_joints"].shape[0]
        print(f"  Frames: {T}  ({T/50:.1f}s @ 50 Hz)\n")

        groups = {
            "── VR 3-point stream (vr3pt mode) ──────────────────────": {
                "vr_3point_local_target":     (T, 9),
                "vr_3point_local_orn_target": (T, 12),
                "action.left_hand_joints":    (T, 6),
                "action.right_hand_joints":   (T, 6),
                "planner/movement_direction": (T, 3),
                "planner/facing_direction":   (T, 3),
                "planner/target_vel":         (T,),
                "planner/height":             (T,),
                "planner/mode":               (T,),
            },
            "── SMPL stream (smpl mode → N1.7 body encoder) ─────────": {
                "smpl_joints": (T, 24, 3),
                "smpl_pose":   (T, 21, 3),
                "body_quat":   (T, 4),      # sent as body_quat_w
                "frame_index": (T,),
            },
        }

        for header, checks in groups.items():
            print(f"  {header}")
            for key, expected in checks.items():
                try:
                    arr = _load(f, key)
                    shape_ok = arr.shape == expected
                    tag = "✓" if shape_ok else "✗ SHAPE"
                    print(f"    {tag}  {key:<40s}  {str(arr.shape)}")
                    if not shape_ok:
                        print(f"         expected {expected}")
                        ok = False
                except KeyError as e:
                    print(f"    ✗  {e}")
                    ok = False
            print()

        # Quaternion norms
        if "vr_3point_local_orn_target" in f:
            orn = f["vr_3point_local_orn_target"][:]
            for name, sl in [("lw", slice(0,4)), ("rw", slice(4,8)), ("head", slice(8,12))]:
                norms = np.linalg.norm(orn[:, sl], axis=1)
                bad = np.sum(np.abs(norms - 1.0) > 1e-3)
                print(f"  {'✓' if bad==0 else '✗'}  orn [{name}] quat norms ({bad} bad frames)")

        # Mode remapping
        try:
            modes = _load(f, "planner/mode")
            remapped = np.sum(modes == 22)
            if remapped:
                print(f"  ⚠  {remapped} frames with mode=22 → will remap to IDLE_SQUAT (4)")
        except KeyError:
            pass

        # Height range
        try:
            h = _load(f, "planner/height")
            print(f"  ✓  planner/height: [{h.min():.3f}, {h.max():.3f}] m")
        except KeyError:
            pass

    print(f"\n  Result: {'PASS ✓' if ok else 'FAIL ✗'}")
    print(f"{'═'*62}\n")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Frame builders
# ─────────────────────────────────────────────────────────────────────────────

_WALK_MODES = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}  # non-idle, non-squat modes


def _build_vr3pt_frame(data: dict, i: int, build_planner_message, vel_scale: float = 1.0) -> bytes:
    """One 'planner' topic message: VR 3-point + discrete lower-body commands.

    VR positions are in the PELVIS BODY FRAME (matching _process_3pt_pose in
    pico_manager), from data["_vr3pt_pos_raw"] (human-scale, no retargeting).

    facing_direction: stored rotation_xy (trajectory tangent) is only reliable
    when the robot is actually locomoting (movement != 0). When standing still,
    the tangent can be backward/arbitrary, so we default to the simulation's
    forward direction [1,0,0] to avoid the robot spinning during warmup/idle.
    """
    mode     = _sanitize_mode(int(data["planner/mode"][i]))
    movement = data["planner/movement_direction"][i].copy()

    if np.linalg.norm(movement) > 0.01:
        # Robot is actively locomoting — stored trajectory tangent is valid.
        facing = data["planner/facing_direction"][i].copy()
    else:
        # Robot standing still (movement=0): rotation_xy (trajectory tangent) is
        # unreliable/backward when computed at rest. Use simulation default so the
        # robot doesn't spin backward during warmup and pre-walk prep phases.
        facing = np.array([1.0, 0.0, 0.0])

    return build_planner_message(
        mode,
        movement.tolist(),
        facing.tolist(),
        speed=float(data["planner/target_vel"][i]) * vel_scale,
        height=float(data["planner/height"][i]),
        left_hand_position=data["action.left_hand_joints"][i].tolist(),
        right_hand_position=data["action.right_hand_joints"][i].tolist(),
        vr_3pt_position=data["_vr3pt_pos_corrected"][i].tolist(),  # 9 floats, G1-scale pelvis frame
        vr_3pt_orientation=data["_vr3pt_orn_corrected"][i].tolist(),
    )


def _build_v1_frame(data: dict, i: int) -> bytes:
    """
    Protocol V1 — direct G1 joint position frame.

    Used with:  bash deploy.sh --input-type zmq --zmq-topic pose sim
    Bypasses the SMPL human body encoder; the robot replays joint angles directly.

    Wire: b"pose" + 1280-byte JSON header + raw little-endian binary
    Fields (order matters):
      joint_pos   (1,29) f32   wrists retargeted from SMPL, rest = 0
      joint_vel   (1,29) f32   zeros
      body_quat_w (1,4)  f32   pelvis quaternion wxyz
      frame_index (1,)   i64   frame counter
      catch_up    (1,)   u8    always 0
    """
    joint_pos   = np.ascontiguousarray(data["_joint_pos"][i:i+1].astype(np.float32))
    joint_vel   = np.zeros((1, 29), dtype=np.float32)
    body_quat_w = np.ascontiguousarray(data["body_quat"][i:i+1].astype(np.float32))
    frame_index = data["frame_index"][i:i+1].astype(np.int64)
    catch_up    = np.zeros(1, dtype=np.uint8)

    fields = [
        {"name": "joint_pos",   "dtype": "f32", "shape": [1, 29]},
        {"name": "joint_vel",   "dtype": "f32", "shape": [1, 29]},
        {"name": "body_quat_w", "dtype": "f32", "shape": [1, 4]},
        {"name": "frame_index", "dtype": "i64", "shape": [1]},
        {"name": "catch_up",    "dtype": "u8",  "shape": [1]},
    ]
    header_json = json.dumps(
        {"v": 1, "endian": "le", "count": 1, "fields": fields},
        separators=(",", ":"),
    ).encode("utf-8")
    header_bytes = header_json.ljust(1280, b"\x00")

    payload = (joint_pos.tobytes() + joint_vel.tobytes() +
               body_quat_w.tobytes() + frame_index.tobytes() + catch_up.tobytes())

    return b"pose" + header_bytes + payload


def _build_smpl_window(data: dict, window: deque, pack_pose_message) -> bytes:
    """
    One 'pose' topic message matching PoseStreamer output exactly.

    window: deque of frame indices, len == num_frames_to_send
    Stacks W frames for smpl_joints, smpl_pose, body_quat_w, joint_pos, joint_vel, frame_index.
    Uses the *last* frame for scalar/current-frame fields (vr_position, hand_joints, etc.).
    """
    idxs = list(window)
    W    = len(idxs)
    i    = idxs[-1]    # current frame

    numpy_data = {
        # Stacked window  (W, ...)
        "smpl_pose":    np.stack([data["smpl_pose"][j]    for j in idxs]).astype(np.float32),
        "smpl_joints":  np.stack([data["smpl_joints"][j]  for j in idxs]).astype(np.float32),
        "body_quat_w":  np.stack([data["body_quat"][j]    for j in idxs]).astype(np.float32),
        "joint_pos":    np.stack([data["_joint_pos"][j]   for j in idxs]).astype(np.float32),
        "joint_vel":    np.zeros((W, 29), dtype=np.float32),
        "frame_index":  np.array([data["frame_index"][j]  for j in idxs], dtype=np.int64),

        # Current-frame VR 3-point — raw human-scale positions (encoder was trained on human data)
        "vr_position":    data["_vr3pt_pos_raw"][i].astype(np.float32),           # (9,)
        "vr_orientation": data["vr_3point_local_orn_target"][i].astype(np.float32), # (12,)

        # Hand joints (current frame)
        "left_hand_joints":  data["action.left_hand_joints"][i].astype(np.float32),
        "right_hand_joints": data["action.right_hand_joints"][i].astype(np.float32),

        # Controller inputs — no live hardware, use neutral values
        "left_trigger":  np.array([0.0], dtype=np.float32),
        "right_trigger": np.array([0.0], dtype=np.float32),
        "left_grip":     np.array([0.0], dtype=np.float32),
        "right_grip":    np.array([0.0], dtype=np.float32),

        # Timestamps (from HDF5 if available, else 0)
        "pico_dt":             np.array([1.0 / 50.0], dtype=np.float32),
        "pico_fps":            np.array([50.0], dtype=np.float32),
        "timestamp_realtime":  np.array([float(data["local_timestamps_ns"][i]) * 1e-9],
                                         dtype=np.float64),
        "timestamp_monotonic": np.array([float(data["local_timestamps_ns"][i]) * 1e-9],
                                         dtype=np.float64),

        # Data-collection toggles — always False for replay
        "toggle_data_collection": np.array([False], dtype=bool),
        "toggle_data_abort":      np.array([False], dtype=bool),

        # Heading — no live joystick, use 0
        "heading_increment": np.array([0.0], dtype=np.float32),
    }

    return pack_pose_message(numpy_data, topic="pose")


# ─────────────────────────────────────────────────────────────────────────────
# Publisher
# ─────────────────────────────────────────────────────────────────────────────

def publish(data: dict, port: int, fps: float, stream: str,
            num_frames_window: int, loop: bool, dry_run: bool,
            groot_wbc_root: str, vel_scale: float = 1.0):

    T  = data["_T"]
    dt = 1.0 / fps

    # v1 mode builds its own wire messages without gear_sonic imports
    build_command_message = build_planner_message = pack_pose_message = None
    if stream != "v1":
        sys.path.insert(0, groot_wbc_root)
        try:
            from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
                build_command_message,
                build_planner_message,
                pack_pose_message,
            )
        except ImportError as e:
            sys.exit(f"Cannot import gear_sonic: {e}\n"
                     f"  Check --groot-wbc-root (currently: {groot_wbc_root})")

    if dry_run:
        print(f"  Dry-run: {T} frames @ {fps} Hz  "
              f"(budget {dt*1000:.1f}ms, stream={stream}, window={num_frames_window})")
        _dry_run(data, T, dt, stream, num_frames_window,
                 build_command_message, build_planner_message, pack_pose_message,
                 vel_scale)
        return

    import zmq
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{port}")

    print(f"\n  ZMQ PUB → tcp://*:{port}  stream={stream}  window={num_frames_window}")
    print(f"  {T} frames  {T/fps:.1f}s  @ {fps} Hz  {'∞ loop' if loop else '1 pass'}")

    if stream == "v1":
        print("  Mode: Protocol V1 — direct G1 joint replay (encoder mode 0)")
        print("  SONIC: bash deploy.sh --input-type zmq --zmq-topic pose sim")
    elif stream == "vr3pt":
        print("  Mode: VR 3-point → kinematic planner")
        print("  SONIC: bash deploy.sh --input-type zmq_manager sim")
        if vel_scale != 1.0:
            raw_vels = data["planner/target_vel"]
            print(f"  vel_scale={vel_scale}×  "
                  f"raw [{raw_vels.min():.3f}, {raw_vels.max():.3f}] m/s → "
                  f"[{raw_vels.min()*vel_scale:.3f}, {raw_vels.max()*vel_scale:.3f}] m/s")
    else:
        print("  Mode: SMPL v3 → N1.7 human body encoder (encoder mode 2)")
        print("  SONIC: bash deploy.sh --input-type zmq sim")

    print(f"  Waiting 1s for subscribers …")
    time.sleep(1.0)

    if stream == "vr3pt":
        sock.send(build_command_message(start=True, stop=False, planner=True))
        print("  → command: start planner (VR 3-point)")

        # ZMQManager's handlePlannerInput() blocks up to 5s waiting for the planner
        # ONNX to initialize. Warm up by repeating frame 0 for 6s so the episode
        # starts from the beginning once SONIC is ready.
        warmup_secs = 6.0
        print(f"  Warming up {warmup_secs:.0f}s for planner ONNX init (hold frame 0) …")
        warmup_end = time.perf_counter() + warmup_secs
        while time.perf_counter() < warmup_end:
            sock.send(_build_vr3pt_frame(data, 0, build_planner_message, vel_scale))
            time.sleep(dt)
        print("  Warmup complete — starting episode")

    elif stream == "smpl":
        sock.send(build_command_message(start=True, stop=False, planner=False))
        print("  → command: start pose (SMPL → N1.7 body encoder)")
        time.sleep(0.2)
    # v1: no command message needed — subscriber is a direct ZMQ reader

    pass_n = 0

    while True:
        pass_n += 1
        t_pass = time.perf_counter()
        frame_lats = []

        window: deque[int] = deque(maxlen=num_frames_window)

        for i in range(T):
            t0 = time.perf_counter()

            if stream == "v1":
                msg = _build_v1_frame(data, i)
                sock.send(msg)
            elif stream == "vr3pt":
                msg = _build_vr3pt_frame(data, i, build_planner_message, vel_scale)
                sock.send(msg)
            else:   # smpl
                window.append(i)
                if len(window) == num_frames_window:  # wait for buffer to fill
                    msg = _build_smpl_window(data, window, pack_pose_message)
                    sock.send(msg)

            elapsed = time.perf_counter() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)
            frame_lats.append((time.perf_counter() - t0) * 1000)

        dur = time.perf_counter() - t_pass
        lat = np.array(frame_lats)
        print(f"  Pass {pass_n:3d} | {T} frames {dur:.2f}s "
              f"| lat mean={lat.mean():.2f}ms max={lat.max():.2f}ms "
              f"budget={dt*1000:.1f}ms overruns={np.sum(lat > dt*1000)}")

        if not loop:
            break

    time.sleep(0.1)
    sock.close()
    ctx.term()


def _dry_run(data, T, dt, stream, W,
             build_command_message, build_planner_message, pack_pose_message,
             vel_scale: float = 1.0):
    lats = []
    window: deque[int] = deque(maxlen=W)

    for i in range(T):
        t0 = time.perf_counter()
        if stream == "v1":
            _build_v1_frame(data, i)
        elif stream == "vr3pt":
            _build_vr3pt_frame(data, i, build_planner_message, vel_scale)
        else:
            window.append(i)
            if len(window) == W:
                _build_smpl_window(data, window, pack_pose_message)
        lats.append((time.perf_counter() - t0) * 1000)

    lat = np.array(lats)

    # Sample message sizes
    if stream == "v1":
        sample = _build_v1_frame(data, 0)
        print(f"  payload/frame: {len(sample)} bytes  (Protocol V1, no command msg)")
    elif stream == "vr3pt":
        sample = _build_vr3pt_frame(data, 0, build_planner_message, vel_scale)
        cmd = build_command_message(start=True, stop=False, planner=True)
        print(f"  command msg  : {len(cmd)} bytes")
        print(f"  payload/frame: {len(sample)} bytes")
    else:
        w = deque(range(min(W, T)), maxlen=W)
        sample = _build_smpl_window(data, w, pack_pose_message)
        cmd = build_command_message(start=True, stop=False, planner=False)
        print(f"  command msg  : {len(cmd)} bytes")
        print(f"  payload/frame: {len(sample)} bytes")

    print(f"  pack overhead: mean={lat.mean():.3f}ms  max={lat.max():.3f}ms  "
          f"budget={dt*1000:.1f}ms  "
          f"{'✓ OK' if lat.max() < dt*1000 else '✗ OVER BUDGET'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Publish SONIC N1.7 episode data to GR00T-WBC simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("hdf5",          type=Path)
    ap.add_argument("--stream",      choices=["vr3pt", "smpl", "v1"], default="vr3pt",
                    help="vr3pt: VR 3-point + kinematic planner  |  "
                         "smpl: SMPL body tracking → N1.7 human body encoder  |  "
                         "v1: Protocol V1 direct joint replay (--input-type zmq --zmq-topic pose)")
    ap.add_argument("--port",        type=int,   default=5556)
    ap.add_argument("--fps",         type=float, default=50.0)
    ap.add_argument("--window",      type=int,   default=5,
                    help="num_frames_to_send window size for smpl mode (default 5)")
    ap.add_argument("--loop",        action="store_true")
    ap.add_argument("--check-only",  action="store_true")
    ap.add_argument("--dry-run",     action="store_true")
    ap.add_argument("--skip-check",  action="store_true")
    ap.add_argument("--groot-wbc-root",
                    default=str(Path.home() / "Projects/GR00T-WholeBodyControl"))
    ap.add_argument("--vel-scale", type=float, default=1.0,
                    help="Multiply target_vel by this factor (vr3pt only). "
                         "Use >1 if recorded velocities are too low for visible locomotion. "
                         "Recommended: 3.0 to scale 0.05-0.10 m/s → 0.15-0.30 m/s.")
    args = ap.parse_args()

    if not args.hdf5.exists():
        sys.exit(f"File not found: {args.hdf5}")

    if not args.skip_check:
        ok = check_compliance(args.hdf5)
        if not ok:
            sys.exit("Compliance check FAILED.")

    if args.check_only:
        return

    print(f"  Loading {args.hdf5.name} …")
    data = load_episode(args.hdf5)
    mb = sum(v.nbytes for v in data.values() if isinstance(v, np.ndarray)) / 1e6
    print(f"  {data['_T']} frames ({mb:.1f} MB)")
    if args.stream in ("smpl", "v1"):
        print(f"  Pre-computed wrist joint_pos: shape {data['_joint_pos'].shape}")

    publish(data, args.port, args.fps, args.stream,
            args.window, args.loop, args.dry_run, args.groot_wbc_root,
            vel_scale=args.vel_scale)


if __name__ == "__main__":
    main()
