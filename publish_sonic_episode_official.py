import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import zmq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--groot-wbc-root",
        default=str(Path.home() / "Projects/GR00T-WholeBodyControl"),
    )
    args = parser.parse_args()

    sys.path.insert(0, args.groot_wbc_root)
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

    with h5py.File(args.file, "r") as f:
        required = [
            "body_quat",
            "frame_index",
            "joint_pos",
            "joint_vel",
            "smpl_joints",
            "smpl_pose",
        ]
        data = {}
        for k in required:
            if k not in f:
                raise KeyError(f"Missing required field: {k}")
            data[k] = f[k][:]

    T = data["smpl_joints"].shape[0]

    data["body_quat"] = data["body_quat"].astype(np.float32)
    data["frame_index"] = data["frame_index"].astype(np.int32)
    data["joint_pos"] = data["joint_pos"].astype(np.float32)
    data["joint_vel"] = data["joint_vel"].astype(np.float32)
    data["smpl_joints"] = data["smpl_joints"].astype(np.float32)
    data["smpl_pose"] = data["smpl_pose"].astype(np.float32)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")

    print(f"Publishing {T} Protocol-v3 frames from {args.file}")
    print(f"ZMQ: tcp://*:{args.port}, topic={args.topic}")
    print("Waiting 1s for subscriber...")
    time.sleep(1.0)

    dt = 1.0 / args.fps

    while True:
        t_start = time.time()

        for i in range(T):
            pose_data = {
                # Common fields
                "body_quat": data["body_quat"][i : i + 1],
                "frame_index": data["frame_index"][i : i + 1],

                # Protocol v3 required fields
                "joint_pos": data["joint_pos"][i : i + 1],
                "joint_vel": data["joint_vel"][i : i + 1],
                "smpl_joints": data["smpl_joints"][i : i + 1],
                "smpl_pose": data["smpl_pose"][i : i + 1],

                # Optional
                "catch_up": np.array([False], dtype=np.bool_),
            }

            msg = pack_pose_message(pose_data, topic=args.topic, version=3)
            sock.send(msg)
            time.sleep(dt)

        print(f"Finished episode in {time.time() - t_start:.2f}s")

        if not args.loop:
            break


if __name__ == "__main__":
    main()
