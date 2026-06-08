import time
import argparse
import h5py
import numpy as np
import zmq
import msgpack
import msgpack_numpy as m

m.patch()


def load_episode(path):
    with h5py.File(path, "r") as f:
        required = [
            "body_quat",
            "frame_index",
            "joint_pos",
            "joint_vel",
            "smpl_joints",
            "smpl_pose",
        ]
        for k in required:
            if k not in f:
                raise KeyError(f"Missing required dataset: {k}")

        data = {k: f[k][:] for k in required}

    T = data["smpl_joints"].shape[0]
    for k, v in data.items():
        if v.shape[0] != T:
            raise ValueError(f"{k} has length {v.shape[0]}, expected {T}")

    data["body_quat"] = data["body_quat"].astype(np.float32)
    data["frame_index"] = data["frame_index"].astype(np.int32)
    data["joint_pos"] = data["joint_pos"].astype(np.float32)
    data["joint_vel"] = data["joint_vel"].astype(np.float32)
    data["smpl_joints"] = data["smpl_joints"].astype(np.float32)
    data["smpl_pose"] = data["smpl_pose"].astype(np.float32)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    data = load_episode(args.file)
    T = data["smpl_joints"].shape[0]

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.host}:{args.port}")

    print(f"Publishing {T} frames from {args.file}")
    print("Waiting 1s for subscriber to connect...")
    time.sleep(1.0)

    dt = 1.0 / args.fps

    while True:
        t0 = time.time()

        for i in range(T):
            msg = {
                "version": 3,
                "body_quat": data["body_quat"][i:i+1],
                "frame_index": data["frame_index"][i:i+1],
                "joint_pos": data["joint_pos"][i:i+1],
                "joint_vel": data["joint_vel"][i:i+1],
                "smpl_joints": data["smpl_joints"][i:i+1],
                "smpl_pose": data["smpl_pose"][i:i+1],
                "catch_up": False,
            }

            payload = msgpack.packb(msg, default=m.encode, use_bin_type=True)
            sock.send_multipart([args.topic.encode("utf-8"), payload])

            time.sleep(dt)

        print(f"Finished episode in {time.time() - t0:.2f}s")

        if not args.loop:
            break


if __name__ == "__main__":
    main()
