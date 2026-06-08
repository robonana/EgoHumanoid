"""
debug_planner_subscriber.py  —  real-time ZMQ decoder for SONIC planner messages

Run this alongside sonic_publisher.py (vr3pt mode) to verify what locomotion
commands are actually arriving on the wire.  Decodes the planner topic wire
format independently (no gear_sonic imports needed).

Wire format recap (from zmq_planner_sender.py):
  Every message = topic_bytes + 1280-byte JSON header + raw little-endian binary
  planner topic fields: mode(i32), movement(f32[3]), facing(f32[3]),
                        speed(f32), height(f32), [optional vr_position(f32[9]), ...]
  command topic fields: start(u8), stop(u8), planner(u8)

Usage:
    python debug_planner_subscriber.py                  # port 5556, print every frame
    python debug_planner_subscriber.py --port 5557
    python debug_planner_subscriber.py --every 10       # print every 10th planner frame
    python debug_planner_subscriber.py --log out.csv    # also write CSV
"""

import argparse
import json
import struct
import sys
import time
from collections import Counter

import numpy as np
import zmq

HEADER_SIZE = 1280

LOCO_NAMES = {
    0: "IDLE", 1: "SLOW_WALK", 2: "WALK", 3: "FAST_WALK",
    4: "IDLE_SQUAT", 22: "mode_22→SQUAT",
}

# ─────────────────────────────────────────────────────────────────────────────
# Wire decoder
# ─────────────────────────────────────────────────────────────────────────────

def _decode_message(raw: bytes) -> tuple[str, dict]:
    """
    Decode a raw ZMQ message into (topic_str, fields_dict).
    fields_dict maps field name → numpy scalar or array.
    Returns (topic, {}) on parse error.
    """
    # Find topic boundary: first null or first non-ASCII printable run
    # The topic is a short ASCII string; header starts at a fixed offset
    # determined by scanning for the first '{' after the topic prefix.
    brace = raw.find(b"{")
    if brace < 0:
        return ("?", {})
    topic = raw[:brace].decode("ascii", errors="replace").rstrip("\x00")

    header_bytes = raw[brace: brace + HEADER_SIZE]
    payload      = raw[brace + HEADER_SIZE:]

    try:
        hdr = json.loads(header_bytes.decode("utf-8").rstrip("\x00"))
    except json.JSONDecodeError:
        return (topic, {})

    fields  = hdr.get("fields", [])
    out     = {}
    offset  = 0

    dtype_map = {
        "f32": (np.float32, 4),
        "f64": (np.float64, 8),
        "i32": (np.int32,   4),
        "i64": (np.int64,   8),
        "u8":  (np.uint8,   1),
        "bool":(np.uint8,   1),
    }

    for f in fields:
        name  = f["name"]
        dtype_str = f.get("dtype", "f32")
        shape = f.get("shape", [1])
        n_elem = 1
        for s in shape:
            n_elem *= s

        np_dtype, item_size = dtype_map.get(dtype_str, (np.float32, 4))
        n_bytes = n_elem * item_size

        chunk = payload[offset: offset + n_bytes]
        offset += n_bytes

        if len(chunk) < n_bytes:
            break

        arr = np.frombuffer(chunk, dtype=np.dtype(np_dtype).newbyteorder("<"))
        out[name] = arr.reshape(shape) if len(shape) > 1 else (arr[0] if n_elem == 1 else arr)

    return (topic, out)


# ─────────────────────────────────────────────────────────────────────────────
# Formatter
# ─────────────────────────────────────────────────────────────name─────────────

def _fmt_vec(arr, n=3) -> str:
    if arr is None:
        return "—"
    v = np.asarray(arr).flat
    return "[" + ", ".join(f"{float(x):+.3f}" for x in list(v)[:n]) + "]"


def _mode_label(m: int) -> str:
    return f"{LOCO_NAMES.get(m, f'mode_{m}')}({m})"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Real-time decoder for SONIC ZMQ planner messages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--port",  type=int, default=5556)
    ap.add_argument("--host",  default="localhost")
    ap.add_argument("--every", type=int, default=1,
                    help="Print every Nth planner frame (default: every frame)")
    ap.add_argument("--log",   default=None,
                    help="Also write CSV log to this path")
    args = ap.parse_args()

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{args.host}:{args.port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"planner")
    sock.setsockopt(zmq.SUBSCRIBE, b"command")
    sock.setsockopt(zmq.RCVTIMEO, 5000)   # 5 s timeout so we can ctrl-C cleanly

    print(f"\n{'═'*70}")
    print(f"  SONIC planner subscriber  tcp://{args.host}:{args.port}")
    print(f"  Topics: planner, command   (print every {args.every} frame)")
    print(f"{'═'*70}")
    print(f"  {'frame':>6}  {'elapsed':>7}  {'mode':<18}  {'movement xyz':>20}  "
          f"{'facing xyz':>20}  {'vel':>6}  {'|move|':>6}  vr?")
    print(f"  {'─'*6}  {'─'*7}  {'─'*18}  {'─'*20}  {'─'*20}  {'─'*6}  {'─'*6}  ───")

    csv_fh = open(args.log, "w") if args.log else None
    if csv_fh:
        csv_fh.write("frame,elapsed_s,mode,move_x,move_y,move_z,face_x,face_y,face_z,"
                     "speed,height,has_vr\n")

    frame_n   = 0
    t_start   = None
    mode_hist = Counter()

    try:
        while True:
            try:
                raw = sock.recv()
            except zmq.Again:
                print("  [timeout – no message in 5 s; is the publisher running?]")
                continue

            topic, fields = _decode_message(raw)

            # ── command message ─────────────────────────────────────────────
            if topic == "command":
                start   = int(fields.get("start",   0))
                stop    = int(fields.get("stop",    0))
                planner = int(fields.get("planner", 0))
                print(f"\n  ▶ COMMAND  start={start}  stop={stop}  planner={planner}\n")
                if start and t_start is None:
                    t_start = time.perf_counter()
                continue

            # ── planner message ─────────────────────────────────────────────
            if topic != "planner":
                continue

            if t_start is None:
                t_start = time.perf_counter()

            frame_n += 1
            elapsed  = time.perf_counter() - t_start

            mode = int(fields.get("mode", -1))
            move = np.asarray(fields.get("movement", [0, 0, 0]), dtype=float)
            face = np.asarray(fields.get("facing",   [1, 0, 0]), dtype=float)
            vel  = float(fields.get("speed",  -1))
            hgt  = float(fields.get("height", -1))
            has_vr = "vr_position" in fields

            mode_hist[mode] += 1
            move_norm = float(np.linalg.norm(move))

            if frame_n % args.every == 0:
                vr_tag = "✓" if has_vr else "✗"
                print(f"  {frame_n:6d}  {elapsed:6.2f}s  {_mode_label(mode):<18}  "
                      f"{_fmt_vec(move):>20}  {_fmt_vec(face):>20}  "
                      f"{vel:6.3f}  {move_norm:6.3f}  {vr_tag}")

            if csv_fh:
                mx, my, mz = float(move[0]), float(move[1]), float(move[2])
                fx, fy, fz = float(face[0]), float(face[1]), float(face[2])
                csv_fh.write(f"{frame_n},{elapsed:.4f},{mode},{mx:.4f},{my:.4f},{mz:.4f},"
                             f"{fx:.4f},{fy:.4f},{fz:.4f},{vel:.4f},{hgt:.4f},"
                             f"{1 if has_vr else 0}\n")
                csv_fh.flush()

    except KeyboardInterrupt:
        pass
    finally:
        # Summary
        print(f"\n{'═'*70}")
        print(f"  Total planner frames received: {frame_n}")
        if frame_n > 0 and t_start is not None:
            dur = time.perf_counter() - t_start
            print(f"  Duration: {dur:.2f}s  ({frame_n/dur:.1f} Hz actual)")
        print(f"\n  Mode distribution:")
        for m, cnt in sorted(mode_hist.items()):
            pct = 100 * cnt / max(frame_n, 1)
            print(f"    {_mode_label(m):<22}: {cnt:5d} frames ({pct:.1f}%)")
        print(f"{'═'*70}\n")

        if csv_fh:
            csv_fh.close()
            print(f"  CSV written to: {args.log}")
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
