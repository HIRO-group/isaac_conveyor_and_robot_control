"""Standalone check of the camera rig's Zenoh contract.

Fetches the camera list, subscribes to one camera's color topic and dumps the
next frame to a PPM file.

Usage (same PYTHONPATH as scripts/run.sh):

    PYTHONPATH=src:/tmp/proto_gen python3 scripts/camera_probe.py [--serial SIM1-PICK] [--out FILE.ppm]

Requires `eclipse-zenoh` (see scripts/setup.sh) and the generated
`sim_camera_pb2` bindings on PYTHONPATH (see gen_proto.sh).
"""

from __future__ import annotations

import argparse
import pathlib
import queue
import sys
import time

try:
    import sim_camera_pb2 as camera
    from conveyor_indexing.topics import Topics
    from conveyor_indexing.zenoh_session import open_session
except ImportError:
    sys.exit("run with PYTHONPATH=src:/tmp/proto_gen after `bash gen_proto.sh` (see scripts/setup.sh)")

TOPICS = Topics.from_env()
LIST_KEY = TOPICS.camera_list
LIST_QUERY_TIMEOUT_S = 5.0
FRAME_WAIT_TIMEOUT_S = 5.0


def _payload_bytes(sample) -> bytes | None:
    payload = getattr(sample, "payload", None)
    if payload is None:
        return None
    return payload.to_bytes() if hasattr(payload, "to_bytes") else bytes(payload)


def _open_session():
    return open_session()


def fetch_camera_list(session) -> camera.CameraList:
    replies = list(session.get(LIST_KEY, timeout=LIST_QUERY_TIMEOUT_S))
    for reply in replies:
        payload = _payload_bytes(reply.ok)
        if payload:
            camera_list = camera.CameraList()
            camera_list.ParseFromString(payload)
            return camera_list
    sys.exit(f"no reply on {LIST_KEY} within {LIST_QUERY_TIMEOUT_S}s - is the sim running?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None, help="Camera serial to probe (default: first in the list)")
    parser.add_argument("--out", default=None, help="Output PPM path (default: <serial>.ppm in the scratchpad)")
    args = parser.parse_args()

    session = _open_session()
    try:
        camera_list = fetch_camera_list(session)
        cameras = list(camera_list.cameras)
        if not cameras:
            sys.exit(f"{LIST_KEY} replied with zero cameras")

        print(f"{len(cameras)} camera(s) on {LIST_KEY}:")
        for info in cameras:
            role_name = camera.CameraRole.Name(info.role)
            print(f"  {info.serial}: {info.width}x{info.height}@{info.fps} {info.format} role={role_name}")

        info = next((c for c in cameras if c.serial == args.serial), cameras[0]) if args.serial else cameras[0]
        expected_len = info.width * info.height * 3
        if info.format.upper() != "RGB8":
            print(f"WARNING: {info.serial}'s format is {info.format!r}, this probe assumes RGB8")

        frame_queue: queue.Queue = queue.Queue()

        def _on_sample(sample) -> None:
            frame_queue.put(sample)

        subscriber = session.declare_subscriber(info.color_topic, _on_sample)
        print(f"subscribed to {info.color_topic}, waiting up to {FRAME_WAIT_TIMEOUT_S}s for a frame...")
        try:
            sample = frame_queue.get(timeout=FRAME_WAIT_TIMEOUT_S)
        except queue.Empty:
            sys.exit(f"no frame received on {info.color_topic} within {FRAME_WAIT_TIMEOUT_S}s")
        finally:
            subscriber.undeclare()

        rgb_bytes = _payload_bytes(sample)
        if rgb_bytes is None or len(rgb_bytes) != expected_len:
            sys.exit(f"frame payload is {0 if rgb_bytes is None else len(rgb_bytes)} bytes, expected {expected_len}")

        attachment = getattr(sample, "attachment", None)
        if attachment is not None:
            att_bytes = attachment.to_bytes() if hasattr(attachment, "to_bytes") else bytes(attachment)
            meta = camera.FrameMetadata()
            meta.ParseFromString(att_bytes)
            age_s = (time.time_ns() // 1_000 - meta.timestamp_published_us) / 1e6
            print(
                f"FrameMetadata: frame_number={meta.frame_number} uuid_v7={meta.uuid_v7} "
                f"published {age_s:.3f}s ago"
            )
        else:
            print("WARNING: no attachment on this sample (expected a FrameMetadata attachment)")

        out_path = pathlib.Path(args.out) if args.out else pathlib.Path(f"{info.serial}.ppm")
        with open(out_path, "wb") as f:
            f.write(f"P6\n{info.width} {info.height}\n255\n".encode("ascii"))
            f.write(rgb_bytes)
        print(f"wrote {out_path} ({info.width}x{info.height})")
    finally:
        session.close()


if __name__ == "__main__":
    main()
