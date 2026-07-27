"""Standalone verification tool for the camera rig - deliberately does not
involve theia at all, so the camera contract (src/cameras/) can be checked
end-to-end on its own.

Connects to the same Zenoh session the sim publishes on
(src/cameras/zenoh_publisher.py), fetches `theia/camera/list`, subscribes to
one camera's color topic, and dumps the next frame to a PPM file (zero
image-library dependencies) for visual inspection.

Usage (run alongside `DISPLAY=:0 bash scripts/run.sh` - peer-to-peer Zenoh
scouting connects the two without a router, matching the sim's own
ZENOH_ROUTER-unset default):

    PYTHONPATH=/tmp/proto_gen python3 scripts/camera_probe.py [--serial SIM1-PICK] [--out FILE.ppm]

Requires `eclipse-zenoh` (see scripts/setup.sh) and the generated
`sim_camera_pb2` bindings on PYTHONPATH (see gen_proto.sh).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import queue
import sys
import time

try:
    import zenoh
except ImportError:
    sys.exit("eclipse-zenoh is required: pip install eclipse-zenoh==1.7.1 (see scripts/setup.sh)")

try:
    import sim_camera_pb2 as camera
except ImportError:
    sys.exit(
        "sim_camera_pb2 not importable - generate it first (bash gen_proto.sh) and put it on "
        "PYTHONPATH, e.g.: PYTHONPATH=/tmp/proto_gen python3 scripts/camera_probe.py"
    )

LIST_KEY = "theia/camera/list"
LIST_QUERY_TIMEOUT_S = 5.0
FRAME_WAIT_TIMEOUT_S = 5.0


def _payload_bytes(sample) -> bytes | None:
    """Mirrors theia's own payload-extraction helper (see
    ~/theia/data_collection/src/data_collection_vol2.py, read-only reference,
    not imported) so this probe's success is a faithful stand-in for theia's.
    """
    payload = getattr(sample, "payload", None)
    if payload is None:
        return None
    return payload.to_bytes() if hasattr(payload, "to_bytes") else bytes(payload)


def _open_session() -> zenoh.Session:
    conf = zenoh.Config()
    router = os.environ.get("ZENOH_ROUTER")
    if router:
        conf.insert_json5("connect/endpoints", f'["{router}"]')
        print(f"connecting to Zenoh router at {router}")
    else:
        print("ZENOH_ROUTER not set; opening in peer-to-peer mode")
    return zenoh.open(conf)


def fetch_camera_list(session: zenoh.Session) -> camera.CameraList:
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
            sys.exit("theia/camera/list replied with zero cameras")

        print(f"{len(cameras)} camera(s) on theia/camera/list:")
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
