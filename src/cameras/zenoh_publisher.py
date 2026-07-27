"""Publishes camera frames on theia's wire contract over Zenoh - see the
top-level README's "Design" section for the exact contract and why this is
a local mirror rather than a theia dependency.

Session setup deliberately mirrors theia's own collector
(`~/theia/data_collection/src/data_collection_vol2.py`, read-only reference,
not imported): `ZENOH_ROUTER` env var set -> connect to that endpoint; unset
-> open in peer mode, so this sim runs standalone (no router needed) as well
as alongside a real theia deployment.
"""

from __future__ import annotations

import logging
import os

from cameras.frame_meta import FrameCounter, now_us, uuid_v7
from cameras.protos import camera
from cameras.specs import color_topic

logger = logging.getLogger(__name__)

try:
    import zenoh
except ImportError as exc:
    raise SystemExit(
        "eclipse-zenoh is required for camera publishing but is not installed "
        "in this interpreter. Install it into Isaac Sim's bundled python:\n"
        "  /home/ubuntu/IsaacSim/python.sh -m pip install eclipse-zenoh==1.7.1\n"
        "(or run scripts/setup.sh, which does this for you - see the "
        "top-level README's 'Setup' section)."
    ) from exc


def _open_session() -> zenoh.Session:
    conf = zenoh.Config()
    router = os.environ.get("ZENOH_ROUTER")
    if router:
        conf.insert_json5("connect/endpoints", f'["{router}"]')
        logger.info("connecting to Zenoh router at %s", router)
    else:
        logger.warning("ZENOH_ROUTER not set; opening Zenoh session in peer-to-peer mode")
    return zenoh.open(conf)


class CameraZenohPublisher:
    """Owns one Zenoh session for the whole camera rig: serves the latched
    camera list, and publishes per-camera color frames with a FrameMetadata
    attachment.
    """

    LIST_KEY = "theia/camera/list"

    def __init__(self, camera_list: camera.CameraList) -> None:
        self._session = _open_session()
        self._frame_counter = FrameCounter()
        self._list_bytes = camera_list.SerializeToString()

        # Latched publisher + queryable on the same key: theia's Python
        # collector does a one-shot session.get() at startup, theia's Rust
        # recorder queries with a timeout - a plain put() alone only satisfies
        # subscribers that were already listening, so both are needed.
        self._list_publisher = self._session.declare_publisher(self.LIST_KEY)
        self._list_publisher.put(self._list_bytes)
        self._list_queryable = self._session.declare_queryable(self.LIST_KEY, self._handle_list_query)

        self._color_publishers = {
            info.serial: self._session.declare_publisher(color_topic(info.serial)) for info in camera_list.cameras
        }
        logger.info("serving theia/camera/list with %d camera(s), publishers ready", len(self._color_publishers))

    def _handle_list_query(self, query: zenoh.Query) -> None:
        query.reply(self.LIST_KEY, self._list_bytes)

    def publish_frame(self, serial: str, rgb_bytes: bytes, capture_ts_us: int) -> None:
        """Publish one raw RGB8 frame (no proto wrapper - see specs.py's
        COLOR_FORMAT docstring) with a FrameMetadata protobuf attachment.
        """
        publisher = self._color_publishers.get(serial)
        if publisher is None:
            logger.warning("publish_frame called for unknown serial %s", serial)
            return
        metadata = camera.FrameMetadata(
            timestamp_camera_us=capture_ts_us,
            timestamp_received_us=capture_ts_us,
            timestamp_published_us=now_us(),
            frame_number=self._frame_counter.next(serial),
            uuid_v7=uuid_v7(),
        )
        publisher.put(rgb_bytes, attachment=metadata.SerializeToString())

    def close(self) -> None:
        self._list_queryable.undeclare()
        self._list_publisher.undeclare()
        for publisher in self._color_publishers.values():
            publisher.undeclare()
        self._session.close()
        logger.info("Zenoh session closed")
