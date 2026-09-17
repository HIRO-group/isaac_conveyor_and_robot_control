"""Publishes the camera list (latched) and per-camera RGB8 frames with a FrameMetadata attachment."""

from __future__ import annotations

import logging

from cameras.frame_meta import FrameCounter, now_us, uuid_v7
from cameras.protos import camera
from conveyor_indexing.topics import Topics
from conveyor_indexing.zenoh_session import open_session

logger = logging.getLogger(__name__)


class CameraZenohPublisher:
    def __init__(self, camera_list: camera.CameraList, topics: Topics | None = None) -> None:
        self.topics = topics or Topics.from_env()
        self._session = open_session()
        self._frame_counter = FrameCounter()
        self._list_bytes = camera_list.SerializeToString()

        # put() for subscribers already listening, queryable for late joiners.
        self._list_publisher = self._session.declare_publisher(self.topics.camera_list)
        self._list_publisher.put(self._list_bytes)
        self._list_queryable = self._session.declare_queryable(self.topics.camera_list, self._handle_list_query)

        self._color_publishers = {
            info.serial: self._session.declare_publisher(info.color_topic) for info in camera_list.cameras
        }
        logger.info("serving %s with %d camera(s)", self.topics.camera_list, len(self._color_publishers))

    def _handle_list_query(self, query) -> None:
        query.reply(self.topics.camera_list, self._list_bytes)

    def publish_frame(self, serial: str, rgb_bytes: bytes, capture_ts_us: int) -> None:
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
        logger.info("camera Zenoh session closed")
