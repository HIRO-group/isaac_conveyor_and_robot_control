"""One sim camera's description, plus the pure helpers that build the camera list message."""

from __future__ import annotations

from dataclasses import dataclass

from cameras.protos import camera
from conveyor_indexing.topics import Topics

COLOR_FORMAT = "RGB8"

# Scene-package role names -> wire enum.
ROLE_BY_NAME = {
    "pick_cam": camera.CameraRole.CAMERA_ROLE_PICK_CAM,
    "place_cam": camera.CameraRole.CAMERA_ROLE_PLACE_CAM,
    "hand_cam": camera.CameraRole.CAMERA_ROLE_HAND_CAM,
}


@dataclass(frozen=True)
class CameraSpec:
    """Pose is relative to `parent_path` when set, else world frame. Color only for now."""

    serial: str  # no '/', it is embedded in the Zenoh key
    role: int  # camera.CameraRole value
    prim_path: str
    parent_path: str | None
    translate: tuple[float, float, float]
    rotation_euler_xyz_deg: tuple[float, float, float]
    width: int
    height: int
    fps: int
    focal_length: float = 18.0

    def __post_init__(self) -> None:
        if "/" in self.serial:
            raise ValueError(f"camera serial must not contain '/': {self.serial!r}")


def camera_info(spec: CameraSpec, topics: Topics | None = None) -> camera.CameraInfo:
    topics = topics or Topics.from_env()
    return camera.CameraInfo(
        serial=spec.serial,
        width=spec.width,
        height=spec.height,
        fps=spec.fps,
        format=COLOR_FORMAT,
        role=spec.role,
        color_topic=topics.camera_color(spec.serial),
        depth_topic=topics.camera_depth(spec.serial),
    )


def build_camera_list(specs: list[CameraSpec], topics: Topics | None = None) -> camera.CameraList:
    topics = topics or Topics.from_env()
    return camera.CameraList(cameras=[camera_info(spec, topics) for spec in specs])
