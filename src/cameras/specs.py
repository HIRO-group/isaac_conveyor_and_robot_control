"""Value bundle describing one sim camera, plus the pure (no-USD, no-Zenoh)
helpers that turn a list of these into wire-contract protobuf messages.
Kept USD-free and Zenoh-free so it's unit-testable without Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass

from cameras.protos import camera

# Downstream consumers reshape color frames as (height, width, 3) and only
# flip channels when "BGR" is in the format string - RGB8 is the safest,
# most-widely-accepted format string (see the top-level README's "Design"
# section).
COLOR_FORMAT = "RGB8"


def color_topic(serial: str) -> str:
    """This camera's color-stream key - a wire-contract constant
    (`sim/camera/{serial}/color`), not something this repo can look up;
    changing it would silently stop consumers from being able to discover
    the stream.
    """
    return f"sim/camera/{serial}/color"


def depth_topic(serial: str) -> str:
    """Advertised for contract completeness even though no depth frames are
    published yet (see CameraSpec's docstring) - kept in sync with color_topic.
    """
    return f"sim/camera/{serial}/depth"


@dataclass(frozen=True)
class CameraSpec:
    """One sim camera. ``translate``/``rotation_euler_xyz_deg`` are in the
    frame of ``parent_path`` when set (e.g. hand cams, parented under the
    UR20 flange so they ride the arm's kinematics) or world frame when
    ``parent_path`` is None (e.g. overhead cams, parented under the identity
    `/World/Cameras` Xform - see `sim_cell.layout.CAMERA_ROOT_PATH`).

    Depth fields are deliberately not modeled here - color only for now (see
    the top-level README's "Design" section); CameraInfo's depth_* fields are
    left zeroed by `camera_info()` below, ready to fill in later without a
    wire-format change.
    """

    serial: str  # No '/' - embedded verbatim in the Zenoh key (color_topic/depth_topic).
    role: int  # A cameras.protos.camera.CameraRole value (PICK_CAM/PLACE_CAM/HAND_CAM).
    prim_path: str
    parent_path: str | None
    translate: tuple[float, float, float]
    rotation_euler_xyz_deg: tuple[float, float, float]
    width: int
    height: int
    fps: int
    focal_length: float = 18.0  # ~60 deg HFOV at a 20.955mm horizontal aperture.

    def __post_init__(self) -> None:
        if "/" in self.serial:
            raise ValueError(f"camera serial must not contain '/' (embedded in Zenoh keys): {self.serial!r}")


def camera_info(spec: CameraSpec) -> camera.CameraInfo:
    """Build the wire-contract CameraInfo for one spec. Depth fields stay at
    their proto zero-value (width=0, height=0, fps=0, format="") - color only.
    """
    return camera.CameraInfo(
        serial=spec.serial,
        width=spec.width,
        height=spec.height,
        fps=spec.fps,
        format=COLOR_FORMAT,
        role=spec.role,
        color_topic=color_topic(spec.serial),
        depth_topic=depth_topic(spec.serial),
    )


def build_camera_list(specs: list[CameraSpec]) -> camera.CameraList:
    return camera.CameraList(cameras=[camera_info(spec) for spec in specs])
