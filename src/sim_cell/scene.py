"""Scene package: the cell description loaded from a directory outside this repo.

A scene package is a directory holding `cell.yaml`, the USD stage and the tuned
camera poses. Its location comes from `--scene` / `SIM_SCENE_DIR`; nothing about
a specific cell lives in this repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SCENE_DIR_ENV = "SIM_SCENE_DIR"
CONFIG_FILENAME = "cell.yaml"


class SceneError(RuntimeError):
    """A scene package is missing or malformed."""


@dataclass(frozen=True)
class LoopConfig:
    """One conveyor line: zone node paths in belt order."""

    zones: tuple[str, ...]
    run_speed_pct: int


@dataclass(frozen=True)
class ZoneRef:
    """(loop index, zone index) into `SceneConfig.loops`."""

    loop: int
    zone: int


@dataclass(frozen=True)
class StationConfig:
    """One pick-and-place robot and the zones it serves."""

    robot_path: str
    pedestal_path: str
    hand_cam_parent: str
    pick_zone: ZoneRef
    place_zone: ZoneRef


@dataclass(frozen=True)
class CameraConfig:
    id: str
    role: str  # pick_cam | place_cam | hand_cam
    station: int  # 1-based index into `SceneConfig.stations`

    @property
    def feature_name(self) -> str:
        return f"{self.role}_{self.station}"


@dataclass(frozen=True)
class CameraDefaults:
    width: int
    height: int
    fps: int
    height_above_belt_m: float
    hand_cam_offset: tuple[float, float, float]
    focal_length_mm: float


@dataclass(frozen=True)
class RobotDefaults:
    position: tuple[float, float, float]
    pedestal_height: float
    place_xy: tuple[float, float]


@dataclass(frozen=True)
class GeometryThresholds:
    floor_z: float
    off_belt_z: float
    off_belt_truck_xy_margin_m: float


@dataclass(frozen=True)
class SceneConfig:
    root: Path
    stage_path: Path
    camera_poses_path: Path
    loops: tuple[LoopConfig, ...]
    stations: tuple[StationConfig, ...]
    cameras: tuple[CameraConfig, ...]
    camera: CameraDefaults
    robot: RobotDefaults
    thresholds: GeometryThresholds
    truck_path: str
    camera_root_path: str
    ground_plane_collision_path: str
    box_prim_name_prefix: str
    conveyor_track_roots: tuple[str, ...]

    @property
    def excluded_structure_roots(self) -> tuple[str, ...]:
        """Prims whose hits are structure, not transported items."""
        station_prims = tuple(p for s in self.stations for p in (s.robot_path, s.pedestal_path))
        return self.conveyor_track_roots + station_prims + (self.truck_path,)

    def zone_path(self, ref: ZoneRef) -> str:
        return self.loops[ref.loop].zones[ref.zone]

    def require_shape(self, loops: int, stations: int) -> None:
        """Fail early for callers that only support a fixed cell shape."""
        if len(self.loops) != loops or len(self.stations) != stations:
            raise SceneError(
                f"{self.root / CONFIG_FILENAME}: this runner supports {loops} loop(s) and "
                f"{stations} station(s); scene has {len(self.loops)} and {len(self.stations)}"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path) -> SceneConfig:
        try:
            return cls(
                root=root,
                stage_path=root / raw["stage"],
                camera_poses_path=root / raw["camera_poses"],
                loops=tuple(
                    LoopConfig(zones=tuple(loop["zones"]), run_speed_pct=int(loop["run_speed_pct"]))
                    for loop in raw["loops"]
                ),
                stations=tuple(
                    StationConfig(
                        robot_path=s["robot_path"],
                        pedestal_path=s["pedestal_path"],
                        hand_cam_parent=s["hand_cam_parent"],
                        pick_zone=ZoneRef(*s["pick_zone"]),
                        place_zone=ZoneRef(*s["place_zone"]),
                    )
                    for s in raw["stations"]
                ),
                cameras=tuple(
                    CameraConfig(id=c["id"], role=c["role"], station=int(c["station"])) for c in raw["cameras"]
                ),
                camera=CameraDefaults(
                    width=int(raw["camera"]["width"]),
                    height=int(raw["camera"]["height"]),
                    fps=int(raw["camera"]["fps"]),
                    height_above_belt_m=float(raw["camera"]["height_above_belt_m"]),
                    hand_cam_offset=tuple(raw["camera"]["hand_cam_offset"]),
                    focal_length_mm=float(raw["camera"]["focal_length_mm"]),
                ),
                robot=RobotDefaults(
                    position=tuple(raw["robot"]["position"]),
                    pedestal_height=float(raw["robot"]["pedestal_height"]),
                    place_xy=tuple(raw["robot"]["place_xy"]),
                ),
                thresholds=GeometryThresholds(
                    floor_z=float(raw["thresholds"]["floor_z"]),
                    off_belt_z=float(raw["thresholds"]["off_belt_z"]),
                    off_belt_truck_xy_margin_m=float(raw["thresholds"]["off_belt_truck_xy_margin_m"]),
                ),
                truck_path=raw["truck_path"],
                camera_root_path=raw["camera_root_path"],
                ground_plane_collision_path=raw["ground_plane_collision_path"],
                box_prim_name_prefix=raw["box_prim_name_prefix"],
                conveyor_track_roots=tuple(raw["conveyor_track_roots"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SceneError(f"{root / CONFIG_FILENAME}: bad or missing field: {exc}") from exc

    @classmethod
    def load(cls, scene_dir: str | os.PathLike) -> SceneConfig:
        root = Path(scene_dir).expanduser().resolve()
        config_path = root / CONFIG_FILENAME
        if not config_path.is_file():
            raise SceneError(f"scene package {root} has no {CONFIG_FILENAME}")
        with config_path.open() as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw, root)


def scene_dir_from_env() -> Path:
    value = os.environ.get(SCENE_DIR_ENV)
    if not value:
        raise SceneError(f"{SCENE_DIR_ENV} is not set; point it at a scene package directory (see README)")
    return Path(value)


@lru_cache(maxsize=1)
def get_scene() -> SceneConfig:
    """The process-wide scene, loaded once from `SIM_SCENE_DIR`."""
    return SceneConfig.load(scene_dir_from_env())
