"""Camera placements derived from the scene's stations and zone geometry.

Tuned poses from the scene's pose file always override the derived defaults.
"""

from __future__ import annotations

from cameras.pose_io import load_pose_overrides
from cameras.specs import ROLE_BY_NAME, CameraSpec
from conveyor_indexing.belt_geometry import compute_belt_bounds
from sim_cell.scene import CameraConfig, SceneConfig, StationConfig, get_scene


def _overhead_spec(cam: CameraConfig, scene: SceneConfig, zone) -> CameraSpec:
    bounds = compute_belt_bounds(zone.belt_prim)
    return CameraSpec(
        serial=cam.id,
        role=ROLE_BY_NAME[cam.role],
        prim_path=f"{scene.camera_root_path}/{cam.id.replace('-', '_')}",
        parent_path=None,
        translate=(bounds.bbox_center[0], bounds.bbox_center[1], bounds.belt_top_z + scene.camera.height_above_belt_m),
        rotation_euler_xyz_deg=(0.0, 0.0, 0.0),  # Z-up stage: an unrotated camera looks straight down.
        width=scene.camera.width,
        height=scene.camera.height,
        fps=scene.camera.fps,
        focal_length=scene.camera.focal_length_mm,
    )


def _hand_spec(cam: CameraConfig, scene: SceneConfig, station: StationConfig) -> CameraSpec:
    return CameraSpec(
        serial=cam.id,
        role=ROLE_BY_NAME[cam.role],
        prim_path=f"{station.hand_cam_parent}/hand_cam",
        parent_path=station.hand_cam_parent,
        translate=scene.camera.hand_cam_offset,
        rotation_euler_xyz_deg=(180.0, 0.0, 0.0),  # look along the flange's +Z
        width=scene.camera.width,
        height=scene.camera.height,
        fps=scene.camera.fps,
        focal_length=scene.camera.focal_length_mm,
    )


def build_camera_specs(loops, scene: SceneConfig | None = None) -> list[CameraSpec]:
    """One spec per scene camera. `loops` are the built line controllers, in scene order."""
    scene = scene or get_scene()
    specs = []
    for cam in scene.cameras:
        station = scene.stations[cam.station - 1]
        if cam.role == "pick_cam":
            ref = station.pick_zone
            specs.append(_overhead_spec(cam, scene, loops[ref.loop].zones[ref.zone]))
        elif cam.role == "place_cam":
            ref = station.place_zone
            specs.append(_overhead_spec(cam, scene, loops[ref.loop].zones[ref.zone]))
        elif cam.role == "hand_cam":
            specs.append(_hand_spec(cam, scene, station))
        else:
            raise ValueError(f"camera {cam.id}: unknown role {cam.role!r}")
    return _apply_pose_overrides(specs, str(scene.camera_poses_path))


def _apply_pose_overrides(specs: list[CameraSpec], poses_path: str) -> list[CameraSpec]:
    overrides = load_pose_overrides(poses_path)
    if not overrides:
        return specs
    tuned = []
    for spec in specs:
        override = overrides.get(spec.serial)
        if override is None:
            tuned.append(spec)
            continue
        tuned.append(
            CameraSpec(
                serial=spec.serial,
                role=spec.role,
                prim_path=spec.prim_path,
                parent_path=spec.parent_path,
                translate=override.translate,
                rotation_euler_xyz_deg=override.rotation_euler_xyz_deg,
                width=spec.width,
                height=spec.height,
                fps=spec.fps,
                focal_length=spec.focal_length,
            )
        )
    return tuned
