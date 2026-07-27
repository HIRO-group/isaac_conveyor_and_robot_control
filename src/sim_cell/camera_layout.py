"""Derives this cell's 6 camera placements (pick_cam/place_cam/hand_cam x 2
stations) from actual zone/robot geometry, the same way `robot_placement.py`
derives station 2's robot position - not hardcoded, since `ConveyorTrack_02`/
`_10` aren't guaranteed to line up with station 1's zones.

Tuned poses saved by the camera-tuning workflow (`sim_cell.camera_tuning`)
always take priority over these derived defaults - see `build_camera_specs`.
"""

from __future__ import annotations

from cameras.pose_io import load_pose_overrides
from cameras.protos import camera
from cameras.specs import CameraSpec
from conveyor_indexing.belt_geometry import compute_belt_bounds
from sim_cell import layout, settings


def _pick_cam_spec(serial: str, zone, prim_path: str) -> CameraSpec:
    bounds = compute_belt_bounds(zone.belt_prim)
    return CameraSpec(
        serial=serial,
        role=camera.CameraRole.CAMERA_ROLE_PICK_CAM,
        prim_path=prim_path,
        parent_path=None,
        translate=(
            bounds.bbox_center[0],
            bounds.bbox_center[1],
            bounds.belt_top_z + settings.CAMERA_HEIGHT_ABOVE_BELT_M,
        ),
        rotation_euler_xyz_deg=(0.0, 0.0, 0.0),  # Z-up stage: unrotated camera already looks straight down.
        width=settings.CAMERA_WIDTH,
        height=settings.CAMERA_HEIGHT,
        fps=settings.CAMERA_FPS,
    )


def _place_cam_spec(serial: str, zone, prim_path: str) -> CameraSpec:
    bounds = compute_belt_bounds(zone.belt_prim)
    return CameraSpec(
        serial=serial,
        role=camera.CameraRole.CAMERA_ROLE_PLACE_CAM,
        prim_path=prim_path,
        parent_path=None,
        translate=(
            bounds.bbox_center[0],
            bounds.bbox_center[1],
            bounds.belt_top_z + settings.CAMERA_HEIGHT_ABOVE_BELT_M,
        ),
        rotation_euler_xyz_deg=(0.0, 0.0, 0.0),
        width=settings.CAMERA_WIDTH,
        height=settings.CAMERA_HEIGHT,
        fps=settings.CAMERA_FPS,
    )


def _hand_cam_spec(serial: str, parent_path: str) -> CameraSpec:
    return CameraSpec(
        serial=serial,
        role=camera.CameraRole.CAMERA_ROLE_HAND_CAM,
        prim_path=parent_path + "/hand_cam",
        parent_path=parent_path,
        translate=settings.HAND_CAM_OFFSET,
        # Local -Z (the camera's look direction) aligned with the flange's
        # outward +Z, so the camera looks where the tool points. Verify via
        # the camera-tuning workflow's live viewport and adjust/save if the
        # image shows sky instead of the tool - see sim_cell.camera_tuning.
        rotation_euler_xyz_deg=(180.0, 0.0, 0.0),
        width=settings.CAMERA_WIDTH,
        height=settings.CAMERA_HEIGHT,
        fps=settings.CAMERA_FPS,
    )


def build_camera_specs(loop1, loop2) -> list[CameraSpec]:
    """Six specs: {pick,place,hand}_cam x {station 1, station 2}. Both pick
    zones live on loop1, both place zones on loop2 - see sim_cell.cell.
    """
    specs = [
        _pick_cam_spec("SIM1-PICK", loop1.zones[layout.PICK_ZONE_INDEX], layout.CAMERA_ROOT_PATH + "/SIM1_PICK"),
        _place_cam_spec("SIM1-PLACE", loop2.zones[layout.PLACE_ZONE_INDEX], layout.CAMERA_ROOT_PATH + "/SIM1_PLACE"),
        _hand_cam_spec("SIM1-HAND", layout.HAND_CAM_PARENT),
        _pick_cam_spec(
            "SIM2-PICK", loop1.zones[layout.PICK_ZONE_INDEX_2], layout.CAMERA_ROOT_PATH + "/SIM2_PICK"
        ),
        _place_cam_spec(
            "SIM2-PLACE", loop2.zones[layout.PLACE_ZONE_INDEX_2], layout.CAMERA_ROOT_PATH + "/SIM2_PLACE"
        ),
        _hand_cam_spec("SIM2-HAND", layout.HAND_CAM_PARENT_2),
    ]
    return _apply_pose_overrides(specs)


def _apply_pose_overrides(specs: list[CameraSpec]) -> list[CameraSpec]:
    """Saved/tuned poses (see cameras.camera_tuning) always win over the
    derived defaults above - a fresh checkout with no pose file yet just
    uses the defaults untouched.
    """
    overrides = load_pose_overrides(layout.CAMERA_POSES_PATH)
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
