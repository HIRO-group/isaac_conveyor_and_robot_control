"""Camera pose persistence: JSON overrides loaded at camera-spec build time,
and a USD-reading save function used by the tuning workflow (see
`sim_cell.camera_tuning`). Poses are stored **local to each camera's parent
prim** - world frame for overhead cams (parented under the identity
`/World/Cameras` Xform, see `sim_cell.layout.CAMERA_ROOT_PATH`), flange
frame for hand cams (parented under the UR20's wrist flange) - so a saved
hand-cam pose stays correct as the arm moves; only the *offset* is tuned.

Reading/writing goes through `UsdGeom.XformCommonAPI` rather than manual
xform-op parsing, so a pose written by dragging the USD gizmo (translate +
rotate, whatever op order the manipulator authors) round-trips correctly as
long as `cameras.rig.CameraRig` authors camera transforms through the same
API - see that module.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoseOverride:
    translate: tuple[float, float, float]
    rotation_euler_xyz_deg: tuple[float, float, float]


def load_pose_overrides(path: str) -> dict[str, PoseOverride]:
    """Read a previously-saved pose file, keyed by camera serial. Returns an
    empty dict (not an error) if the file doesn't exist yet - a fresh
    checkout has no tuned poses, so `camera_layout.build_camera_specs` falls
    back to its derived defaults.
    """
    pose_path = pathlib.Path(path)
    if not pose_path.exists():
        return {}
    raw = json.loads(pose_path.read_text())
    overrides = {}
    for serial, entry in raw.items():
        overrides[serial] = PoseOverride(
            translate=tuple(entry["translate"]),
            rotation_euler_xyz_deg=tuple(entry["rotation_euler_xyz_deg"]),
        )
    logger.info("loaded %d camera pose override(s) from %s", len(overrides), path)
    return overrides


def save_camera_poses(path: str, stage, specs) -> None:
    """Read each camera prim's current *local* transform (post-tuning, i.e.
    after the gizmo has been dragged) and write it to `path` as JSON, sorted
    by serial so the diff is stable in git.
    """
    from pxr import Usd, UsdGeom

    poses = {}
    for spec in specs:
        prim = stage.GetPrimAtPath(spec.prim_path)
        if not prim.IsValid():
            logger.warning("skipping save for %s - prim %s not found", spec.serial, spec.prim_path)
            continue
        xform_api = UsdGeom.XformCommonAPI(prim)
        translation, rotation, _scale, _pivot, _rot_order = xform_api.GetXformVectors(Usd.TimeCode.Default())
        poses[spec.serial] = PoseOverride(
            translate=(translation[0], translation[1], translation[2]),
            rotation_euler_xyz_deg=(rotation[0], rotation[1], rotation[2]),
        )

    pose_path = pathlib.Path(path)
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {serial: asdict(pose) for serial, pose in sorted(poses.items())}
    pose_path.write_text(json.dumps(serializable, indent=2) + "\n")
    logger.info("saved %d camera pose(s) to %s", len(poses), path)
