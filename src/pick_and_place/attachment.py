"""Magic-attach: rigidly join a box to the wrist via a PhysX FixedJoint, so it
moves as a real extension of the arm rather than being teleported to a
computed offset.
"""

from __future__ import annotations

import logging

import numpy as np
from pxr import Gf, Sdf, UsdPhysics

import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.xform as xform_utils

from pick_and_place.transforms import rotation_matrix_to_quaternion_wxyz

logger = logging.getLogger(__name__)


def attach_box(box_rigid_prim, wrist_link_path: str, attach_joint_path: str) -> None:
    """Create the FixedJoint attaching `box_rigid_prim` to `wrist_link_path`.

    Re-enables the box's rigid body (disabled since WAITING) in the same call
    the joint is created, so there's no gap where it could fall under gravity first.
    """
    box_path = box_rigid_prim.paths[0]
    relative_transform = xform_utils.get_relative_transform(box_path, wrist_link_path)
    local_pos0 = relative_transform[:3, 3]
    # CubeBox_* prims carry a non-unity xformOp:scale, so this transform's rotation
    # block isn't unit-length; normalize each column before extracting the quaternion,
    # or the box snaps to the wrong orientation when the FixedJoint is created.
    rotation_block = relative_transform[:3, :3]
    rotation_block = rotation_block / np.linalg.norm(rotation_block, axis=0, keepdims=True)
    local_rot0 = rotation_matrix_to_quaternion_wxyz(rotation_block)
    logger.debug(
        "attaching: box_path=%s local_pos0(rel. wrist_3_link)=%s local_rot0(wxyz)=%s",
        box_path, local_pos0, local_rot0,
    )

    box_rigid_prim.set_enabled_rigid_bodies([True])

    stage = stage_utils.get_current_stage()
    joint = UsdPhysics.FixedJoint.Define(stage, attach_joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(wrist_link_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(box_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0.tolist()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(local_rot0[0]), Gf.Vec3f(*local_rot0[1:].tolist())))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))


def detach_box(attach_joint_path: str) -> None:
    """Remove the FixedJoint created by attach_box, releasing the box to fall/settle
    naturally under gravity from wherever it currently is.
    """
    stage_utils.delete_prim(attach_joint_path)
