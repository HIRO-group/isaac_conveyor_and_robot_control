"""Static pedestal + UR20 spawning."""

from __future__ import annotations

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cylinder
from isaacsim.core.experimental.prims import Articulation
from isaacsim.storage.native import get_assets_root_path

from pick_and_place.ur20 import UR20_DEFAULT_JOINT_POSITIONS


def create_pedestal_and_robot(
    stage: Usd.Stage,
    robot_path: str,
    pedestal_path: str,
    position: tuple,
    pedestal_height: float,
    pedestal_radius: float = 0.15,
) -> Articulation:
    """Create a simple static cylindrical pedestal and a UR20 on top of it.

    Args:
        position: (x, y, z) of the pedestal's base (ground contact point).
        pedestal_height: Pedestal column height; the robot is placed at
            z = position[2] + pedestal_height.
    """
    px, py, pz = position
    pedestal = Cylinder(
        paths=pedestal_path,
        positions=[px, py, pz + pedestal_height / 2.0],
        radii=pedestal_radius,
        heights=pedestal_height,
        colors="gray",
    )
    # Static collider only - no RigidBodyAPI, so it doesn't fall under gravity
    # and isn't mistaken for a kinematic/dynamic body by anything else.
    UsdPhysics.CollisionAPI.Apply(pedestal.prims[0])
    usd_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur20/ur20.usd"
    robot_prim = stage_utils.add_reference_to_stage(usd_path=usd_path, path=robot_path, variants=[])
    xformable = UsdGeom.Xformable(robot_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(px, py, pz + pedestal_height))

    robot = Articulation(robot_path)
    robot.set_default_state(dof_positions=UR20_DEFAULT_JOINT_POSITIONS)
    return robot
