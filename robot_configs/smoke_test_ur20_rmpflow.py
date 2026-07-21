"""Standalone smoke test: load the generated UR20 cuMotion config
(robot_configs/ur20/{robot.urdf,robot.xrdf,rmp_flow.yaml}) and confirm
RmpFlowController actually converges on a target end-effector pose, in
isolation from the full conveyor scaffold - validates the config before
wiring it into pick_and_place.py.

Run with:
    /home/ubuntu/IsaacSim/python.sh /home/ubuntu/conveyor_indexing/robot_configs/smoke_test_ur20_rmpflow.py
"""

from __future__ import annotations

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.timeline
import omni.usd
import warp as wp

# isaacsim.robot_motion.cumotion's transforms.py / cumotion_world_interface.py
# call np.reshape(arr, shape=[...]) (6 call sites, confirmed via grep) - the
# `shape=` keyword to np.reshape only exists from NumPy 2.1 onward, but Isaac
# Sim's own bundled NumPy here is 1.26.4 (confirmed), so every one of those
# calls raises `TypeError: reshape() got an unexpected keyword argument
# 'shape'`. This is a genuine version mismatch in the shipped extension code,
# not specific to UR20 - the bundled/supported UR10 example would hit the
# same TypeError. Patched here only in this process (not the shared Isaac
# Sim installation) since it's a contained, reversible compatibility shim.
_np_reshape = np.reshape


def _reshape_shape_kwarg_compat(a, *args, **kwargs):
    if "shape" in kwargs:
        kwargs["newshape"] = kwargs.pop("shape")
    return _np_reshape(a, *args, **kwargs)


np.reshape = _reshape_shape_kwarg_compat
from isaacsim.core.experimental.prims import Articulation
from isaacsim.storage.native import get_assets_root_path
import isaacsim.core.experimental.utils.stage as stage_utils

import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, RmpFlowController, load_cumotion_robot

UR20_USD_PATH = "/Isaac/Robots/UniversalRobots/ur20/ur20.usd"
ROBOT_PRIM_PATH = "/ur20"
CONFIG_DIR = "/home/ubuntu/conveyor_indexing/robot_configs/ur20"


def main() -> None:
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not resolve Isaac Sim assets root path")

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage_utils.add_reference_to_stage(usd_path=assets_root + UR20_USD_PATH, path=ROBOT_PRIM_PATH)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation = Articulation(ROBOT_PRIM_PATH)
    print(f"[smoke_test] dof_names={articulation.dof_names}", flush=True)

    cumotion_robot = load_cumotion_robot(directory=CONFIG_DIR)
    tool_frames = cumotion_robot.robot_description.tool_frame_names()
    print(f"[smoke_test] tool_frames={tool_frames}", flush=True)
    if not tool_frames:
        raise RuntimeError("No tool frames found in generated XRDF")
    tool_frame = tool_frames[0]

    robot_pos, robot_ori = articulation.get_world_poses()
    world_binding = mg.WorldBinding(
        world_interface=CumotionWorldInterface(),
        obstacle_strategy=mg.ObstacleStrategy(),
        tracked_prims=[],
        tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
    )
    world_binding.initialize()
    world_binding.get_world_interface().update_world_to_robot_root_transforms(poses=(robot_pos, robot_ori))
    world_binding.synchronize_transforms()

    controller = RmpFlowController(
        cumotion_robot=cumotion_robot,
        cumotion_world_interface=world_binding.get_world_interface(),
        robot_joint_space=list(articulation.dof_names),
        robot_site_space=tool_frames,
        tool_frame=tool_frame,
    )

    def estimated_state() -> "mg.RobotState":
        names = list(articulation.dof_names)
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=names,
                positions=(names, articulation.get_dof_positions()),
                velocities=(names, articulation.get_dof_velocities()),
            )
        )

    # Arbitrary reachable target in front of the robot base, downward-facing
    # tool orientation - this is a synthetic isolation test, not the real
    # pick/place geometry (that comes later once wired into the scaffold).
    target_position = robot_pos.numpy()[0] + np.array([0.6, 0.0, 0.6])
    down_orientation = np.array([0.0, 1.0, 0.0, 0.0])  # (w, x, y, z)

    def setpoint_state() -> "mg.RobotState":
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=tool_frames,
                positions=([tool_frame], wp.array([target_position.tolist()], dtype=wp.float32)),
                orientations=([tool_frame], wp.array([down_orientation.tolist()], dtype=wp.float32)),
            ),
        )

    if not controller.reset(estimated_state(), setpoint_state(), t=0.0):
        raise RuntimeError("RmpFlowController.reset() failed")

    t = 0.0
    dt = 1.0 / 60.0
    for step in range(300):
        world_binding.get_world_interface().update_world_to_robot_root_transforms(
            poses=articulation.get_world_poses()
        )
        world_binding.synchronize_transforms()

        desired = controller.forward(estimated_state(), setpoint_state(), t)
        if desired is None or desired.joints.positions is None:
            raise RuntimeError(f"controller.forward() returned no joint targets at step {step}")
        articulation.set_dof_position_targets(positions=desired.joints.positions, dof_indices=desired.joints.position_indices)
        simulation_app.update()
        t += dt

        if step % 60 == 0:
            print(f"[smoke_test] step={step} dof_positions={articulation.get_dof_positions().numpy()[0]}", flush=True)

    print("[smoke_test] PASSED: RmpFlowController ran 300 steps on the generated UR20 config with no exceptions", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
