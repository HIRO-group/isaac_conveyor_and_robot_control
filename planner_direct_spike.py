"""Spike: plan with the low-level cumotion API and NO SimulationApp.

The current planner subprocess boots a full headless SimulationApp (Kit + USD +
render stack) just to give cuMotion an obstacle world - which is what made the
load hang / feel too heavy. But cuMotion's own low-level library
(create_world / add_obstacle / create_motion_planner / plan_to_cspace_target)
needs none of that; only the *Isaac wrapper* that scans the USD stage does.

This proves the make-or-break: that the low-level pipeline loads the UR20 from
robot_configs/ur20 and plans a path with just `warp` + `cumotion` initialized -
no SimulationApp. If it passes, the planner subprocess can drop the whole Kit
stack (fast startup, small footprint) and hand-build its obstacle world.

Run with Isaac Sim's bundled python from this directory - it should finish in
seconds, unlike the SimulationApp path:
    ./python.sh planner_direct_spike.py

Success = "SUCCESS: low-level cumotion planned with NO SimulationApp".
It also probes (non-fatally) whether the Kit trajectory-time-parameterization
class imports without an app, which tells us whether we must vendor path.py.
"""

from __future__ import annotations

import glob
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
UR20_DIR = os.path.join(REPO, "robot_configs", "ur20")

# warp + cumotion ship as Isaac Sim extensions; their sys.path entries are
# normally added by Kit's extension manager when SimulationApp boots. Since we
# deliberately don't boot Kit, add them by hand. (In the real subprocess the
# parent - which HAS Kit - will discover these from warp.__file__/cumotion.__file__
# and pass them down, so nothing is hardcoded there. Hardcoded here only because
# the spike is standalone.)
_EXT_PATHS = [
    "/home/ubuntu/IsaacSim-source/_build/target-deps/isaac_cumotion_prebundle",
    *sorted(glob.glob("/home/ubuntu/IsaacSim/extscache/omni.warp.core-*")),
]
for _p in _EXT_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# --- Vendored URDF normalization (Apache-2.0, condensed from
# isaacsim.robot_motion.cumotion.impl.urdf_normalize): our robot.urdf has
# <limit> without `effort`, which the urdfdom parser statically linked into
# libcumotion.so rejects. Importing the real module would pull in the Kit-heavy
# package __init__, which is exactly what we're trying to avoid here. ---
_TYPES_REQUIRING_LIMIT = frozenset({"revolute", "prismatic"})


def _normalize_joint(joint: ET.Element) -> bool:
    modified = False
    limit = joint.find("limit")
    if joint.get("type") in _TYPES_REQUIRING_LIMIT and limit is None:
        ET.SubElement(joint, "limit", attrib={"lower": "0", "upper": "0", "effort": "1000", "velocity": "1000"})
        modified = True
    elif limit is not None:
        if "effort" not in limit.attrib:
            limit.set("effort", "1000")
            modified = True
        if "velocity" not in limit.attrib:
            limit.set("velocity", "1000")
            modified = True
    safety = joint.find("safety_controller")
    if safety is not None and "k_velocity" not in safety.attrib:
        safety.set("k_velocity", "0")
        modified = True
    dynamics = joint.find("dynamics")
    if dynamics is not None and "damping" not in dynamics.attrib and "friction" not in dynamics.attrib:
        joint.remove(dynamics)
        modified = True
    mimic = joint.find("mimic")
    if mimic is not None and "joint" not in mimic.attrib:
        joint.remove(mimic)
        modified = True
    return modified


def normalize_urdf(urdf_text: str) -> str:
    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError:
        return urdf_text
    modified = False
    for joint in root.iter("joint"):
        modified |= _normalize_joint(joint)
    return ET.tostring(root, encoding="unicode") if modified else urdf_text


def _stage(name: str) -> None:
    print(f"[spike] {name}...", flush=True)


def main() -> None:
    t_start = time.monotonic()

    _stage("import warp + init (NO SimulationApp)")
    import warp as wp

    wp.init()

    # cuMotion's shipped code calls np.reshape(arr, shape=[...]); the `shape=`
    # kwarg only exists from NumPy 2.1 and this env ships 1.26. Shim it before
    # any cumotion call (capture the real reshape first to avoid recursion).
    _np_reshape = np.reshape

    def _reshape_shim(a, *args, **kwargs):
        if "shape" in kwargs:
            kwargs["newshape"] = kwargs.pop("shape")
        return _np_reshape(a, *args, **kwargs)

    np.reshape = _reshape_shim

    _stage("import cumotion")
    import cumotion

    _stage("load UR20 from XRDF/URDF via cumotion.load_robot_from_memory")
    xrdf_text = open(os.path.join(UR20_DIR, "robot.xrdf")).read()
    urdf_text = normalize_urdf(open(os.path.join(UR20_DIR, "robot.urdf")).read())
    robot_description = cumotion.load_robot_from_memory(xrdf_text, urdf_text)
    ndof = robot_description.num_cspace_coords()
    joint_names = [robot_description.cspace_coord_name(i) for i in range(ndof)]
    print(f"[spike]   ndof={ndof} joints={joint_names} tool_frames={robot_description.tool_frame_names()}", flush=True)

    _stage("build world + one distant cuboid obstacle")
    world = cumotion.create_world()
    obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, np.array([0.5, 0.5, 0.5]))
    world.add_obstacle(obstacle, cumotion.Pose3.from_translation(np.array([5.0, 5.0, 5.0])))
    world_view = world.add_world_view()

    _stage("create default motion planner config + planner")
    config = cumotion.create_default_motion_planner_config(
        robot_description=robot_description, tool_frame_name="tool0", world_view=world_view
    )
    planner = cumotion.create_motion_planner(config=config)

    _stage("plan_to_cspace_target")
    world_view.update()
    q_initial = np.array([1.5708, -1.5708, -1.5708, -1.5708, 1.5708, 3.1415], dtype=np.float64)
    q_target = np.array([-1.5708, -1.5708, -1.5708, -1.5708, 1.5708, 3.1415], dtype=np.float64)
    t0 = time.monotonic()
    result = planner.plan_to_cspace_target(q_initial, q_target)
    solve_s = time.monotonic() - t0
    print(f"[spike]   path_found={result.path_found} waypoints={len(result.path)} solve={solve_s:.3f}s", flush=True)

    if not result.path_found:
        print("\n[spike] PLAN FAILED (no path); stopping", flush=True)
        return

    # Time-parameterize the RRT waypoints into a trajectory using cuMotion's
    # OWN generator (no Kit, no vendoring) - this is the native equivalent of
    # the Kit Path.to_minimal_time_joint_trajectory the old code used.
    _stage("generate time-optimal trajectory (cumotion.CSpaceTrajectoryGenerator)")
    generator = cumotion.create_cspace_trajectory_generator(robot_description.kinematics())
    generator.set_velocity_limits(np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
    generator.set_acceleration_limits(np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
    trajectory = generator.generate_trajectory(result.path)
    domain = trajectory.domain()
    duration = domain.upper - domain.lower
    n_samples = int(duration / (1.0 / 120.0)) + 1
    first = trajectory.eval(domain.lower)
    last = trajectory.eval(domain.upper)
    print(
        f"[spike]   duration={duration:.3f}s -> {n_samples} samples @120Hz; "
        f"q(0)={np.round(first, 3)} q(T)={np.round(last, 3)}",
        flush=True,
    )

    print(
        f"\n[spike] SUCCESS: full low-level pipeline (load + world + plan + trajectory) "
        f"with NO SimulationApp (total {time.monotonic() - t_start:.1f}s from process start)",
        flush=True,
    )


if __name__ == "__main__":
    main()
