"""Out-of-process cuMotion planner (implementation) - direct low-level API.

Runs in the planner subprocess launched by conveyor_indexer.py via
planner_server.py. Uses cuMotion's low-level library directly - NO
SimulationApp, NO Kit/USD, NO rendering - so it starts in ~1-2s and stays
light. (An earlier design booted a second headless SimulationApp just to scan
the USD stage for obstacles; that was heavy and hung on load. cuMotion's core
lib needs none of it - validated by planner_direct_spike.py.)

A single cuMotion solve is a monolithic C++/CUDA call that holds the GIL for
its whole duration, so it runs here in a separate process (own GIL + CUDA
context) rather than blocking the main sim loop.

Import only AFTER the launcher has added the warp + cumotion extension dirs to
sys.path (Kit's extension manager normally does that; it's absent here). The
launcher guarantees this before importing this module.

Protocol (pickled dicts over a multiprocessing.connection Connection):
  parent -> child:
    {"type":"init", "physics_dt", "robots":[{ "robot_id", "xrdf_path",
        "urdf_path", "base_position"[3], "base_orientation"[4] wxyz,
        "tool_frame", "max_velocities"[n], "max_accelerations"[n],
        "obstacles":[{"side_lengths"[3], "position"[3] world,
                      "orientation"[4] wxyz world}, ...] }]}
    {"type":"plan", "request_id", "robot_id", "mode": cspace|pose|ik_cspace,
        "q_initial"[n], "q_target"[n]|None, "position"[3]|None,
        "orientation"[4] wxyz|None}
    {"type":"stop"}
  child -> parent:
    {"type":"ready"}
    {"type":"plan_result", "request_id", "robot_id", "ok",
        "positions" (T,n)|None, "joint_names"[n]|None, "error"|None}

`positions` are in `joint_names` order; the parent maps those names to its own
articulation dof indices (so ordering never has to match implicitly).
"""

from __future__ import annotations

import math
import traceback
import xml.etree.ElementTree as ET

import numpy as np

# cuMotion's shipped code calls np.reshape(arr, shape=[...]); the `shape=`
# kwarg only exists from NumPy 2.1 and this env ships 1.26. Shim it before any
# cumotion call (capture the real reshape first to avoid infinite recursion).
_np_reshape = np.reshape


def _reshape_shim(a, *args, **kwargs):
    if "shape" in kwargs:
        kwargs["newshape"] = kwargs.pop("shape")
    return _np_reshape(a, *args, **kwargs)


np.reshape = _reshape_shim

import warp as wp  # noqa: E402  (import after sys.path set up by launcher)

wp.init()

import cumotion  # noqa: E402

IK_CSPACE_LIMIT_BIASING_WEIGHT = 1.0  # relative weight; see IkConfig docs

# ----------------------------------------------------------------------------
# Vendored URDF normalization (Apache-2.0, from
# isaacsim.robot_motion.cumotion.impl.urdf_normalize): the urdfdom parser
# statically linked into libcumotion.so rejects <limit> without `effort`
# (our robot.urdf omits it), <safety_controller> without k_velocity, empty
# <dynamics>, and <mimic> without a joint. Importing the real module would drag
# in the Kit-heavy package __init__, which this process specifically avoids.
# ----------------------------------------------------------------------------
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


def _pose3(position, orientation_wxyz) -> "cumotion.Pose3":
    """Build a cumotion.Pose3 from a translation and a (w,x,y,z) quaternion."""
    w, x, y, z = (float(v) for v in orientation_wxyz)
    return cumotion.Pose3(cumotion.Rotation3(w, x, y, z), np.asarray(position, dtype=np.float64))


class RobotPlanner:
    """One robot's cuMotion planner + static obstacle world + trajectory generator.

    The robot base pose and obstacles are static in this process (no physics
    stepping), so the world is built once here and reused for every plan.
    """

    def __init__(self, cfg: dict, physics_dt: float) -> None:
        self._physics_dt = physics_dt
        self._tool_frame = cfg["tool_frame"]

        with open(cfg["xrdf_path"], encoding="utf-8") as f:
            xrdf_text = f.read()
        with open(cfg["urdf_path"], encoding="utf-8") as f:
            urdf_text = normalize_urdf(f.read())
        self._robot_description = cumotion.load_robot_from_memory(xrdf_text, urdf_text)
        self._kinematics = self._robot_description.kinematics()
        self._joint_names = [
            self._robot_description.cspace_coord_name(i)
            for i in range(self._robot_description.num_cspace_coords())
        ]

        # world->base, so world-frame targets/obstacles can be expressed in the
        # robot base frame cuMotion plans in.
        base_world = _pose3(cfg["base_position"], cfg["base_orientation"])
        self._base_from_world = base_world.inverse()

        world = cumotion.create_world()
        for obs in cfg.get("obstacles", []):
            obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
            obstacle.set_attribute(
                cumotion.Obstacle.Attribute.SIDE_LENGTHS, np.asarray(obs["side_lengths"], dtype=np.float64)
            )
            pose_base = self._base_from_world * _pose3(obs["position"], obs["orientation"])
            world.add_obstacle(obstacle, pose_base)
        self._world = world
        self._world_view = world.add_world_view()

        self._planner_config = cumotion.create_default_motion_planner_config(
            robot_description=self._robot_description,
            tool_frame_name=self._tool_frame,
            world_view=self._world_view,
        )

        self._traj_generator = cumotion.create_cspace_trajectory_generator(self._kinematics)
        self._traj_generator.set_velocity_limits(np.asarray(cfg["max_velocities"], dtype=np.float64))
        self._traj_generator.set_acceleration_limits(np.asarray(cfg["max_accelerations"], dtype=np.float64))

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    def _solve_ik(self, position, orientation, q_initial: np.ndarray) -> np.ndarray:
        target_pose_base = self._base_from_world * _pose3(position, orientation)
        ik_config = cumotion.IkConfig()
        ik_config.bfgs_cspace_limit_biasing = cumotion.IkConfig.CSpaceLimitBiasing.ENABLE
        ik_config.bfgs_cspace_limit_biasing_weight = IK_CSPACE_LIMIT_BIASING_WEIGHT
        ik_config.cspace_seeds = [q_initial]
        result = cumotion.solve_ik(
            kinematics=self._kinematics,
            target_pose=target_pose_base,
            target_frame=self._tool_frame,
            config=ik_config,
        )
        if not result.success:
            raise RuntimeError(f"solve_ik found no configuration for position={position} orientation={orientation}")
        return np.asarray(result.cspace_position, dtype=np.float64)

    def plan(self, mode, q_initial, q_target, position, orientation) -> np.ndarray | None:
        """Plan and time-parameterize; return sampled positions (T, n) or None on failure."""
        q_initial = np.asarray(q_initial, dtype=np.float64)
        # Recreate the planner per call (matches the shipped GraphBasedMotionPlanner
        # pattern) and refresh the world view before planning.
        planner = cumotion.create_motion_planner(config=self._planner_config)
        self._world_view.update()

        if mode == "cspace":
            result = planner.plan_to_cspace_target(q_initial, np.asarray(q_target, dtype=np.float64))
        elif mode == "ik_cspace":
            q_final = self._solve_ik(position, orientation, q_initial)
            result = planner.plan_to_cspace_target(q_initial, q_final)
        elif mode == "pose":
            result = planner.plan_to_pose_target(q_initial, self._base_from_world * _pose3(position, orientation))
        else:
            raise ValueError(f"unknown plan mode {mode!r}")

        if not result.path_found:
            return None
        return self._sample(result.path)

    def _sample(self, waypoints) -> np.ndarray | None:
        """Time-parameterize `waypoints` and sample at physics_dt into a (T, n) array."""
        trajectory = self._traj_generator.generate_trajectory(waypoints)
        domain = trajectory.domain()
        duration = domain.upper - domain.lower
        n_samples = max(int(math.ceil(duration / self._physics_dt)) + 1, 1)
        positions = []
        for i in range(n_samples):
            t = min(domain.lower + i * self._physics_dt, domain.upper)
            positions.append(np.asarray(trajectory.eval(t), dtype=np.float64).flatten())
        return np.stack(positions) if positions else None


def serve(address: str, authkey: bytes) -> None:
    """Connect back to the parent, build one RobotPlanner per robot, then serve
    plan requests until told to stop."""
    from multiprocessing.connection import Client

    conn = Client(address, authkey=authkey)
    try:
        init = conn.recv()
        if init.get("type") != "init":
            raise RuntimeError(f"expected init message, got {init.get('type')!r}")
        physics_dt = init["physics_dt"]
        planners: dict[int, RobotPlanner] = {}
        for cfg in init["robots"]:
            planners[cfg["robot_id"]] = RobotPlanner(cfg, physics_dt)
            print(
                f"[planner_server] built planner for robot_id={cfg['robot_id']} "
                f"({len(cfg.get('obstacles', []))} obstacles)",
                flush=True,
            )
        conn.send({"type": "ready"})
        print("[planner_server] ready, serving plan requests", flush=True)

        while True:
            msg = conn.recv()  # blocking; process idles here between requests
            msg_type = msg.get("type")
            if msg_type == "stop":
                break
            if msg_type != "plan":
                continue
            request_id = msg["request_id"]
            robot_id = msg["robot_id"]
            try:
                positions = planners[robot_id].plan(
                    msg["mode"], msg["q_initial"], msg.get("q_target"), msg.get("position"), msg.get("orientation")
                )
                ok = positions is not None
                conn.send(
                    {
                        "type": "plan_result",
                        "request_id": request_id,
                        "robot_id": robot_id,
                        "ok": ok,
                        "positions": positions,
                        "joint_names": planners[robot_id].joint_names if ok else None,
                        "error": None if ok else "no collision-free path found",
                    }
                )
            except Exception as exc:
                print(
                    f"[planner_server] plan for robot_id={robot_id} raised: {exc!r}\n{traceback.format_exc()}",
                    flush=True,
                )
                conn.send(
                    {
                        "type": "plan_result",
                        "request_id": request_id,
                        "robot_id": robot_id,
                        "ok": False,
                        "positions": None,
                        "joint_names": None,
                        "error": f"{exc!r}",
                    }
                )
    except (EOFError, OSError):
        # Parent connection closed (EOFError on recv) or broke mid-loop
        # (BrokenPipeError/ConnectionReset on send, both OSError) - the parent
        # is gone, so shut down. PR_SET_PDEATHSIG is the backstop if we were
        # blocked and never noticed.
        print("[planner_server] parent connection gone; shutting down", flush=True)
    finally:
        conn.close()
