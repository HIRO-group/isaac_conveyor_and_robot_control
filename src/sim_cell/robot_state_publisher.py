"""Publishes live per-arm joint/suction state and conveyor state over Zenoh,
so an external controller (e.g. a trained policy) can observe the sim the
same way it would observe a real robot - see the top-level README's "Design"
section. Session setup deliberately mirrors cameras.zenoh_publisher's
(duplicated by the same existing convention rather than shared).
"""

from __future__ import annotations

import logging
import os
from typing import ClassVar

from conveyor_indexing.protos import plc
from sim_cell.protos import robot_state, sim_state

logger = logging.getLogger(__name__)

try:
    import zenoh
except ImportError as exc:
    raise SystemExit(
        "eclipse-zenoh is required for robot-state publishing but is not installed "
        "in this interpreter. Install it into Isaac Sim's bundled python:\n"
        "  /home/ggbrisco/isaacsim/_build/linux-x86_64/release/python.sh -m pip install eclipse-zenoh==1.7.1\n"
        "(or run scripts/setup.sh, which does this for you - see the "
        "top-level README's 'Setup' section)."
    ) from exc

# Suction on + all 8 cups on - matches sim_cell.runner's _DIO_HOLDING/_DIO_EMPTY
# and the same bit layout conveyor_indexing.mcap_recorder already writes for
# this arm's position_status channel.
_DIO_HOLDING = 0x10000 | 0xFF
_DIO_EMPTY = 0


def _open_session() -> zenoh.Session:
    conf = zenoh.Config()
    router = os.environ.get("ZENOH_ROUTER")
    if router:
        conf.insert_json5("connect/endpoints", f'["{router}"]')
        logger.info("connecting to Zenoh router at %s", router)
    else:
        logger.warning("ZENOH_ROUTER not set; opening Zenoh session in peer-to-peer mode")
    return zenoh.open(conf)


class RobotStateZenohPublisher:
    """One Zenoh session publishing both arms' PositionStatus plus the shared
    conveyor StateConveyors - the live counterpart of what
    conveyor_indexing.mcap_recorder already writes to MCAP.
    """

    ARM_TOPICS: ClassVar[dict] = {1: "theia/robot/arm1/position_status", 2: "theia/robot/arm2/position_status"}
    CONVEYOR_TOPIC = "theia/plc/state_conveyors"
    # `sim/` prefix, not `theia/` - matches sim/conveyor/command and
    # sim/arm/<n>/action_command's convention for sim-only channels with no
    # production equivalent (BoxState/BoxStates are sim ground truth, not
    # something a real cell's PLC would ever report).
    BOX_STATE_TOPIC = "sim/box_states"
    # Live flange pose per arm (2026-09-09, capability-diffusion's grasp-
    # feedback work). Same name as the MCAP channel mcap_recorder.record_
    # tool_pose already writes, same ArmToolPose message, same source prim
    # (pick_and_place.controller.tool_world_pose - wrist_3_link/flange, the
    # exact point external_control's EXTERNAL_ATTACH_MAX_DISTANCE gate
    # measures from). Until now this was recorded but never published, so
    # an external policy could only estimate tool position by running its
    # own forward kinematics on joint readback - which lands on tool0, a
    # different frame from the flange, and left it unable to measure the
    # same cup-to-box distance the attach gate uses.
    TOOL_POSE_TOPICS: ClassVar[dict] = {1: "sim/arm/1/tool_pose", 2: "sim/arm/2/tool_pose"}
    # The autonomous indexer's per-tick belt DECISION (2026-09-09), as a
    # SimConveyorCommands. In external-action mode `theia/plc/state_conveyors`
    # deliberately reports what was actually commanded externally, so a
    # client that wants to mirror the state machine (the collectors' echo
    # bridge) cannot read the decision from telemetry - it would be echoing
    # its own previous command. This topic carries the decision itself.
    AUTONOMOUS_DECISION_TOPIC = "sim/conveyor/autonomous_decision"

    def __init__(self) -> None:
        self._session = _open_session()
        self._arm_publishers = {arm: self._session.declare_publisher(topic) for arm, topic in self.ARM_TOPICS.items()}
        self._conveyor_publisher = self._session.declare_publisher(self.CONVEYOR_TOPIC)
        self._box_state_publisher = self._session.declare_publisher(self.BOX_STATE_TOPIC)
        self._tool_pose_publishers = {
            arm: self._session.declare_publisher(topic) for arm, topic in self.TOOL_POSE_TOPICS.items()
        }
        self._decision_publisher = self._session.declare_publisher(self.AUTONOMOUS_DECISION_TOPIC)
        logger.info(
            "robot-state publishers ready: %s, %s, %s, %s",
            list(self.ARM_TOPICS.values()), self.CONVEYOR_TOPIC, self.BOX_STATE_TOPIC,
            list(self.TOOL_POSE_TOPICS.values()),
        )

    def publish_tool_pose(self, arm: int, sim_time_s: float, position, orientation_wxyz) -> None:
        """Live counterpart of `mcap_recorder.record_tool_pose` - identical
        message, identical source. Published every control tick regardless
        of control mode (the flange GeomPrim is live even when the phase
        machine is not running)."""
        publisher = self._tool_pose_publishers.get(arm)
        if publisher is None:
            logger.warning("publish_tool_pose called for unknown arm %s", arm)
            return
        msg = sim_state.ArmToolPose(
            sim_time_s=sim_time_s,
            arm=arm,
            position=sim_state.Vec3(x=float(position[0]), y=float(position[1]), z=float(position[2])),
            orientation=sim_state.Quat(
                w=float(orientation_wxyz[0]),
                x=float(orientation_wxyz[1]),
                y=float(orientation_wxyz[2]),
                z=float(orientation_wxyz[3]),
            ),
        )
        publisher.put(msg.SerializeToString())

    def publish_autonomous_decision(self, commands_msg) -> None:
        """`commands_msg`: the `SimConveyorCommands` the line controllers filled
        in this tick from their own state machines (before any external
        override)."""
        self._decision_publisher.put(commands_msg.SerializeToString())

    def publish_arm_state(self, arm: int, joint_degrees, holding: bool, capture_ts_us: int) -> None:
        publisher = self._arm_publishers.get(arm)
        if publisher is None:
            logger.warning("publish_arm_state called for unknown arm %s", arm)
            return
        msg = robot_state.PositionStatus(
            joint_degrees=[float(v) for v in joint_degrees],
            dio_blocks=[_DIO_HOLDING if holding else _DIO_EMPTY],
            # Same synchronized capture instant this tick's camera frames carry
            # (see cameras.frame_meta.now_us / sim_cell.runner's capture_ts_us) -
            # an external observer can align this arm-state sample with the
            # camera frames published the same tick without a separate clock.
            recv_timestamp_us=capture_ts_us,
        )
        publisher.put(msg.SerializeToString())

    def publish_conveyor_state(self, state_msg: plc.StateConveyors) -> None:
        self._conveyor_publisher.put(state_msg.SerializeToString())

    def publish_box_states(self, sim_time_s: float, boxes: list, truck_deliveries_count: int = 0) -> None:
        """`boxes`: list of `sim_state_pb2.BoxState`, e.g. from
        `sim_cell.recording.build_box_states` - live counterpart of what
        `conveyor_indexing.mcap_recorder.record_box_states` already writes
        to MCAP (see Stage 5b, docs/progress-tracker.md), so an external
        controller can react to real box position/hold state the same way
        it already can for arm/conveyor state, without needing camera
        perception.

        `truck_deliveries_count` (2026-09-02, capability-diffusion's
        Stage 7c investigation): the sim's own authoritative running count
        of `despawn_boxes_in_truck` despawns SPECIFICALLY - see
        sim_state.proto's `BoxStates.truck_deliveries_count` for why a
        client can't reconstruct this from `boxes` alone (a box vanishing
        from this list is equally consistent with a floor-despawn or a
        stale-despawn, neither of which is a real delivery).
        """
        msg = sim_state.BoxStates(
            sim_time_s=sim_time_s, boxes=boxes, truck_deliveries_count=truck_deliveries_count
        )
        self._box_state_publisher.put(msg.SerializeToString())

    def close(self) -> None:
        for publisher in self._arm_publishers.values():
            publisher.undeclare()
        self._conveyor_publisher.undeclare()
        self._box_state_publisher.undeclare()
        for publisher in self._tool_pose_publishers.values():
            publisher.undeclare()
        self._decision_publisher.undeclare()
        self._session.close()
        logger.info("robot-state Zenoh session closed")
