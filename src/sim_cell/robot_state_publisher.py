"""Publishes live arm, conveyor, box and tool-pose telemetry over Zenoh."""

from __future__ import annotations

import logging
import os
from typing import ClassVar

from sim_cell.protos import sim_state, telemetry

logger = logging.getLogger(__name__)

try:
    import zenoh
except ImportError as exc:
    raise SystemExit("eclipse-zenoh is not installed in this interpreter; see scripts/setup.sh") from exc


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
    """One Zenoh session for all live telemetry the sim emits."""

    ARM_TOPICS: ClassVar[dict] = {1: "sim/arm/1/state", 2: "sim/arm/2/state"}
    CONVEYOR_TOPIC = "sim/conveyor/state"
    BOX_STATE_TOPIC = "sim/box_states"
    TOOL_POSE_TOPICS: ClassVar[dict] = {1: "sim/arm/1/tool_pose", 2: "sim/arm/2/tool_pose"}
    # The autonomous indexer's decision before any external override.
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
        logger.info("telemetry publishers ready: %s, %s", list(self.ARM_TOPICS.values()), self.CONVEYOR_TOPIC)

    def publish_arm_state(
        self,
        arm: int,
        joint_positions_rad,
        joint_velocities_rad_s,
        holding: bool,
        tool_position,
        tool_orientation_wxyz,
        sim_time_us: int,
    ) -> None:
        publisher = self._arm_publishers.get(arm)
        if publisher is None:
            logger.warning("publish_arm_state called for unknown arm %s", arm)
            return
        msg = telemetry.SimArmState(
            arm=arm,
            joint_positions_rad=[float(v) for v in joint_positions_rad],
            joint_velocities_rad_s=[float(v) for v in joint_velocities_rad_s],
            holding=holding,
            tool_position=sim_state.Vec3(x=float(tool_position[0]), y=float(tool_position[1]), z=float(tool_position[2])),
            tool_orientation=sim_state.Quat(
                w=float(tool_orientation_wxyz[0]),
                x=float(tool_orientation_wxyz[1]),
                y=float(tool_orientation_wxyz[2]),
                z=float(tool_orientation_wxyz[3]),
            ),
            sim_time_us=sim_time_us,
        )
        publisher.put(msg.SerializeToString())

    def publish_tool_pose(self, arm: int, sim_time_s: float, position, orientation_wxyz) -> None:
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
        self._decision_publisher.put(commands_msg.SerializeToString())

    def publish_conveyor_state(self, state_msg: telemetry.SimConveyorStates) -> None:
        self._conveyor_publisher.put(state_msg.SerializeToString())

    def publish_box_states(self, sim_time_s: float, boxes: list, truck_deliveries_count: int = 0) -> None:
        msg = sim_state.BoxStates(sim_time_s=sim_time_s, boxes=boxes, truck_deliveries_count=truck_deliveries_count)
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
        logger.info("telemetry Zenoh session closed")
