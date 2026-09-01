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
        "  /home/ubuntu/IsaacSim/python.sh -m pip install eclipse-zenoh==1.7.1\n"
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

    def __init__(self) -> None:
        self._session = _open_session()
        self._arm_publishers = {arm: self._session.declare_publisher(topic) for arm, topic in self.ARM_TOPICS.items()}
        self._conveyor_publisher = self._session.declare_publisher(self.CONVEYOR_TOPIC)
        self._box_state_publisher = self._session.declare_publisher(self.BOX_STATE_TOPIC)
        logger.info(
            "robot-state publishers ready: %s, %s, %s",
            list(self.ARM_TOPICS.values()), self.CONVEYOR_TOPIC, self.BOX_STATE_TOPIC,
        )

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

    def publish_box_states(self, sim_time_s: float, boxes: list) -> None:
        """`boxes`: list of `sim_state_pb2.BoxState`, e.g. from
        `sim_cell.recording.build_box_states` - live counterpart of what
        `conveyor_indexing.mcap_recorder.record_box_states` already writes
        to MCAP (see Stage 5b, docs/progress-tracker.md), so an external
        controller can react to real box position/hold state the same way
        it already can for arm/conveyor state, without needing camera
        perception.
        """
        msg = sim_state.BoxStates(sim_time_s=sim_time_s, boxes=boxes)
        self._box_state_publisher.put(msg.SerializeToString())

    def close(self) -> None:
        for publisher in self._arm_publishers.values():
            publisher.undeclare()
        self._conveyor_publisher.undeclare()
        self._box_state_publisher.undeclare()
        self._session.close()
        logger.info("robot-state Zenoh session closed")
