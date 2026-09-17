"""Publishes live arm, conveyor, box, tool-pose, clock and run-metadata telemetry over Zenoh."""

from __future__ import annotations

import logging

from conveyor_indexing.topics import Topics
from conveyor_indexing.zenoh_session import open_session
from sim_cell.protos import sim_state, telemetry

logger = logging.getLogger(__name__)


def _vec3(v) -> sim_state.Vec3:
    return sim_state.Vec3(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def _quat(wxyz) -> sim_state.Quat:
    return sim_state.Quat(w=float(wxyz[0]), x=float(wxyz[1]), y=float(wxyz[2]), z=float(wxyz[3]))


class RobotStateZenohPublisher:
    """One Zenoh session for all live telemetry the sim emits."""

    def __init__(self, arms: tuple[int, ...] = (1, 2), topics: Topics | None = None) -> None:
        self.topics = topics or Topics.from_env()
        self._session = open_session()
        declare = self._session.declare_publisher
        self._arm_publishers = {arm: declare(self.topics.arm_state(arm)) for arm in arms}
        self._tool_pose_publishers = {arm: declare(self.topics.arm_tool_pose(arm)) for arm in arms}
        self._conveyor_publisher = declare(self.topics.conveyor_state)
        self._box_state_publisher = declare(self.topics.boxes_state)
        self._decision_publisher = declare(self.topics.conveyor_autonomous_decision)
        self._clock_publisher = declare(self.topics.clock)
        self._run_metadata_publisher = declare(self.topics.run_metadata)
        self._run_metadata_bytes: bytes | None = None
        self._run_metadata_queryable = self._session.declare_queryable(
            self.topics.run_metadata, self._handle_run_metadata_query
        )
        logger.info("telemetry publishers ready under prefix %r for arms %s", self.topics.prefix, list(arms))

    def serve_run_metadata(self, run_metadata: sim_state.RunMetadata) -> None:
        """Latched: put once, and answer queries for late joiners."""
        self._run_metadata_bytes = run_metadata.SerializeToString()
        self._run_metadata_publisher.put(self._run_metadata_bytes)

    def _handle_run_metadata_query(self, query) -> None:
        if self._run_metadata_bytes is not None:
            query.reply(self.topics.run_metadata, self._run_metadata_bytes)

    def publish_clock(self, sim_time_us: int, realtime_factor: float) -> None:
        msg = telemetry.SimClock(sim_time_us=sim_time_us, realtime_factor=realtime_factor)
        self._clock_publisher.put(msg.SerializeToString())

    def publish_arm_state(
        self, arm: int, joint_positions_rad, joint_velocities_rad_s, holding: bool,
        tool_position, tool_orientation_wxyz, sim_time_us: int,
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
            tool_position=_vec3(tool_position),
            tool_orientation=_quat(tool_orientation_wxyz),
            sim_time_us=sim_time_us,
        )
        publisher.put(msg.SerializeToString())

    def publish_tool_pose(self, arm: int, sim_time_s: float, position, orientation_wxyz) -> None:
        publisher = self._tool_pose_publishers.get(arm)
        if publisher is None:
            logger.warning("publish_tool_pose called for unknown arm %s", arm)
            return
        msg = sim_state.ArmToolPose(
            sim_time_s=sim_time_s, arm=arm, position=_vec3(position), orientation=_quat(orientation_wxyz)
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
        self._run_metadata_queryable.undeclare()
        for publisher in (
            *self._arm_publishers.values(),
            *self._tool_pose_publishers.values(),
            self._conveyor_publisher,
            self._box_state_publisher,
            self._decision_publisher,
            self._clock_publisher,
            self._run_metadata_publisher,
        ):
            publisher.undeclare()
        self._session.close()
        logger.info("telemetry Zenoh session closed")
