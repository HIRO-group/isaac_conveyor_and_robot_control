"""Zenoh loopback tests for RobotStateZenohPublisher (peer mode, no Isaac Sim)."""

from __future__ import annotations

import queue

import pytest

from sim_cell.protos import telemetry
from sim_cell.robot_state_publisher import RobotStateZenohPublisher


def _recv(q: queue.Queue):
    sample = q.get(timeout=5.0)
    payload = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)
    msg = telemetry.SimArmState()
    msg.ParseFromString(payload)
    return msg


def test_publish_arm_state_round_trip():
    publisher = RobotStateZenohPublisher()
    try:
        received: queue.Queue = queue.Queue()
        sub = publisher._session.declare_subscriber(publisher.ARM_TOPICS[1], received.put)
        try:
            publisher.publish_arm_state(
                1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.0] * 6, True, (1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0), 123_456
            )
            msg = _recv(received)
            assert msg.arm == 1
            assert msg.sim_time_us == 123_456
            assert msg.holding is True
            assert msg.joint_positions_rad[2] == pytest.approx(0.3)
            assert (msg.tool_position.x, msg.tool_orientation.w) == (1.0, 1.0)
        finally:
            sub.undeclare()
    finally:
        publisher.close()


def test_publish_arm_state_distinct_time_per_call():
    publisher = RobotStateZenohPublisher()
    try:
        received: queue.Queue = queue.Queue()
        sub = publisher._session.declare_subscriber(publisher.ARM_TOPICS[2], received.put)
        try:
            for t in (1_000, 2_000):
                publisher.publish_arm_state(2, [0.0] * 6, [0.0] * 6, False, (0, 0, 0), (1, 0, 0, 0), t)
            assert [_recv(received).sim_time_us for _ in range(2)] == [1_000, 2_000]
        finally:
            sub.undeclare()
    finally:
        publisher.close()
