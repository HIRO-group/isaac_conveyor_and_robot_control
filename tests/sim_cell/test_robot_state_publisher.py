"""Integration-style test for sim_cell.robot_state_publisher's P9 fix
(recv_timestamp_us = capture_ts_us). Real eclipse-zenoh peer-to-peer
pub/sub loopback - no Isaac Sim needed (see tests/conftest.py); zenoh opens
a local session with no network/router dependency.
"""

from __future__ import annotations

import queue

from sim_cell.protos import robot_state, sim_state
from sim_cell.robot_state_publisher import RobotStateZenohPublisher


def test_publish_arm_state_sets_recv_timestamp_us():
    publisher = RobotStateZenohPublisher()
    try:
        received: queue.Queue = queue.Queue()
        sub = publisher._session.declare_subscriber("theia/robot/arm1/position_status", received.put)
        try:
            capture_ts_us = 1_723_000_000_123_456
            publisher.publish_arm_state(1, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], True, capture_ts_us)

            sample = received.get(timeout=5.0)
            payload = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)
            msg = robot_state.PositionStatus()
            msg.ParseFromString(payload)

            assert msg.recv_timestamp_us == capture_ts_us
            assert msg.joint_degrees[0] == 10.0
            assert msg.dio_blocks[0] == 0x10000 | 0xFF  # holding=True
        finally:
            sub.undeclare()
    finally:
        publisher.close()


def test_publish_arm_state_distinct_capture_ts_per_call():
    """Different ticks must carry their own capture_ts_us, not a stale/fixed
    value - regression guard against accidentally hoisting it to __init__.
    """
    publisher = RobotStateZenohPublisher()
    try:
        received: queue.Queue = queue.Queue()
        sub = publisher._session.declare_subscriber("theia/robot/arm2/position_status", received.put)
        try:
            publisher.publish_arm_state(2, [0.0] * 6, False, 111)
            publisher.publish_arm_state(2, [0.0] * 6, False, 222)

            seen = []
            for _ in range(2):
                sample = received.get(timeout=5.0)
                payload = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)
                msg = robot_state.PositionStatus()
                msg.ParseFromString(payload)
                seen.append(msg.recv_timestamp_us)
            assert seen == [111, 222]
        finally:
            sub.undeclare()
    finally:
        publisher.close()


def test_publish_box_states_round_trips():
    """Stage 5b: box-state telemetry live over Zenoh, the plumbing an
    external policy needs to react to real box position/hold state
    without camera perception - see docs/progress-tracker.md."""
    publisher = RobotStateZenohPublisher()
    try:
        received: queue.Queue = queue.Queue()
        sub = publisher._session.declare_subscriber("sim/box_states", received.put)
        try:
            box = sim_state.BoxState(
                box_id="/World/BoxPool/box_00",
                variant="small",
                position=sim_state.Vec3(x=1.0, y=2.0, z=0.5),
                orientation=sim_state.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
                held_by_arm=1,
            )
            publisher.publish_box_states(12.5, [box])

            sample = received.get(timeout=5.0)
            payload = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)
            msg = sim_state.BoxStates()
            msg.ParseFromString(payload)

            assert msg.sim_time_s == 12.5
            assert len(msg.boxes) == 1
            assert msg.boxes[0].box_id == "/World/BoxPool/box_00"
            assert msg.boxes[0].held_by_arm == 1
            assert msg.boxes[0].position.x == 1.0
        finally:
            sub.undeclare()
    finally:
        publisher.close()
