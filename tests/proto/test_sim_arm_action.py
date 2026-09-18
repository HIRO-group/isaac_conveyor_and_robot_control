"""proto/sim_arm_action.proto: the tool_target alternative to joint_targets (external mode)."""

from __future__ import annotations

import sim_arm_action_pb2
import sim_state_pb2


def test_joint_command_has_no_tool_target():
    cmd = sim_arm_action_pb2.SimArmActionCommand(joint_targets=[0.0] * 6, suction=False, seq=1)
    assert not cmd.HasField("tool_target")
    assert not sim_arm_action_pb2.SimArmActionCommand.FromString(cmd.SerializeToString()).HasField("tool_target")


def test_tool_target_round_trips_with_world_frame_pose():
    cmd = sim_arm_action_pb2.SimArmActionCommand(
        suction=True,
        seq=7,
        tool_target=sim_arm_action_pb2.ToolTarget(
            position=sim_state_pb2.Vec3(x=-3.0, y=2.18, z=1.2),
            orientation=sim_state_pb2.Quat(w=-0.5, x=0.5, y=-0.5, z=0.5),
        ),
    )
    back = sim_arm_action_pb2.SimArmActionCommand.FromString(cmd.SerializeToString())
    assert back.HasField("tool_target") and back.seq == 7 and back.suction
    assert back.tool_target.position.x == -3.0 and abs(back.tool_target.position.z - 1.2) < 1e-6
    assert back.tool_target.orientation.w == -0.5
    assert list(back.joint_targets) == []
