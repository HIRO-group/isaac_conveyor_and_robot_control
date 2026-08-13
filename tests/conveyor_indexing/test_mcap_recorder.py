"""Integration-style tests for conveyor_indexing.mcap_recorder - the P3
(on-policy action-log channels), P5 (tool_pose channel), and P9 (persisted
drop count) additions. Real McapRecorder instances writing to tmp_path - no
Isaac Sim needed (mcap_recorder.py only imports generated protobuf bindings
+ mcap/mcap_protobuf - see tests/conftest.py).
"""

from __future__ import annotations

import json

import sim_arm_action_pb2
import sim_state_pb2
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from conveyor_indexing.mcap_recorder import McapRecorder
from conveyor_indexing.protos import sim_action


def _read_topics(mcap_path) -> list[str]:
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        return [channel.topic for _, channel, _ in reader.iter_messages()]


def _read_messages_on(mcap_path, topic: str) -> list:
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        return [
            decoded
            for _, channel, _, decoded in reader.iter_decoded_messages()
            if channel.topic == topic
        ]


def _single_mcap_file(output_dir) -> object:
    files = list(output_dir.glob("*.mcap"))
    assert len(files) == 1, f"expected exactly one .mcap file, found {files}"
    return files[0]


def test_record_arm_action_command_and_conveyor_command_channels(tmp_path):
    metadata = sim_state_pb2.RunMetadata(conveyor_indexing_git_sha="deadbeef0000")
    recorder = McapRecorder(output_dir=str(tmp_path), run_metadata=metadata, rotate_period_s=1000.0)

    arm_cmd = sim_arm_action_pb2.SimArmActionCommand(joint_targets=[0.1] * 6, suction=True, seq=7)
    recorder.record_arm_action_command(1, 0.5, arm_cmd)

    conveyor_cmds = sim_action.SimConveyorCommands()
    c = conveyor_cmds.commands.add()
    c.zone_index = 1
    c.conveyor_node_path = "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode"
    c.run = True
    c.speed = 55
    c.direction = 1
    recorder.record_conveyor_command(0.5, conveyor_cmds)

    recorder.close()

    mcap_file = _single_mcap_file(tmp_path)
    topics = _read_topics(mcap_file)
    assert "sim/arm/1/action_command" in topics
    assert "sim/conveyor/command" in topics
    assert "sim/run_metadata" in topics

    [decoded_arm_cmd] = _read_messages_on(mcap_file, "sim/arm/1/action_command")
    assert decoded_arm_cmd.seq == 7
    assert decoded_arm_cmd.suction is True

    [decoded_conveyor_cmds] = _read_messages_on(mcap_file, "sim/conveyor/command")
    assert len(decoded_conveyor_cmds.commands) == 1
    assert decoded_conveyor_cmds.commands[0].speed == 55


def test_record_tool_pose_channel_both_arms(tmp_path):
    metadata = sim_state_pb2.RunMetadata()
    recorder = McapRecorder(output_dir=str(tmp_path), run_metadata=metadata, rotate_period_s=1000.0)

    recorder.record_tool_pose(1, 0.25, (1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    recorder.record_tool_pose(2, 0.25, (4.0, 5.0, 6.0), (0.0, 1.0, 0.0, 0.0))
    recorder.close()

    mcap_file = _single_mcap_file(tmp_path)
    topics = _read_topics(mcap_file)
    assert "sim/arm/1/tool_pose" in topics
    assert "sim/arm/2/tool_pose" in topics

    [pose_1] = _read_messages_on(mcap_file, "sim/arm/1/tool_pose")
    assert (pose_1.position.x, pose_1.position.y, pose_1.position.z) == (1.0, 2.0, 3.0)
    assert pose_1.arm == 1

    [pose_2] = _read_messages_on(mcap_file, "sim/arm/2/tool_pose")
    assert pose_2.orientation.x == 1.0  # orientation_wxyz=(0,1,0,0) -> w=0, x=1
    assert pose_2.arm == 2


def test_recorder_stats_persisted_at_close(tmp_path):
    """P9: the recorder's drop count must be persisted (not just logged) at
    run end - see McapRecorder._persist_stats.
    """
    metadata = sim_state_pb2.RunMetadata()
    recorder = McapRecorder(output_dir=str(tmp_path), run_metadata=metadata, rotate_period_s=1000.0)
    recorder.record_tool_pose(1, 0.1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    recorder.close()

    stats_path = tmp_path / "recorder_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["dropped"] == 0
    # messages_written counts only queued record_* calls, not the
    # sim/run_metadata header written directly by _rotate() - one tool_pose
    # message here.
    assert stats["messages_written"] == 1
    assert "run_epoch_ns" in stats


def test_recorder_stats_reflects_forced_drops(tmp_path):
    """Deterministic (non-racy) check of the persistence logic itself: drive
    the internal counters directly rather than relying on real queue-full
    timing, and confirm _persist_stats() writes them faithfully.
    """
    metadata = sim_state_pb2.RunMetadata()
    recorder = McapRecorder(output_dir=str(tmp_path), run_metadata=metadata, rotate_period_s=1000.0)
    recorder._dropped = 3
    recorder._messages_written = 41
    recorder.close()

    stats = json.loads((tmp_path / "recorder_stats.json").read_text())
    assert stats["dropped"] == 3
    assert stats["messages_written"] == 41
