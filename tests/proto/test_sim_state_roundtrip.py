"""Proto round-trip tests for proto/sim_state.proto's P4/P5 additions
(Aabb, ZoneGeometry, Transform, PoolVariant, ArmToolPose, and RunMetadata's
new field numbers 10-18) - equivalent to the plan's P0 test-plan item "proto
round-trip after submodule fix". Requires `bash gen_proto.sh` to have been
run first (see tests/conftest.py).
"""

from __future__ import annotations

import sim_state_pb2


def test_run_metadata_new_field_numbers_dont_collide_with_existing():
    """Field numbers 1-9 must keep their exact wire numbers - added fields
    start at 10 (see the proto's own RunMetadata comment).
    """
    fields_by_name = {f.name: f.number for f in sim_state_pb2.RunMetadata.DESCRIPTOR.fields}
    assert fields_by_name["conveyor_indexing_git_sha"] == 1
    assert fields_by_name["instance_index"] == 2
    assert fields_by_name["spawn_seed"] == 3
    assert fields_by_name["physics_dt"] == 4
    assert fields_by_name["control_hz"] == 5
    assert fields_by_name["camera_fps"] == 6
    assert fields_by_name["camera_width"] == 7
    assert fields_by_name["camera_height"] == 8
    assert fields_by_name["cameras"] == 9

    assert fields_by_name["control_source"] == 10
    assert fields_by_name["run_label"] == 11
    assert fields_by_name["zone_geometry"] == 12
    assert fields_by_name["truck_bed_aabb"] == 13
    assert fields_by_name["robot_1_base_transform"] == 14
    assert fields_by_name["robot_2_base_transform"] == 15
    assert fields_by_name["attach_max_distance_m"] == 16
    assert fields_by_name["pool_variants"] == 17
    assert fields_by_name["camera_horizontal_aperture_mm"] == 18

    # No duplicate field numbers anywhere in the message.
    numbers = list(fields_by_name.values())
    assert len(numbers) == len(set(numbers))


def test_run_metadata_full_round_trip():
    metadata = sim_state_pb2.RunMetadata(
        conveyor_indexing_git_sha="abc123def456",
        instance_index=2,
        spawn_seed=99,
        control_source="policy:r5-line1",
        run_label="overnight pilot",
        attach_max_distance_m=0.005,
        camera_horizontal_aperture_mm=20.955,
    )
    zone = metadata.zone_geometry.add()
    zone.node_path = "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode"
    zone.aabb.min.x = -1.0
    zone.aabb.max.x = 1.0
    zone.belt_top_z = 0.9
    zone.travel_direction.x = 1.0
    zone.stop_fraction = 0.8
    zone.speed_m_per_s = 1.1
    zone.is_hold_zone = True
    zone.line_id = 1

    metadata.truck_bed_aabb.min.x = -2.0
    metadata.truck_bed_aabb.max.x = 2.0
    metadata.robot_1_base_transform.translate.y = 1.0928
    metadata.robot_1_base_transform.orientation.w = 1.0
    metadata.robot_2_base_transform.translate.y = 3.0

    variant = metadata.pool_variants.add()
    variant.variant = "CubeBox_A04_26cm_PR_NVD"
    variant.asset_url = "https://example.com/CubeBox_A04_26cm_PR_NVD_01.usd"
    variant.count = 12
    variant.half_extent.x = 0.13

    raw = metadata.SerializeToString()
    round_tripped = sim_state_pb2.RunMetadata()
    round_tripped.ParseFromString(raw)

    assert round_tripped == metadata
    assert round_tripped.control_source == "policy:r5-line1"
    assert round_tripped.zone_geometry[0].is_hold_zone is True
    assert round_tripped.pool_variants[0].count == 12


def test_arm_tool_pose_round_trip():
    msg = sim_state_pb2.ArmToolPose(sim_time_s=12.5, arm=2)
    msg.position.x, msg.position.y, msg.position.z = 1.0, 2.0, 3.0
    msg.orientation.w = 1.0

    round_tripped = sim_state_pb2.ArmToolPose()
    round_tripped.ParseFromString(msg.SerializeToString())
    assert round_tripped == msg
    assert round_tripped.arm == 2


def test_zone_geometry_defaults_are_falsy_not_missing():
    """proto3 has no field presence for scalars without `optional` - a zone
    with is_hold_zone unset must read as False (0.0 stop_fraction), matching
    sim_cell.recording's non-hold-zone convention (documented in the
    ZoneGeometry proto comment), not raise/omit.
    """
    zone = sim_state_pb2.ZoneGeometry(node_path="/World/ConveyorTrack_02/ConveyorBeltGraph/ConveyorNode")
    assert zone.is_hold_zone is False
    assert zone.stop_fraction == 0.0
    assert zone.line_id == 0
