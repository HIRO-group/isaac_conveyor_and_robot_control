"""Unit tests for sim_cell.recording's pure logic - the P1/P2/P4 fixes from
the multi-policy/VLM plan's sim-repo-changes list. No Isaac Sim/omni/carb
needed (see tests/conftest.py); sim_state_pb2 comes from `bash gen_proto.sh`.
"""

from __future__ import annotations

import pytest

from sim_cell import recording

# -- P1: recording + external-action exclusivity relaxation -----------------


def test_validate_external_action_recording_allows_mcap_only():
    """external_action=1 + CONVEYOR_INDEXING_RECORD_MCAP=1 (no episode
    recorder) must be allowed - this is the whole point of the P1 fix.
    """
    recording.validate_external_action_recording(external_action=True, episode_recorder_enabled=False)


def test_validate_external_action_recording_still_rejects_episode_recorder():
    """external_action=1 + CONVEYOR_INDEXING_RECORD=1 (the 30Hz episode/
    parquet recorder) must still raise - episode segmentation is meaningless
    once an external controller owns the phase machine.
    """
    with pytest.raises(SystemExit):
        recording.validate_external_action_recording(external_action=True, episode_recorder_enabled=True)


def test_validate_external_action_recording_autonomous_mode_always_ok():
    recording.validate_external_action_recording(external_action=False, episode_recorder_enabled=True)
    recording.validate_external_action_recording(external_action=False, episode_recorder_enabled=False)


# -- P2: frozen-state telemetry (held_by_arm/dio) fix ------------------------


def test_resolve_arm_telemetry_external_action_uses_held_box_paths():
    """The actual bug: in external_action mode, holding/held_by_arm must
    come from held_box_path_1/2, not the (frozen, always-WAITING/never-
    holding) phase-machine state - even when the phase machine claims it's
    holding (which shouldn't happen, but the fix must not depend on that).
    """
    holding_1, holding_2, held_by_arm = recording.resolve_arm_telemetry(
        external_action=True,
        held_box_path_1="/World/CubeBox_A03_21cm_PR_NVD_P00",
        held_box_path_2=None,
        pick_place_1_holding=False,  # frozen phase machine says "not holding"
        pick_place_2_holding=False,
        pick_place_1_held_box_path=None,
        pick_place_2_held_box_path=None,
    )
    assert holding_1 is True
    assert holding_2 is False
    assert held_by_arm == {"/World/CubeBox_A03_21cm_PR_NVD_P00": 1}


def test_resolve_arm_telemetry_external_action_both_arms_holding():
    holding_1, holding_2, held_by_arm = recording.resolve_arm_telemetry(
        external_action=True,
        held_box_path_1="/World/CubeBox_A",
        held_box_path_2="/World/CubeBox_B",
        pick_place_1_holding=False,
        pick_place_2_holding=False,
        pick_place_1_held_box_path=None,
        pick_place_2_held_box_path=None,
    )
    assert (holding_1, holding_2) == (True, True)
    assert held_by_arm == {"/World/CubeBox_A": 1, "/World/CubeBox_B": 2}


def test_resolve_arm_telemetry_external_action_neither_holding():
    holding_1, holding_2, held_by_arm = recording.resolve_arm_telemetry(
        external_action=True,
        held_box_path_1=None,
        held_box_path_2=None,
        pick_place_1_holding=True,  # dormant phase machine value must be ignored
        pick_place_2_holding=True,
        pick_place_1_held_box_path="/World/StaleFromDormantMachine",
        pick_place_2_held_box_path="/World/AlsoStale",
    )
    assert (holding_1, holding_2) == (False, False)
    assert held_by_arm == {}


def test_resolve_arm_telemetry_autonomous_mode_uses_phase_machine():
    """Autonomous (external_action=False) mode is unchanged: it uses the
    phase machine's own holding_box/held_box_path.
    """
    holding_1, holding_2, held_by_arm = recording.resolve_arm_telemetry(
        external_action=False,
        held_box_path_1=None,  # never populated outside external_action mode
        held_box_path_2=None,
        pick_place_1_holding=True,
        pick_place_2_holding=False,
        pick_place_1_held_box_path="/World/CubeBox_X",
        pick_place_2_held_box_path=None,
    )
    assert (holding_1, holding_2) == (True, False)
    assert held_by_arm == {"/World/CubeBox_X": 1}


# -- P4: control_source resolution -------------------------------------------


def test_resolve_control_source_explicit_env_wins():
    assert recording.resolve_control_source(True, "policy:ckpt-42") == "policy:ckpt-42"
    assert recording.resolve_control_source(False, "policy:ckpt-42") == "policy:ckpt-42"


def test_resolve_control_source_autonomous_default():
    assert recording.resolve_control_source(False, None) == "scripted"
    assert recording.resolve_control_source(False, "") == "scripted"


def test_resolve_control_source_external_action_default_is_not_scripted():
    """An on-policy run must never default to "scripted" - see Workstream 2's
    manifest rule that policy runs must never enter a training superset by
    accident.
    """
    assert recording.resolve_control_source(True, None) == "policy:unknown"
    assert recording.resolve_control_source(True, "") == "policy:unknown"


# -- P4: RunMetadata geometry/config proto builders --------------------------


def test_build_zone_geometry_proto_aabb_from_center_and_half_extent():
    zone = recording.ZoneGeometryInput(
        node_path="/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode",
        bbox_center=(1.0, 2.0, 3.0),
        bbox_half_extent=(0.5, 0.25, 0.1),
        belt_top_z=0.9,
        travel_direction=(1.0, 0.0, 0.0),
        stop_fraction=0.8,
        speed_m_per_s=1.1,
        is_hold_zone=True,
        line_id=1,
    )
    proto = recording.build_zone_geometry_proto(zone)
    assert proto.node_path == zone.node_path
    assert (proto.aabb.min.x, proto.aabb.min.y, proto.aabb.min.z) == pytest.approx((0.5, 1.75, 2.9))
    assert (proto.aabb.max.x, proto.aabb.max.y, proto.aabb.max.z) == pytest.approx((1.5, 2.25, 3.1))
    assert proto.belt_top_z == pytest.approx(0.9)
    assert (proto.travel_direction.x, proto.travel_direction.y, proto.travel_direction.z) == pytest.approx(
        (1.0, 0.0, 0.0)
    )
    assert proto.stop_fraction == pytest.approx(0.8)
    assert proto.speed_m_per_s == pytest.approx(1.1)
    assert proto.is_hold_zone is True
    assert proto.line_id == 1


def test_build_zone_geometry_proto_non_hold_zone_zero_stop_fraction():
    zone = recording.ZoneGeometryInput(
        node_path="/World/ConveyorTrack_02/ConveyorBeltGraph/ConveyorNode",
        bbox_center=(0.0, 0.0, 0.0),
        bbox_half_extent=(1.0, 1.0, 1.0),
        belt_top_z=0.9,
        travel_direction=(0.0, 0.0, 0.0),  # curved-zone fallback
        stop_fraction=0.0,
        speed_m_per_s=1.1,
        is_hold_zone=False,
        line_id=1,
    )
    proto = recording.build_zone_geometry_proto(zone)
    assert proto.is_hold_zone is False
    assert proto.stop_fraction == pytest.approx(0.0)


def test_build_aabb_proto():
    aabb = recording.build_aabb_proto((0.0, 1.0, 2.0), (3.0, 4.0, 5.0))
    assert (aabb.min.x, aabb.min.y, aabb.min.z) == pytest.approx((0.0, 1.0, 2.0))
    assert (aabb.max.x, aabb.max.y, aabb.max.z) == pytest.approx((3.0, 4.0, 5.0))


def test_build_transform_proto_isaac_wxyz_convention():
    transform = recording.build_transform_proto((1.0, 2.0, 3.0), (0.7071, 0.0, 0.0, 0.7071))
    assert (transform.translate.x, transform.translate.y, transform.translate.z) == pytest.approx((1.0, 2.0, 3.0))
    assert transform.orientation.w == pytest.approx(0.7071)
    assert transform.orientation.z == pytest.approx(0.7071)


def test_build_pool_variant_proto():
    p = recording.PoolVariantInput(
        variant="CubeBox_A03_21cm_PR_NVD",
        asset_url="https://example.com/CubeBox_A03_21cm_PR_NVD_01.usd",
        count=12,
        half_extent=(0.105, 0.105, 0.105),
    )
    proto = recording.build_pool_variant_proto(p)
    assert proto.variant == "CubeBox_A03_21cm_PR_NVD"
    assert proto.asset_url.endswith(".usd")
    assert proto.count == 12
    assert (proto.half_extent.x, proto.half_extent.y, proto.half_extent.z) == pytest.approx((0.105, 0.105, 0.105))


# -- P4: full RunMetadata build (proto round-trip) ---------------------------


class _FakeCameraSpec:
    """Minimal stand-in for cameras.specs.CameraSpec - only the attributes
    _build_run_metadata actually reads, so this test doesn't need
    cameras.specs (itself Isaac-free, but keeping this test's fixture
    self-contained and explicit about the contract).
    """

    def __init__(self, serial: str, role: int):
        self.serial = serial
        self.role = role
        self.translate = (0.1, 0.2, 0.3)
        self.rotation_euler_xyz_deg = (0.0, 0.0, 0.0)
        self.parent_path = None
        self.focal_length = 18.0
        self.width = 640
        self.height = 480
        self.fps = 30


def _make_extras() -> recording.RunMetadataExtras:
    zone = recording.ZoneGeometryInput(
        node_path="/World/ConveyorTrack/ConveyorBeltGraph/ConveyorNode",
        bbox_center=(0.0, 0.0, 0.0),
        bbox_half_extent=(1.0, 1.0, 1.0),
        belt_top_z=0.9,
        travel_direction=(1.0, 0.0, 0.0),
        stop_fraction=0.0,
        speed_m_per_s=1.1,
        is_hold_zone=False,
        line_id=1,
    )
    pool_variant = recording.PoolVariantInput(
        variant="CubeBox_A03_21cm_PR_NVD",
        asset_url="https://example.com/box.usd",
        count=12,
        half_extent=(0.105, 0.105, 0.105),
    )
    return recording.RunMetadataExtras(
        zone_geometry=[zone],
        truck_bed_min=(-1.0, -1.0, 0.0),
        truck_bed_max=(1.0, 1.0, 1.0),
        robot_1_transform=((-3.0, 1.0928, 0.0), (1.0, 0.0, 0.0, 0.0)),
        robot_2_transform=((-3.0, 3.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        attach_max_distance_m=0.005,
        pool_variants=[pool_variant],
        camera_horizontal_aperture_mm=20.955,
    )


def test_build_run_metadata_includes_p4_fields(monkeypatch):
    monkeypatch.delenv(recording.CONTROL_SOURCE_ENV_VAR, raising=False)
    monkeypatch.delenv(recording._EXTERNAL_ACTION_ENV_VAR, raising=False)
    monkeypatch.delenv(recording.RUN_LABEL_ENV_VAR, raising=False)

    camera_specs = [_FakeCameraSpec("SIM1-PICK", 1)]
    metadata = recording._build_run_metadata(camera_specs, spawn_seed=42, extras=_make_extras())

    assert metadata.control_source == "scripted"
    assert metadata.run_label == ""
    assert len(metadata.zone_geometry) == 1
    assert metadata.zone_geometry[0].node_path == "/World/ConveyorTrack/ConveyorBeltGraph/ConveyorNode"
    assert (metadata.truck_bed_aabb.min.x, metadata.truck_bed_aabb.max.x) == pytest.approx((-1.0, 1.0))
    assert metadata.robot_1_base_transform.translate.y == pytest.approx(1.0928)
    assert metadata.robot_2_base_transform.translate.y == pytest.approx(3.0)
    assert metadata.attach_max_distance_m == pytest.approx(0.005, abs=1e-6)
    assert len(metadata.pool_variants) == 1
    assert metadata.pool_variants[0].variant == "CubeBox_A03_21cm_PR_NVD"
    assert metadata.camera_horizontal_aperture_mm == pytest.approx(20.955, abs=1e-3)
    # Pre-existing fields (1-9) still populated correctly - the P4 additions
    # didn't clobber or renumber anything.
    assert metadata.spawn_seed == 42
    assert len(metadata.cameras) == 1


def test_build_run_metadata_control_source_env_override(monkeypatch):
    monkeypatch.setenv(recording.CONTROL_SOURCE_ENV_VAR, "policy:r3-line1")
    monkeypatch.setenv(recording.RUN_LABEL_ENV_VAR, "overnight-pilot")
    metadata = recording._build_run_metadata([], spawn_seed=1, extras=_make_extras())
    assert metadata.control_source == "policy:r3-line1"
    assert metadata.run_label == "overnight-pilot"


def test_run_metadata_serializes_and_parses_round_trip(monkeypatch):
    """Proto round-trip smoke test - equivalent to the plan's P0 test-plan
    item "proto round-trip after submodule fix", exercised here against the
    P4 additions specifically.
    """
    monkeypatch.delenv(recording.CONTROL_SOURCE_ENV_VAR, raising=False)
    metadata = recording._build_run_metadata([], spawn_seed=7, extras=_make_extras())
    raw = metadata.SerializeToString()

    import sim_state_pb2

    round_tripped = sim_state_pb2.RunMetadata()
    round_tripped.ParseFromString(raw)
    assert round_tripped == metadata
