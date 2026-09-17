"""Wave sizing and grid layout of sim_cell.box_spawner (2026-09-10): waves
above one row become lanes across the belt; the single-row case is the
historical layout so recorded seeds replay. Pure geometry, no Isaac: the
module's Isaac-only imports are stubbed at load time.
"""

from __future__ import annotations

import importlib
import math
import sys
import types

import pytest


@pytest.fixture(scope="module")
def spawner_module():
    stubs = {}
    for name in ("conveyor_indexing.belt_geometry", "conveyor_indexing.zone", "sim_cell.stage_setup.box_pool"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.compute_belt_bounds = lambda *_a, **_k: None
            m.ConveyorZone = object
            m.BoxPool = object
            sys.modules[name] = m
            stubs[name] = m
    try:
        sys.modules.pop("sim_cell.box_spawner", None)
        yield importlib.import_module("sim_cell.box_spawner")
    finally:
        for name in stubs:
            sys.modules.pop(name, None)


# ConveyorTrack: 2.0 m along travel, 0.9 m across; the 26 cm box is the widest
TRAVEL_HALF, LATERAL_HALF = 1.0, 0.45
R26 = math.hypot(0.13, 0.13)


def test_one_row_holds_four_and_matches_the_historical_clamp(spawner_module):
    gl = spawner_module.grid_layout
    for n in (1, 2, 3, 4):
        assert gl(n, TRAVEL_HALF, LATERAL_HALF, R26) == [n]


def test_larger_waves_add_a_second_lane_up_to_eight(spawner_module):
    gl = spawner_module.grid_layout
    assert gl(5, TRAVEL_HALF, LATERAL_HALF, R26) == [3, 2]
    assert gl(8, TRAVEL_HALF, LATERAL_HALF, R26) == [4, 4]
    # nothing beyond two lanes fits across 0.9 m; the wave is clamped
    assert sum(gl(12, TRAVEL_HALF, LATERAL_HALF, R26)) == 8


def test_two_lanes_cannot_overlap_at_worst_case_jitter(spawner_module):
    m = spawner_module
    lanes = m.grid_layout(8, TRAVEL_HALF, LATERAL_HALF, R26)
    lane_half = LATERAL_HALF / len(lanes)
    lateral_room = max(0.0, lane_half - m.LANE_CLEARANCE_M - R26)
    # two worst-case circles in neighbouring lanes, both jittered toward each other
    assert 2 * lane_half - 2 * lateral_room >= 2 * R26 + 2 * m.LANE_CLEARANCE_M - 1e-9
    slot_half = TRAVEL_HALF / lanes[0]
    travel_room = max(0.0, slot_half - R26 - m.SPAWN_SLOT_CLEARANCE_M)
    assert 2 * slot_half - 2 * travel_room >= 2 * R26 + 2 * m.SPAWN_SLOT_CLEARANCE_M - 1e-9


def test_wave_bounds_come_from_env_and_default_to_the_historical_draw(spawner_module):
    wb = spawner_module.wave_bounds_from_env
    assert wb({}) == (spawner_module.WAVE_COUNT_MIN, spawner_module.WAVE_COUNT_MAX) == (1, 4)
    assert wb({"CONVEYOR_INDEXING_WAVE_MIN": "4", "CONVEYOR_INDEXING_WAVE_MAX": "8"}) == (4, 8)
    with pytest.raises(ValueError):
        wb({"CONVEYOR_INDEXING_WAVE_MIN": "0"})
    with pytest.raises(ValueError):
        wb({"CONVEYOR_INDEXING_WAVE_MIN": "5", "CONVEYOR_INDEXING_WAVE_MAX": "4"})
