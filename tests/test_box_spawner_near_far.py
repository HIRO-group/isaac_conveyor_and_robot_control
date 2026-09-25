"""The near/far demonstration layout of sim_cell.box_spawner: one pair per
wave, side by side at the zone centre, the near box on the robot's side. The
module's Isaac-only imports are stubbed at load time, as in
test_box_spawner_waves.py."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture
def spawner_module():
    saved = {name: sys.modules.get(name) for name in
             ("conveyor_indexing.belt_geometry", "conveyor_indexing.zone", "sim_cell.stage_setup.box_pool", "sim_cell.box_spawner")}
    geometry = types.ModuleType("conveyor_indexing.belt_geometry")
    geometry.compute_belt_bounds = lambda *_a, **_k: SimpleNamespace(belt_top_z=1.0)
    zone = types.ModuleType("conveyor_indexing.zone")
    zone.ConveyorZone = object
    pool = types.ModuleType("sim_cell.stage_setup.box_pool")
    pool.BoxPool = object
    sys.modules.update({"conveyor_indexing.belt_geometry": geometry, "conveyor_indexing.zone": zone, "sim_cell.stage_setup.box_pool": pool})
    sys.modules.pop("sim_cell.box_spawner", None)
    try:
        yield importlib.import_module("sim_cell.box_spawner")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class FakeRigidPrim:
    def __init__(self):
        self.position = None
        self.enabled = None
        self.visible = None
        self._physics_rigid_body_view = SimpleNamespace(wake_up=lambda: None)

    def set_world_poses(self, positions=None, orientations=None):
        self.position = positions[0]

    def set_enabled_rigid_bodies(self, flags):
        self.enabled = flags[0]

    def set_visibilities(self, flags):
        self.visible = flags[0]


def _spawner(module, near_side_sign, layout="near_far"):
    zone = SimpleNamespace(
        belt_prim=None, node_path="/World/ConveyorTrack", world_travel_direction=(-1.0, 0.0, 0.0),
        bbox_center=(-1.0, -0.2, 1.0), bbox_half_extent=(1.0, 0.45, 0.1),
    )
    paths = [f"/World/CubeBox_{i}" for i in range(4)]
    pool = SimpleNamespace(
        paths_by_variant={"c26": paths}, half_extents_by_variant={"c26": (0.13, 0.13, 0.13)}, all_paths=lambda: list(paths)
    )
    prims = {p: FakeRigidPrim() for p in paths}
    spawner = module.BoxSpawner(zone, prims, pool, seed=1, layout=layout, near_side_sign=near_side_sign)
    return spawner, prims


def test_near_far_spawns_one_pair_at_the_zone_centre(spawner_module):
    spawner, prims = _spawner(spawner_module, near_side_sign=+1.0)
    spawned = spawner.update(sim_time=0.0, zone_occupied=False)
    assert len(spawned) == 2
    (p_near, _, pos_near, q_near), (p_far, _, pos_far, _) = spawned
    assert pos_near[0] == pos_far[0] == -1.0  # same travel coordinate
    assert pos_near[1] > -0.2 > pos_far[1]  # near on the robot's (+y) side, far opposite
    assert q_near == (1.0, 0.0, 0.0, 0.0)
    assert prims[p_near].enabled and prims[p_near].visible and prims[p_far].position == pos_far


def test_near_far_mirrors_when_the_robot_is_on_the_other_side(spawner_module):
    spawner, _ = _spawner(spawner_module, near_side_sign=-1.0)
    (_, _, pos_near, _), (_, _, pos_far, _) = spawner.update(sim_time=0.0, zone_occupied=False)
    assert pos_near[1] < -0.2 < pos_far[1]


def test_near_far_needs_two_boxes_or_spawns_nothing(spawner_module):
    spawner, _ = _spawner(spawner_module, near_side_sign=+1.0)
    assert len(spawner.update(0.0, False)) == 2
    # the zone must read empty through the debounce before the next wave
    assert spawner.update(10.0, False) == []
    assert len(spawner.update(11.0, False)) == 2  # pool of four: a second pair
    assert spawner.update(20.0, False) == []
    assert spawner.update(21.0, False) == []  # pool exhausted: a pair or nothing


def test_far_layout_spawns_the_far_box_alone(spawner_module):
    spawner, prims = _spawner(spawner_module, near_side_sign=+1.0, layout="far")
    pair, _ = _spawner(spawner_module, near_side_sign=+1.0)
    (_, _, pos_pair_far, _) = pair.update(0.0, False)[1]
    spawned = spawner.update(sim_time=0.0, zone_occupied=False)
    assert len(spawned) == 1
    path, _, pos, q = spawned[0]
    assert pos == pos_pair_far and q == (1.0, 0.0, 0.0, 0.0)  # exactly where the pair's far box goes
    assert prims[path].enabled and prims[path].visible
    assert spawner.waves_spawned == 1


def test_max_waves_caps_the_automatic_spawning(spawner_module, monkeypatch):
    monkeypatch.setenv(spawner_module.MAX_WAVES_ENV_VAR, "1")
    spawner, _ = _spawner(spawner_module, near_side_sign=+1.0, layout="far")
    assert len(spawner.update(0.0, False)) == 1
    assert spawner.update(10.0, False) == [] and spawner.update(11.0, False) == []  # would have spawned
    assert spawner._total_available() == 3
    monkeypatch.setenv(spawner_module.MAX_WAVES_ENV_VAR, "0")
    with pytest.raises(ValueError, match="need >= 1"):
        _spawner(spawner_module, near_side_sign=+1.0, layout="far")


def _two_variant_spawner(module, monkeypatch, variant_filter):
    monkeypatch.setenv(module.BOX_VARIANT_ENV_VAR, variant_filter)
    zone = SimpleNamespace(
        belt_prim=None, node_path="/World/ConveyorTrack", world_travel_direction=(-1.0, 0.0, 0.0),
        bbox_center=(-1.0, -0.2, 1.0), bbox_half_extent=(1.0, 0.45, 0.1),
    )
    small = [f"/World/CubeBox_21_{i}" for i in range(4)]
    large = [f"/World/CubeBox_26_{i}" for i in range(4)]
    pool = SimpleNamespace(
        paths_by_variant={"CubeBox_A03_21cm": small, "CubeBox_A04_26cm": large},
        half_extents_by_variant={"CubeBox_A03_21cm": (0.105,) * 3, "CubeBox_A04_26cm": (0.13,) * 3},
        all_paths=lambda: small + large,
    )
    prims = {p: FakeRigidPrim() for p in small + large}
    return module.BoxSpawner(zone, prims, pool, seed=1, layout="near_far", near_side_sign=+1.0)


def test_box_variant_filter_spawns_only_the_matching_size(spawner_module, monkeypatch):
    spawner = _two_variant_spawner(spawner_module, monkeypatch, "21cm")
    seen = set()
    for t in (0.0, 100.0, 200.0):
        spawner._last_spawn_time = None  # force a wave regardless of debounce/cooldown
        seen |= {variant for _, variant, _, _ in spawner.update(sim_time=t, zone_occupied=False)}
    assert seen == {"CubeBox_A03_21cm"}
    # A recycled small box comes back; a large one (never spawned) is ignored.
    spawner.release(["/World/CubeBox_21_0", "/World/CubeBox_26_0"])
    assert spawner._total_available() == 1


def test_box_variant_filter_rejects_an_unknown_size(spawner_module, monkeypatch):
    with pytest.raises(ValueError, match="matches none"):
        _two_variant_spawner(spawner_module, monkeypatch, "42cm")
