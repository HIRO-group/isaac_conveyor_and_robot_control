"""despawn_boxes_off_belt (2026-09-10): a box clearly below belt height and not
over the truck bed is parked at once; one over the truck bed is left for
despawn_boxes_in_truck; one still at belt height is untouched. Pure logic,
pxr stubbed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def truck_module():
    # Load truck.py by file path so the package __init__ (which imports
    # omni.kit) is never executed; only pxr needs a stand-in.
    added = []
    if "pxr" not in sys.modules:
        pxr = types.ModuleType("pxr")
        for n in ("Gf", "Usd", "UsdGeom", "UsdPhysics"):
            setattr(pxr, n, types.SimpleNamespace())
        sys.modules["pxr"] = pxr
        added.append("pxr")
    try:
        path = Path(__file__).resolve().parents[1] / "src" / "sim_cell" / "stage_setup" / "truck.py"
        spec = importlib.util.spec_from_file_location("truck_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        for n in added:
            sys.modules.pop(n, None)


class FakePrim:
    def __init__(self):
        self.enabled = True
        self.visible = True
        self.pose = None

    def set_enabled_rigid_bodies(self, v):
        self.enabled = v[0]

    def set_visibilities(self, v):
        self.visible = v[0]

    def set_world_poses(self, positions=None, orientations=None):
        self.pose = positions[0]


BED_MIN, BED_MAX = (-3.6, 3.0, 0.265), (-2.4, 5.0, 0.95)


def test_falling_box_off_the_line_end_is_parked_now(truck_module):
    prims = {"/W/a": FakePrim(), "/W/b": FakePrim(), "/W/c": FakePrim()}
    positions = {
        "/W/a": (-6.4, 0.1, 1.2),   # off the end of zone 2, mid-fall
        "/W/b": (-4.5, 0.0, 1.78),  # on a belt
        "/W/c": (-3.0, 4.0, 1.1),   # dropping into the truck bed
    }
    out = truck_module.despawn_boxes_off_belt(prims, positions, 1.40, BED_MIN, BED_MAX, 0.5)
    assert out == ["/W/a"]
    assert prims["/W/a"].enabled is False and prims["/W/a"].visible is False
    assert prims["/W/a"].pose == truck_module.DESPAWNED_BOX_PARK_POSITION
    assert prims["/W/b"].enabled and prims["/W/c"].enabled


def test_truck_margin_keeps_a_box_dropping_just_outside_the_bed_edge(truck_module):
    prims = {"/W/d": FakePrim(), "/W/e": FakePrim()}
    positions = {"/W/d": (-3.9, 4.0, 1.0), "/W/e": (-4.3, 4.0, 1.0)}  # 0.3 m and 0.7 m outside the bed's x
    out = truck_module.despawn_boxes_off_belt(prims, positions, 1.40, BED_MIN, BED_MAX, 0.5)
    assert out == ["/W/e"]
