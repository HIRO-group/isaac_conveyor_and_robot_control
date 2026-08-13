"""Unit tests for sim_cell.robot_placement - derive_station_2_geometry
(pre-existing pure geometry) and zone_geometry_inputs (new, P4's
RunMetadata.zone_geometry extraction, shared by sim_cell.cell and
scripts/export_zone_geometry.py).

zone_geometry_inputs' belt_top_z(zone.belt_prim) call needs a real
Usd.Prim - built here with an in-memory stage via the `usd-core` PyPI
package (plain core USD, NOT Isaac Sim/omni/carb - see tests/conftest.py's
docstring on what this test suite avoids).
"""

from __future__ import annotations

from types import SimpleNamespace

from pxr import Gf, Usd, UsdGeom

from conveyor_indexing.state_machine import HOLD_ZONE_STOP_FRACTION
from sim_cell.robot_placement import belt_top_z, derive_station_2_geometry, zone_geometry_inputs


def _belt_prim(stage: Usd.Stage, path: str, translate_z: float, half_extent_xy: tuple) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(2.0)  # unit cube -> [-1, 1] before scale
    xform = UsdGeom.XformCommonAPI(cube.GetPrim())
    xform.SetTranslate(Gf.Vec3d(0.0, 0.0, translate_z))
    xform.SetScale(Gf.Vec3f(half_extent_xy[0], half_extent_xy[1], 0.05))
    return cube.GetPrim()


def _fake_zone(stage, path, node_path, bbox_center, bbox_half_extent, travel_direction):
    return SimpleNamespace(
        node_path=node_path,
        bbox_center=bbox_center,
        bbox_half_extent=bbox_half_extent,
        belt_prim=_belt_prim(stage, path, translate_z=bbox_center[2], half_extent_xy=bbox_half_extent[:2]),
        world_travel_direction=travel_direction,
    )


def test_zone_geometry_inputs_hold_zone_vs_non_hold_zone():
    stage = Usd.Stage.CreateInMemory()
    zone_0 = _fake_zone(
        stage, "/Belt0", "/World/ConveyorTrack/ConveyorBeltGraph/ConveyorNode",
        bbox_center=(0.0, 0.0, 0.5), bbox_half_extent=(1.0, 0.3, 0.1), travel_direction=Gf.Vec3f(1.0, 0.0, 0.0),
    )
    zone_1 = _fake_zone(
        stage, "/Belt1", "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode",
        bbox_center=(2.0, 0.0, 0.5), bbox_half_extent=(1.0, 0.3, 0.1), travel_direction=Gf.Vec3f(1.0, 0.0, 0.0),
    )

    inputs = zone_geometry_inputs(
        [zone_0, zone_1], speed_m_per_s=1.1, line_id=1, hold_zone_indices=frozenset({1})
    )

    assert len(inputs) == 2
    zero, one = inputs
    assert zero.node_path == "/World/ConveyorTrack/ConveyorBeltGraph/ConveyorNode"
    assert zero.is_hold_zone is False
    assert zero.stop_fraction == 0.0  # non-hold zone: no defined stop point
    assert zero.line_id == 1
    assert zero.speed_m_per_s == 1.1
    assert zero.travel_direction == (1.0, 0.0, 0.0)

    assert one.is_hold_zone is True
    assert one.stop_fraction == HOLD_ZONE_STOP_FRACTION


def test_zone_geometry_inputs_belt_top_z_matches_direct_query():
    stage = Usd.Stage.CreateInMemory()
    zone = _fake_zone(
        stage, "/Belt", "/World/ConveyorTrack_09/ConveyorBeltGraph/ConveyorNode",
        bbox_center=(0.0, 2.1857, 0.9), bbox_half_extent=(1.0, 0.3, 0.5), travel_direction=Gf.Vec3f(0.0, 1.0, 0.0),
    )
    expected_top_z = belt_top_z(zone.belt_prim)

    [geometry] = zone_geometry_inputs([zone], speed_m_per_s=1.0, line_id=2, hold_zone_indices=frozenset())
    assert geometry.belt_top_z == expected_top_z
    assert geometry.line_id == 2


def test_zone_geometry_inputs_curved_zone_falls_back_to_zero_vector():
    """world_travel_direction is None for curved zones (see
    conveyor_indexing.directions.fix_zone_directions) - must not raise, and
    must record (0, 0, 0) rather than crash on None indexing.
    """
    stage = Usd.Stage.CreateInMemory()
    zone = _fake_zone(
        stage, "/Belt", "/World/CurvedZone/ConveyorBeltGraph/ConveyorNode",
        bbox_center=(0.0, 0.0, 0.5), bbox_half_extent=(0.5, 0.5, 0.1), travel_direction=None,
    )
    [geometry] = zone_geometry_inputs([zone], speed_m_per_s=1.0, line_id=1, hold_zone_indices=frozenset())
    assert geometry.travel_direction == (0.0, 0.0, 0.0)


# -- pre-existing function, previously untested ------------------------------


def test_derive_station_2_geometry_reach_balanced_midpoint():
    pick_zone = SimpleNamespace(bbox_center=(0.0, 0.0, 0.5), bbox_half_extent=(0.5, 0.5, 0.1))
    place_zone = SimpleNamespace(bbox_center=(0.0, 2.0, 0.5), bbox_half_extent=(0.5, 0.5, 0.1))
    geometry = derive_station_2_geometry(pick_zone, place_zone)
    assert geometry.robot_position[0] == 0.0
    # Y midpoint between pick zone's far edge (0.5) and place zone's near edge (1.5).
    assert geometry.robot_position[1] == 1.0
    assert geometry.place_xy == (0.0, 2.0)
