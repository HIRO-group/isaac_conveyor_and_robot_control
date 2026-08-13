"""Unit tests for scripts/export_zone_geometry.py's pure JSON-shaping logic
(P7). The script's main() needs Isaac Sim (opens a real USD stage) - not
exercised here; zone_geometry_to_dict/build_export are standalone pure
functions specifically so this part is testable without it (see
tests/conftest.py, which puts scripts/ on sys.path).
"""

from __future__ import annotations

from types import SimpleNamespace

import export_zone_geometry as ezg


def _zone(**overrides):
    defaults = {
        "node_path": "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode",
        "bbox_center": (1.0, 2.0, 3.0),
        "bbox_half_extent": (0.5, 0.25, 0.1),
        "belt_top_z": 0.9,
        "travel_direction": (1.0, 0.0, 0.0),
        "stop_fraction": 0.8,
        "speed_m_per_s": 1.1,
        "is_hold_zone": True,
        "line_id": 1,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_zone_geometry_to_dict_shape_matches_zonegeometry_proto_fields():
    d = ezg.zone_geometry_to_dict(_zone())
    assert d["node_path"] == "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode"
    assert d["aabb"]["min"] == [0.5, 1.75, 2.9]
    assert d["aabb"]["max"] == [1.5, 2.25, 3.1]
    assert d["belt_top_z"] == 0.9
    assert d["travel_direction"] == [1.0, 0.0, 0.0]
    assert d["stop_fraction"] == 0.8
    assert d["speed_m_per_s"] == 1.1
    assert d["is_hold_zone"] is True
    assert d["line_id"] == 1


def test_build_export_keys_by_git_sha():
    zones = [_zone(node_path="/A", line_id=1), _zone(node_path="/B", line_id=2)]
    export = ezg.build_export(zones, git_sha="abc123def456")
    assert export["conveyor_indexing_git_sha"] == "abc123def456"
    assert len(export["zones"]) == 2
    assert [z["node_path"] for z in export["zones"]] == ["/A", "/B"]


def test_build_export_is_json_serializable(tmp_path):
    import json

    export = ezg.build_export([_zone()], git_sha="sha")
    out_file = tmp_path / "zone_geometry.json"
    out_file.write_text(json.dumps(export, indent=2))
    round_tripped = json.loads(out_file.read_text())
    assert round_tripped == export
