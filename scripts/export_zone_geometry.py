"""One-time dump of conveyor zone AABB/geometry constants (`sim_cell.layout`
+ the actual USD stage) into a committed JSON file, keyed by the current sim
git sha.

This is a checked-in *reference snapshot* of zone geometry, independent of
any particular recorded run - contrast with `proto/sim_state.proto`'s
`RunMetadata.zone_geometry`, which every mcap capture already carries
per-run (see `sim_cell.robot_placement.zone_geometry_inputs` /
`sim_cell.recording.build_zone_geometry_proto`, the exact same extraction
this script reuses). Useful for quick sanity checks or feeding eval tooling
before any run/mcap exists, and as a regression check on zone geometry
drift across `environments/5_conv_env.usd` edits - same committed-sidecar
convention as `environments/camera_poses.json`.

Run with Isaac Sim's bundled python (needs the conveyor extension + a real
opened USD stage, same as scripts/download_assets.py):

    ~/IsaacSim/python.sh ~/conveyor_indexing/scripts/export_zone_geometry.py [--out FILE.json]

Writes to environments/zone_geometry.json by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# This script (like scripts/download_assets.py) is invoked directly, not via
# scripts/run.sh, so it puts src/ on sys.path itself.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

DEFAULT_OUT = os.path.join(_REPO_ROOT, "environments", "zone_geometry.json")


def zone_geometry_to_dict(zone) -> dict:
    """One `sim_cell.recording.ZoneGeometryInput` -> a plain
    JSON-serializable dict, field-for-field matching
    `proto/sim_state.proto`'s `ZoneGeometry` message (see
    `sim_cell.recording.build_zone_geometry_proto`). Kept as a standalone
    pure function (no Isaac Sim / protobuf dependency) so this script's
    actual output shape is unit-testable without Isaac Sim - see
    tests/scripts/test_export_zone_geometry.py.
    """
    cx, cy, cz = zone.bbox_center
    hx, hy, hz = zone.bbox_half_extent
    return {
        "node_path": zone.node_path,
        "aabb": {
            "min": [cx - hx, cy - hy, cz - hz],
            "max": [cx + hx, cy + hy, cz + hz],
        },
        "belt_top_z": zone.belt_top_z,
        "travel_direction": list(zone.travel_direction),
        "stop_fraction": zone.stop_fraction,
        "speed_m_per_s": zone.speed_m_per_s,
        "is_hold_zone": zone.is_hold_zone,
        "line_id": zone.line_id,
    }


def build_export(zone_geometry: list, git_sha: str) -> dict:
    """{conveyor_indexing_git_sha, zones: [...]} - this script's committed
    JSON's top-level shape, keyed by the sim git sha so a stale snapshot is
    always identifiable against the checkout that produced it.
    """
    return {
        "conveyor_indexing_git_sha": git_sha,
        "zones": [zone_geometry_to_dict(z) for z in zone_geometry],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output JSON path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    # SimulationApp must be constructed before any omni.*/carb/isaacsim/
    # cumotion import - deferred until here, inside main(), rather than
    # module scope (unlike download_assets.py) so zone_geometry_to_dict/
    # build_export above stay importable - and unit-testable - without
    # Isaac Sim; same deferred-import convention scripts/run_conveyor_indexing.py
    # already uses for the same reason.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    from conveyor_indexing.line_controller import ConveyorLineController
    from conveyor_indexing.mcap_recorder import git_sha
    from conveyor_indexing.zone import ZONE_RUN_VELOCITY
    from sim_cell import layout, settings
    from sim_cell.robot_placement import zone_geometry_inputs
    from sim_cell.stage_setup import prepare_stage

    # Only the stage + zones are needed for geometry - no robots/cuMotion/
    # world.reset() (see sim_cell.cell.build_cell for the full cell, not
    # needed here): ConveyorZone's bbox/belt-top-z queries and
    # fix_zone_directions's travel-direction correction are pure USD bbox/
    # attribute reads, independent of physics.
    stage_prep = prepare_stage()
    loop1 = ConveyorLineController(
        stage_prep.stage,
        layout.ZONE_NODE_PATHS_LOOP1,
        layout.EXCLUDED_STRUCTURE_ROOTS,
        hold_zone_indices=frozenset({layout.PICK_ZONE_INDEX, layout.PICK_ZONE_INDEX_2}),
        closed_loop=False,
        run_speed_pct=settings.LOOP1_RUN_SPEED_PCT,
    )
    loop2 = ConveyorLineController(
        stage_prep.stage,
        layout.ZONE_NODE_PATHS_LOOP2,
        layout.EXCLUDED_STRUCTURE_ROOTS,
        closed_loop=False,
        run_speed_pct=settings.LOOP2_RUN_SPEED_PCT,
    )

    zone_geometry = zone_geometry_inputs(
        loop1.zones,
        ZONE_RUN_VELOCITY * settings.LOOP1_RUN_SPEED_PCT / 100.0,
        1,
        frozenset({layout.PICK_ZONE_INDEX, layout.PICK_ZONE_INDEX_2}),
    ) + zone_geometry_inputs(
        loop2.zones,
        ZONE_RUN_VELOCITY * settings.LOOP2_RUN_SPEED_PCT / 100.0,
        2,
        frozenset({layout.PLACE_ZONE_INDEX, layout.PLACE_ZONE_INDEX_2}),
    )

    export = build_export(zone_geometry, git_sha(settings.REPO_ROOT))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(export, f, indent=2)
        f.write("\n")
    print(f"wrote {len(zone_geometry)} zone(s) to {args.out} (sha={export['conveyor_indexing_git_sha']})")

    simulation_app.close()


if __name__ == "__main__":
    main()
