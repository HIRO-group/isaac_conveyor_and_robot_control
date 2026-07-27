"""Opens `5_conv_env.usd` and prepares it for simulation: enables the conveyor
extension, localizes assets, deactivates frame meshes, adds truck/box physics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import omni.kit.app
import omni.usd
from pxr import Gf, Usd

from sim_cell import layout
from sim_cell.assets import localize_asset_references
from sim_cell.stage_setup.boxes import apply_box_physics, discover_box_prim_paths, resolve_box_overlaps
from sim_cell.stage_setup.tracks import deactivate_frame_meshes
from sim_cell.stage_setup.truck import apply_truck_collision, truck_body_world_bounds

logger = logging.getLogger(__name__)


@dataclass
class StagePrep:
    stage: Usd.Stage
    box_paths: list
    truck_bed_min: Gf.Vec3d
    truck_bed_max: Gf.Vec3d


def prepare_stage() -> StagePrep:
    # isaacsim.asset.gen.conveyor isn't guaranteed enabled by the resolved app config;
    # enable it explicitly before the stage (and its ConveyorBeltGraphs) is opened.
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.asset.gen.conveyor", True
    )

    # Open the stage before constructing World: World() attaches at construction time,
    # and opening after leaves it referencing a stale, now-gone stage.
    localize_asset_references(layout.STAGE_PATH)
    logger.info("opening stage %s", layout.STAGE_PATH)
    ctx = omni.usd.get_context()
    ctx.open_stage(layout.STAGE_PATH)
    stage = ctx.get_stage()
    deactivate_frame_meshes(stage, layout.CONVEYOR_TRACK_ROOTS)
    apply_truck_collision(stage, layout.TRUCK_PATH)
    truck_bed_min, truck_bed_max = truck_body_world_bounds(stage, layout.TRUCK_PATH)
    box_paths = discover_box_prim_paths(stage, layout.BOX_PRIM_NAME_PREFIX)
    if not box_paths:
        raise RuntimeError(
            f"No prims named '{layout.BOX_PRIM_NAME_PREFIX}*' found in {layout.STAGE_PATH} - "
            "expected the pre-authored box pallet on ConveyorTrack"
        )
    resolve_box_overlaps(stage, box_paths)
    apply_box_physics(stage, box_paths)
    logger.info("stage prepared, %d boxes discovered", len(box_paths))
    return StagePrep(stage=stage, box_paths=box_paths, truck_bed_min=truck_bed_min, truck_bed_max=truck_bed_max)
