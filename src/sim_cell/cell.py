"""Builds the whole cell: the World, both conveyor lines, both robots, both
pick-and-place controllers, wired together for `5_conv_env.usd`.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass

from isaacsim.core.api import World
from isaacsim.core.experimental.prims import Articulation, RigidPrim

from cameras.rig import CameraRig
from cameras.specs import build_camera_list
from cameras.zenoh_publisher import CameraZenohPublisher
from conveyor_indexing.line_controller import ConveyorLineController
from conveyor_indexing.parquet_logger import ConveyorIndexingLogger
from pick_and_place import UR20_PRE_PLACE_JOINT_POSITIONS_AWAY, MagicAttachPickPlace, create_pedestal_and_robot
from sim_cell import layout, settings
from sim_cell.box_spawner import BoxSpawner
from sim_cell.camera_layout import build_camera_specs
from sim_cell.camera_tuning import maybe_enable_camera_tuning
from sim_cell.robot_placement import belt_top_z, derive_station_2_geometry
from sim_cell.stage_setup import StagePrep

logger = logging.getLogger(__name__)


@dataclass
class Cell:
    world: World
    loop1: ConveyorLineController
    loop2: ConveyorLineController
    robot: Articulation
    robot2: Articulation
    pick_place: MagicAttachPickPlace
    pick_place_2: MagicAttachPickPlace
    tick_logger: ConveyorIndexingLogger
    camera_rig: CameraRig
    camera_publisher: CameraZenohPublisher
    box_rigid_prims: dict
    box_paths_ordered: list
    box_positions_view: RigidPrim
    truck_bed_min: tuple
    truck_bed_max: tuple
    robot_xy: tuple
    robot_2_xy: tuple
    spawner: BoxSpawner


def build_cell(stage_prep: StagePrep) -> Cell:
    stage = stage_prep.stage
    world = World(physics_dt=settings.PHYSICS_DT, rendering_dt=settings.RENDERING_DT, stage_units_in_meters=1.0)
    logger.info("World constructed, building logger + zones")

    # Pre-create the log directory - on a clean checkout, ConveyorIndexingLogger's
    # exists-at-call-time heuristic otherwise mistakes it for a file stem.
    pathlib.Path(settings.LOG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    tick_logger = ConveyorIndexingLogger(settings.LOG_OUTPUT_DIR)

    loop1 = ConveyorLineController(
        stage,
        layout.ZONE_NODE_PATHS_LOOP1,
        layout.EXCLUDED_STRUCTURE_ROOTS,
        hold_zone_indices=frozenset({layout.PICK_ZONE_INDEX, layout.PICK_ZONE_INDEX_2}),
        closed_loop=False,
        run_speed_pct=settings.LOOP1_RUN_SPEED_PCT,
    )
    loop2 = ConveyorLineController(
        stage,
        layout.ZONE_NODE_PATHS_LOOP2,
        layout.EXCLUDED_STRUCTURE_ROOTS,
        closed_loop=False,
        run_speed_pct=settings.LOOP2_RUN_SPEED_PCT,
    )
    logger.info("zones built, %d boxes discovered, creating pedestal + robot", len(stage_prep.box_paths))

    robot = create_pedestal_and_robot(
        stage,
        robot_path=layout.ROBOT_PATH,
        pedestal_path=layout.PEDESTAL_PATH,
        position=settings.ROBOT_POSITION,
        pedestal_height=settings.PEDESTAL_HEIGHT,
    )

    # Reach-balanced position for the second robot, derived from actual zone geometry
    # rather than hardcoded - ConveyorTrack_02/_10 aren't guaranteed to line up in X.
    pick_zone_2 = loop1.zones[layout.PICK_ZONE_INDEX_2]
    place_zone_2 = loop2.zones[layout.PLACE_ZONE_INDEX_2]
    station_2 = derive_station_2_geometry(pick_zone_2, place_zone_2)

    robot2 = create_pedestal_and_robot(
        stage,
        robot_path=layout.ROBOT_PATH_2,
        pedestal_path=layout.PEDESTAL_PATH_2,
        position=station_2.robot_position,
        pedestal_height=settings.PEDESTAL_HEIGHT,
    )

    # Place target Z depends on box height, computed per-cycle inside MagicAttachPickPlace;
    # only the belt-top Z is fixed here.
    place_belt_prim = loop2.zones[layout.PLACE_ZONE_INDEX].belt_prim
    place_belt_top_z = belt_top_z(place_belt_prim)
    place_belt_top_z_2 = belt_top_z(place_zone_2.belt_prim)

    box_rigid_prims = {path: RigidPrim(path) for path in stage_prep.box_paths}
    # Single batched view over every box for position reads - the per-box RigidPrims
    # above stay in use for the writes (despawn, magic attach) that only ever touch
    # one box at a time; reads are the hot path (evaluate_pick_station,
    # ConveyorLineController.step, despawn_boxes_in_truck all need every box's
    # position every control tick) so those go through one GPU->host transfer here
    # instead of one per box per call site.
    box_paths_ordered = stage_prep.box_paths
    box_positions_view = RigidPrim(box_paths_ordered)
    # Only loop1 has hold zones needing is_past_center() checks against a real box position.
    loop1.set_box_rigid_prims(box_rigid_prims)
    logger.info("robots ready, calling world.reset()")

    world.reset()
    # World.reset() has no knowledge of isaacsim.core.experimental prims like this
    # robot and never calls reset_to_default_state() on them - must be triggered explicitly.
    robot.reset_to_default_state()
    robot2.reset_to_default_state()
    check_pos, _ = robot.get_world_poses()
    logger.debug("robot base pose AFTER world.reset(): %s", check_pos.numpy())
    logger.debug("robot dof_positions AFTER reset_to_default_state(): %s", robot.get_dof_positions().numpy())
    logger.info("world.reset() done")

    # Randomizes what's on ConveyorTrack (loop1 zone 0) between training runs - see
    # sim_cell.box_spawner. Built after world.reset() since it writes through the
    # same RigidPrim tensor views as everything else here.
    spawner = BoxSpawner(loop1.zones[0], box_rigid_prims, stage_prep.pool)

    # MagicAttachPickPlace builds the cuMotion RmpFlowController, which needs a valid
    # PhysX tensor entity - must happen after world.reset().
    # pre_place_joint_positions override: robot 2 sits on this robot's -X side, so its
    # pick<->place swing must arc away from it - see UR20_PRE_PLACE_JOINT_POSITIONS_AWAY.
    pick_place = MagicAttachPickPlace(
        robot=robot,
        robot_path=layout.ROBOT_PATH,
        place_xy=settings.PLACE_XY,
        place_belt_top_z=place_belt_top_z,
        box_rigid_prims=box_rigid_prims,
        physics_dt=world.get_physics_dt(),
        get_pick_zone_occupant_paths=loop1.zones[layout.PICK_ZONE_INDEX].get_occupying_prim_paths,
        extra_exclude_obstacle_paths=[layout.GROUND_PLANE_COLLISION_PATH],
        pre_place_joint_positions=UR20_PRE_PLACE_JOINT_POSITIONS_AWAY,
        disable_obstacle_tracking=settings.DISABLE_OBSTACLE_TRACKING,
    )
    pick_place_2 = MagicAttachPickPlace(
        robot=robot2,
        robot_path=layout.ROBOT_PATH_2,
        place_xy=station_2.place_xy,
        place_belt_top_z=place_belt_top_z_2,
        box_rigid_prims=box_rigid_prims,
        physics_dt=world.get_physics_dt(),
        get_pick_zone_occupant_paths=loop1.zones[layout.PICK_ZONE_INDEX_2].get_occupying_prim_paths,
        extra_exclude_obstacle_paths=[layout.GROUND_PLANE_COLLISION_PATH],
        disable_obstacle_tracking=settings.DISABLE_OBSTACLE_TRACKING,
    )
    loop1.set_hold_zone_ready_check(layout.PICK_ZONE_INDEX, lambda: pick_place.phase_name == "WAITING")
    loop1.set_hold_zone_ready_check(layout.PICK_ZONE_INDEX_2, lambda: pick_place_2.phase_name == "WAITING")
    logger.info("pick/place controllers ready")

    # After world.reset()/robot resets - camera prims (esp. hand cams, parented
    # under the flange) need the referenced robot geometry already in place.
    # Missing eclipse-zenoh (see cameras.zenoh_publisher) fails here, before
    # the main loop, rather than mid-run.
    camera_specs = build_camera_specs(loop1, loop2)
    camera_rig = CameraRig(stage, camera_specs)
    camera_publisher = CameraZenohPublisher(build_camera_list(camera_specs))
    maybe_enable_camera_tuning(stage, camera_specs)

    return Cell(
        world=world,
        loop1=loop1,
        loop2=loop2,
        robot=robot,
        robot2=robot2,
        pick_place=pick_place,
        pick_place_2=pick_place_2,
        tick_logger=tick_logger,
        camera_rig=camera_rig,
        camera_publisher=camera_publisher,
        box_rigid_prims=box_rigid_prims,
        box_paths_ordered=box_paths_ordered,
        box_positions_view=box_positions_view,
        truck_bed_min=stage_prep.truck_bed_min,
        truck_bed_max=stage_prep.truck_bed_max,
        robot_xy=settings.ROBOT_POSITION[:2],
        robot_2_xy=station_2.robot_position[:2],
        spawner=spawner,
    )
