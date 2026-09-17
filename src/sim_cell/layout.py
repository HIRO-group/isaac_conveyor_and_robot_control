"""Prim-layout view of the loaded scene (`sim_cell.scene`).

Module-level names are kept for the Isaac-side callers; the values come from
the scene package. New code should read `get_scene()` directly.
"""

from __future__ import annotations

from sim_cell.scene import get_scene

_scene = get_scene()
_scene.require_shape(loops=2, stations=2)

STAGE_PATH = str(_scene.stage_path)
CAMERA_POSES_PATH = str(_scene.camera_poses_path)

ZONE_NODE_PATHS_LOOP1 = list(_scene.loops[0].zones)
ZONE_NODE_PATHS_LOOP2 = list(_scene.loops[1].zones)

_station_1, _station_2 = _scene.stations
PICK_ZONE_INDEX = _station_1.pick_zone.zone
PLACE_ZONE_INDEX = _station_1.place_zone.zone
PICK_ZONE_INDEX_2 = _station_2.pick_zone.zone
PLACE_ZONE_INDEX_2 = _station_2.place_zone.zone

ROBOT_PATH = _station_1.robot_path
PEDESTAL_PATH = _station_1.pedestal_path
ROBOT_PATH_2 = _station_2.robot_path
PEDESTAL_PATH_2 = _station_2.pedestal_path
HAND_CAM_PARENT = _station_1.hand_cam_parent
HAND_CAM_PARENT_2 = _station_2.hand_cam_parent

TRUCK_PATH = _scene.truck_path
CAMERA_ROOT_PATH = _scene.camera_root_path
GROUND_PLANE_COLLISION_PATH = _scene.ground_plane_collision_path
BOX_PRIM_NAME_PREFIX = _scene.box_prim_name_prefix
CONVEYOR_TRACK_ROOTS = _scene.conveyor_track_roots
EXCLUDED_STRUCTURE_ROOTS = _scene.excluded_structure_roots
