"""Run-time tuning: rates, output location, and cell values re-exported from the scene."""

from __future__ import annotations

import os
from pathlib import Path

from sim_cell.scene import get_scene

REPO_ROOT = Path(__file__).resolve().parents[2]
# One directory per sim instance; parallel runs must not share it.
LOG_OUTPUT_DIR = os.environ.get("CONVEYOR_INDEXING_DATA_DIR", str(REPO_ROOT / "data"))

CONTROL_HZ = 120.0  # matches physics rate; 30Hz let boxes drift past hold points
PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 60.0

# Debug escape hatch: skip the planner's obstacle scan.
DISABLE_OBSTACLE_TRACKING = False

_scene = get_scene()

LOOP1_RUN_SPEED_PCT = _scene.loops[0].run_speed_pct
LOOP2_RUN_SPEED_PCT = _scene.loops[1].run_speed_pct

ROBOT_POSITION = _scene.robot.position
PEDESTAL_HEIGHT = _scene.robot.pedestal_height
PLACE_XY = _scene.robot.place_xy

# Box despawn thresholds; see the scene package for the belt-height reasoning.
FLOOR_Z_THRESHOLD = _scene.thresholds.floor_z
OFF_BELT_Z_THRESHOLD = _scene.thresholds.off_belt_z
OFF_BELT_TRUCK_XY_MARGIN_M = _scene.thresholds.off_belt_truck_xy_margin_m

CAMERA_WIDTH = _scene.camera.width
CAMERA_HEIGHT = _scene.camera.height
CAMERA_FPS = _scene.camera.fps
CAMERA_HEIGHT_ABOVE_BELT_M = _scene.camera.height_above_belt_m
HAND_CAM_OFFSET = _scene.camera.hand_cam_offset
