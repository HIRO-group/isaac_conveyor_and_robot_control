"""Run-time tuning for this cell: control rate, physics rate, per-line speeds,
robot placement, and logging output location.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../conveyor_indexing/src/sim_cell/settings.py -> .../conveyor_indexing
REPO_ROOT = Path(__file__).resolve().parents[2]
# Overridable so N parallel sim instances (e.g. one per Vertex AI job) can each
# write to their own directory instead of colliding on a shared checkout's
# data/ - see sim_cell.recording, which derives the recordings/mcap subdirs
# from this, and conveyor_indexing.parquet_logger's tick log.
LOG_OUTPUT_DIR = os.environ.get("CONVEYOR_INDEXING_DATA_DIR", str(REPO_ROOT / "data"))

CONTROL_HZ = 120.0  # matches physics rate; 30Hz let boxes drift past hold points before the belt reacted
PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 60.0

# Debug escape hatch for pick_and_place.motion_planner.build_motion_planner - skips
# the AABB obstacle scan entirely (planner sees an empty world). Off in normal use.
DISABLE_OBSTACLE_TRACKING = False

# Slowed for a comfortable pick cadence. A bigger global slowdown (~1.0 m/s) previously
# stalled the arm's no-timeout ATTACH phase indefinitely - not fully root-caused.
LOOP1_RUN_SPEED_PCT = 55

# Tuned so boxes land inside the truck bed instead of overshooting it (at full speed
# they cleared the truck's far wall and landed on the ground beyond it).
LOOP2_RUN_SPEED_PCT = 50

# Both loops already sit close enough for the UR20 (1.75m reach) at the Y midpoint
# without any runtime repositioning.
ROBOT_POSITION = (-3.0, 1.0928, 0.0)  # (x, y, z-of-ground-contact); Y = loop midpoint
PEDESTAL_HEIGHT = 1.6
PLACE_XY = (-3.0, 2.1857)  # ConveyorTrack_09's belt-top Y center

# All 5 conveyor zones share the same real belt-top Z (confirmed via
# scripts/export_zone_geometry.py, 2026-09-02: every zone's belt_top_z ~=
# 1.7805m). A box knocked off any belt onto the floor falls well below this
# (observed z in [-0.0, 0.21] for grounded boxes vs. 1.78 for belt-resting
# ones) - this threshold gates sim_cell.stage_setup.truck.despawn_boxes_
# below_floor, recycling a permanently-grounded box back into the spawner
# pool the same way a truck-landed one is, instead of leaving it on the
# floor forever shrinking the usable box pool for any client.
#
# CRITICAL FIX (2026-09-02, capability-diffusion's Stage 7c part 8/9
# investigation): 1.0 was ABOVE the truck bed's own real top surface -
# confirmed via a one-off prepare_stage() AABB dump: truck_bed_max z =
# 0.950 (truck_bed_min z = 0.265). Since despawn_boxes_in_truck and
# despawn_boxes_below_floor both run every tick on the same box_positions,
# and a box legitimately falling from belt height (1.78) into the truck
# bed passes through z<1.0 WHILE STILL MID-AIR, well before its (x, y) has
# traveled far enough to satisfy despawn_boxes_in_truck's tighter AABB
# check - the OLD threshold caught every genuine truck-bound box first and
# recycled it as a "floor despawn," so despawn_boxes_in_truck's own check
# never got a chance to fire. This silently produced a real, measured 0%
# truck-delivery rate for an entire investigation (capability-diffusion's
# docs/progress-tracker.md "part 5" through "part 8") even under fully
# nominal conditions with a correctly-placed box - not a pick/place/grip
# bug at all, a floor-threshold misconfiguration undermining every
# genuine delivery. 0.24 sits strictly between the observed genuine-
# ground-drop max (0.21) and the truck bed's own floor (0.265), so a real
# floor-grounded box is still caught while a box settling into the truck
# bed range is not.
FLOOR_Z_THRESHOLD = 0.24

# Camera rig tuning (see src/cameras/, sim_cell.camera_layout). 640x480@30 RGB8
# matches theia's production default camera config (~theia/infra/etcd/bootstrap/
# seed/defaults.json's camera.config.default), so sim data looks like real data.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Overhead cameras look straight down (Z-up stage, unrotated camera already
# looks down -Z); this height plus an 18mm focal length (~60deg horizontal FOV
# at a 20.955mm aperture, see cameras.rig.HORIZONTAL_APERTURE_MM) covers the
# ~2m-long belt zone with margin on both sides.
CAMERA_HEIGHT_ABOVE_BELT_M = 2.5

# Local offset from the UR20 flange (see layout.HAND_CAM_PARENT) - just enough
# to clear the flange's own mesh so the camera doesn't render its own housing.
HAND_CAM_OFFSET = (0.0, 0.0, 0.10)
