"""Run-time tuning for this cell: control rate, physics rate, per-line speeds,
robot placement, and logging output location.
"""

from __future__ import annotations

from pathlib import Path

# .../conveyor_indexing/src/sim_cell/settings.py -> .../conveyor_indexing
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_OUTPUT_DIR = str(REPO_ROOT / "data")

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
