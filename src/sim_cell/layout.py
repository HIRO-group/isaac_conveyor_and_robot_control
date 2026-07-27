"""Everything specific to `5_conv_env.usd`'s prim layout: two independent open
(non-looping) lines - loop 1 runs along Y=0, loop 2 along Y~2.186. Loop 2's
far end sits at the near wall of SteelBoxTruck_A01_01; boxes run off the belt
there and drop into the truck bed rather than handing off to another zone.
"""

from __future__ import annotations

import os

STAGE_PATH = os.path.join(os.path.expanduser("~"), "5_conv_env.usd")

ZONE_NODE_PATHS_LOOP1 = [
    "/World/ConveyorTrack/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_01/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_02/ConveyorBeltGraph/ConveyorNode",
]
ZONE_NODE_PATHS_LOOP2 = [
    "/World/ConveyorTrack_09/ConveyorBeltGraph/ConveyorNode",
    "/World/ConveyorTrack_10/ConveyorBeltGraph/ConveyorNode",
]

# ConveyorTrack_01/_09 face each other at local X=-3 - the natural spot for a fixed
# pick/place robot. ConveyorTrack_02 (X=-5) is out of a UR20's reach from there, so
# it's left as an unused upstream buffer.
PICK_ZONE_INDEX = 1  # ConveyorTrack_01 within ZONE_NODE_PATHS_LOOP1
PLACE_ZONE_INDEX = 0  # ConveyorTrack_09 within ZONE_NODE_PATHS_LOOP2

# Second pick/place station, one zone further downstream on each loop - see
# ConveyorLineController.set_hold_zone_ready_check for overflow between them.
PICK_ZONE_INDEX_2 = 2  # ConveyorTrack_02 within ZONE_NODE_PATHS_LOOP1
PLACE_ZONE_INDEX_2 = 1  # ConveyorTrack_10 within ZONE_NODE_PATHS_LOOP2

ROBOT_PATH = "/World/PickPlaceRobot"
PEDESTAL_PATH = "/World/PickPlacePedestal"
ROBOT_PATH_2 = "/World/PickPlaceRobot_02"
PEDESTAL_PATH_2 = "/World/PickPlacePedestal_02"
TRUCK_PATH = "/World/SteelBoxTruck_A01_01"

# The ground plane's collider breaks cuMotion's obstacle scan (recursion bug in
# pick_and_place's np.reshape shim); excluded via extra_exclude_obstacle_paths instead.
GROUND_PLANE_COLLISION_PATH = "/World/GroundPlane/CollisionPlane"

# 5_conv_env.usd ships ~18 pre-placed CubeBox_* prims with no physics schemas of their
# own; discovered at runtime (sim_cell.stage_setup.boxes.discover_box_prim_paths) and
# given physics by apply_box_physics.
BOX_PRIM_NAME_PREFIX = "CubeBox_"

CONVEYOR_TRACK_ROOTS = (
    "/World/ConveyorTrack",
    "/World/ConveyorTrack_01",
    "/World/ConveyorTrack_02",
    "/World/ConveyorTrack_09",
    "/World/ConveyorTrack_10",
)

# Any occupancy hit whose prim path falls under one of these roots is belt/
# structure/robot/truck geometry, not a transported item, and is excluded
# from occupancy detection.
EXCLUDED_STRUCTURE_ROOTS = CONVEYOR_TRACK_ROOTS + (
    ROBOT_PATH,
    PEDESTAL_PATH,
    ROBOT_PATH_2,
    PEDESTAL_PATH_2,
    TRUCK_PATH,
)
