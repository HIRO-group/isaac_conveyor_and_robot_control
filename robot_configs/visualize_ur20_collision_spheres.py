"""Standalone sanity-check viewer: overlay the UR20's actual RMPflow collision
spheres (Lula-generated, lula_spheres/*.json -> robot.xrdf via
generate_ur20_xrdf.py) directly on the robot mesh, so they can be visually
compared against the real arm geometry.

Only covers shoulder_link/upper_arm_link/forearm_link/wrist_1_link/
wrist_2_link - the 5 links generate_ur20_spheres_lula.py generates spheres
for (see that file's docstring for why base_link/wrist_3_link aren't
included: they're covered instead by rmp_flow.yaml's own body_capsules/
body_collision_controllers, a separate mechanism not visualized here).

Spawns each sphere as a translucent child prim of its link's real prim path
(same link-path resolution convention as pick_and_place.py's
TOOL_FRAME_LIVE_PRIM_SUBPATH lookup), so they move with the arm. Applies the
same MIN_SPHERE_RADIUS_M filter as generate_ur20_xrdf.py so what's shown here
matches what's actually in robot.xrdf, not raw unfiltered generator output.

Run with (opens a viewer window - not headless, this is for visual
inspection):
    /home/ubuntu/IsaacSim/python.sh /home/ubuntu/conveyor_indexing/robot_configs/visualize_ur20_collision_spheres.py
"""

from __future__ import annotations

import json
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
import omni.usd
from omni.kit.viewport.utility import frame_viewport_prims
from isaacsim.core.experimental.materials import OmniPbrMaterial
from isaacsim.core.experimental.objects import DistantLight, GroundPlane, Sphere
from isaacsim.core.experimental.prims import Articulation
from isaacsim.storage.native import get_assets_root_path

import isaacsim.core.experimental.utils.stage as stage_utils

UR20_USD_PATH = "/Isaac/Robots/UniversalRobots/ur20/ur20.usd"
ROBOT_PRIM_PATH = "/ur20"
CONFIG_DIR = Path("/home/ubuntu/conveyor_indexing/robot_configs/ur20")
LULA_SPHERES_DIR = CONFIG_DIR / "lula_spheres"

# Same mapping and filter as generate_ur20_xrdf.py, so this shows exactly
# what ended up in robot.xrdf (not raw unfiltered generator output).
LULA_LINK_FILES = {
    "shoulder_link": "shoulder_link.json",
    "upper_arm_link": "upper_arm_link.json",
    "forearm_link": "forearm_link.json",
    "wrist_1_link": "wrist_1_link.json",
    "wrist_2_link": "wrist_2_link.json",
}
MIN_SPHERE_RADIUS_M = 0.005

# Same "ready" pose used elsewhere (pick_and_place.py's
# UR20_DEFAULT_JOINT_POSITIONS) - not load-bearing for this sanity check, just
# a natural, non-singular pose to inspect the spheres in.
DEFAULT_JOINT_POSITIONS = [2.583766, -0.523898, -0.007470, -0.872193, 1.125949, 0.148515]

SPHERE_COLOR = (0.1, 0.6, 1.0)
SPHERE_OPACITY = 0.4


def main() -> None:
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not resolve Isaac Sim assets root path")

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage_utils.add_reference_to_stage(usd_path=assets_root + UR20_USD_PATH, path=ROBOT_PRIM_PATH)
    GroundPlane(paths="/World/GroundPlane")
    DistantLight(paths="/World/DistantLight")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    robot = Articulation(ROBOT_PRIM_PATH)
    robot.set_dof_position_targets(positions=DEFAULT_JOINT_POSITIONS)
    for _ in range(60):
        simulation_app.update()

    material = OmniPbrMaterial(paths="/CollisionSphereViz/Material")
    material.set_input_values("diffuse_color_constant", list(SPHERE_COLOR))
    material.set_input_values("enable_opacity", [True])
    material.set_input_values("opacity_constant", [SPHERE_OPACITY])

    link_names = list(robot.link_names)
    total_spheres = 0
    for link_name, filename in LULA_LINK_FILES.items():
        json_path = LULA_SPHERES_DIR / filename
        if not json_path.exists():
            raise RuntimeError(f"Missing Lula sphere output for {link_name}: {json_path}")
        with open(json_path) as f:
            data = json.load(f)

        link_path = robot.link_paths[0][link_names.index(link_name)]

        added = 0
        for i, (center, radius) in enumerate(zip(data["centers"], data["radii"])):
            if radius < MIN_SPHERE_RADIUS_M:
                continue
            sphere_path = f"{link_path}/CollisionSphereViz_{i}"
            sphere = Sphere(paths=sphere_path, radii=radius, translations=center)
            sphere.apply_visual_materials(material)
            added += 1
        print(f"[visualize_ur20_collision_spheres] {link_name}: {added} spheres from {filename}", flush=True)
        total_spheres += added

    print(f"[visualize_ur20_collision_spheres] {total_spheres} spheres total - close the viewer window to exit", flush=True)

    # The default camera on a fresh stage isn't pointed at the robot at all -
    # frame it explicitly rather than leaving the viewer on an empty grid.
    frame_viewport_prims(prims=[ROBOT_PRIM_PATH])

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
