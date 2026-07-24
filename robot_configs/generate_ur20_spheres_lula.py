"""Step 2a of UR20 cuMotion config generation: run Lula's own built-in
collision-sphere generator (lula.create_collision_sphere_generator, the same
algorithm isaacsim.robot_setup.xrdf_editor's CollisionSphereEditor.generate_spheres()
uses) on each UR20 link's collision mesh, producing per-link {centers, radii}
JSON files later consumed by generate_ur20_xrdf.py.

Supersedes generate_ur20_spheres_morphit.py: visually sanity-checked via
visualize_ur20_collision_spheres.py (spheres rendered translucent over the
robot mesh) and MorphIt-V's output - trained here with the library's own
generic default hyperparameters, no UR20/MorphIt-specific tuning - visibly
bulged well outside the real link geometry at several joints, most visibly at
the shoulder/wrist. Lula's generator is a deterministic geometric sphere-
packing algorithm (not a trained model), and is what actually produced the
bundled UR10/Franka robot.xrdf configs this project has been mirroring the
sphere-count convention from, so it should track the real mesh surface
directly with no training/convergence to verify by eye.

Operates on the same exported collision-mesh OBJ files
generate_ur20_spheres_morphit.py used (robot_configs/ur20/meshes/mesh_*.obj)
rather than the live USD stage's meshes, since the bundled ur20.usd marks all
visual/collision meshes Instanceable - confirmed via
isaacsim.robot_setup.xrdf_editor.sphere_generation.find_link_meshes(), which
explicitly cannot generate spheres for instanceable meshes (logs a warning
and returns an empty mesh list per link). Loading the same static OBJ files
directly with trimesh and feeding vertices/faces straight into
lula.create_collision_sphere_generator() sidesteps that entirely - no live
stage or SimulationApp needed for this step at all.

Per-link sphere counts mirror UR10's own bundled counts (same kinematic
topology, just physically larger) - unchanged from generate_ur20_spheres_morphit.py,
see that file's docstring for the full rationale:
    shoulder_link: 2, upper_arm_link: 12, forearm_link: 15, wrist_1_link: 2,
    wrist_2_link: 7

radius_offset=0.0 (no padding, no shrink) - the tightest fit the generator
supports, matching the goal of letting the robot approach real obstacles
closely (see pick_and_place.py's obstacle-representation tightening) rather
than baking extra clearance into the robot's own collision geometry.

Run with the isolated lula PYTHONPATH (needs the compiled lula module, not
otherwise on Isaac Sim's default path - same requirement as
generate_ur20_xrdf.py) - no SimulationApp/Isaac Sim runtime startup required:

    PYTHONPATH=/home/ubuntu/IsaacSim-source/_build/target-deps/isaac_lula_prebundle \
    /home/ubuntu/IsaacSim/python.sh /home/ubuntu/conveyor_indexing/robot_configs/generate_ur20_spheres_lula.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lula
import numpy as np
import trimesh

MESH_DIR = Path("/home/ubuntu/conveyor_indexing/robot_configs/ur20/meshes")
OUTPUT_DIR = Path("/home/ubuntu/conveyor_indexing/robot_configs/ur20/lula_spheres")

# link name -> (collision mesh filename, num_spheres) - same mapping as
# generate_ur20_spheres_morphit.py.
LINKS = {
    "shoulder_link": ("mesh_15.obj", 2),
    "upper_arm_link": ("mesh_16.obj", 12),
    "forearm_link": ("mesh_17.obj", 15),
    "wrist_1_link": ("mesh_18.obj", 2),
    "wrist_2_link": ("mesh_19.obj", 7),
}

RADIUS_OFFSET = 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for link_name, (mesh_filename, num_spheres) in LINKS.items():
        mesh_path = MESH_DIR / mesh_filename
        if not mesh_path.exists():
            raise RuntimeError(f"Expected collision mesh not found: {mesh_path}")

        mesh = trimesh.load(str(mesh_path), force="mesh")
        points = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.faces, dtype=np.int32)

        generator = lula.create_collision_sphere_generator(points, triangles)
        spheres = generator.generate_spheres(num_spheres, RADIUS_OFFSET)

        centers = [s.center.tolist() for s in spheres]
        radii = [float(s.radius) for s in spheres]

        output_path = OUTPUT_DIR / f"{link_name}.json"
        with open(output_path, "w") as f:
            json.dump({"centers": centers, "radii": radii}, f, indent=4)

        print(
            f"[generate_ur20_spheres_lula] {link_name}: {mesh_filename} -> "
            f"{len(spheres)} spheres (requested {num_spheres}) -> {output_path}",
            flush=True,
        )

    print("[generate_ur20_spheres_lula] done", flush=True)


if __name__ == "__main__":
    main()
