"""Step 2a of UR20 cuMotion config generation: run MorphIt (gradient-based
sphere-packing optimizer, https://github.com/HIRO-group/MorphIt-1) on each
UR20 link's collision mesh, producing per-link {centers, radii} JSON files
later consumed by generate_ur20_xrdf.py to populate the actual XRDF/Lula
collision-sphere entries (via CollisionSphereEditor.add_sphere(), bypassing
Lula's own simpler sphere generator).

Mirrors UR10's own collision-sphere convention (read from the bundled
robot_configurations/ur10/robot.xrdf): only upper_arm_link, forearm_link,
wrist_1_link, wrist_2_link get spheres in the main obstacle-avoidance group,
plus shoulder_link in a self-collision-only group. base_link and
wrist_3_link/tool0 are instead covered by rmp_flow.yaml's own
body_capsules/body_collision_controllers (hand-authored in a later step from
actual mesh bounds), matching UR10's split exactly - not arbitrarily decided
here.

Per-link sphere counts mirror UR10's own bundled counts (same kinematic
topology, just physically larger), not arbitrary numbers:
    shoulder_link: 2 (UR10's self_only_collision_spheres count)
    upper_arm_link: 12, forearm_link: 15, wrist_1_link: 2, wrist_2_link: 7
      (UR10's ur10_collision_spheres per-link counts)

MorphIt-V (volume-focused/conservative) is used rather than the repo's own
default "-B" (balanced) example config, because this robot's task is pure
transit obstacle-avoidance (magic-attach, no real contact-rich grasping) -
MorphIt-V is the variant the project's own docs recommend for that case.

Run with the isolated MorphIt venv (NOT Isaac Sim's python - MorphIt's pinned
torch/numpy/vtk versions are independent of Isaac Sim's own environment by
design, to avoid destabilizing it):

    source /tmp/.../venv-morphit/bin/activate
    python /home/ubuntu/conveyor_indexing/robot_configs/generate_ur20_spheres_morphit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

MORPHIT_SRC = "/tmp/claude-1000/-home-ubuntu-conveyor-indexing/fc5f92c6-d423-495e-b08a-ac73bf40a38f/scratchpad/MorphIt-1/src"
sys.path.insert(0, MORPHIT_SRC)

from config import get_config, update_config_from_dict  # noqa: E402
from morphit import MorphIt  # noqa: E402

MESH_DIR = Path("/home/ubuntu/conveyor_indexing/robot_configs/ur20/meshes")
OUTPUT_DIR = Path("/home/ubuntu/conveyor_indexing/robot_configs/ur20/morphit_spheres")

# link name -> (collision mesh filename, num_spheres)
LINKS = {
    "shoulder_link": ("mesh_15.obj", 2),
    "upper_arm_link": ("mesh_16.obj", 12),
    "forearm_link": ("mesh_17.obj", 15),
    "wrist_1_link": ("mesh_18.obj", 2),
    "wrist_2_link": ("mesh_19.obj", 7),
}

CONFIG_UPDATES_TEMPLATE = {
    "training.iterations": 500,
    "training.verbose_frequency": 50,
    "training.logging_enabled": False,
    "training.density_control_min_interval": 150,
    "training.radius_lr": 0.1,
    "visualization.enabled": False,
    "visualization.save_video": False,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for link_name, (mesh_filename, num_spheres) in LINKS.items():
        mesh_path = MESH_DIR / mesh_filename
        if not mesh_path.exists():
            raise RuntimeError(f"Expected collision mesh not found: {mesh_path}")

        output_name = f"{link_name}.json"
        print(f"[generate_ur20_spheres_morphit] {link_name}: {mesh_filename} -> {num_spheres} spheres", flush=True)

        config = get_config("MorphIt-V")
        updates = dict(CONFIG_UPDATES_TEMPLATE)
        updates["model.mesh_path"] = str(mesh_path)
        updates["model.num_spheres"] = num_spheres
        updates["output_filename"] = output_name
        updates["results_dir"] = str(OUTPUT_DIR)
        config = update_config_from_dict(config, updates)

        model = MorphIt(config)
        model.train()
        model.save_results()

    print("[generate_ur20_spheres_morphit] done", flush=True)


if __name__ == "__main__":
    main()
