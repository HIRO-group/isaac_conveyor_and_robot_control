"""Step 2b of UR20 cuMotion config generation: build robot.xrdf + the Lula
robot_description YAML from the UR20 articulation, using collision spheres
optimized by MorphIt (generate_ur20_spheres_morphit.py) rather than Lula's
own simpler built-in generator.

isaacsim.robot_setup.xrdf_editor is a real Isaac Sim extension, but its
`lula` dependency (isaacsim.robot_motion.lula) is not part of the runtime
distribution's registered extensions - only the compiled lula.cpython-*.so
exists, under IsaacSim-source's build output. Run with that directory on
PYTHONPATH so the plain `import lula` inside collision_sphere_editor.py
resolves; isaacsim.robot_setup.xrdf_editor itself IS a registered runtime
extension, enabled the normal way (set_extension_enabled_immediate), which is
what correctly wires isaacsim.robot_setup's namespace path (a raw PYTHONPATH
addition alone was tried first and failed - Isaac Sim's `isaacsim` package
has a fixed, non-namespace __path__, unlike a plain PEP 420 package).

    PYTHONPATH=/home/ubuntu/IsaacSim-source/_build/target-deps/isaac_lula_prebundle \
    /home/ubuntu/IsaacSim/python.sh /home/ubuntu/conveyor_indexing/robot_configs/generate_ur20_xrdf.py
"""

from __future__ import annotations

import json
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.storage.native import get_assets_root_path

ext_manager = omni.kit.app.get_app().get_extension_manager()
if not ext_manager.set_extension_enabled_immediate("isaacsim.robot_setup.xrdf_editor", True):
    raise RuntimeError("Failed to enable isaacsim.robot_setup.xrdf_editor")

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.robot_setup.xrdf_editor import EditorState

UR20_USD_PATH = "/Isaac/Robots/UniversalRobots/ur20/ur20.usd"
ROBOT_PRIM_PATH = "/ur20"
CONFIG_DIR = Path("/home/ubuntu/conveyor_indexing/robot_configs/ur20")
MORPHIT_SPHERES_DIR = CONFIG_DIR / "morphit_spheres"

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Same bent-elbow seed pose as the bundled UR10 config's own
# default_joint_positions (robot_configurations/ur10/robot.xrdf) - the
# articulation's raw rest pose (all ~0 rad) is a stretched/singular UR
# configuration, same convention issue UR10's own config was already
# authored to avoid. Reused here rather than left at the raw rest pose,
# since it's the same joint order/family, not a fabricated guess.
DEFAULT_JOINT_POSITIONS_RAD = {
    "shoulder_pan_joint": -1.57,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": -1.57,
    "wrist_1_joint": -1.57,
    "wrist_2_joint": 1.57,
    "wrist_3_joint": 0.0,
}

# base_link and wrist_3_link deliberately have no entry here - they're
# covered instead by rmp_flow.yaml's own body_capsules/body_collision_controllers,
# mirroring the bundled UR10 config's exact split (see generate_ur20_spheres_morphit.py).
MORPHIT_LINK_FILES = {
    "shoulder_link": "shoulder_link.json",
    "upper_arm_link": "upper_arm_link.json",
    "forearm_link": "forearm_link.json",
    "wrist_1_link": "wrist_1_link.json",
    "wrist_2_link": "wrist_2_link.json",
}

TOOL_FRAME_LINK_NAME = "tool0"

# Spheres below this radius are optimizer artifacts (e.g. a sphere collapsed
# to ~0.1mm while its center stayed technically inside the mesh, so the
# final-iteration escaped-sphere prune in MorphIt doesn't catch it) - they
# contribute no real collision coverage and are dropped rather than exported.
MIN_SPHERE_RADIUS_M = 0.005


def _add_morphit_spheres(editor_state: EditorState) -> None:
    for link_name, filename in MORPHIT_LINK_FILES.items():
        json_path = MORPHIT_SPHERES_DIR / filename
        if not json_path.exists():
            raise RuntimeError(f"Missing MorphIt sphere output for {link_name}: {json_path}")
        with open(json_path) as f:
            data = json.load(f)

        link_path = editor_state.link_path(f"/{link_name}")
        added, skipped = 0, 0
        for center, radius in zip(data["centers"], data["radii"]):
            if radius < MIN_SPHERE_RADIUS_M:
                skipped += 1
                continue
            editor_state.collision_sphere_editor.add_sphere(link_path, np.array(center, dtype=float), float(radius))
            added += 1
        print(
            f"[generate_ur20_xrdf] {link_name}: added {added} MorphIt spheres "
            f"(skipped {skipped} below {MIN_SPHERE_RADIUS_M} m)",
            flush=True,
        )


def _add_tool_frames_field(xrdf_path: str) -> None:
    """Insert `tool_frames: ["tool0"]` after `cspace:`.

    EditorState.export_xrdf() / XrdfWriteInputs have no tool_frames field at
    all (confirmed by reading xrdf_io.py) - the tool only round-trips it via
    merge_existing from a prior XRDF, which doesn't exist yet for a from-
    scratch UR20 config. cuMotion's RmpFlowController needs at least one tool
    frame (robot_description.tool_frame_names()), and "tool0" is the real
    fixed frame we ourselves appended to robot.urdf (see
    generate_ur20_urdf.py) off wrist_3_link's actual authored flange
    transform - not a fabricated name.
    """
    with open(xrdf_path, encoding="utf-8") as f:
        content = f.read()
    marker = "\nworld_collision: "
    if marker not in content:
        raise RuntimeError(f"Expected 'world_collision:' section not found in {xrdf_path}")
    content = content.replace(marker, f'\ntool_frames: \n  - "{TOOL_FRAME_LINK_NAME}"\n{marker}', 1)
    with open(xrdf_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not resolve Isaac Sim assets root path")

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage_utils.add_reference_to_stage(usd_path=assets_root + UR20_USD_PATH, path=ROBOT_PRIM_PATH)

    # Articulation.get_dof_positions() (used internally by
    # EditorState.select_articulation) asserts on a valid PhysX tensor
    # entity, which only exists once the timeline has actually played at
    # least once - same requirement as World.reset() elsewhere in this repo.
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    editor_state = EditorState()
    editor_state.select_articulation(ROBOT_PRIM_PATH)

    print(f"[generate_ur20_xrdf] dof_names: {editor_state.dof_names}", flush=True)
    print(f"[generate_ur20_xrdf] link_to_meshes keys: {list(editor_state.link_to_meshes.keys())}", flush=True)

    for i, name in enumerate(editor_state.dof_names):
        editor_state.active_joints[i] = name in ARM_JOINT_NAMES
        if name in DEFAULT_JOINT_POSITIONS_RAD:
            editor_state.joint_positions[i] = DEFAULT_JOINT_POSITIONS_RAD[name]
    if not all(editor_state.active_joints):
        missing = [n for n in ARM_JOINT_NAMES if n not in editor_state.dof_names]
        if missing:
            raise RuntimeError(f"Expected arm joints not found in articulation dof_names: {missing}")

    _add_morphit_spheres(editor_state)

    xrdf_path = str(CONFIG_DIR / "robot.xrdf")
    lula_path = str(CONFIG_DIR / "robot_description.yaml")
    editor_state.export_xrdf(xrdf_path, format_version=2.0)
    editor_state.export_lula(lula_path)
    _add_tool_frames_field(xrdf_path)
    print(f"[generate_ur20_xrdf] wrote {xrdf_path}", flush=True)
    print(f"[generate_ur20_xrdf] wrote {lula_path}", flush=True)

    editor_state.on_shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
