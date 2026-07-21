"""Step 1 of UR20 cuMotion config generation: export a URDF from the bundled
Isaac Sim ur20.usd asset via isaacsim.asset.exporter.urdf.UsdToUrdfConverter.

isaacsim.asset.exporter.urdf / isaacsim.asset.importer.utils /
isaacsim.robot_setup.xrdf_editor / the `lula` python bindings are not part of
the runtime Isaac Sim distribution at /home/ubuntu/IsaacSim - they only exist
in the built extensions under /home/ubuntu/IsaacSim-source's _build output.
Run with:

    PYTHONPATH=\
/home/ubuntu/IsaacSim-source/_build/linux-x86_64/release/exts/isaacsim.asset.exporter.urdf:\
/home/ubuntu/IsaacSim-source/_build/linux-x86_64/release/exts/isaacsim.asset.importer.utils \
    /home/ubuntu/IsaacSim/python.sh /home/ubuntu/conveyor_indexing/robot_configs/generate_ur20_urdf.py
"""

from __future__ import annotations

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import math

from pxr import Usd, UsdGeom
from isaacsim.storage.native import get_assets_root_path
from isaacsim.asset.exporter.urdf import UsdToUrdfConverter

UR20_USD_PATH = "/Isaac/Robots/UniversalRobots/ur20/ur20.usd"
OUTPUT_DIR = "/home/ubuntu/conveyor_indexing/robot_configs/ur20"

# ur20.usd's tool-mounting frame is an Xform named "flange" nested under
# wrist_3_link (confirmed by traversing the stage - no "tool0" prim exists in
# this asset, unlike ur10.usd which bakes one in). UsdToUrdfConverter only
# emits links connected via USD PhysicsJoints, so this plain Xform frame is
# silently dropped from the export. cuMotion's XRDF/rmp_flow config
# convention (mirrored from the bundled ur10 config) expects a "tool0" frame
# reachable via a fixed joint off the last wrist link, so it's added back
# here as a fixed joint, using "flange"'s ACTUAL authored local transform
# relative to wrist_3_link (read from the USD, not guessed).
TOOL_FRAME_PARENT_LINK = "wrist_3_link"
TOOL_FRAME_SOURCE_PRIM = "/ur20/wrist_3_link/flange"
TOOL_FRAME_LINK_NAME = "tool0"


def _add_tool0_frame(stage: Usd.Stage, urdf_path: str) -> None:
    flange_prim = stage.GetPrimAtPath(TOOL_FRAME_SOURCE_PRIM)
    if not flange_prim.IsValid():
        raise RuntimeError(f"Expected tool frame source prim not found at {TOOL_FRAME_SOURCE_PRIM}")
    local_transform = UsdGeom.Xformable(flange_prim).GetLocalTransformation()
    translation = local_transform.ExtractTranslation()
    rotation_matrix = local_transform.ExtractRotationMatrix()

    # URDF <origin rpy="..."> is intrinsic XYZ (R = Rz(yaw) * Ry(pitch) * Rx(roll)).
    r = [[rotation_matrix[i][j] for j in range(3)] for i in range(3)]
    pitch = math.asin(-r[2][0])
    roll = math.atan2(r[2][1], r[2][2])
    yaw = math.atan2(r[1][0], r[0][0])

    # Inserted as raw text rather than an ElementTree parse/rewrite round-trip,
    # which would silently drop the exporter's `isaac:source_drive` XML
    # comments (PhysX drive gains, useful if this URDF is ever re-imported).
    with open(urdf_path, encoding="utf-8") as f:
        content = f.read()
    insertion = (
        f'  <link name="{TOOL_FRAME_LINK_NAME}"/>\n'
        f'  <joint name="{TOOL_FRAME_PARENT_LINK}-{TOOL_FRAME_LINK_NAME}_fixed_joint" type="fixed">\n'
        f'    <origin xyz="{translation[0]:.8f} {translation[1]:.8f} {translation[2]:.8f}" '
        f'rpy="{roll:.8f} {pitch:.8f} {yaw:.8f}"/>\n'
        f'    <parent link="{TOOL_FRAME_PARENT_LINK}"/>\n'
        f'    <child link="{TOOL_FRAME_LINK_NAME}"/>\n'
        f"  </joint>\n"
    )
    closing_tag = "</robot>"
    if closing_tag not in content:
        raise RuntimeError(f"Expected closing '{closing_tag}' tag not found in {urdf_path}")
    content = content.replace(closing_tag, insertion + closing_tag)
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(
        f"[generate_ur20_urdf] appended '{TOOL_FRAME_LINK_NAME}' link + fixed joint off "
        f"'{TOOL_FRAME_PARENT_LINK}' (xyz={translation}, rpy=({roll:.6f},{pitch:.6f},{yaw:.6f})) "
        f"sourced from {TOOL_FRAME_SOURCE_PRIM}'s actual authored transform",
        flush=True,
    )


def main() -> None:
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not resolve Isaac Sim assets root path (nucleus/asset server unreachable)")
    ur20_url = assets_root + UR20_USD_PATH
    print(f"[generate_ur20_urdf] opening {ur20_url}", flush=True)

    stage = Usd.Stage.Open(ur20_url)
    if stage is None:
        raise RuntimeError(f"Failed to open stage at {ur20_url}")

    default_prim = stage.GetDefaultPrim()
    print(f"[generate_ur20_urdf] default prim: {default_prim.GetPath() if default_prim else None}", flush=True)
    print("[generate_ur20_urdf] top-level children of default prim:", flush=True)
    if default_prim:
        for child in default_prim.GetChildren():
            print(f"  {child.GetPath()} ({child.GetTypeName()})", flush=True)

    converter = UsdToUrdfConverter(stage, root_prim_path=str(default_prim.GetPath()) if default_prim else None)
    output_path = converter.convert(f"{OUTPUT_DIR}/robot.urdf")
    print(f"[generate_ur20_urdf] wrote URDF to {output_path}", flush=True)

    _add_tool0_frame(stage, output_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
