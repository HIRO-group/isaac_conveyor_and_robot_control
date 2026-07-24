"""Scene-construction helpers shared by the main sim (conveyor_indexer.py)
and the out-of-process motion planner (planner_server.py).

Both processes open the same ~/5_conv_env.usd and apply the same runtime-only
mutations, so the planner subprocess builds its cuMotion obstacle world from a
stage identical to the one the arm actually moves in. Factored out here (rather
than left in conveyor_indexer.py) precisely because planner_server.py cannot
import conveyor_indexer.py - importing that module creates a visible
SimulationApp at module load (see its top-level `SimulationApp(...)` call),
which the planner subprocess must not do.

Import only AFTER a SimulationApp exists in the importing process (same
convention as pick_and_place.py) - the pxr imports below are fine, but callers
rely on a live stage.
"""

from __future__ import annotations

import os

from pxr import Sdf, Usd, UsdPhysics


def deactivate_frame_meshes(stage: Usd.Stage, track_roots: tuple) -> None:
    """Deactivate every track's `SM_ConveyorBelt_A06_02` frame/upright-posts mesh.

    Applied at runtime (stage.SetActive(False)) rather than edited into
    5_conv_env.usd, so the authored scene file is left untouched. This
    removes the mesh from rendering AND from PhysX collision - unlike the
    `Belt` mesh, this frame mesh otherwise IS tracked by cuMotion's obstacle
    world, so deactivating it means the arm no longer avoids the frame/upright
    posts. Both the main sim and the planner subprocess must do this
    identically, or their obstacle sets would differ.

    Unlike racetrack.usd (8 straight + curved tracks per loop, the curved
    ones using a different frame asset, `SM_ConveyorBelt_A12`), every track in
    5_conv_env.usd is the same straight `ConveyorBelt_A06` asset - so every
    root is expected to match; a miss is a real error, not an
    allowed-for curved-track gap.
    """
    deactivated = []
    for root in track_roots:
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim.IsValid():
            raise RuntimeError(f"Expected conveyor track prim not found at {root}")
        matched = False
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == "SM_ConveyorBelt_A06_02":
                prim.SetActive(False)
                deactivated.append(str(prim.GetPath()))
                matched = True
        if not matched:
            raise RuntimeError(f"No SM_ConveyorBelt_A06_02 mesh found under: {root}")
    print(f"[scene_setup] deactivated {len(deactivated)} SM_ConveyorBelt_A06_02 frame meshes", flush=True)


def apply_truck_collision(stage: Usd.Stage, truck_path: str) -> None:
    """Add a static collider to the truck bed so falling boxes land in it
    instead of clipping straight through.

    `/World/SteelBoxTruck_A01_01` (like the boxes) is a pure visual payload -
    confirmed via direct inspection, no physics schemas anywhere in its
    subtree. Only its body mesh (`sm_steelboxtruck_a01_body_01`, the
    bed/frame shape boxes actually land in/on) gets a collider - the wheel
    meshes don't need one for this scaffold's purposes. Static (CollisionAPI
    only, no RigidBodyAPI) since the truck itself never moves; the default
    exact-triangle-mesh approximation is fine here (unlike the boxes, this
    is a static collider, not one in dynamic-dynamic contact every tick).

    The main sim needs this so boxes land in the truck; the planner subprocess
    needs it so the truck registers as the same obstacle the arm must avoid.
    """
    body_prim = stage.GetPrimAtPath(f"{truck_path}/sm_steelboxtruck_a01_body_01")
    if not body_prim.IsValid():
        raise RuntimeError(f"Expected truck body mesh not found under {truck_path}")
    UsdPhysics.CollisionAPI.Apply(body_prim)
    print(f"[scene_setup] added static collision to {body_prim.GetPath()}", flush=True)


def localize_asset_references(stage_path: str, remote_root: str, local_root: str) -> None:
    """Rewrite every reference/payload asset path under remote_root to
    local_root, in the in-memory Sdf.Layer for stage_path, BEFORE it's ever
    opened as a Usd.Stage.

    Order matters: opening stage_path as a Usd.Stage composes every
    referenced/payloaded prim - e.g. each ConveyorTrack's referenced Belt/frame
    geometry - which is exactly when Kit would fetch these assets over the
    network. Sdf.Layer.FindOrOpen, unlike Usd.Stage.Open, just parses the raw
    layer content (prim specs and their un-resolved reference/payload list ops)
    without triggering that composition/fetch - so every rewrite here happens
    first, and by the time the stage is opened (finding this same, now-modified
    layer already resident in Sdf's process-global layer registry, keyed by
    resolved identifier, rather than re-reading the file from disk) it only
    ever resolves local paths. Not saved back to disk, so 5_conv_env.usd itself
    is never touched - same "runtime-only" convention as every other mutation
    here (e.g. deactivate_frame_meshes).

    A no-op (falls back to fetching from remote_root, same as before this
    function existed) if local_root doesn't exist - e.g. `download_assets.py`
    (see README) hasn't been run yet on this machine.

    Only handles `prepend` list-op items (`prependedItems`) - confirmed via
    direct inspection that every reference/payload in 5_conv_env.usd is
    authored that way (`prepend references = @url@` / `prepend payload =
    @url@`), never `explicit`/`append`/`delete`.
    """
    if not os.path.isdir(local_root):
        print(
            f"[scene_setup] {local_root} not found - fetching assets from {remote_root} instead "
            "(see README's download_assets.py note to cache them locally)",
            flush=True,
        )
        return

    layer = Sdf.Layer.FindOrOpen(stage_path)
    if layer is None:
        raise RuntimeError(f"Could not open {stage_path} as an Sdf.Layer")

    def _iter_prim_specs(root_specs):
        stack = list(root_specs)
        while stack:
            spec = stack.pop()
            yield spec
            stack.extend(spec.nameChildren.values())

    def _localized(asset_path: str) -> str:
        return os.path.join(local_root, asset_path[len(remote_root):])

    rewritten = 0
    for spec in _iter_prim_specs(layer.rootPrims.values()):
        refs = spec.referenceList
        if refs.prependedItems:
            new_items = []
            for item in refs.prependedItems:
                if item.assetPath.startswith(remote_root):
                    item = Sdf.Reference(_localized(item.assetPath), item.primPath, item.layerOffset, item.customData)
                    rewritten += 1
                new_items.append(item)
            refs.prependedItems = new_items

        payloads = spec.payloadList
        if payloads.prependedItems:
            new_items = []
            for item in payloads.prependedItems:
                if item.assetPath.startswith(remote_root):
                    item = Sdf.Payload(_localized(item.assetPath), item.primPath, item.layerOffset)
                    rewritten += 1
                new_items.append(item)
            payloads.prependedItems = new_items

    print(f"[scene_setup] localized {rewritten} asset reference(s)/payload(s) to {local_root}", flush=True)
