"""Frame/upright-posts mesh deactivation."""

from __future__ import annotations

import logging

from pxr import Usd

logger = logging.getLogger(__name__)


def deactivate_frame_meshes(stage: Usd.Stage, track_roots: tuple) -> None:
    """Deactivate every track's frame/upright-posts mesh at runtime (not edited into
    the USD) - removes it from both rendering and cuMotion's obstacle tracking, so
    the arm no longer avoids the frame/posts, only the belt-top zone bboxes.
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
    logger.info("deactivated %d SM_ConveyorBelt_A06_02 frame meshes", len(deactivated))
