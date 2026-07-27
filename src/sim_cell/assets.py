"""Rewrite `5_conv_env.usd`'s remote asset references to the local mirror,
before the stage is ever opened.
"""

from __future__ import annotations

import logging
import os

from pxr import Sdf

from sim_cell.asset_paths import LOCAL_ASSET_ROOT, REMOTE_ASSET_ROOT, local_path_for

logger = logging.getLogger(__name__)


def localize_asset_references(stage_path: str) -> None:
    """Rewrite reference/payload asset paths from REMOTE_ASSET_ROOT to LOCAL_ASSET_ROOT
    in stage_path's Sdf.Layer, before it's ever opened as a Usd.Stage (opening triggers
    composition, which is when Kit would fetch the un-rewritten paths over the
    network). Not saved to disk. No-op if LOCAL_ASSET_ROOT doesn't exist yet.
    """
    if not os.path.isdir(LOCAL_ASSET_ROOT):
        logger.info(
            "%s not found - fetching assets from %s instead "
            "(see README's download_assets.py note to cache them locally)",
            LOCAL_ASSET_ROOT, REMOTE_ASSET_ROOT,
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

    rewritten = 0
    for spec in _iter_prim_specs(layer.rootPrims.values()):
        refs = spec.referenceList
        if refs.prependedItems:
            new_items = []
            for item in refs.prependedItems:
                if item.assetPath.startswith(REMOTE_ASSET_ROOT):
                    item = Sdf.Reference(local_path_for(item.assetPath), item.primPath, item.layerOffset, item.customData)
                    rewritten += 1
                new_items.append(item)
            refs.prependedItems = new_items

        payloads = spec.payloadList
        if payloads.prependedItems:
            new_items = []
            for item in payloads.prependedItems:
                if item.assetPath.startswith(REMOTE_ASSET_ROOT):
                    item = Sdf.Payload(local_path_for(item.assetPath), item.primPath, item.layerOffset)
                    rewritten += 1
                new_items.append(item)
            payloads.prependedItems = new_items

    logger.info("localized %d asset reference(s)/payload(s) to %s", rewritten, LOCAL_ASSET_ROOT)
