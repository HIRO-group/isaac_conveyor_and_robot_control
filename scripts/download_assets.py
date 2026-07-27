"""Mirror 5_conv_env.usd's referenced Omniverse-bucket assets locally.

Recursively downloads the 5 top-level assets 5_conv_env.usd references
(conveyor belt, truck, 3 box variants) AND everything each one references in
turn (sublayers, materials, MDL shaders, textures), preserving the bucket's
own relative directory structure under LOCAL_ASSET_ROOT - so once local,
each asset's own relative references resolve to the local mirror without any
further editing. See sim_cell.assets.localize_asset_references, which points
the running sim at this mirror instead of the network at runtime
(REMOTE_ASSET_ROOT -> LOCAL_ASSET_ROOT is a clean string-prefix swap,
confirmed against every path discovered here).

Idempotent: skips any file already present locally, so re-running after an
interrupted download only fetches what's missing.

Run with Isaac Sim's bundled python (needs omni.client + pxr.UsdUtils, both
only available inside a Kit process):
    ~/IsaacSim/python.sh ~/conveyor_indexing/scripts/download_assets.py
"""

from __future__ import annotations

import logging
import os
import sys

# This script (unlike scripts/run_conveyor_indexing.py) is invoked directly,
# not via scripts/run.sh, so it puts src/ on sys.path itself.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, UsdUtils
import omni.client

from sim_cell.asset_paths import LOCAL_ASSET_ROOT, REMOTE_ASSET_ROOT, TOP_LEVEL_URLS, local_path_for

logger = logging.getLogger("download_assets")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


def _download_one(url: str, stats: dict) -> None:
    local_path = local_path_for(url)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        stats["skipped"] += 1
        return
    result, _version, content = omni.client.read_file(url)
    if result != omni.client.Result.OK:
        stats["failed"].append((url, str(result)))
        logger.error("FAILED to fetch %s: %s", url, result)
        return
    data = bytes(memoryview(content))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
    stats["downloaded"] += 1
    stats["bytes"] += len(data)


def main() -> None:
    os.makedirs(LOCAL_ASSET_ROOT, exist_ok=True)
    stats = {"downloaded": 0, "skipped": 0, "bytes": 0, "failed": []}

    for top_url in TOP_LEVEL_URLS:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(top_url))
        dep_urls = {top_url}
        dep_urls.update(layer.identifier for layer in layers if layer.identifier.startswith(REMOTE_ASSET_ROOT))
        dep_urls.update(asset for asset in assets if asset.startswith(REMOTE_ASSET_ROOT))
        if unresolved:
            logger.warning("unresolved paths for %s: %s", top_url, unresolved)

        logger.info("%s: %d dependency file(s)", top_url, len(dep_urls))
        for url in sorted(dep_urls):
            _download_one(url, stats)

    logger.info(
        "done: %d downloaded (%.1f MB), %d already present, %d failed",
        stats["downloaded"], stats["bytes"] / 1e6, stats["skipped"], len(stats["failed"]),
    )
    for url, err in stats["failed"]:
        logger.error("  FAILED: %s (%s)", url, err)

    simulation_app.close()


if __name__ == "__main__":
    main()
