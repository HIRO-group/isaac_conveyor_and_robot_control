"""Mirror 5_conv_env.usd's referenced Omniverse-bucket assets locally.

Recursively downloads the 5 top-level assets 5_conv_env.usd references
(conveyor belt, truck, 3 box variants) AND everything each one references in
turn (sublayers, materials, MDL shaders, textures), preserving the bucket's
own relative directory structure under LOCAL_ASSET_ROOT - so once local,
each asset's own relative references resolve to the local mirror without any
further editing. See conveyor_indexer.py's _localize_asset_references, which
points the running sim at this mirror instead of the network at runtime
(REMOTE_ASSET_ROOT -> LOCAL_ASSET_ROOT is a clean string-prefix swap,
confirmed against every path discovered here).

Idempotent: skips any file already present locally, so re-running after an
interrupted download only fetches what's missing.

Run with Isaac Sim's bundled python (needs omni.client + pxr.UsdUtils, both
only available inside a Kit process):
    ~/IsaacSim/python.sh ~/conveyor_indexing/download_assets.py
"""

import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, UsdUtils
import omni.client

REMOTE_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
LOCAL_ASSET_ROOT = os.path.join(os.path.expanduser("~"), "isaac_assets")

TOP_LEVEL_URLS = [
    REMOTE_ASSET_ROOT + "Assets/Isaac/6.0/Isaac/Props/Conveyors/ConveyorBelt_A06.usd",
    REMOTE_ASSET_ROOT + "Assets/DigitalTwin/Assets/Warehouse/Equipment/Carts/SteelBoxTruck_A/SteelBoxTruck_A01_01.usd",
    REMOTE_ASSET_ROOT + "Assets/DigitalTwin/Assets/Warehouse/Shipping/Cardboard_Boxes/Cube_A/CubeBox_A04_26cm_PR_NVD_01.usd",
    REMOTE_ASSET_ROOT + "Assets/DigitalTwin/Assets/Warehouse/Shipping/Cardboard_Boxes/Cube_A/CubeBox_A06_42cm_PR_NVD_01.usd",
    REMOTE_ASSET_ROOT + "Assets/DigitalTwin/Assets/Warehouse/Shipping/Cardboard_Boxes/Cube_A/CubeBox_A03_21cm_PR_NVD_01.usd",
]


def _local_path_for(url: str) -> str:
    assert url.startswith(REMOTE_ASSET_ROOT), f"URL not under {REMOTE_ASSET_ROOT}: {url}"
    return os.path.join(LOCAL_ASSET_ROOT, url[len(REMOTE_ASSET_ROOT):])


def _download_one(url: str, stats: dict) -> None:
    local_path = _local_path_for(url)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        stats["skipped"] += 1
        return
    result, _version, content = omni.client.read_file(url)
    if result != omni.client.Result.OK:
        stats["failed"].append((url, str(result)))
        print(f"[download_assets] FAILED to fetch {url}: {result}", flush=True)
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
            print(f"[download_assets] WARNING: unresolved paths for {top_url}: {unresolved}", flush=True)

        print(f"[download_assets] {top_url}: {len(dep_urls)} dependency file(s)", flush=True)
        for url in sorted(dep_urls):
            _download_one(url, stats)

    print(
        f"[download_assets] done: {stats['downloaded']} downloaded "
        f"({stats['bytes'] / 1e6:.1f} MB), {stats['skipped']} already present, "
        f"{len(stats['failed'])} failed",
        flush=True,
    )
    for url, err in stats["failed"]:
        print(f"[download_assets]   FAILED: {url} ({err})", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
