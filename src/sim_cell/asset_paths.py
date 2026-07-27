"""Where `5_conv_env.usd`'s referenced assets live remotely, and where they're
mirrored locally. Pure stdlib - shared by `scripts/download_assets.py` and
`sim_cell.assets` alike.

5_conv_env.usd fetches its assets from this public S3 bucket over HTTPS every
run unless localized; `scripts/download_assets.py` mirrors them locally, and
`sim_cell.assets.localize_asset_references` points the running sim at that
mirror instead of the network at runtime.
"""

from __future__ import annotations

import os

REMOTE_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
LOCAL_ASSET_ROOT = os.path.join(os.path.expanduser("~"), "isaac_assets")

TOP_LEVEL_URLS = [
    REMOTE_ASSET_ROOT + "Assets/Isaac/6.0/Isaac/Props/Conveyors/ConveyorBelt_A06.usd",
    REMOTE_ASSET_ROOT + "Assets/DigitalTwin/Assets/Warehouse/Equipment/Carts/SteelBoxTruck_A/SteelBoxTruck_A01_01.usd",
    REMOTE_ASSET_ROOT + "Assets/DigitalTwin/Assets/Warehouse/Shipping/Cardboard_Boxes/Cube_A/CubeBox_A04_26cm_PR_NVD_01.usd",
    REMOTE_ASSET_ROOT + "Assets/DigitalTwin/Assets/Warehouse/Shipping/Cardboard_Boxes/Cube_A/CubeBox_A06_42cm_PR_NVD_01.usd",
    REMOTE_ASSET_ROOT + "Assets/DigitalTwin/Assets/Warehouse/Shipping/Cardboard_Boxes/Cube_A/CubeBox_A03_21cm_PR_NVD_01.usd",
]


def local_path_for(url: str) -> str:
    assert url.startswith(REMOTE_ASSET_ROOT), f"URL not under {REMOTE_ASSET_ROOT}: {url}"
    return os.path.join(LOCAL_ASSET_ROOT, url[len(REMOTE_ASSET_ROOT) :])
