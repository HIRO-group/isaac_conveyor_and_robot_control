"""pytest bootstrap: puts `src/` and the generated protobuf bindings on
sys.path, mirroring scripts/run.sh's PYTHONPATH (see the top-level README's
"Setup" section) - so tests import this repo's packages (`sim_cell`,
`conveyor_indexing`, `pick_and_place`, `cameras`) and their generated `*_pb2`
modules exactly like the real sim process does, without a packaging step
(see pyproject.toml's comment on why this repo has no build backend).

Every test module here is written to run WITHOUT Isaac Sim/omni/carb - see
each package's own import-weight docstring (`conveyor_indexing/__init__.py`,
`sim_cell/__init__.py`): only modules that avoid `isaacsim`/`omni`/`carb`
imports (directly or via `pxr.PhysxSchema`/`omni.physics.core`) are exercised
here. Modules that require Isaac Sim (`sim_cell.runner`, `sim_cell.cell`,
everything under `pick_and_place` except a couple of pure helper modules,
`conveyor_indexing.zone`/`occupancy`/`line_controller`) are validated by
reading + by the nne2 GPU session instead - see the top-level plan's test
report for exactly which.

Generated bindings: run `bash gen_proto.sh` first (see the top-level README),
which writes to /tmp/proto_gen by default - override with
CONVEYOR_INDEXING_PROTO_GEN_DIR if you generated elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_PROTO_GEN_DIR = Path(os.environ.get("CONVEYOR_INDEXING_PROTO_GEN_DIR", "/tmp/proto_gen"))

for path in (_SRC_DIR, _SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

if _PROTO_GEN_DIR.is_dir():
    proto_gen_str = str(_PROTO_GEN_DIR)
    if proto_gen_str not in sys.path:
        sys.path.insert(0, proto_gen_str)
else:
    raise RuntimeError(
        f"generated protobuf bindings not found at {_PROTO_GEN_DIR} - run `bash gen_proto.sh` first "
        "(see the top-level README's 'Setup' section), or set CONVEYOR_INDEXING_PROTO_GEN_DIR to "
        "wherever you generated them. Note gen_proto.sh's THEIA_ROOT assumes a sibling ~/theia "
        "checkout at /home/ubuntu/theia - override that path in your own environment if it differs."
    )
