"""Entrypoint for the out-of-process cuMotion planner subprocess.

Launched by conveyor_indexer.py via subprocess.Popen - not meant to be run by
hand. Uses cuMotion's low-level library directly, with NO SimulationApp (no
Kit/USD/render), so it starts in ~1-2s.

warp + cumotion ship as Isaac Sim extensions whose sys.path entries are
normally added by Kit's extension manager. Since this process never boots Kit,
the parent (which does have Kit) discovers those directories and passes them in
via --ext-path; we insert them BEFORE importing planner_server_impl, whose
module-level `import warp` / `import cumotion` then resolve.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="cuMotion planner subprocess (launched by conveyor_indexer.py)")
    parser.add_argument("--addr", required=True, help="multiprocessing.connection address of the parent Listener")
    parser.add_argument("--authkey", required=True, help="hex-encoded connection authkey")
    parser.add_argument(
        "--ext-path",
        action="append",
        default=[],
        help="directory to add to sys.path so warp/cumotion import (repeatable)",
    )
    args = parser.parse_args()

    for path in args.ext_path:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    import planner_server_impl

    planner_server_impl.serve(args.addr, bytes.fromhex(args.authkey))


if __name__ == "__main__":
    main()
