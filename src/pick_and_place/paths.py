"""Repo-relative paths, anchored off this file's location rather than the
process's current working directory.
"""

from __future__ import annotations

from pathlib import Path

# .../conveyor_indexing/src/pick_and_place/paths.py -> .../conveyor_indexing
REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_CONFIGS_DIR = REPO_ROOT / "robot_configs"
