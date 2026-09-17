"""SceneConfig loading and derived views."""

from __future__ import annotations

from pathlib import Path

import pytest

from sim_cell import scene as scene_mod
from sim_cell.scene import SCENE_DIR_ENV, SceneConfig, SceneError, ZoneRef

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "scene"


def test_load_resolves_paths_relative_to_scene_root():
    cfg = SceneConfig.load(FIXTURE)
    assert cfg.stage_path == FIXTURE / "5_conv_env_empty.usd"
    assert cfg.camera_poses_path == FIXTURE / "camera_poses.json"


def test_loops_stations_cameras():
    cfg = SceneConfig.load(FIXTURE)
    assert [len(loop.zones) for loop in cfg.loops] == [3, 2]
    assert cfg.zone_path(cfg.stations[1].pick_zone) == cfg.loops[0].zones[2]
    assert cfg.zone_path(ZoneRef(1, 0)) == cfg.loops[1].zones[0]
    assert [c.feature_name for c in cfg.cameras][:3] == ["pick_cam_1", "place_cam_1", "hand_cam_1"]


def test_excluded_structure_roots_cover_tracks_robots_truck():
    cfg = SceneConfig.load(FIXTURE)
    roots = cfg.excluded_structure_roots
    assert set(cfg.conveyor_track_roots) <= set(roots)
    assert "/World/PickPlaceRobot_02" in roots and cfg.truck_path in roots


def test_require_shape():
    cfg = SceneConfig.load(FIXTURE)
    cfg.require_shape(loops=2, stations=2)
    with pytest.raises(SceneError):
        cfg.require_shape(loops=1, stations=2)


def test_missing_field_is_a_scene_error(tmp_path):
    (tmp_path / "cell.yaml").write_text("stage: x.usd\n")
    with pytest.raises(SceneError):
        SceneConfig.load(tmp_path)


def test_missing_config_file(tmp_path):
    with pytest.raises(SceneError):
        SceneConfig.load(tmp_path)


def test_env_unset_is_a_scene_error(monkeypatch):
    monkeypatch.delenv(SCENE_DIR_ENV, raising=False)
    with pytest.raises(SceneError):
        scene_mod.scene_dir_from_env()
