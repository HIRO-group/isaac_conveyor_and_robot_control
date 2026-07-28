"""Sim-side glue for training-data recording: env-var gating, camera-serial ->
dataset-role mapping, the cell-wide episode key, and the observation.state
vector layout.

Kept separate from conveyor_indexing.episode_recorder (the pure parquet
writer) so the recorder stays unit-testable without Isaac Sim, and separate
from runner.py so the main loop only gains a few calls.

observation.state layout (30 float32s, conveyor dims appended at conversion
time from plc_state_conveyors - see episode_recorder's module docstring):

  [0:6]    arm 1 joint positions, radians, Articulation dof order
  [6]      arm 1 suction (1.0 while holding a box, else 0.0)
  [7:15]   arm 1 cups 1-8 (mirror suction - the sim's magic attach has no
           per-cup actuation, so all 8 toggle together with attach/detach)
  [15:30]  arm 2, same block
"""

from __future__ import annotations

import logging
import os

import numpy as np

from conveyor_indexing.episode_recorder import EpisodeRecorder
from sim_cell import settings

logger = logging.getLogger(__name__)

RECORD_ENV_VAR = "CONVEYOR_INDEXING_RECORD"
RECORD_OUTPUT_DIR = str(settings.REPO_ROOT / "data" / "recordings")

# Dataset role keys become LeRobot feature names (observation.images.<role>);
# the _1/_2 suffix distinguishes the two stations for the dual-robot model.
SERIAL_TO_ROLE = {
    "SIM1-PICK": "pick_cam_1",
    "SIM1-PLACE": "place_cam_1",
    "SIM1-HAND": "hand_cam_1",
    "SIM2-PICK": "pick_cam_2",
    "SIM2-PLACE": "place_cam_2",
    "SIM2-HAND": "hand_cam_2",
}

_STATE_DIMS_PER_ARM = 15  # 6 joints + suction + 8 cups
_WAITING = "WAITING"


def maybe_build_recorder(camera_specs) -> EpisodeRecorder | None:
    """An EpisodeRecorder when CONVEYOR_INDEXING_RECORD=1, else None (default off)."""
    if os.environ.get(RECORD_ENV_VAR, "0") != "1":
        return None
    serials = {spec.serial for spec in camera_specs}
    if serials != set(SERIAL_TO_ROLE):
        raise ValueError(
            f"camera serials {sorted(serials)} don't match recording roles {sorted(SERIAL_TO_ROLE)} - "
            "update sim_cell.recording.SERIAL_TO_ROLE alongside sim_cell.camera_layout"
        )
    logger.info("episode recording enabled -> %s", RECORD_OUTPUT_DIR)
    return EpisodeRecorder(
        output_dir=RECORD_OUTPUT_DIR,
        serial_to_role=SERIAL_TO_ROLE,
        width=settings.CAMERA_WIDTH,
        height=settings.CAMERA_HEIGHT,
    )


class EpisodeTracker:
    """Cell-wide int64 episode key: +1 whenever either arm starts a new pick
    (WAITING -> anything edge). This is the recording's default segmentation;
    theia-side conversion can re-segment from the recorded phase/sim-time
    columns instead. Call update() every physics step so no edge is missed
    between 30Hz recorded rows.
    """

    def __init__(self) -> None:
        self._episode_id = 0
        self._prev_phase_1 = _WAITING
        self._prev_phase_2 = _WAITING

    def update(self, phase_name_1: str, phase_name_2: str) -> int:
        if (self._prev_phase_1 == _WAITING and phase_name_1 != _WAITING) or (
            self._prev_phase_2 == _WAITING and phase_name_2 != _WAITING
        ):
            self._episode_id += 1
        self._prev_phase_1 = phase_name_1
        self._prev_phase_2 = phase_name_2
        return self._episode_id

    @property
    def episode_id(self) -> int:
        return self._episode_id


def build_observation_state(robot, robot2, holding_1: bool, holding_2: bool) -> np.ndarray:
    """float32 (30,) per the layout in the module docstring. Joint reads are a
    6-float device->host copy per arm - negligible at 30Hz.
    """
    state = np.empty(2 * _STATE_DIMS_PER_ARM, dtype=np.float32)
    for i, (arm, holding) in enumerate(((robot, holding_1), (robot2, holding_2))):
        block = i * _STATE_DIMS_PER_ARM
        state[block : block + 6] = arm.get_dof_positions().numpy()[0]
        state[block + 6 : block + 15] = 1.0 if holding else 0.0
    return state
