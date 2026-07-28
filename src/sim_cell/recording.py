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
from pathlib import Path

import numpy as np

import sim_state_pb2
from cameras.protos import camera
from conveyor_indexing.episode_recorder import EpisodeRecorder
from conveyor_indexing.mcap_recorder import McapRecorder, git_sha
from sim_cell import settings

logger = logging.getLogger(__name__)

RECORD_ENV_VAR = "CONVEYOR_INDEXING_RECORD"
RECORD_OUTPUT_DIR = str(Path(settings.LOG_OUTPUT_DIR) / "recordings")

# MCAP ("record everything", no episode concept) is gated independently of the
# 30Hz parquet recorder above - see conveyor_indexing.mcap_recorder.
MCAP_RECORD_ENV_VAR = "CONVEYOR_INDEXING_RECORD_MCAP"
MCAP_OUTPUT_DIR = str(Path(settings.LOG_OUTPUT_DIR) / "mcap")

# Which parallel instance this process is (0-based) - purely descriptive
# metadata (see RunMetadata below); distinct from
# CONVEYOR_INDEXING_EPISODE_ID_BASE, which is derived from it but set
# separately by the launcher so this module never has to reverse that math.
INSTANCE_INDEX_ENV_VAR = "CONVEYOR_INDEXING_INSTANCE_INDEX"

_HELD_BY_NONE = 0

# Re-exported so sim_cell.runner can record box events without importing the
# generated sim_state_pb2 module directly.
BOX_EVENT_SPAWNED = sim_state_pb2.BoxEventType.BOX_EVENT_SPAWNED
BOX_EVENT_DESPAWNED = sim_state_pb2.BoxEventType.BOX_EVENT_DESPAWNED

# Base value for EpisodeTracker's episode counter - set to a distinct value per
# parallel instance (e.g. instance_index * 10_000_000) so IDs stay globally
# unique when a conversion pass walks multiple instances' recordings together;
# otherwise instance N's last episode and instance N+1's first episode could
# collide on the same id and get silently merged by an id-based segmenter.
EPISODE_ID_BASE_ENV_VAR = "CONVEYOR_INDEXING_EPISODE_ID_BASE"

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
    queue_maxsize_env = os.environ.get("CONVEYOR_INDEXING_RECORD_QUEUE")
    kwargs = {"queue_maxsize": int(queue_maxsize_env)} if queue_maxsize_env else {}
    return EpisodeRecorder(
        output_dir=RECORD_OUTPUT_DIR,
        serial_to_role=SERIAL_TO_ROLE,
        width=settings.CAMERA_WIDTH,
        height=settings.CAMERA_HEIGHT,
        **kwargs,
    )


def _build_run_metadata(camera_specs, spawn_seed: int) -> sim_state_pb2.RunMetadata:
    cameras = [
        sim_state_pb2.CameraStaticInfo(
            serial=spec.serial,
            role=camera.CameraRole.Name(spec.role),
            translate=sim_state_pb2.Vec3(x=spec.translate[0], y=spec.translate[1], z=spec.translate[2]),
            rotation_euler_xyz_deg=sim_state_pb2.Vec3(
                x=spec.rotation_euler_xyz_deg[0], y=spec.rotation_euler_xyz_deg[1], z=spec.rotation_euler_xyz_deg[2]
            ),
            parent_path=spec.parent_path or "",
            focal_length_mm=spec.focal_length,
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
        )
        for spec in camera_specs
    ]
    return sim_state_pb2.RunMetadata(
        conveyor_indexing_git_sha=git_sha(settings.REPO_ROOT),
        instance_index=int(os.environ.get(INSTANCE_INDEX_ENV_VAR, "0")),
        spawn_seed=spawn_seed,
        physics_dt=settings.PHYSICS_DT,
        control_hz=settings.CONTROL_HZ,
        camera_fps=settings.CAMERA_FPS,
        camera_width=settings.CAMERA_WIDTH,
        camera_height=settings.CAMERA_HEIGHT,
        cameras=cameras,
    )


def maybe_build_mcap_recorder(camera_specs, spawn_seed: int) -> McapRecorder | None:
    """An McapRecorder when CONVEYOR_INDEXING_RECORD_MCAP=1, else None (default off).
    Independent of maybe_build_recorder's CONVEYOR_INDEXING_RECORD - either,
    both, or neither can be enabled for a given run.
    """
    if os.environ.get(MCAP_RECORD_ENV_VAR, "0") != "1":
        return None
    logger.info("mcap recording enabled -> %s", MCAP_OUTPUT_DIR)
    recorder = McapRecorder(
        output_dir=MCAP_OUTPUT_DIR,
        run_metadata=_build_run_metadata(camera_specs, spawn_seed),
    )
    recorder.write_move_target_stub(1)
    recorder.write_move_target_stub(2)
    return recorder


def build_box_states(
    active_box_paths, box_positions: dict, box_orientations: dict, box_linear_vel: dict, box_angular_vel: dict,
    box_id_to_variant: dict, held_by_arm: dict,
) -> list:
    """One sim_state_pb2.BoxState per path in ``active_box_paths`` (spawned,
    not yet despawned - see sim_cell.runner's active_box_paths bookkeeping).
    The four ``box_*`` dicts are {box_path: value}, keyed by every box path
    (sim_cell.runner builds them from one batched RigidPrim read per control
    tick). ``held_by_arm`` is {box_path: 1|2} for whichever box(es) are
    currently attached - see
    pick_and_place.controller.MagicAttachPickPlace.held_box_path.
    """
    boxes = []
    for path in active_box_paths:
        pos = box_positions[path]
        quat = box_orientations[path]  # (w, x, y, z) - Isaac Sim convention, see proto/sim_state.proto
        lin = box_linear_vel[path]
        ang = box_angular_vel[path]
        boxes.append(
            sim_state_pb2.BoxState(
                box_id=path,
                variant=box_id_to_variant.get(path, ""),
                position=sim_state_pb2.Vec3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                orientation=sim_state_pb2.Quat(w=float(quat[0]), x=float(quat[1]), y=float(quat[2]), z=float(quat[3])),
                linear_velocity=sim_state_pb2.Vec3(x=float(lin[0]), y=float(lin[1]), z=float(lin[2])),
                angular_velocity=sim_state_pb2.Vec3(x=float(ang[0]), y=float(ang[1]), z=float(ang[2])),
                held_by_arm=held_by_arm.get(path, _HELD_BY_NONE),
            )
        )
    return boxes


class EpisodeTracker:
    """Cell-wide int64 episode key: +1 whenever either arm starts a new pick
    (WAITING -> anything edge). This is the recording's default segmentation;
    theia-side conversion can re-segment from the recorded phase/sim-time
    columns instead. Call update() every physics step so no edge is missed
    between 30Hz recorded rows.
    """

    def __init__(self) -> None:
        self._episode_id = int(os.environ.get(EPISODE_ID_BASE_ENV_VAR, "0"))
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
