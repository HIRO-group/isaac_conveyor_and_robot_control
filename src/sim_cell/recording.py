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
from dataclasses import dataclass
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
_ZERO_VEC = (0.0, 0.0, 0.0)

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

# RunMetadata.control_source / run_label - see resolve_control_source and
# _build_run_metadata below. Both are free-form operator/launcher-set
# metadata; neither has a default derived from sim state the way
# instance_index/spawn_seed do.
CONTROL_SOURCE_ENV_VAR = "CONVEYOR_INDEXING_CONTROL_SOURCE"
RUN_LABEL_ENV_VAR = "CONVEYOR_INDEXING_RUN_LABEL"
_CONTROL_SOURCE_SCRIPTED = "scripted"
_CONTROL_SOURCE_POLICY_UNKNOWN = "policy:unknown"

# Same env var sim_cell.runner/sim_cell.cell read directly (see their own
# EXTERNAL_ACTION_ENV_VAR / "CONVEYOR_INDEXING_EXTERNAL_ACTION" literals) -
# duplicated here rather than imported, since importing from either of those
# would pull this Isaac-Sim-optional module into their heavier import chain
# (both already import *this* module, so the reverse import would cycle).
_EXTERNAL_ACTION_ENV_VAR = "CONVEYOR_INDEXING_EXTERNAL_ACTION"

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


def validate_external_action_recording(external_action: bool, episode_recorder_enabled: bool) -> None:
    """Raises SystemExit if `external_action` (CONVEYOR_INDEXING_EXTERNAL_ACTION=1)
    is incompatible with the given recorder configuration - see sim_cell.runner.

    Only CONVEYOR_INDEXING_RECORD (the 30Hz episode/parquet recorder,
    `episode_recorder_enabled`) still conflicts: its episode segmentation has
    no defined meaning once an external controller owns the phase machine.
    CONVEYOR_INDEXING_RECORD_MCAP (the episode-free ground-truth recorder) is
    fine at the same time as external_action - needed for on-policy eval
    recording - as long as the caller also skips phase-transition recording
    while external_action is set (the phase state machine is dormant there;
    see sim_cell.runner's mcap_recorder.record_phase_transition guard).
    """
    if external_action and episode_recorder_enabled:
        raise SystemExit(
            f"{_EXTERNAL_ACTION_ENV_VAR}=1 is incompatible with {RECORD_ENV_VAR} (the 30Hz "
            "episode/parquet recorder) - episode segmentation has no defined meaning once an "
            f"external controller owns the phase machine. {MCAP_RECORD_ENV_VAR} is fine at the "
            "same time (needed for on-policy eval recording) - only disable CONVEYOR_INDEXING_RECORD."
        )


def resolve_arm_telemetry(
    external_action: bool,
    held_box_path_1: str | None,
    held_box_path_2: str | None,
    pick_place_1_holding: bool,
    pick_place_2_holding: bool,
    pick_place_1_held_box_path: str | None,
    pick_place_2_held_box_path: str | None,
) -> tuple[bool, bool, dict]:
    """Derive (holding_1, holding_2, held_by_arm) for mcap ground-truth
    telemetry (PositionStatus.dio_blocks + BoxState.held_by_arm) from the
    right source depending on control mode.

    In external_action mode, MagicAttachPickPlace.holding_box/held_box_path
    are frozen (forward() never runs - see sim_cell.runner): the only ground
    truth is held_box_path_1/2, tracked by pick_and_place.apply_suction_edge.
    Before this fix, sim_cell.runner's mcap-recording block always read
    cell.pick_place(_2).holding_box/held_box_path regardless of control mode,
    so every policy-run KPI silently read "never holding" (the phase machine
    never advances past WAITING while dormant). Autonomous mode is unchanged:
    it uses the phase machine's own holding_box/held_box_path, same as before.
    """
    if external_action:
        holding_1 = held_box_path_1 is not None
        holding_2 = held_box_path_2 is not None
        held_by_arm: dict = {}
        if held_box_path_1:
            held_by_arm[held_box_path_1] = 1
        if held_box_path_2:
            held_by_arm[held_box_path_2] = 2
        return holding_1, holding_2, held_by_arm

    held_by_arm = {}
    if pick_place_1_held_box_path:
        held_by_arm[pick_place_1_held_box_path] = 1
    if pick_place_2_held_box_path:
        held_by_arm[pick_place_2_held_box_path] = 2
    return pick_place_1_holding, pick_place_2_holding, held_by_arm


def resolve_control_source(external_action: bool, env_value: str | None) -> str:
    """RunMetadata.control_source for this run - "scripted" | "policy:<id>".

    An explicit CONVEYOR_INDEXING_CONTROL_SOURCE always wins. Left unset: an
    autonomous (non-external-action) run defaults to "scripted" (its actual
    behavior); an external-action run defaults to "policy:unknown" rather
    than silently mislabeling an on-policy run as scripted - so it never
    enters a training superset by accident (see the multi-policy/VLM plan's
    Workstream 2 manifest rule).
    """
    if env_value:
        return env_value
    return _CONTROL_SOURCE_POLICY_UNKNOWN if external_action else _CONTROL_SOURCE_SCRIPTED


@dataclass(frozen=True)
class ZoneGeometryInput:
    """Plain-value snapshot of one conveyor zone's static geometry/tuning -
    everything RunMetadata.zone_geometry needs, extracted from a live
    conveyor_indexing.zone.ConveyorZone (+ its line's run_speed_pct, hold-zone
    status, and conveyor_indexing.state_machine.HOLD_ZONE_STOP_FRACTION) by
    sim_cell.cell at run-metadata build time. Kept as plain floats/tuples
    (not the zone object itself) so build_zone_geometry_proto stays
    unit-testable without Isaac Sim/pxr.
    """

    node_path: str
    bbox_center: tuple
    bbox_half_extent: tuple
    belt_top_z: float
    travel_direction: tuple  # (0.0, 0.0, 0.0) if unset (curved zone - see ZoneGeometry's proto comment)
    stop_fraction: float
    speed_m_per_s: float
    is_hold_zone: bool
    line_id: int


def build_zone_geometry_proto(zone: ZoneGeometryInput) -> sim_state_pb2.ZoneGeometry:
    cx, cy, cz = zone.bbox_center
    hx, hy, hz = zone.bbox_half_extent
    return sim_state_pb2.ZoneGeometry(
        node_path=zone.node_path,
        aabb=sim_state_pb2.Aabb(
            min=sim_state_pb2.Vec3(x=cx - hx, y=cy - hy, z=cz - hz),
            max=sim_state_pb2.Vec3(x=cx + hx, y=cy + hy, z=cz + hz),
        ),
        belt_top_z=zone.belt_top_z,
        travel_direction=sim_state_pb2.Vec3(
            x=zone.travel_direction[0], y=zone.travel_direction[1], z=zone.travel_direction[2]
        ),
        stop_fraction=zone.stop_fraction,
        speed_m_per_s=zone.speed_m_per_s,
        is_hold_zone=zone.is_hold_zone,
        line_id=zone.line_id,
    )


def build_aabb_proto(min_xyz: tuple, max_xyz: tuple) -> sim_state_pb2.Aabb:
    return sim_state_pb2.Aabb(
        min=sim_state_pb2.Vec3(x=min_xyz[0], y=min_xyz[1], z=min_xyz[2]),
        max=sim_state_pb2.Vec3(x=max_xyz[0], y=max_xyz[1], z=max_xyz[2]),
    )


def build_transform_proto(translate: tuple, orientation_wxyz: tuple) -> sim_state_pb2.Transform:
    return sim_state_pb2.Transform(
        translate=sim_state_pb2.Vec3(x=translate[0], y=translate[1], z=translate[2]),
        orientation=sim_state_pb2.Quat(
            w=orientation_wxyz[0], x=orientation_wxyz[1], y=orientation_wxyz[2], z=orientation_wxyz[3]
        ),
    )


@dataclass(frozen=True)
class PoolVariantInput:
    """Plain-value snapshot of one box_pool variant - see build_pool_variant_proto."""

    variant: str
    asset_url: str
    count: int
    half_extent: tuple


def build_pool_variant_proto(pool_variant: PoolVariantInput) -> sim_state_pb2.PoolVariant:
    return sim_state_pb2.PoolVariant(
        variant=pool_variant.variant,
        asset_url=pool_variant.asset_url,
        count=pool_variant.count,
        half_extent=sim_state_pb2.Vec3(
            x=pool_variant.half_extent[0], y=pool_variant.half_extent[1], z=pool_variant.half_extent[2]
        ),
    )


@dataclass(frozen=True)
class RunMetadataExtras:
    """Everything RunMetadata's P4 geometry/config additions need, beyond
    camera_specs/spawn_seed - bundled here so sim_cell.cell (the only caller
    with live access to zones/robots/the box pool) has one object to build
    and pass through maybe_build_mcap_recorder -> _build_run_metadata.
    """

    zone_geometry: list  # list[ZoneGeometryInput]
    truck_bed_min: tuple
    truck_bed_max: tuple
    robot_1_transform: tuple  # (translate_xyz, orientation_wxyz)
    robot_2_transform: tuple  # (translate_xyz, orientation_wxyz) - see Transform's proto comment
    attach_max_distance_m: float
    pool_variants: list  # list[PoolVariantInput]
    camera_horizontal_aperture_mm: float


def _build_run_metadata(camera_specs, spawn_seed: int, extras: RunMetadataExtras) -> sim_state_pb2.RunMetadata:
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
        control_source=resolve_control_source(
            os.environ.get(_EXTERNAL_ACTION_ENV_VAR) == "1", os.environ.get(CONTROL_SOURCE_ENV_VAR)
        ),
        run_label=os.environ.get(RUN_LABEL_ENV_VAR, ""),
        zone_geometry=[build_zone_geometry_proto(z) for z in extras.zone_geometry],
        truck_bed_aabb=build_aabb_proto(extras.truck_bed_min, extras.truck_bed_max),
        robot_1_base_transform=build_transform_proto(*extras.robot_1_transform),
        robot_2_base_transform=build_transform_proto(*extras.robot_2_transform),
        attach_max_distance_m=extras.attach_max_distance_m,
        pool_variants=[build_pool_variant_proto(p) for p in extras.pool_variants],
        camera_horizontal_aperture_mm=extras.camera_horizontal_aperture_mm,
    )


def maybe_build_mcap_recorder(camera_specs, spawn_seed: int, extras: RunMetadataExtras) -> McapRecorder | None:
    """An McapRecorder when CONVEYOR_INDEXING_RECORD_MCAP=1, else None (default off).
    Independent of maybe_build_recorder's CONVEYOR_INDEXING_RECORD - either,
    both, or neither can be enabled for a given run.
    """
    if os.environ.get(MCAP_RECORD_ENV_VAR, "0") != "1":
        return None
    logger.info("mcap recording enabled -> %s", MCAP_OUTPUT_DIR)
    recorder = McapRecorder(
        output_dir=MCAP_OUTPUT_DIR,
        run_metadata=_build_run_metadata(camera_specs, spawn_seed, extras),
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

    ``box_linear_vel``/``box_angular_vel`` are looked up with a zero-vector
    default (Stage 5b, docs/progress-tracker.md): they're only populated
    when an MCAP recorder is attached (an extra PhysX velocity-sync call a
    live-only box-telemetry publish - a policy needs position/hold state,
    not velocity - shouldn't have to pay for).
    """
    boxes = []
    for path in active_box_paths:
        pos = box_positions[path]
        quat = box_orientations[path]  # (w, x, y, z) - Isaac Sim convention, see proto/sim_state.proto
        lin = box_linear_vel.get(path, _ZERO_VEC)
        ang = box_angular_vel.get(path, _ZERO_VEC)
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
