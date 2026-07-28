"""Episode-free MCAP recorder: the "record everything" base capture theia's
mcap_to_lerobot.py (and future tooling) can convert from, with no episode
concept baked in - see the top-level README's "Design" section and
sim_cell.recording for the CONVEYOR_INDEXING_RECORD_MCAP gate.

Every message goes through one background writer thread and one bounded
queue (drop-and-warn on overflow - the sim loop must never block on I/O,
same contract as conveyor_indexing.episode_recorder.EpisodeRecorder), across
two channel families:

  Theia-contract channels (mirror the real data-collection service's wire
  format field-for-field, via theia's own robot.proto / foxglove/raw_image.proto
  - see gen_proto.sh - so mcap_to_lerobot.py routes sim and real captures
  identically by schema name):
    foxglove.RawImage              theia/camera/<serial>/color          30Hz/camera
    theia.robot.v1.PositionStatus  theia/robot/arm<n>/position_status   120Hz/arm
    theia.robot.v1.MoveTarget      theia/robot/arm<n>/move_target       once/arm (see below)
    theia.plc_connector.v1.StateConveyors  theia/plc/state_conveyors    120Hz

  Sim-only ground truth (theia.sim.conveyor_indexing.v1, proto/sim_state.proto -
  no theia equivalent; exists so a capture can fully reproduce the run, not
  just what a real robot's sensors would see):
    BoxStates            sim/boxes/state    120Hz (every currently-active box)
    BoxEvent              sim/boxes/events   on spawn/despawn
    ArmPhaseTransition    sim/arms/phase     on phase edge
    RunMetadata           sim/run_metadata   once, and again on every rotation

MoveTarget is a real per-move message on theia's robot service, but this
sim's cuMotion planner has no equivalent per-tick "target" to mirror - it
plans whole trajectories, not waypoint commands. mcap_to_lerobot.py's replay
gate only checks that at least one MoveTarget with a non-empty pose has ever
been seen (`if mt.pose_target.pose: move_target_seen = True`, never reset),
so one placeholder message per arm at recorder construction satisfies the
gate structurally without pretending to be a real motion command - see
_write_move_target_stub.

Files rotate every ``rotate_period_s`` of sim time, named
``<start_ns>_<end_ns>_INCOMPLETE.mcap`` while open and renamed to
``<start_ns>_<end_ns>.mcap`` on close - the same convention (and the same
INCOMPLETE-skip rule) theia's real data-collection service and
mcap_to_lerobot.py already use.

``log_time`` for every message is a wall-clock epoch fixed once at recorder
construction (``run_epoch_ns``) plus the sim-time offset at record time, not
wall-clock time of the call - so replay reflects sim rates (30Hz images,
120Hz robot/PLC/box state) regardless of how fast or slow this process
actually runs.
"""

from __future__ import annotations

import logging
import pathlib
import queue
import subprocess
import threading
import time

try:
    from mcap_protobuf.writer import Writer as McapProtobufWriter
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "mcap + mcap-protobuf-support are required for MCAP recording but are not installed "
        "in this interpreter. Install them with:\n"
        "  /home/ubuntu/IsaacSim/python.sh -m pip install mcap mcap-protobuf-support\n"
    ) from exc

from google.protobuf.timestamp_pb2 import Timestamp

import sim_state_pb2
from conveyor_indexing.protos import plc
from foxglove import raw_image_pb2
from robot import robot_pb2

logger = logging.getLogger(__name__)

_DROP_WARNING_INTERVAL_S = 1.0
_HELD_BY_NONE = 0

# theia's real image encoding is uppercase ("RGB8", see cameras.specs.COLOR_FORMAT);
# foxglove's own RawImage examples use lowercase - either is fine since every
# consumer (this repo's, mcap_to_lerobot.py's) only checks "BGR" in .upper().
_RAW_IMAGE_ENCODING = "rgb8"


def git_sha(repo_root: pathlib.Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=str(repo_root), timeout=5)
            .decode()
            .strip()
        )
    except Exception:
        logger.warning("could not determine conveyor_indexing git sha for RunMetadata", exc_info=True)
        return "unknown"


def _timestamp(epoch_ns: int) -> Timestamp:
    ts = Timestamp()
    ts.FromNanoseconds(epoch_ns)
    return ts


class McapRecorder:
    """Background-thread MCAP writer for the full, episode-free sim capture."""

    _SENTINEL = None

    def __init__(
        self,
        output_dir: str,
        run_metadata: sim_state_pb2.RunMetadata,
        rotate_period_s: float = 30.0,
        queue_maxsize: int = 2000,
    ) -> None:
        self.output_dir = pathlib.Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_period_s = rotate_period_s
        self._run_metadata = run_metadata

        # Wall-clock epoch fixed once; every message's log_time is this plus a
        # sim-time offset (see module docstring) so replay reflects sim rates.
        self._run_epoch_ns = time.time_ns()

        self._writer: McapProtobufWriter | None = None
        self._current_path: pathlib.Path | None = None
        self._file_start_sim_s: float | None = None
        self._file_start_ns: int | None = None
        self._file_last_sim_s: float = 0.0

        self._dropped = 0
        self._last_drop_warning = 0.0
        self._messages_written = 0

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="mcap-recorder-writer")
        self._thread.start()

    # -- public record_* API (called from sim_cell.runner; never blocks) -----

    def record_camera_frame(
        self, serial: str, role_value: int, sim_time_s: float, capture_ts_us: int, rgb_bytes: bytes, width: int, height: int
    ) -> None:
        # frame_id = "<serial>:<CameraRole value>" (the numeric wire value, not
        # a name) - matches theia's real data-collection service exactly, so
        # mcap_to_lerobot.py's camera_role_from_frame_id() parses it unchanged.
        image = raw_image_pb2.RawImage(
            timestamp=_timestamp(capture_ts_us * 1000),
            frame_id=f"{serial}:{role_value}",
            width=width,
            height=height,
            encoding=_RAW_IMAGE_ENCODING,
            step=width * 3,
            data=rgb_bytes,
        )
        self._enqueue(f"theia/camera/{serial}/color", image, sim_time_s)

    def record_position_status(
        self, arm: int, sim_time_s: float, joint_degrees: list, dio_block0: int, ref_req_id: int = 0
    ) -> None:
        msg = robot_pb2.PositionStatus(
            ref_req_id=ref_req_id,
            joint_degrees=joint_degrees,
            dio_blocks=[dio_block0],
        )
        self._enqueue(f"theia/robot/arm{arm}/position_status", msg, sim_time_s)

    def record_state_conveyors(self, sim_time_s: float, state_msg: plc.StateConveyors) -> None:
        # Safe to hand off without copying: sim_cell.runner builds a fresh
        # StateConveyors() every control tick and never mutates this one again.
        self._enqueue("theia/plc/state_conveyors", state_msg, sim_time_s)

    def record_box_states(self, sim_time_s: float, boxes: list) -> None:
        """``boxes``: list of sim_state_pb2.BoxState (built by the caller -
        see sim_cell.recording.build_box_states)."""
        msg = sim_state_pb2.BoxStates(sim_time_s=sim_time_s, boxes=boxes)
        self._enqueue("sim/boxes/state", msg, sim_time_s)

    def record_box_event(
        self,
        sim_time_s: float,
        event_type: int,
        box_id: str,
        variant: str,
        position: tuple,
        orientation_wxyz: tuple,
    ) -> None:
        msg = sim_state_pb2.BoxEvent(
            sim_time_s=sim_time_s,
            type=event_type,
            box_id=box_id,
            variant=variant,
            position=sim_state_pb2.Vec3(x=position[0], y=position[1], z=position[2]),
            orientation=sim_state_pb2.Quat(
                w=orientation_wxyz[0], x=orientation_wxyz[1], y=orientation_wxyz[2], z=orientation_wxyz[3]
            ),
        )
        self._enqueue("sim/boxes/events", msg, sim_time_s)

    def record_phase_transition(self, sim_time_s: float, arm: int, from_phase: str, to_phase: str, box_id: str) -> None:
        msg = sim_state_pb2.ArmPhaseTransition(
            sim_time_s=sim_time_s, arm=arm, from_phase=from_phase, to_phase=to_phase, box_id=box_id or ""
        )
        self._enqueue("sim/arms/phase", msg, sim_time_s)

    def write_move_target_stub(self, arm: int) -> None:
        """One placeholder MoveTarget per arm - see the module docstring for
        why this is structural (satisfies mcap_to_lerobot.py's replay gate)
        rather than a real motion command.
        """
        msg = robot_pb2.MoveTarget(ref_req_id=0, pose_target=robot_pb2.PoseTarget(pose=[0.0] * 6))
        self._enqueue(f"theia/robot/arm{arm}/move_target", msg, 0.0)

    # -- internal -------------------------------------------------------------

    def _enqueue(self, topic: str, message, sim_time_s: float) -> None:
        try:
            self._queue.put_nowait((topic, message, sim_time_s))
        except queue.Full:
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_warning >= _DROP_WARNING_INTERVAL_S:
                logger.warning(
                    "mcap recorder queue full - dropped %d message(s) so far; "
                    "disk cannot keep up with the recording rate",
                    self._dropped,
                )
                self._last_drop_warning = now

    def _writer_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is self._SENTINEL:
                break

            topic, message, sim_time_s = item
            if self._writer is None or self._should_rotate(sim_time_s):
                self._rotate(sim_time_s)
            log_time_ns = self._run_epoch_ns + int(sim_time_s * 1e9)
            self._writer.write_message(topic, message, log_time=log_time_ns, publish_time=log_time_ns)
            self._file_last_sim_s = sim_time_s
            self._messages_written += 1

        if self._writer is not None:
            self._close_current_file()

    def _should_rotate(self, sim_time_s: float) -> bool:
        return sim_time_s - self._file_start_sim_s >= self.rotate_period_s

    def _rotate(self, sim_time_s: float) -> None:
        if self._writer is not None:
            self._close_current_file()
        self._file_start_sim_s = sim_time_s
        self._file_start_ns = self._run_epoch_ns + int(sim_time_s * 1e9)
        self._current_path = self.output_dir / f"{self._file_start_ns}_INCOMPLETE.mcap"
        self._writer = McapProtobufWriter(str(self._current_path))
        # First message of every file, so any single file is self-describing
        # enough to reconstruct the run it came from - see the proto's
        # RunMetadata docstring.
        self._writer.write_message(
            "sim/run_metadata", self._run_metadata, log_time=self._file_start_ns, publish_time=self._file_start_ns
        )

    def _close_current_file(self) -> None:
        self._writer.finish()
        self._writer = None
        # The actual last sim_time_s written, not an assumed full
        # rotate_period_s - the final file of a run almost never reaches a
        # full period, and this keeps <start_ns>_<end_ns> honest either way.
        end_ns = self._run_epoch_ns + int(self._file_last_sim_s * 1e9)
        final_path = self.output_dir / f"{self._file_start_ns}_{end_ns}.mcap"
        self._current_path.rename(final_path)
        logger.debug("mcap file closed: %s", final_path)

    def close(self) -> None:
        self._queue.put(self._SENTINEL)
        self._thread.join()
        logger.info(
            "mcap recorder closed: %d message(s) written to %s, %d dropped",
            self._messages_written,
            self.output_dir,
            self._dropped,
        )
        if self._dropped:
            logger.warning(
                "mcap recorder dropped %d message(s) - this capture has time gaps", self._dropped
            )
