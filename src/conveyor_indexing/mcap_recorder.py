"""Episode-free MCAP recorder: every live channel, one background writer thread.

Channels (all generic sim schemas):
  foxglove.RawImage     sim/camera/<id>/color     per camera frame
  SimArmState           sim/arm/<n>/state         per control tick
  SimConveyorStates     sim/conveyor/state        per control tick
  BoxStates/BoxEvent    sim/boxes/state, sim/boxes/events
  ArmPhaseTransition    sim/arms/phase            autonomous mode only
  ArmToolPose           sim/arm/<n>/tool_pose
  SimArmActionCommand   sim/arm/<n>/action_command   external-action mode only
  SimConveyorCommands   sim/conveyor/command         external-action mode only
  RunMetadata           sim/run_metadata          first message of every file

Files rotate every `rotate_period_s` of sim time; `<start>_<end>_INCOMPLETE.mcap`
while open, renamed on close. log_time = a fixed wall epoch + sim time, so
replay runs at sim rates.
"""

from __future__ import annotations

import json
import logging
import pathlib
import queue
import subprocess
import threading
import time

try:
    from mcap_protobuf.writer import Writer as McapProtobufWriter
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("mcap + mcap-protobuf-support are not installed in this interpreter; see scripts/setup.sh") from exc

import sim_arm_action_pb2
import sim_state_pb2
from foxglove import raw_image_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from conveyor_indexing.protos import sim_action, telemetry

logger = logging.getLogger(__name__)

_DROP_WARNING_INTERVAL_S = 1.0
_HELD_BY_NONE = 0

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
        # frame_id = "<serial>:<CameraRole value>".
        image = raw_image_pb2.RawImage(
            timestamp=_timestamp(capture_ts_us * 1000),
            frame_id=f"{serial}:{role_value}",
            width=width,
            height=height,
            encoding=_RAW_IMAGE_ENCODING,
            step=width * 3,
            data=rgb_bytes,
        )
        self._enqueue(f"sim/camera/{serial}/color", image, sim_time_s)

    def record_arm_state(
        self, arm: int, sim_time_s: float, joint_positions_rad, joint_velocities_rad_s, holding: bool,
        tool_position, tool_orientation_wxyz,
    ) -> None:
        msg = telemetry.SimArmState(
            arm=arm,
            joint_positions_rad=[float(v) for v in joint_positions_rad],
            joint_velocities_rad_s=[float(v) for v in joint_velocities_rad_s],
            holding=holding,
            tool_position=sim_state_pb2.Vec3(
                x=float(tool_position[0]), y=float(tool_position[1]), z=float(tool_position[2])
            ),
            tool_orientation=sim_state_pb2.Quat(
                w=float(tool_orientation_wxyz[0]), x=float(tool_orientation_wxyz[1]),
                y=float(tool_orientation_wxyz[2]), z=float(tool_orientation_wxyz[3]),
            ),
            sim_time_us=int(sim_time_s * 1e6),
        )
        self._enqueue(f"sim/arm/{arm}/state", msg, sim_time_s)

    def record_conveyor_states(self, sim_time_s: float, state_msg: telemetry.SimConveyorStates) -> None:
        # The runner builds a fresh message every tick and never mutates it afterwards.
        self._enqueue("sim/conveyor/state", state_msg, sim_time_s)

    def record_box_states(self, sim_time_s: float, boxes: list, truck_deliveries_count: int = 0) -> None:
        """``boxes``: list of sim_state_pb2.BoxState (built by the caller -
        see sim_cell.recording.build_box_states). ``truck_deliveries_count``
        (added - was missing here even though the live
        ``robot_state_publisher.publish_box_states`` counterpart already
        takes and threads it through; the caller's own running total was
        silently dropped on the MCAP path, so every post-hoc reader of
        `sim/boxes/state` alone saw a permanent 0, unable to tell a genuine
        truck delivery apart from a floor/stale despawn from this channel
        alone - exactly the ambiguity `BoxStates.truck_deliveries_count`
        exists to resolve, see sim_state.proto's own comment on that field).
        """
        msg = sim_state_pb2.BoxStates(sim_time_s=sim_time_s, boxes=boxes, truck_deliveries_count=truck_deliveries_count)
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

    def record_tool_pose(self, arm: int, sim_time_s: float, position: tuple, orientation_wxyz: tuple) -> None:
        """Arm `arm`'s tool-frame (wrist_3_link/flange) world pose this
        control tick - see pick_and_place.controller.MagicAttachPickPlace.
        tool_world_pose(). Independent of control mode (unlike
        holding_box/held_box_path, this GeomPrim read stays live whether or
        not the phase state machine is running), recorded at the same 120Hz
        cadence as BoxStates/PositionStatus so forward kinematics is never
        needed downstream to reconstruct tool position for the 0.35m
        proximity-based error detectors.
        """
        msg = sim_state_pb2.ArmToolPose(
            sim_time_s=sim_time_s,
            arm=arm,
            position=sim_state_pb2.Vec3(x=float(position[0]), y=float(position[1]), z=float(position[2])),
            orientation=sim_state_pb2.Quat(
                w=float(orientation_wxyz[0]),
                x=float(orientation_wxyz[1]),
                y=float(orientation_wxyz[2]),
                z=float(orientation_wxyz[3]),
            ),
        )
        self._enqueue(f"sim/arm/{arm}/tool_pose", msg, sim_time_s)

    def record_arm_action_command(self, arm: int, sim_time_s: float, cmd: sim_arm_action_pb2.SimArmActionCommand) -> None:
        """The externally-supplied per-tick arm command actually applied this
        physics step (CONVEYOR_INDEXING_EXTERNAL_ACTION=1 only - see
        sim_cell.runner / sim_cell.external_command_bridge): the on-policy
        action log needed for eval + DAgger. `cmd` is the exact message
        applied this tick (carries its own sender-assigned monotonic `seq`
        field verbatim, since the whole message is recorded); `sim_time_s` is
        when it was actually applied to the sim, not when it was received off
        Zenoh - same convention as every other record_* method here.
        """
        self._enqueue(f"sim/arm/{arm}/action_command", cmd, sim_time_s)

    def record_conveyor_command(self, sim_time_s: float, cmd: sim_action.SimConveyorCommands) -> None:
        """The externally-supplied conveyor command actually applied this
        control tick (CONVEYOR_INDEXING_EXTERNAL_ACTION=1 only) - same
        on-policy action log rationale as record_arm_action_command.
        SimConveyorCommand carries no per-message seq (unlike
        SimArmActionCommand) - only sim_time_s orders these on replay.
        """
        self._enqueue("sim/conveyor/command", cmd, sim_time_s)

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
        self._persist_stats()

    def _persist_stats(self) -> None:
        """Write final message/drop counts to a small sidecar JSON in
        output_dir, next to the mcap files themselves - process log output
        alone doesn't reliably survive an unattended multi-hour collection
        run (see scripts/collect_local.py), the way a file under output_dir
        does (collect_local.py already streams that whole directory to GCS).
        Eval's QA gate (recording-integrity checks) can read this directly
        instead of re-deriving a drop count from mcap gaps.
        """
        stats_path = self.output_dir / "recorder_stats.json"
        try:
            stats_path.write_text(
                json.dumps(
                    {
                        "messages_written": self._messages_written,
                        "dropped": self._dropped,
                        "run_epoch_ns": self._run_epoch_ns,
                    },
                    indent=2,
                )
            )
        except OSError:
            logger.warning("could not write recorder_stats.json to %s", self.output_dir, exc_info=True)
