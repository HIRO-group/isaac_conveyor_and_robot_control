"""30Hz training-data recorder: synchronized camera frames + robot/cell state.

Written for consumption by theia's ``dc_to_lerobot.py`` converter, which reads
these columns by name (extra columns are tolerated and ignored):

  reference_req_id     int64    episode key; the converter starts a new episode
                                whenever the value changes. The sim writes its
                                default segmentation here (see
                                sim_cell.recording.EpisodeTracker); alternative
                                segmentations are derived at conversion time
                                from the extra columns below.
  observation.state    binary   float32 vector bytes - two 15-float arm blocks
                                (6 joints rad, suction, 8 cups); conveyor dims
                                are appended by the converter from
                                plc_state_conveyors, matching theia's collector
  observation.images   binary   numpy.savez NPZ bytes, one CHW uint8 array per
                                camera role key (savez uncompressed - parquet
                                zstd below does the compression)
  plc_state_conveyors  binary   plc_connector_pb2.StateConveyors bytes, same
                                message the 120Hz tick log records

Extra columns for conversion-time re-segmentation and debugging:

  tick                 int64    control-loop tick at capture time
  sim_time_s           double   elapsed sim time at capture time
  phase_1 / phase_2    string   each arm's pick-and-place phase name

Rows arrive at CAMERA_FPS and are ~5.5MB raw (6 x 640x480 RGB), so unlike
ConveyorIndexingLogger this writer:
  - keeps row groups small (``batch_size``): the converter re-reads a full row
    group per written frame when loading images, so row-group size multiplies
    its decode cost;
  - bounds the queue and drops rows (with a warning) rather than let a slow
    disk grow memory without limit - the sim loop must never block on I/O;
  - rotates output files (``rotate_after_rows``), deferring the cut to the
    next reference_req_id change so the converter's don't-cross-file-boundary
    rule never truncates a window mid-episode.

HWC->CHW transposes and NPZ encoding happen on the writer thread; ``record()``
only builds a dict and enqueues (frames from CameraRig.capture_all() are
freshly allocated per call, so handing them off is safe).
"""

from __future__ import annotations

import io
import logging
import pathlib
import queue
import threading
import time

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "pyarrow is required for episode_recorder.py. Install it with: pip install pyarrow"
    ) from exc

logger = logging.getLogger(__name__)

_DROP_WARNING_INTERVAL_S = 1.0


class EpisodeRecorder:
    """Background-thread batched parquet writer for 30Hz image+state rows."""

    _SENTINEL = None

    def __init__(
        self,
        output_dir: str,
        serial_to_role: dict[str, str],
        width: int,
        height: int,
        batch_size: int = 15,
        rotate_after_rows: int = 900,
        queue_maxsize: int = 60,
        compression: str | None = "zstd",
    ) -> None:
        self.output_dir = pathlib.Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.serial_to_role = dict(serial_to_role)
        self.expected_serials = frozenset(serial_to_role)
        self.width = width
        self.height = height
        self.batch_size = max(1, batch_size)
        self.rotate_after_rows = max(1, rotate_after_rows)
        self.compression = compression

        # Fixed per run so files sort chronologically across runs sharing the dir.
        self._run_prefix = int(time.time() * 1e3)
        self._file_seq = 0
        self._writer: pq.ParquetWriter | None = None
        self._rows_in_file = 0
        self._last_rri: int | None = None
        self._rows_written = 0

        self._dropped = 0
        self._last_drop_warning = 0.0

        self._schema = pa.schema(
            [
                ("reference_req_id", pa.int64()),
                ("observation.state", pa.binary()),
                ("observation.images", pa.binary()),
                ("plc_state_conveyors", pa.binary()),
                ("tick", pa.int64()),
                ("sim_time_s", pa.float64()),
                ("phase_1", pa.string()),
                ("phase_2", pa.string()),
            ]
        )

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="episode-recorder-writer")
        self._thread.start()

    def record(
        self,
        reference_req_id: int,
        observation_state: np.ndarray,
        frames: dict[str, bytes],
        plc_state_conveyors: bytes,
        tick: int,
        sim_time_s: float,
        phase_1: str,
        phase_2: str,
    ) -> None:
        """Queue one row; drops (with a rate-limited warning) if the writer is behind.

        ``frames`` is {serial: raw RGB8 HWC bytes} and must cover every serial in
        ``serial_to_role`` - the caller (sim_cell.runner) skips iterations where
        capture_all() returned a partial set (annotator warm-up).
        """
        row = {
            "reference_req_id": reference_req_id,
            "observation.state": np.asarray(observation_state, dtype=np.float32).tobytes(),
            "frames": frames,
            "plc_state_conveyors": plc_state_conveyors,
            "tick": tick,
            "sim_time_s": sim_time_s,
            "phase_1": phase_1,
            "phase_2": phase_2,
        }
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_warning >= _DROP_WARNING_INTERVAL_S:
                logger.warning(
                    "episode recorder queue full - dropped %d frame(s) so far; "
                    "disk cannot keep up with the recording rate",
                    self._dropped,
                )
                self._last_drop_warning = now

    def _encode_images(self, frames: dict[str, bytes]) -> bytes:
        arrays = {}
        for serial, rgb_bytes in frames.items():
            hwc = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(self.height, self.width, 3)
            arrays[self.serial_to_role[serial]] = hwc.transpose(2, 0, 1)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        return buf.getvalue()

    def _writer_loop(self) -> None:
        pending = []
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                if pending:
                    self._flush_rows(pending)
                    pending = []
                continue

            if item is self._SENTINEL:
                if pending:
                    self._flush_rows(pending)
                break

            item["observation.images"] = self._encode_images(item.pop("frames"))
            if self._should_rotate(item["reference_req_id"], len(pending)):
                # Flush before cutting so rows queued ahead of the boundary stay
                # in the old file - the cut must fall exactly between episodes.
                if pending:
                    self._flush_rows(pending)
                    pending = []
                self._writer.close()
                self._writer = None
                self._rows_in_file = 0
            self._last_rri = item["reference_req_id"]
            pending.append(item)
            if len(pending) >= self.batch_size:
                self._flush_rows(pending)
                pending = []

    def _should_rotate(self, reference_req_id: int, pending_count: int) -> bool:
        """Cut a new file once past rotate_after_rows, deferred to the next episode-key
        change (so no converter window straddles a file boundary mid-episode), with a
        2x hard cap in case one episode runs unusually long.
        """
        if self._writer is None:
            return False
        rows_so_far = self._rows_in_file + pending_count
        if rows_so_far < self.rotate_after_rows:
            return False
        episode_boundary = self._last_rri is not None and reference_req_id != self._last_rri
        return episode_boundary or rows_so_far >= 2 * self.rotate_after_rows

    def _current_path(self) -> pathlib.Path:
        return self.output_dir / f"{self._run_prefix}_{self._file_seq:04d}.parquet"

    def _flush_rows(self, rows: list) -> None:
        columns = {name: [r[name] for r in rows] for name in self._schema.names}
        table = pa.Table.from_pydict(columns, schema=self._schema)
        if self._writer is None:
            self._file_seq += 1
            self._writer = pq.ParquetWriter(
                where=str(self._current_path()),
                schema=self._schema,
                compression=self.compression,
            )
        self._writer.write_table(table)
        self._rows_in_file += len(rows)
        self._rows_written += len(rows)

    def close(self) -> None:
        self._queue.put(self._SENTINEL)
        self._thread.join()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        logger.info(
            "episode recorder closed: %d row(s) written to %s, %d dropped",
            self._rows_written,
            self.output_dir,
            self._dropped,
        )
        if self._dropped:
            logger.warning(
                "episode recorder dropped %d frame(s) - the recording has time gaps "
                "and should not be used for training",
                self._dropped,
            )
