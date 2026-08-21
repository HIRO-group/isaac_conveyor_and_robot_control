"""Per-tick parquet logger for the conveyor indexing sim.

Schema mirrors theia's real data collection layout (see
``~/theia/data_collection/src/data_collection_vol2.py``): binary protobuf
columns, written in background-thread batches so the sim control loop never
blocks on I/O.

Columns:
  tick                 int64    control-loop tick counter, monotonic per run
  sim_time_s           double   elapsed sim time in seconds
  plc_state_conveyors  binary   plc_connector_pb2.StateConveyors bytes -
                                one StateConveyors_ConveyorsItem per zone,
                                same message theia's real PLC connector
                                publishes on theia/plc/v1/state/Conveyors
  conveyor_commands    binary   sim_conveyor_action_pb2.SimConveyorCommands
                                bytes - one SimConveyorCommand per zone

This is intentionally NOT wired to a live Zenoh session (see this
directory's README for why) - it's a standalone parquet file, schema-
compatible with production so it can be merged with real collected data
later, or replayed through the same tooling (test_collected_data_parquet.py)
once plc_connector_pb2 bindings are on PYTHONPATH.

Queue is bounded and instrumented the same way conveyor_indexing.mcap_recorder.
McapRecorder already is (drop-and-warn on overflow, persisted close-time
stats) - this logger previously used an UNBOUNDED queue.Queue() with a plain
blocking put() and zero drop/backpressure signal of any kind, unlike every
other writer thread in this package. That couldn't itself stall the sim's
main thread (an unbounded put() never blocks the caller), but it was a real,
silent-growth risk with no way to see it happening - this fix closes that gap
and gives a capability-diffusion real-collection smoke test a concrete
queue_depth()/dropped-count signal to watch, whatever turns out to actually
cause the still-unexplained ~195s recording stall.
"""

from __future__ import annotations

import json
import logging
import pathlib
import queue
import threading
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

_DROP_WARNING_INTERVAL_S = 1.0

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "pyarrow is required for conveyor_indexing_logger.py. Install it with: "
        "pip install pyarrow"
    ) from exc


class ConveyorIndexingLogger:
    """Background-thread batched parquet writer for conveyor indexing ticks."""

    _SENTINEL = None

    def __init__(
        self,
        output_path: str,
        batch_size: int = 100,
        compression: Optional[str] = "zstd",
        queue_maxsize: int = 2000,
    ) -> None:
        self.output_file = self._resolve_parquet_path(output_path)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.batch_size = max(1, batch_size)
        self.compression = compression
        self._writer: Optional[pq.ParquetWriter] = None
        self._schema = pa.schema(
            [
                ("tick", pa.int64()),
                ("sim_time_s", pa.float64()),
                ("plc_state_conveyors", pa.binary()),
                ("conveyor_commands", pa.binary()),
            ]
        )

        self._rows_logged = 0
        self._dropped = 0
        self._last_drop_warning = 0.0

        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_maxsize)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="conveyor-indexing-writer")
        self._thread.start()

    def queue_depth(self) -> int:
        """Current backlog size - a smoke test's most direct signal for
        whether this writer is falling behind (see module docstring)."""
        return self._queue.qsize()

    @staticmethod
    def _resolve_parquet_path(output_path: str) -> pathlib.Path:
        path = pathlib.Path(output_path).expanduser()
        if path.exists() and path.is_dir():
            unique_name = f"{int(time.time() * 1e3)}_{uuid.uuid4().hex}.parquet"
            return path / unique_name
        if path.suffix == ".parquet":
            return path
        if path.suffix:
            raise AttributeError("Output path must be a directory or .parquet file")
        return path.with_suffix(".parquet")

    def log_tick(
        self,
        tick: int,
        sim_time_s: float,
        plc_state_conveyors: bytes,
        conveyor_commands: bytes,
    ) -> None:
        """Queue one control-tick row for background serialization/writing -
        never blocks the sim loop (put_nowait; drop-and-warn on overflow,
        same contract as conveyor_indexing.mcap_recorder.McapRecorder)."""
        try:
            self._queue.put_nowait(
                {
                    "tick": tick,
                    "sim_time_s": sim_time_s,
                    "plc_state_conveyors": plc_state_conveyors,
                    "conveyor_commands": conveyor_commands,
                }
            )
        except queue.Full:
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_warning >= _DROP_WARNING_INTERVAL_S:
                logger.warning(
                    "conveyor indexing tick logger queue full - dropped %d row(s) so far; "
                    "disk cannot keep up with the logging rate",
                    self._dropped,
                )
                self._last_drop_warning = now

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

            pending.append(item)
            if len(pending) >= self.batch_size:
                self._flush_rows(pending)
                pending = []

    def _flush_rows(self, rows: list) -> None:
        columns = {
            "tick": [r["tick"] for r in rows],
            "sim_time_s": [r["sim_time_s"] for r in rows],
            "plc_state_conveyors": [r["plc_state_conveyors"] for r in rows],
            "conveyor_commands": [r["conveyor_commands"] for r in rows],
        }
        table = pa.Table.from_pydict(columns, schema=self._schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                where=str(self.output_file),
                schema=self._schema,
                compression=self.compression,
            )
        self._writer.write_table(table)
        self._rows_logged += len(rows)

    def close(self) -> None:
        self._queue.put(self._SENTINEL)
        self._thread.join()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        logger.info(
            "conveyor indexing tick logger closed: %d row(s) written to %s, %d dropped",
            self._rows_logged,
            self.output_file,
            self._dropped,
        )
        if self._dropped:
            logger.warning(
                "conveyor indexing tick logger dropped %d row(s) - this capture has gaps", self._dropped
            )
        self._persist_stats()

    def _persist_stats(self) -> None:
        """Mirrors McapRecorder._persist_stats: a sidecar JSON survives an
        unattended run even if process log output doesn't."""
        stats_path = self.output_file.with_suffix(".stats.json")
        try:
            stats_path.write_text(
                json.dumps({"rows_logged": self._rows_logged, "dropped": self._dropped}, indent=2)
            )
        except OSError:
            logger.warning("could not write tick logger stats to %s", stats_path, exc_info=True)
