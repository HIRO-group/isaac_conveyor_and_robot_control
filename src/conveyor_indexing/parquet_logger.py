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
"""

from __future__ import annotations

import pathlib
import queue
import threading
import time
import uuid
from typing import Optional

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

        self._queue: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="conveyor-indexing-writer")
        self._thread.start()

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
        """Queue one control-tick row for background serialization/writing."""
        self._queue.put(
            {
                "tick": tick,
                "sim_time_s": sim_time_s,
                "plc_state_conveyors": plc_state_conveyors,
                "conveyor_commands": conveyor_commands,
            }
        )

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

    def close(self) -> None:
        self._queue.put(self._SENTINEL)
        self._thread.join()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
