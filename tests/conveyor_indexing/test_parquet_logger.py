"""Tests for conveyor_indexing.parquet_logger.ConveyorIndexingLogger's
bounded-queue + drop-and-warn + persisted-stats hardening - previously this
logger used an unbounded queue.Queue() with a plain blocking put() and no
drop/backpressure signal at all, unlike every other writer thread in this
package (conveyor_indexing.mcap_recorder.McapRecorder,
conveyor_indexing.episode_recorder.EpisodeRecorder). Mirrors
test_mcap_recorder.py's style: real ConveyorIndexingLogger instances writing
to tmp_path, no Isaac Sim needed.
"""

from __future__ import annotations

import json
import queue

import pyarrow.parquet as pq

from conveyor_indexing.parquet_logger import ConveyorIndexingLogger


class _AlwaysFullQueue:
    """Stands in for self._queue to deterministically exercise log_tick's
    `except queue.Full` path - reliably forcing a REAL queue.Full against
    a live writer thread is inherently racy (the writer drains almost as
    fast as the main thread can fill it), which is also why
    test_mcap_recorder.py's own forced-drop test drives the drop counter
    directly rather than real queue timing.
    """

    def put_nowait(self, item) -> None:
        raise queue.Full()


def test_log_tick_writes_rows_and_persists_stats(tmp_path):
    output_file = tmp_path / "ticks.parquet"
    logger_ = ConveyorIndexingLogger(str(output_file), batch_size=2)

    logger_.log_tick(tick=0, sim_time_s=0.0, plc_state_conveyors=b"a", conveyor_commands=b"x")
    logger_.log_tick(tick=1, sim_time_s=1.0 / 120, plc_state_conveyors=b"b", conveyor_commands=b"y")
    logger_.close()

    table = pq.read_table(output_file)
    assert table.num_rows == 2
    assert table.column("tick").to_pylist() == [0, 1]

    stats_path = output_file.with_suffix(".stats.json")
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats == {"rows_logged": 2, "dropped": 0}


def test_log_tick_drops_and_counts_on_queue_full_without_raising(tmp_path):
    """log_tick must never raise/block on overflow - drop-and-warn, the
    same contract McapRecorder already has. The real writer thread is
    stopped first (close()), then _queue is swapped for a stand-in that
    always raises queue.Full, isolating log_tick's own exception-handling
    logic from writer-thread timing entirely."""
    output_file = tmp_path / "ticks.parquet"
    logger_ = ConveyorIndexingLogger(str(output_file))
    logger_.close()
    logger_._queue = _AlwaysFullQueue()

    logger_.log_tick(tick=0, sim_time_s=0.0, plc_state_conveyors=b"a", conveyor_commands=b"b")
    logger_.log_tick(tick=1, sim_time_s=1.0, plc_state_conveyors=b"a", conveyor_commands=b"b")

    assert logger_._dropped == 2


def test_queue_depth_is_zero_once_drained_after_close(tmp_path):
    """queue_depth() (self._queue.qsize()) is inherently racy against the
    live writer thread while a run is in progress - a real smoke test
    would sample it periodically over a run, not assert an exact mid-run
    value here. What's deterministic and worth checking: it's callable
    without error, and settles to 0 once close() has fully drained and
    joined the writer thread."""
    output_file = tmp_path / "ticks.parquet"
    logger_ = ConveyorIndexingLogger(str(output_file), batch_size=100, queue_maxsize=2000)

    for i in range(10):
        logger_.log_tick(tick=i, sim_time_s=float(i), plc_state_conveyors=b"a", conveyor_commands=b"b")
    logger_.close()

    assert logger_.queue_depth() == 0


def test_recorder_stats_reflects_forced_drops(tmp_path):
    """Mirrors test_mcap_recorder.py's own forced-drop persistence check:
    drive the internal counter directly rather than real queue timing."""
    output_file = tmp_path / "ticks.parquet"
    logger_ = ConveyorIndexingLogger(str(output_file))
    logger_._dropped = 5
    logger_._rows_logged = 17
    logger_.close()

    stats = json.loads(output_file.with_suffix(".stats.json").read_text())
    assert stats == {"rows_logged": 17, "dropped": 5}
