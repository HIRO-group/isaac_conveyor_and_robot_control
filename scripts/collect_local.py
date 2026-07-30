#!/usr/bin/env python3
"""Local MCAP data-collection supervisor: run the sim headless with the MCAP
recorder on, stream closed .mcap files to GCS as the run progresses, and
(optionally) delete them locally so a multi-hour run never fills the disk.

Stdlib-only on stock python3 - the sim itself still runs under Isaac Sim's
bundled interpreter via scripts/run.sh; this process only supervises.

Smoke run (files kept local for Foxglove AND uploaded):
  python3 scripts/collect_local.py --sim-seconds 600 --keep-local

Long run (stream + delete local; ~0.23x realtime on an L4, so 14400 sim-s
is ~17-18h wall):
  nohup python3 scripts/collect_local.py --sim-seconds 14400 > /dev/null 2>&1 &
  tail -f data/collect/<run_id>/collect.log

Crash/reboot recovery (upload whatever a dead run left behind):
  python3 scripts/collect_local.py --sweep-only --run-id <run_id>

Layout in GCS mirrors the Vertex batch-collection convention:
  <gcs-prefix>/<run_id>/instance_00/{mcap/*.mcap, sim.log, collect.log, summary.json}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
GCLOUD = "/snap/bin/gcloud"  # absolute: nohup/cron environments may lack /snap/bin on PATH
DEFAULT_GCS_PREFIX = "gs://por-theia-1/data_collection/sim"
SWEEP_INTERVAL_S = 15.0
DISK_WARN_PCT = 80.0
DISK_ABORT_PCT = 90.0
SIM_STOP_GRACE_S = 60.0

logger = logging.getLogger("collect_local")


def closed_mcap_files(mcap_dir: pathlib.Path) -> list[pathlib.Path]:
    """Closed = final name. The recorder renames <start>_INCOMPLETE.mcap ->
    <start>_<end>.mcap atomically on close, so name alone is sufficient."""
    if not mcap_dir.is_dir():
        return []
    return sorted(p for p in mcap_dir.glob("*.mcap") if "INCOMPLETE" not in p.name)


def incomplete_mcap_files(mcap_dir: pathlib.Path) -> list[pathlib.Path]:
    if not mcap_dir.is_dir():
        return []
    return sorted(mcap_dir.glob("*INCOMPLETE*.mcap"))


class GcsUploader:
    """Sweep closed mcap files to GCS; verify size before any local delete."""

    def __init__(self, mcap_dir: pathlib.Path, gcs_mcap_prefix: str, keep_local: bool) -> None:
        self.mcap_dir = mcap_dir
        self.gcs_mcap_prefix = gcs_mcap_prefix.rstrip("/")
        self.keep_local = keep_local
        self.uploaded_names: set[str] = set()
        self.uploaded_bytes = 0
        self.upload_failures = 0
        self.incomplete_uploaded: list[str] = []

    def upload_one(self, src: pathlib.Path, delete_local: bool) -> bool:
        url = f"{self.gcs_mcap_prefix}/{src.name}"
        size = src.stat().st_size
        cp = subprocess.run([GCLOUD, "storage", "cp", str(src), url], capture_output=True, text=True, check=False)
        if cp.returncode != 0:
            self.upload_failures += 1
            logger.error("upload failed (%s): %s", src.name, cp.stderr.strip().splitlines()[-1:] or cp.returncode)
            return False
        # gcloud already CRC32C-validates; the size check is belt-and-braces
        # before we destroy the only other copy.
        desc = subprocess.run(
            [GCLOUD, "storage", "objects", "describe", url, "--format=value(size)"],
            capture_output=True, text=True, check=False,
        )
        if desc.returncode != 0 or desc.stdout.strip() != str(size):
            self.upload_failures += 1
            logger.error("size verify failed (%s): local=%d remote=%r", src.name, size, desc.stdout.strip())
            return False
        self.uploaded_names.add(src.name)
        self.uploaded_bytes += size
        if delete_local:
            src.unlink()
        logger.info("uploaded %s (%.1f MiB)%s", src.name, size / 2**20, "" if delete_local else " [kept local]")
        return True

    def sweep(self, final: bool = False) -> None:
        for src in closed_mcap_files(self.mcap_dir):
            if src.name in self.uploaded_names:
                continue  # only reachable with keep_local
            self.upload_one(src, delete_local=not self.keep_local)
        if final:
            # A leftover INCOMPLETE file means the sim died without a clean
            # close. Upload it under its INCOMPLETE name (downstream tooling
            # skips those by convention) but never delete the local copy.
            for src in incomplete_mcap_files(self.mcap_dir):
                if src.name not in self.uploaded_names and self.upload_one(src, delete_local=False):
                    self.incomplete_uploaded.append(src.name)

    def backlog(self) -> int:
        return len([p for p in closed_mcap_files(self.mcap_dir) if p.name not in self.uploaded_names])


def disk_used_pct(path: pathlib.Path) -> float:
    du = shutil.disk_usage(path)
    return 100.0 * du.used / du.total


def upload_file(local: pathlib.Path, url: str) -> bool:
    if not local.exists():
        return False
    return subprocess.run([GCLOUD, "storage", "cp", str(local), url], capture_output=True, check=False).returncode == 0


def preflight(gcs_run_prefix: str, run_info: dict, data_dir: pathlib.Path) -> None:
    """Fail fast on auth/bucket problems before paying an Isaac Sim startup."""
    marker = data_dir / "run_started.json"
    marker.write_text(json.dumps(run_info, indent=2) + "\n")
    if not upload_file(marker, f"{gcs_run_prefix}/run_started.json"):
        raise SystemExit(f"preflight upload to {gcs_run_prefix}/ failed - check gcloud auth / bucket access")
    auth = subprocess.run(
        [GCLOUD, "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    logger.info("preflight ok: destination %s, active account %s", gcs_run_prefix, auth or "<unknown>")
    if auth and "gserviceaccount" not in auth:
        logger.warning(
            "user credentials (%s) - a multi-hour run can outlive the token; "
            "if uploads start failing, re-auth then run --sweep-only", auth,
        )


def scrape_mcap_summary(sim_log: pathlib.Path) -> str | None:
    """The recorder's close line, e.g. 'mcap recorder closed: 123 message(s)
    written to ..., 0 dropped' - the run's authoritative drop count."""
    if not sim_log.exists():
        return None
    for line in reversed(sim_log.read_text(errors="replace").splitlines()):
        if "mcap recorder closed" in line:
            return line.strip()
    return None


def run_sim(args, data_dir: pathlib.Path, sim_log: pathlib.Path, uploader: GcsUploader | None) -> tuple[int, float]:
    env = os.environ.copy()
    env.update(
        CONVEYOR_INDEXING_HEADLESS="1",
        CONVEYOR_INDEXING_RECORD_MCAP="1",
        CONVEYOR_INDEXING_MAX_SIM_SECONDS=str(args.sim_seconds),
        CONVEYOR_INDEXING_DATA_DIR=str(data_dir),
        CONVEYOR_INDEXING_SPAWN_SEED=str(args.seed),
        CONVEYOR_INDEXING_INSTANCE_INDEX="0",
    )
    env.pop("CONVEYOR_INDEXING_RECORD", None)  # parquet recorder stays off

    start = time.monotonic()
    with open(sim_log, "ab") as log_fh:
        proc = subprocess.Popen(
            ["bash", str(REPO / "scripts" / "run.sh")],
            env=env, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True,
        )
    logger.info("sim started (pid %d): %s sim-seconds, seed %s -> %s", proc.pid, args.sim_seconds, args.seed, data_dir)

    # start_new_session=True makes proc.pid both the session and process-group
    # leader. This matters because scripts/run.sh's `exec` puts IsaacSim/python.sh
    # (a bash wrapper) at proc.pid, but python.sh itself launches the real sim as
    # a plain child (`$python_exe ... || error_exit`, not `exec`) - so signalling
    # proc.pid alone only reaches the wrapper and orphans the actual sim process,
    # which then keeps running (and keeps appending to its MCAP file) unsupervised.
    # Signal the whole group instead so the real sim gets SIGTERM directly too.
    pgid = proc.pid

    def group_alive() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False

    def signal_group(sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass

    stopping = threading.Event()
    stop_requested_at = [0.0]

    def request_stop(signum, _frame):
        logger.warning("received signal %d - stopping sim (recorder will flush)", signum)
        if not stopping.is_set():
            stopping.set()
            stop_requested_at[0] = time.monotonic()
            signal_group(signal.SIGTERM)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    peak_disk = 0.0

    def sweeper():
        nonlocal peak_disk
        while group_alive():
            time.sleep(SWEEP_INTERVAL_S)
            if uploader is not None:
                uploader.sweep()
            pct = disk_used_pct(data_dir)
            peak_disk = max(peak_disk, pct)
            backlog = uploader.backlog() if uploader is not None else len(closed_mcap_files(data_dir / "mcap"))
            logger.info(
                "heartbeat: uploaded=%d files / %.2f GiB, backlog=%d, disk=%.0f%%, sim_alive=%s",
                len(uploader.uploaded_names) if uploader else 0,
                (uploader.uploaded_bytes if uploader else 0) / 2**30,
                backlog, pct, group_alive(),
            )
            if pct >= DISK_ABORT_PCT and (backlog > 0 or uploader is None or uploader.keep_local):
                logger.error("disk at %.0f%% with local data still present - stopping sim early", pct)
                if not stopping.is_set():
                    stopping.set()
                    stop_requested_at[0] = time.monotonic()
                    signal_group(signal.SIGTERM)
                return
            if pct >= DISK_WARN_PCT:
                logger.warning("disk at %.0f%%", pct)

    sweep_thread = threading.Thread(target=sweeper, daemon=True, name="uploader-sweep")
    sweep_thread.start()

    # Wait for every process in the group to exit, not just proc (see the
    # comment above pgid) - proc.wait() alone would return as soon as the
    # python.sh wrapper dies, while the real sim (and its MCAP writer) is
    # potentially still running and still appending to the current file.
    while True:
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        if not group_alive():
            break
        if stopping.is_set() and time.monotonic() - stop_requested_at[0] > SIM_STOP_GRACE_S:
            logger.error("sim group ignored SIGTERM for %.0fs - killing", SIM_STOP_GRACE_S)
            signal_group(signal.SIGKILL)
            stop_requested_at[0] = time.monotonic()
    exit_code = proc.returncode
    sweep_thread.join(timeout=SWEEP_INTERVAL_S + 30)
    return exit_code, time.monotonic() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-seconds", type=float, help="sim-time duration (required unless --sweep-only)")
    parser.add_argument("--run-id", default=None, help="default: run_<UTC timestamp>")
    parser.add_argument("--seed", type=int, default=None, help="spawn seed; default derived from start time")
    parser.add_argument("--keep-local", action="store_true", help="upload but never delete local mcap files")
    parser.add_argument("--no-upload", action="store_true", help="record locally only")
    parser.add_argument("--sweep-only", action="store_true", help="no sim: upload leftovers from --data-dir/--run-id")
    parser.add_argument("--gcs-prefix", default=DEFAULT_GCS_PREFIX)
    parser.add_argument("--data-dir", type=pathlib.Path, default=None, help="default: <repo>/data/collect/<run_id>")
    args = parser.parse_args()

    if not args.sweep_only and args.sim_seconds is None:
        parser.error("--sim-seconds is required unless --sweep-only")
    if args.run_id is None:
        args.run_id = "run_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if args.seed is None:
        args.seed = int(time.time()) % 1_000_000
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error(f"--run-id must be GCS-path-safe, got {args.run_id!r}")

    data_dir = args.data_dir or (REPO / "data" / "collect" / args.run_id)
    data_dir.mkdir(parents=True, exist_ok=True)
    collect_log = data_dir / "collect.log"
    sim_log = data_dir / "sim.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(collect_log)],
    )

    gcs_run_prefix = f"{args.gcs_prefix.rstrip('/')}/{args.run_id}/instance_00"
    uploader = None if args.no_upload else GcsUploader(data_dir / "mcap", f"{gcs_run_prefix}/mcap", args.keep_local)

    if args.sweep_only:
        if uploader is None:
            parser.error("--sweep-only with --no-upload makes no sense")
        uploader.sweep(final=True)
        logger.info("sweep-only done: %d files / %.2f GiB uploaded, %d failures, backlog=%d",
                    len(uploader.uploaded_names), uploader.uploaded_bytes / 2**30,
                    uploader.upload_failures, uploader.backlog())
        return 1 if uploader.backlog() or uploader.upload_failures else 0

    run_info = {
        "run_id": args.run_id, "sim_seconds_requested": args.sim_seconds, "seed": args.seed,
        "keep_local": args.keep_local, "host": os.uname().nodename,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if uploader is not None:
        preflight(gcs_run_prefix, run_info, data_dir)

    if not pathlib.Path("/tmp/proto_gen/sim_state_pb2.py").exists():
        logger.info("proto bindings missing - running gen_proto.sh")
        subprocess.run(["bash", str(REPO / "gen_proto.sh")], check=True)

    exit_code, wall_s = run_sim(args, data_dir, sim_log, uploader)
    logger.info("sim exited with code %s after %.0fs wall", exit_code, wall_s)

    if uploader is not None:
        uploader.sweep(final=True)

    mcap_line = scrape_mcap_summary(sim_log)
    summary = {
        **run_info,
        "wall_seconds": round(wall_s, 1),
        "sim_exit_code": exit_code,
        "uploaded_files": len(uploader.uploaded_names) if uploader else 0,
        "uploaded_bytes": uploader.uploaded_bytes if uploader else 0,
        "upload_failures": uploader.upload_failures if uploader else 0,
        "backlog_remaining": uploader.backlog() if uploader else None,
        "incomplete_files_uploaded": uploader.incomplete_uploaded if uploader else [],
        "local_mcap_files_remaining": len(list((data_dir / "mcap").glob("*.mcap"))) if (data_dir / "mcap").is_dir() else 0,
        "mcap_recorder_summary": mcap_line,
        "gcs_destination": None if uploader is None else gcs_run_prefix,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path = data_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("summary: %s", json.dumps(summary))

    if uploader is not None:
        upload_file(summary_path, f"{gcs_run_prefix}/summary.json")
        upload_file(sim_log, f"{gcs_run_prefix}/sim.log")
        upload_file(collect_log, f"{gcs_run_prefix}/collect.log")

    failed = (exit_code != 0) or (uploader is not None and (uploader.backlog() > 0 or uploader.upload_failures > 0))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
