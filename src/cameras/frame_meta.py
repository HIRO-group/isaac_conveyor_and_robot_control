"""Per-frame metadata helpers: wall-clock microsecond timestamps (theia's
FrameMetadata convention), a UUIDv7 generator (Python's stdlib `uuid` module
doesn't gain `uuid7()` until 3.14 - this Isaac Sim install bundles 3.12), and
a monotonic per-camera frame counter.
"""

from __future__ import annotations

import secrets
import time
import uuid
from collections import defaultdict


def now_us() -> int:
    """Wall-clock epoch microseconds, matching theia's FrameMetadata convention."""
    return time.time_ns() // 1_000


def uuid_v7() -> str:
    """RFC 9562 UUIDv7: 48-bit ms timestamp, 4-bit version, 12-bit random,
    2-bit variant, 62-bit random - same layout `uuid.uuid7()` will produce
    once this repo can rely on Python 3.14+.
    """
    unix_ts_ms = time.time_ns() // 1_000_000
    rand_bytes = secrets.token_bytes(10)  # 12 bits of rand_a + 62 bits of rand_b + slack
    rand_a = int.from_bytes(rand_bytes[0:2], "big") & 0x0FFF
    rand_b = int.from_bytes(rand_bytes[2:10], "big") & 0x3FFF_FFFF_FFFF_FFFF

    uuid_int = (
        (unix_ts_ms & 0xFFFF_FFFF_FFFF) << 80
        | (0x7 << 76)  # version 7
        | (rand_a << 64)
        | (0b10 << 62)  # variant
        | rand_b
    )
    return str(uuid.UUID(int=uuid_int))


class FrameCounter:
    """Sequential, monotonic frame numbers, tracked independently per serial."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)

    def next(self, serial: str) -> int:
        count = self._counts[serial]
        self._counts[serial] = count + 1
        return count
