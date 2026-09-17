"""Exit 0 once the sim publishes on its clock topic; used as a container healthcheck."""

from __future__ import annotations

import argparse
import sys
import threading

from conveyor_indexing.topics import Topics
from conveyor_indexing.zenoh_session import open_session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for one sample")
    args = parser.parse_args()

    seen = threading.Event()
    session = open_session()
    sub = session.declare_subscriber(Topics.from_env().clock, lambda _sample: seen.set())
    try:
        return 0 if seen.wait(args.timeout) else 1
    finally:
        sub.undeclare()
        session.close()


if __name__ == "__main__":
    sys.exit(main())
