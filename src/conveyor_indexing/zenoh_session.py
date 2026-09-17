"""One way to open the sim's Zenoh sessions."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ROUTER_ENV = "ZENOH_ROUTER"

try:
    import zenoh
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("eclipse-zenoh is not installed in this interpreter; see scripts/setup.sh") from exc


def open_session(router: str | None = None) -> zenoh.Session:
    """Connect to `router` (default `$ZENOH_ROUTER`), or open in peer mode when unset."""
    router = router if router is not None else os.environ.get(ROUTER_ENV)
    conf = zenoh.Config()
    if router:
        conf.insert_json5("connect/endpoints", f'["{router}"]')
        logger.info("connecting to Zenoh router at %s", router)
    else:
        logger.warning("%s not set; opening Zenoh session in peer-to-peer mode", ROUTER_ENV)
    return zenoh.open(conf)


def payload_bytes(sample) -> bytes:
    payload = sample.payload
    return payload.to_bytes() if hasattr(payload, "to_bytes") else bytes(payload)
