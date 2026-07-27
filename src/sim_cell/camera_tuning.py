"""Optional camera-tuning workflow: one small viewport per camera, locked to
that camera's feed, plus an F10 hotkey that saves the current transforms to
`layout.CAMERA_POSES_PATH` (see `cameras.pose_io.save_camera_poses`).

Enabled only via `CONVEYOR_INDEXING_CAMERA_TUNING=1` (naming convention
matches `CONVEYOR_INDEXING_DEBUG_LOGGERS` - see `sim_cell.log_setup`); a
normal run does none of this - no extra viewports, no keyboard hook.

Workflow: run with the env var set, drag each camera prim in the main
viewport/stage tree with the ordinary USD gizmo while watching its locked
viewport update live, then press F10 once satisfied. Saving is an explicit,
one-shot action - accidental nudges are never auto-persisted.
"""

from __future__ import annotations

import logging
import os

import carb.input
import omni.appwindow

from cameras.pose_io import save_camera_poses
from sim_cell import layout

logger = logging.getLogger(__name__)

_ENV_VAR = "CONVEYOR_INDEXING_CAMERA_TUNING"
_SAVE_KEY = carb.input.KeyboardInput.F10
_VIEWPORT_SIZE = (480, 360)

# Holds the subscription id AND the callback closure for the process
# lifetime. Storing only the int id (as an earlier version of this module
# did) is not enough - carb's subscribe_to_keyboard_events binding is not
# guaranteed to keep its own strong reference to the passed Python callable,
# so with nothing else referencing the `_on_keyboard_event` closure, it's
# eligible for garbage collection the moment `maybe_enable_camera_tuning`
# returns, which would make F10 silently stop firing at some later, GC-timing
# -dependent point. Keeping the closure itself alive here is what's actually
# load-bearing; the id is kept alongside it for completeness/debugging.
_keyboard_subscription = None
_keyboard_callback = None


def is_enabled() -> bool:
    return os.environ.get(_ENV_VAR, "") == "1"


def maybe_enable_camera_tuning(stage, specs) -> None:
    """No-op unless `CONVEYOR_INDEXING_CAMERA_TUNING=1`."""
    if not is_enabled():
        return

    from omni.kit.viewport.utility import create_viewport_window

    for i, spec in enumerate(specs):
        create_viewport_window(
            name=f"Camera tuning: {spec.serial}",
            camera_path=spec.prim_path,
            width=_VIEWPORT_SIZE[0],
            height=_VIEWPORT_SIZE[1],
            position_x=_VIEWPORT_SIZE[0] * (i % 3),
            position_y=_VIEWPORT_SIZE[1] * (i // 3),
        )
    logger.info(
        "camera tuning enabled: %d viewport(s) open, press F10 to save poses to %s",
        len(specs), layout.CAMERA_POSES_PATH,
    )

    def _on_keyboard_event(event: carb.input.KeyboardEvent) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS or event.input != _SAVE_KEY:
            return True
        logger.info("F10 pressed - saving camera poses")
        try:
            save_camera_poses(layout.CAMERA_POSES_PATH, stage, specs)
        except Exception:
            logger.exception("failed to save camera poses to %s", layout.CAMERA_POSES_PATH)
        return True

    input_iface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    global _keyboard_subscription, _keyboard_callback
    _keyboard_callback = _on_keyboard_event
    _keyboard_subscription = input_iface.subscribe_to_keyboard_events(keyboard, _keyboard_callback)
