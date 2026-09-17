"""Zenoh key expressions the sim publishes and subscribes to, built from one prefix."""

from __future__ import annotations

import os
from dataclasses import dataclass

PREFIX_ENV = "SIM_ZENOH_PREFIX"
DEFAULT_PREFIX = "sim"


@dataclass(frozen=True)
class Topics:
    prefix: str = DEFAULT_PREFIX

    @classmethod
    def from_env(cls) -> Topics:
        return cls(os.environ.get(PREFIX_ENV, DEFAULT_PREFIX).strip("/"))

    # telemetry (published by the sim)
    def arm_state(self, arm: int) -> str:
        return f"{self.prefix}/arm/{arm}/state"

    def arm_tool_pose(self, arm: int) -> str:
        return f"{self.prefix}/arm/{arm}/tool_pose"

    @property
    def arms_phase(self) -> str:
        return f"{self.prefix}/arms/phase"

    @property
    def conveyor_state(self) -> str:
        return f"{self.prefix}/conveyor/state"

    @property
    def conveyor_autonomous_decision(self) -> str:
        return f"{self.prefix}/conveyor/autonomous_decision"

    @property
    def camera_list(self) -> str:
        return f"{self.prefix}/camera/list"

    def camera_color(self, camera_id: str) -> str:
        return f"{self.prefix}/camera/{camera_id}/color"

    def camera_depth(self, camera_id: str) -> str:
        return f"{self.prefix}/camera/{camera_id}/depth"

    @property
    def boxes_state(self) -> str:
        return f"{self.prefix}/boxes/state"

    @property
    def boxes_events(self) -> str:
        return f"{self.prefix}/boxes/events"

    @property
    def clock(self) -> str:
        return f"{self.prefix}/clock"

    @property
    def run_metadata(self) -> str:
        return f"{self.prefix}/run_metadata"

    @property
    def status(self) -> str:
        return f"{self.prefix}/status"

    # control (queryable + subscriber on the sim)
    @property
    def control(self) -> str:
        return f"{self.prefix}/control"

    # commands (subscribed by the sim)
    def arm_action(self, arm: int) -> str:
        return f"{self.prefix}/arm/{arm}/action_command"

    @property
    def conveyor_command(self) -> str:
        return f"{self.prefix}/conveyor/command"

    @property
    def boxes_command(self) -> str:
        return f"{self.prefix}/boxes/command"
