"""Live event monitor for a CONVEYOR_INDEXING_EXTERNAL_ACTION=1 run: prints a
line every time an arm's suction (EE) toggles on/off, or a conveyor's
commanded run/speed/direction changes, plus a periodic snapshot so a quiet
sim isn't mistaken for a dead one. Read-only observer over the same Zenoh bus
the sim and an external controller (e.g. theia's sim_bridge) already use - no
interference with either process.

Start this BEFORE the external controller, right after the sim itself is up -
Zenoh has no message replay, so a subscriber only ever sees samples published
after it declared its subscription. Starting this first (and confirming the
"Zenoh session open" line below) is what guarantees no early command/state
transition is missed. See the top-level README's "Running a trained policy in
closed loop" section for the full three-step order.

Usage (same PYTHONPATH as scripts/run.sh - see that script/gen_proto.sh):
  PYTHONPATH=/tmp/proto_gen /home/ubuntu/IsaacSim/python.sh \
    scripts/monitor_external_action.py
"""

from __future__ import annotations

import datetime
import threading
import time

import plc_connector_pb2
import sim_arm_action_pb2
import sim_conveyor_action_pb2
import sim_robot_state_pb2
import zenoh

DIO_SUCTION_BIT = 0x10000
SNAPSHOT_INTERVAL_S = 5.0

_lock = threading.Lock()
_state = {
    "commanded_suction": {1: None, 2: None},
    "actual_holding": {1: None, 2: None},
    "commanded_conveyor": {},  # node_path -> (run, speed, direction)
    "actual_conveyor": {},  # node_path -> (machine_name, speed, direction)
}


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _short(node_path: str) -> str:
    parts = node_path.split("/")
    return parts[2] if len(parts) > 2 else node_path


def _on_arm_action(arm: int, sample) -> None:
    msg = sim_arm_action_pb2.SimArmActionCommand()
    msg.ParseFromString(bytes(sample.payload))
    with _lock:
        prev = _state["commanded_suction"][arm]
        if prev is not None and prev != msg.suction:
            print(f"[{_ts()}] EE COMMAND arm{arm}: {'ON' if prev else 'OFF'} -> {'ON' if msg.suction else 'OFF'}", flush=True)
        _state["commanded_suction"][arm] = msg.suction


def _on_arm_state(arm: int, sample) -> None:
    msg = sim_robot_state_pb2.PositionStatus()
    msg.ParseFromString(bytes(sample.payload))
    holding = bool(msg.dio_blocks and (msg.dio_blocks[0] & DIO_SUCTION_BIT))
    with _lock:
        prev = _state["actual_holding"][arm]
        if prev is not None and prev != holding:
            print(f"[{_ts()}] EE ACTUAL  arm{arm}: {'ON' if prev else 'OFF'} -> {'ON' if holding else 'OFF'}", flush=True)
        _state["actual_holding"][arm] = holding


def _on_conveyor_command(sample) -> None:
    msg = sim_conveyor_action_pb2.SimConveyorCommands()
    msg.ParseFromString(bytes(sample.payload))
    with _lock:
        for cmd in msg.commands:
            key = cmd.conveyor_node_path
            prev = _state["commanded_conveyor"].get(key)
            cur = (cmd.run, cmd.speed, cmd.direction)
            if prev is not None and prev != cur:
                print(
                    f"[{_ts()}] CONVEYOR CMD {_short(key)}: "
                    f"run={prev[0]}->{cur[0]} speed={prev[1]}->{cur[1]} dir={prev[2]}->{cur[2]}",
                    flush=True,
                )
            _state["commanded_conveyor"][key] = cur


def _on_conveyor_state(sample) -> None:
    msg = plc_connector_pb2.StateConveyors()
    msg.ParseFromString(bytes(sample.payload))
    with _lock:
        for item in msg.Conveyors:
            key = item.Name
            machine_name = plc_connector_pb2.ConveyorStateMachineCode.Name(item.Machine)
            cur = (machine_name, item.Speed, item.Direction)
            _state["actual_conveyor"][key] = cur


def _print_snapshot() -> None:
    with _lock:
        arm_bits = " ".join(
            f"arm{a}=[cmd:{'ON' if _state['commanded_suction'][a] else 'off'} "
            f"actual:{'ON' if _state['actual_holding'][a] else 'off'}]"
            for a in (1, 2)
        )
        conv_bits = " ".join(
            f"{_short(k)}={'RUN' if v[0] else 'stop'}(spd={v[1]},dir={v[2]})"
            for k, v in sorted(_state["commanded_conveyor"].items())
        )
        actual_bits = " ".join(
            f"{_short(k)}={v[0]}(spd={v[1]},dir={v[2]})" for k, v in sorted(_state["actual_conveyor"].items())
        )
    print(f"[{_ts()}] snapshot -- cmd: {arm_bits} || {conv_bits} -- actual: {actual_bits}", flush=True)


def main() -> None:
    conf = zenoh.Config()
    session = zenoh.open(conf)
    print(f"[{_ts()}] monitor: Zenoh session open (peer-to-peer), watching arm1/2 suction + all conveyor commands", flush=True)

    subs = [
        session.declare_subscriber("sim/arm/1/action_command", lambda s: _on_arm_action(1, s)),
        session.declare_subscriber("sim/arm/2/action_command", lambda s: _on_arm_action(2, s)),
        session.declare_subscriber("theia/robot/arm1/position_status", lambda s: _on_arm_state(1, s)),
        session.declare_subscriber("theia/robot/arm2/position_status", lambda s: _on_arm_state(2, s)),
        session.declare_subscriber("sim/conveyor/command", _on_conveyor_command),
        session.declare_subscriber("theia/plc/state_conveyors", _on_conveyor_state),
    ]

    try:
        while True:
            time.sleep(SNAPSHOT_INTERVAL_S)
            _print_snapshot()
    except KeyboardInterrupt:
        pass
    finally:
        for sub in subs:
            sub.undeclare()
        session.close()


if __name__ == "__main__":
    main()
