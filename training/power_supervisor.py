"""Battery-aware supervisor for long-running training jobs.

Starts the child immediately. Once battery falls below the low-water mark
while discharging, pauses the child via SIGINT and waits for the battery
to recover to the high-water mark before restarting.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone


BATTERY_BASE = "/sys/class/power_supply/BAT1"
AC_BASE = "/sys/class/power_supply/ACAD"


def _read_text(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


@dataclass
class PowerState:
    capacity_pct: int | None
    battery_status: str
    ac_online: int | None

    @property
    def is_discharging(self) -> bool:
        return self.battery_status.lower() == "discharging"


def read_power_state() -> PowerState:
    return PowerState(
        capacity_pct=_read_int(os.path.join(BATTERY_BASE, "capacity")),
        battery_status=_read_text(os.path.join(BATTERY_BASE, "status")) or "",
        ac_online=_read_int(os.path.join(AC_BASE, "online")),
    )


def build_command(base_cmd: list[str], resume_arg: str, resume_path: str) -> list[str]:
    if resume_arg in base_cmd:
        return list(base_cmd)
    if os.path.exists(resume_path):
        return [*base_cmd, resume_arg, resume_path]
    return list(base_cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pause/resume training based on battery level")
    parser.add_argument("--low", type=int, default=20, help="Pause when battery <= this percent")
    parser.add_argument("--high", type=int, default=80, help="Resume when battery >= this percent")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--resume-arg", type=str, default="--resume")
    parser.add_argument("--resume-path", type=str, required=True)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.high <= args.low:
        raise SystemExit("--high must be greater than --low")
    if not args.command:
        raise SystemExit("Provide the training command after '--'")

    base_cmd = args.command
    if base_cmd and base_cmd[0] == "--":
        base_cmd = base_cmd[1:]
    if not base_cmd:
        raise SystemExit("Provide the training command after '--'")

    child: subprocess.Popen | None = None
    should_exit = False
    last_wait_key: tuple[int | None, str, int | None] | None = None
    restart_backoff = max(5.0, args.poll_seconds)
    next_start_time = 0.0
    # Start immediately — only enter wait-for-charge after first low-battery pause
    waiting_for_charge = False

    writer = None
    log_fh = None
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        exists = os.path.exists(args.log_file)
        log_fh = open(args.log_file, "a", newline="")
        writer = csv.DictWriter(
            log_fh,
            fieldnames=[
                "timestamp_utc",
                "event",
                "capacity_pct",
                "battery_status",
                "ac_online",
                "pid",
                "command",
            ],
        )
        if not exists:
            writer.writeheader()
            log_fh.flush()

    def log_event(event: str, state: PowerState, pid: int | None):
        message = (
            f"[{datetime.now(timezone.utc).isoformat()}] {event}: "
            f"cap={state.capacity_pct} status={state.battery_status} "
            f"ac={state.ac_online} pid={pid or '-'}"
        )
        print(message, flush=True)
        if writer is not None:
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    "capacity_pct": "" if state.capacity_pct is None else state.capacity_pct,
                    "battery_status": state.battery_status,
                    "ac_online": "" if state.ac_online is None else state.ac_online,
                    "pid": "" if pid is None else pid,
                    "command": shlex.join(build_command(base_cmd, args.resume_arg, args.resume_path)),
                }
            )
            assert log_fh is not None
            log_fh.flush()

    def maybe_log_wait(state: PowerState):
        nonlocal last_wait_key
        key = (state.capacity_pct, state.battery_status, state.ac_online)
        if key != last_wait_key:
            log_event("waiting-for-charge", state, None)
            last_wait_key = key

    def handle_signal(signum, _frame):
        nonlocal should_exit
        should_exit = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    old_sigint = signal.signal(signal.SIGINT, handle_signal)
    old_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not should_exit:
            state = read_power_state()

            if child is None or child.poll() is not None:
                # Child exited or not started yet
                if child is not None and child.poll() is not None:
                    log_event(f"child-exit-{child.returncode}", state, child.pid)
                    if child.returncode != 0:
                        next_start_time = time.monotonic() + restart_backoff
                    child = None

                # Decide whether to start
                can_start = time.monotonic() >= next_start_time
                if waiting_for_charge:
                    # After a low-battery pause, wait for high-water mark
                    can_start = can_start and (
                        state.capacity_pct is not None
                        and state.capacity_pct >= args.high
                    )
                    if not can_start:
                        maybe_log_wait(state)

                if can_start:
                    cmd = build_command(base_cmd, args.resume_arg, args.resume_path)
                    child = subprocess.Popen(cmd)
                    last_wait_key = None
                    waiting_for_charge = False
                    log_event("start", state, child.pid)
            else:
                # Child running — check if we need to pause
                if (
                    state.capacity_pct is not None
                    and state.capacity_pct <= args.low
                    and state.is_discharging
                ):
                    log_event("pause-request", state, child.pid)
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=600)
                    except subprocess.TimeoutExpired:
                        log_event("pause-escalate-term", state, child.pid)
                        child.terminate()
                        try:
                            child.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            log_event("pause-escalate-kill", state, child.pid)
                            child.kill()
                            child.wait()
                    log_event(f"paused-{child.returncode}", state, child.pid)
                    child = None
                    last_wait_key = None
                    waiting_for_charge = True

            slept = 0.0
            while slept < args.poll_seconds and not should_exit:
                time.sleep(min(1.0, args.poll_seconds - slept))
                slept += 1.0
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if log_fh is not None:
            log_fh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
