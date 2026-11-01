"""Lightweight system-health monitor for long training runs.

Logs AC/battery state, basic thermal info, load, and available memory.
Useful for diagnosing sudden shutdowns or sleep/power events after the fact.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone


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


def _power_supply_row() -> dict[str, str]:
    row: dict[str, str] = {}
    base = "/sys/class/power_supply"
    ac_online = _read_int(os.path.join(base, "ACAD", "online"))
    row["ac_online"] = "" if ac_online is None else str(ac_online)

    bat = os.path.join(base, "BAT1")
    row["battery_status"] = _read_text(os.path.join(bat, "status")) or ""
    capacity = _read_int(os.path.join(bat, "capacity"))
    row["battery_capacity_pct"] = "" if capacity is None else str(capacity)

    power_now = _read_int(os.path.join(bat, "power_now"))
    if power_now is None:
        current_now = _read_int(os.path.join(bat, "current_now"))
        voltage_now = _read_int(os.path.join(bat, "voltage_now"))
        if current_now is not None and voltage_now is not None:
            power_now = (current_now * voltage_now) // 1_000_000
    row["battery_power_uw"] = "" if power_now is None else str(power_now)
    return row


def _thermal_row() -> dict[str, str]:
    temps: list[tuple[str, float]] = []
    base = "/sys/class/thermal"
    for name in os.listdir(base):
        if not name.startswith("thermal_zone"):
            continue
        zone = os.path.join(base, name)
        temp = _read_int(os.path.join(zone, "temp"))
        if temp is None:
            continue
        ttype = _read_text(os.path.join(zone, "type")) or name
        temps.append((ttype, temp / 1000.0))
    if not temps:
        return {"thermal_max_c": "", "thermal_summary": ""}
    max_temp = max(t for _, t in temps)
    summary = "; ".join(f"{name}={temp:.1f}" for name, temp in temps[:6])
    return {
        "thermal_max_c": f"{max_temp:.1f}",
        "thermal_summary": summary,
    }


def _mem_available_mb() -> str:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024.0:.1f}"
    except OSError:
        pass
    return ""


def main():
    parser = argparse.ArgumentParser(description="Log system health periodically")
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    log_dir = os.path.dirname(args.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    exists = os.path.exists(args.log_file)
    with open(args.log_file, "a", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "timestamp_utc",
                "uptime_load_1m",
                "uptime_load_5m",
                "uptime_load_15m",
                "mem_available_mb",
                "ac_online",
                "battery_status",
                "battery_capacity_pct",
                "battery_power_uw",
                "thermal_max_c",
                "thermal_summary",
            ],
        )
        if not exists:
            writer.writeheader()
            fh.flush()

        while True:
            load1, load5, load15 = os.getloadavg()
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "uptime_load_1m": f"{load1:.2f}",
                "uptime_load_5m": f"{load5:.2f}",
                "uptime_load_15m": f"{load15:.2f}",
                "mem_available_mb": _mem_available_mb(),
                **_power_supply_row(),
                **_thermal_row(),
            }
            writer.writerow(row)
            fh.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
