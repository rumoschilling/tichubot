"""Passive throughput watcher for a live ring run (stdlib only).

Reads, never touches: /proc for the trainer and its pool workers, NVML via
nvidia-smi, and the run's stdout log. It sends no signals, attaches to
nothing, and writes only into ``--out``. Stops itself after ``--duration``.

Outputs (all flushed per row):
  gpu.csv      t, gpu_util%, mem_util%, mem_used_mib       (nvidia-smi, 1 s)
  pmon.txt     per-process SM% / memory                     (nvidia-smi pmon, 1 s)
  procs.csv    t, pid, role, cpu% since last sample, state, rss_mib
  events.csv   t, kind, line   for log lines marking rollout end / update end

Usage (on the server, low priority):
  nice -n 19 python3 throughput_watch.py --log ~/ring_runs/ring1.log \
      --out ~/throughput_watch/run1 --duration 3600
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

CLK = os.sysconf("SC_CLK_TCK")
EVENT_PREFIXES = {
    "[ring ": "update_end",
    "  [rollout profile]": "rollout_end",
    "  parity check:": "parity_end",
    "  [worker cuda]": "worker_cuda",
}


def read_cmdline(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return []


def read_stat(pid: int) -> tuple[str, int, int, int] | None:
    """(state, ppid, utime+stime ticks, rss pages)."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = raw[raw.rindex(")") + 2:].split()
    return fields[0], int(fields[1]), int(fields[11]) + int(fields[12]), int(fields[21])


def find_processes() -> dict[int, str]:
    """{pid: role}: the train_ring python process and its direct children."""
    pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    trainer = [
        p for p in pids
        if (cmd := read_cmdline(p)) and cmd[0].endswith("python")
        and "training.train_ring" in cmd
    ]
    roles = {p: "trainer" for p in trainer}
    for p in pids:
        st = read_stat(p)
        if st and st[1] in roles and roles[st[1]] == "trainer":
            cmd = " ".join(read_cmdline(p))
            roles[p] = "worker" if "spawn_main" in cmd else "child"
    return roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=3600)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    gpu_fh = open(out / "gpu.csv", "w")
    gpu = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used",
         "--format=csv,noheader,nounits", "-lms", "1000"],
        stdout=gpu_fh, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    pmon = subprocess.Popen(
        ["nvidia-smi", "pmon", "-s", "um", "-d", "1", "-o", "T", "-f", str(out / "pmon.txt")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)

    procs_fh = open(out / "procs.csv", "w", newline="")
    procs = csv.writer(procs_fh)
    procs.writerow(["t", "pid", "role", "cpu_pct", "state", "rss_mib"])
    events_fh = open(out / "events.csv", "w", newline="")
    events = csv.writer(events_fh)
    events.writerow(["t", "kind", "line"])
    (out / "started").write_text(f"{time.time():.3f}\n")

    log_path = Path(args.log).expanduser()
    log_fh = open(log_path, "r", errors="replace")
    log_fh.seek(0, os.SEEK_END)
    partial = ""
    last: dict[int, tuple[float, int]] = {}
    roles: dict[int, str] = {}
    next_scan = 0.0
    deadline = time.time() + args.duration
    try:
        while time.time() < deadline:
            now = time.time()
            if now >= next_scan:
                roles = find_processes()
                next_scan = now + 30
                if not roles:
                    events.writerow([f"{now:.3f}", "no_trainer", ""])
            page_mib = os.sysconf("SC_PAGE_SIZE") / 2**20
            for pid, role in roles.items():
                st = read_stat(pid)
                if st is None:
                    continue
                state, _, ticks, rss = st
                if pid in last:
                    t0, ticks0 = last[pid]
                    pct = 100.0 * (ticks - ticks0) / CLK / max(now - t0, 1e-6)
                    procs.writerow([f"{now:.3f}", pid, role, f"{pct:.1f}", state, f"{rss * page_mib:.0f}"])
                last[pid] = (now, ticks)
            chunk = log_fh.read()
            if chunk:
                partial += chunk
                *lines, partial = partial.split("\n")
                for line in lines:
                    for prefix, kind in EVENT_PREFIXES.items():
                        if line.startswith(prefix):
                            events.writerow([f"{now:.3f}", kind, line.strip()[:200]])
                            break
            procs_fh.flush()
            events_fh.flush()
            time.sleep(max(0.0, args.interval - (time.time() - now)))
    finally:
        for p in (gpu, pmon):
            p.terminate()
        for p in (gpu, pmon):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        (out / "finished").write_text(f"{time.time():.3f}\n")


if __name__ == "__main__":
    main()
