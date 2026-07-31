#!/usr/bin/env python3
"""Write timestamped GPU process samples for ownership acceptance."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def read_nvidia_processes() -> list[dict[str, object]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows: list[dict[str, object]] = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row or not any(item.strip() for item in row):
            continue
        if len(row) != 4:
            raise ValueError("nvidia-smi returned a malformed process row")
        gpu_uuid, pid, process_name, used_memory = (item.strip() for item in row)
        rows.append(
            {
                "gpuUuid": gpu_uuid,
                "pid": int(pid),
                "processName": process_name,
                "usedMemory": int(used_memory),
                "containerLabel": container_label_for_pid(int(pid)),
            }
        )
    return rows


def container_label_for_pid(pid: int) -> str | None:
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
    except OSError:
        return None
    for token in cgroup.replace("/", ":").split(":"):
        if len(token) >= 12 and all(char in "0123456789abcdef" for char in token.lower()):
            return token
    return None


def sample_ownership(
    duration_seconds: int,
    interval_seconds: int,
    output: Path,
    *,
    read_rows: Callable[[], list[dict[str, object]]] = read_nvidia_processes,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    if duration_seconds <= 0 or interval_seconds <= 0:
        raise ValueError("duration and interval must be positive")
    if duration_seconds % interval_seconds != 0:
        raise ValueError("duration must be divisible by interval")
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = duration_seconds // interval_seconds
    with output.open("w", encoding="utf-8") as stream:
        for index in range(samples):
            stream.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "processes": read_rows(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            if index + 1 < samples:
                sleep_fn(interval_seconds)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample_ownership(args.duration, args.interval, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
