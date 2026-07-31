#!/usr/bin/env python3
"""Fail-closed GPU ownership preflight for model services."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml


class PolicyError(ValueError):
    """Raised when the policy or nvidia-smi output is invalid."""


@dataclass(frozen=True)
class GpuProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory: int


@dataclass(frozen=True)
class Workload:
    allowed_gpu_uuids: tuple[str, ...]
    allowed_process_patterns: tuple[str, ...]
    co_residency_group: str | None = None


@dataclass(frozen=True)
class OwnershipReport:
    policy_version: int
    checked_at: str
    allocations: tuple[GpuProcess, ...]
    violations: tuple[dict[str, Any], ...]
    blocking_pids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "policyVersion": self.policy_version,
            "checkedAt": self.checked_at,
            "allocations": [asdict(item) for item in self.allocations],
            "violations": list(self.violations),
            "blockingPids": list(self.blocking_pids),
        }


def load_policy(policy_path: Path) -> tuple[int, dict[str, Workload]]:
    try:
        raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PolicyError(f"could not load policy: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise PolicyError("policy version must be 1")
    raw_workloads = raw.get("workloads")
    if not isinstance(raw_workloads, dict) or not raw_workloads:
        raise PolicyError("policy workloads must be a non-empty mapping")
    workloads: dict[str, Workload] = {}
    for name, value in raw_workloads.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise PolicyError("each workload must be a named mapping")
        gpu_uuids = value.get("allowed_gpu_uuids")
        patterns = value.get("allowed_process_patterns")
        if (
            not isinstance(gpu_uuids, list)
            or not gpu_uuids
            or any(not isinstance(item, str) or not item.startswith("GPU-") for item in gpu_uuids)
        ):
            raise PolicyError(f"{name}: allowed_gpu_uuids must contain GPU UUIDs")
        if not isinstance(patterns, list) or not patterns or any(not isinstance(item, str) or not item for item in patterns):
            raise PolicyError(f"{name}: allowed_process_patterns must be non-empty strings")
        group = value.get("co_residency_group")
        if group is not None and not isinstance(group, str):
            raise PolicyError(f"{name}: co_residency_group must be a string")
        workloads[name] = Workload(tuple(gpu_uuids), tuple(patterns), group)
    return 1, workloads


def parse_process_rows(output: str) -> list[GpuProcess]:
    rows: list[GpuProcess] = []
    for row in csv.reader(output.splitlines()):
        if not row or not any(item.strip() for item in row):
            continue
        if len(row) != 4:
            raise PolicyError("nvidia-smi returned a malformed process row")
        gpu_uuid, pid, process_name, used_memory = (item.strip() for item in row)
        try:
            parsed_pid = int(pid)
            parsed_memory = int(used_memory)
        except ValueError as exc:
            raise PolicyError("nvidia-smi returned a non-numeric pid or memory value") from exc
        if not gpu_uuid or not process_name or parsed_pid <= 0 or parsed_memory < 0:
            raise PolicyError("nvidia-smi returned an invalid process row")
        rows.append(GpuProcess(gpu_uuid, parsed_pid, process_name, parsed_memory))
    return rows


def read_nvidia_processes() -> list[GpuProcess]:
    try:
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
    except (OSError, subprocess.SubprocessError) as exc:
        raise PolicyError(f"nvidia-smi preflight failed: {exc}") from exc
    return parse_process_rows(result.stdout)


def read_visible_gpu_uuids() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PolicyError(f"could not query visible GPUs: {exc}") from exc
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not uuids:
        raise PolicyError("nvidia-smi returned no visible GPUs")
    return uuids


def _matches(pattern: str, process_name: str) -> bool:
    process = process_name.casefold()
    candidate = pattern.casefold()
    return fnmatch(process, candidate) or candidate in process


def find_violations(
    workloads: dict[str, Workload],
    process_rows: list[GpuProcess],
    target_service: str | None,
    visible_gpu_uuids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if target_service is not None and target_service not in workloads:
        raise PolicyError(f"unknown target service: {target_service}")
    target = workloads.get(target_service) if target_service else None
    known_gpu_uuids = {
        gpu_uuid for workload in workloads.values() for gpu_uuid in workload.allowed_gpu_uuids
    }
    if visible_gpu_uuids is not None:
        unknown_gpu_uuids = set(visible_gpu_uuids) - known_gpu_uuids
        if unknown_gpu_uuids:
            raise PolicyError(f"visible GPU UUIDs are not in policy: {sorted(unknown_gpu_uuids)}")
        if target is not None and not set(target.allowed_gpu_uuids).issubset(visible_gpu_uuids):
            raise PolicyError(f"{target_service} target GPU is not visible")
    violations: list[dict[str, Any]] = []
    for row in process_rows:
        if target is not None:
            if row.gpu_uuid not in target.allowed_gpu_uuids:
                continue
            peer_workloads = [
                workload
                for workload in workloads.values()
                if workload.co_residency_group
                and workload.co_residency_group == target.co_residency_group
                and row.gpu_uuid in workload.allowed_gpu_uuids
            ]
            allowed_patterns = [
                pattern
                for workload in [target, *peer_workloads]
                for pattern in workload.allowed_process_patterns
            ]
            if any(_matches(pattern, row.process_name) for pattern in allowed_patterns):
                continue
            violations.append(
                {
                    "reason": "foreign_process",
                    "workload": target_service,
                    "gpuUuid": row.gpu_uuid,
                    "pid": row.pid,
                    "processName": row.process_name,
                }
            )
            continue
        if any(
            row.gpu_uuid in workload.allowed_gpu_uuids
            and any(_matches(pattern, row.process_name) for pattern in workload.allowed_process_patterns)
            for workload in workloads.values()
        ):
            continue
        if any(row.gpu_uuid in workload.allowed_gpu_uuids for workload in workloads.values()):
            violations.append(
                {
                    "reason": "unrecognized_process",
                    "gpuUuid": row.gpu_uuid,
                    "pid": row.pid,
                    "processName": row.process_name,
                }
            )
    return violations


def check_gpu_ownership(
    policy_path: Path,
    process_rows: list[GpuProcess],
    target_service: str | None = None,
    visible_gpu_uuids: list[str] | None = None,
) -> OwnershipReport:
    policy_version, workloads = load_policy(policy_path)
    violations = find_violations(workloads, process_rows, target_service, visible_gpu_uuids)
    return OwnershipReport(
        policy_version=policy_version,
        checked_at=datetime.now(timezone.utc).isoformat(),
        allocations=tuple(process_rows),
        violations=tuple(violations),
        blocking_pids=tuple(sorted({int(item["pid"]) for item in violations})),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--target-service")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = check_gpu_ownership(
            args.policy,
            read_nvidia_processes(),
            args.target_service,
            read_visible_gpu_uuids(),
        )
    except PolicyError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 3 if report.blocking_pids else 0


if __name__ == "__main__":
    sys.exit(main())
