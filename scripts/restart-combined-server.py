#!/usr/bin/env python3
"""Fail-closed restart adapter for the verified combined-server manager."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml


@dataclass(frozen=True)
class ServiceControl:
    manager: str
    service_name: str
    health_url: str
    restart_timeout_seconds: float
    session_name: str | None = None
    start_script: str | None = None


def load_service_control(path: Path) -> ServiceControl:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load service-control config: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("service-control config must be a mapping")
    combined = raw.get("combined_server")
    if not isinstance(combined, dict):
        raise ValueError("combined_server config is required")
    manager = combined.get("manager")
    if manager not in {"systemd", "docker-compose", "tmux"}:
        raise ValueError("manager must be systemd, docker-compose, or tmux")
    service_name = combined.get("service_name")
    if not isinstance(service_name, str) or not service_name.strip():
        raise ValueError("service_name is required")
    health_url = combined.get("health_url")
    parsed = urlparse(health_url) if isinstance(health_url, str) else None
    if not parsed or parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("health_url must use a loopback host")
    timeout = combined.get("restart_timeout_seconds")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("restart_timeout_seconds must be positive")

    session_name = None
    start_script = None
    if manager == "tmux":
        session_name = combined.get("session_name")
        if not isinstance(session_name, str) or not session_name.strip():
            raise ValueError("session_name is required for the tmux manager")
        start_script = combined.get("start_script")
        if not isinstance(start_script, str) or not start_script.strip():
            raise ValueError("start_script is required for the tmux manager")
        session_name = session_name.strip()
        start_script = start_script.strip()

    return ServiceControl(
        manager,
        service_name.strip(),
        health_url,
        float(timeout),
        session_name=session_name,
        start_script=start_script,
    )


def wait_for_health(
    health_url: str,
    timeout_seconds: float,
    *,
    open_url: Callable[[str, float], object] = urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with open_url(health_url, 2.0):
                return
        except Exception as exc:  # health may be unavailable during restart
            last_error = exc
            sleep_fn(1.0)
    raise TimeoutError(f"combined server health did not recover: {last_error}")


def _restart_command(config: ServiceControl) -> list[str]:
    if config.manager == "systemd":
        return ["systemctl", "restart", config.service_name]
    if config.manager == "docker-compose":
        return ["docker", "compose", "restart", config.service_name]
    # tmux: kill any existing session (ignore absence), then respawn via the
    # start script. tmux has no "restart" verb, so this is the equivalent.
    assert config.session_name and config.start_script
    shell = (
        f"tmux kill-session -t {shlex.quote(config.session_name)} 2>/dev/null; "
        f"exec {shlex.quote(config.start_script)}"
    )
    return ["sh", "-c", shell]


def restart_combined_server(
    config: ServiceControl,
    *,
    run: Callable[..., object] = subprocess.run,
    wait: Callable[[str, float], None] = wait_for_health,
) -> None:
    run(_restart_command(config), check=True, timeout=config.restart_timeout_seconds)
    wait(config.health_url, config.restart_timeout_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    restart_combined_server(load_service_control(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
