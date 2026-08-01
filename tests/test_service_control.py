from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


service_control = _load("restart-combined-server")
sample = _load("sample-gpu-ownership")


def config(tmp_path, **overrides):
    values = {
        "manager": "docker-compose",
        "service_name": "model-server",
        "health_url": "http://127.0.0.1:8001/health",
        "restart_timeout_seconds": 120,
    }
    values.update(overrides)
    body = "combined_server:\n" + "\n".join(f"  {key}: {value!r}" for key, value in values.items()) + "\n"
    path = tmp_path / "service-control.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_service_control_accepts_verified_compose_shape(tmp_path):
    result = service_control.load_service_control(config(tmp_path))
    assert result.service_name == "model-server"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"manager": "kubernetes"}, "manager"),
        ({"service_name": ""}, "service_name"),
        ({"health_url": "http://example.test/health"}, "loopback"),
        ({"restart_timeout_seconds": 0}, "timeout"),
    ],
)
def test_service_control_rejects_unsafe_config(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        service_control.load_service_control(config(tmp_path, **overrides))


def test_restart_uses_compose_service_and_waits_for_health(tmp_path):
    control = service_control.load_service_control(config(tmp_path))
    commands = []
    health = []
    service_control.restart_combined_server(
        control,
        run=lambda command, **kwargs: commands.append((command, kwargs)),
        wait=lambda url, timeout: health.append((url, timeout)),
    )
    assert commands[0][0] == ["docker", "compose", "restart", "model-server"]
    assert health[0][0] == "http://127.0.0.1:8001/health"


def test_wait_for_health_passes_timeout_as_a_keyword_not_request_body():
    calls = []

    def fake_open_url(url, timeout):
        calls.append((url, timeout))
        return _OkResponse()

    service_control.wait_for_health(
        "http://127.0.0.1:8001/health",
        2.0,
        open_url=fake_open_url,
        sleep_fn=lambda _: None,
    )
    assert calls == [("http://127.0.0.1:8001/health", 2.0)]


class _OkResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_wait_for_health_default_open_url_uses_urlopen_timeout_kwarg(monkeypatch):
    captured = {}

    def fake_urlopen(url, timeout=None, **kwargs):
        captured["url"] = url
        captured["timeout"] = timeout
        return _OkResponse()

    monkeypatch.setattr(service_control, "urlopen", fake_urlopen)
    service_control.wait_for_health("http://127.0.0.1:8001/health", 2.0, sleep_fn=lambda _: None)
    assert captured == {"url": "http://127.0.0.1:8001/health", "timeout": 2.0}


def tmux_config(tmp_path, **overrides):
    values = {
        "manager": "tmux",
        "service_name": "model-server",
        "session_name": "model-server",
        "start_script": "/home/long/Host-Inference-Models/start_model_server.sh",
        "health_url": "http://127.0.0.1:8001/health",
        "restart_timeout_seconds": 120,
    }
    values.update(overrides)
    body = "combined_server:\n" + "\n".join(f"  {key}: {value!r}" for key, value in values.items()) + "\n"
    path = tmp_path / "service-control.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_service_control_rejects_tmux_config_missing_session_name(tmp_path):
    with pytest.raises(ValueError, match="session_name"):
        service_control.load_service_control(tmux_config(tmp_path, session_name=""))


def test_service_control_rejects_tmux_config_missing_start_script(tmp_path):
    with pytest.raises(ValueError, match="start_script"):
        service_control.load_service_control(tmux_config(tmp_path, start_script=""))


def test_service_control_accepts_verified_tmux_shape(tmp_path):
    result = service_control.load_service_control(tmux_config(tmp_path))
    assert result.manager == "tmux"
    assert result.session_name == "model-server"
    assert result.start_script == "/home/long/Host-Inference-Models/start_model_server.sh"


def test_restart_uses_tmux_kill_and_respawn_and_waits_for_health(tmp_path):
    control = service_control.load_service_control(tmux_config(tmp_path))
    commands = []
    health = []
    service_control.restart_combined_server(
        control,
        run=lambda command, **kwargs: commands.append((command, kwargs)),
        wait=lambda url, timeout: health.append((url, timeout)),
    )
    command = commands[0][0]
    assert command[0] == "sh"
    assert "tmux kill-session -t model-server" in command[2]
    assert "start_model_server.sh" in command[2]
    assert health[0][0] == "http://127.0.0.1:8001/health"


def test_gpu_sampler_writes_one_row_per_interval(tmp_path):
    output = tmp_path / "sample.jsonl"
    count = sample.sample_ownership(
        3,
        1,
        output,
        read_rows=lambda: [{"gpuUuid": "GPU-test", "pid": 12, "usedMemory": 1}],
        sleep_fn=lambda _: None,
    )
    assert count == 3
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3
