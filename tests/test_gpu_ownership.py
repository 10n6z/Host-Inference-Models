from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("gpu_ownership", ROOT / "scripts/check-gpu-ownership.py")
assert SPEC and SPEC.loader
gpu_ownership = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gpu_ownership
SPEC.loader.exec_module(gpu_ownership)

GPU_0_UUID = "GPU-abffce46-a266-a0b8-8f86-e40d19fd546e"
GPU_1_UUID = "GPU-47cba434-13c2-1aca-e112-f8d1c9659d5a"
GPU_2_UUID = "GPU-c808af26-9750-cfe0-9b8a-b18871c6cfed"
GPU_3_UUID = "GPU-b8a1d704-80b3-37f8-c2ed-e764700c5326"


def test_rejects_generation_process_on_computer_vision_gpu():
    rows = [gpu_ownership.GpuProcess(GPU_1_UUID, 412, "image-server", 426)]
    report = gpu_ownership.check_gpu_ownership(ROOT / "config/gpu-policy.yaml", rows, "vision-ocr")
    assert report.blocking_pids == (412,)


def test_allows_planner_and_ocr_to_share_gpu_1():
    rows = [
        gpu_ownership.GpuProcess(GPU_1_UUID, 501, "vision-planner", 8192),
        gpu_ownership.GpuProcess(GPU_1_UUID, 502, "vision-ocr", 4096),
    ]
    report = gpu_ownership.check_gpu_ownership(ROOT / "config/gpu-policy.yaml", rows)
    assert report.blocking_pids == ()


def test_ignores_processes_on_other_gpus_for_target_service():
    rows = [gpu_ownership.GpuProcess(GPU_0_UUID, 700, "vision-ocr", 1)]
    report = gpu_ownership.check_gpu_ownership(ROOT / "config/gpu-policy.yaml", rows, "vision-ocr")
    assert report.blocking_pids == ()


def test_rejects_malformed_policy(tmp_path: Path):
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 2\nworkloads: {}\n", encoding="utf-8")
    try:
        gpu_ownership.check_gpu_ownership(policy, [])
    except gpu_ownership.PolicyError as exc:
        assert "version" in str(exc)
    else:
        raise AssertionError("malformed policy was accepted")


def test_policy_uses_immutable_gpu_uuids():
    policy = yaml.safe_load((ROOT / "config/gpu-policy.yaml").read_text(encoding="utf-8"))
    assert policy["workloads"]["image-generation"]["allowed_gpu_uuids"] == [GPU_0_UUID]
    assert policy["workloads"]["video-generation"]["allowed_gpu_uuids"] == [GPU_2_UUID]
    assert policy["workloads"]["vision-detection"]["allowed_gpu_uuids"] == [GPU_3_UUID]


def test_launcher_and_routes_pin_each_modality():
    launcher = (ROOT / "start_all.sh").read_text(encoding="utf-8")
    standalone_launcher = (ROOT / "start_model_server.sh").read_text(encoding="utf-8")
    assert f'IMAGE_GPU_UUID="{GPU_0_UUID}"' in launcher
    assert f'VIDEO_GPU_UUID="{GPU_2_UUID}"' in launcher
    assert f'IMAGE_GPU_UUID="{GPU_0_UUID}"' in standalone_launcher
    assert f'VIDEO_GPU_UUID="{GPU_2_UUID}"' in standalone_launcher
    assert "IMAGE_GPU_UUID" in (ROOT / "servers/routes/image.py").read_text(encoding="utf-8")
    assert "VIDEO_GPU_UUID" in (ROOT / "servers/routes/video.py").read_text(encoding="utf-8")
