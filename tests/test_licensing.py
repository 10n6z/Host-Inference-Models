"""License resolution + commercial-coverage tests for the model gateway.

Runs under pytest (repo convention) and also standalone:
    python3 tests/test_licensing.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_DIR = REPO_ROOT / "model-gateway"
sys.path.insert(0, str(GATEWAY_DIR))

from licensing import resolve_license  # noqa: E402

TARGET_PER_CATEGORY = 20


def _category(entry):
    raw = f"{entry.get('modality') or entry.get('task') or ''}".lower()
    if "image" in raw:
        return "image"
    if "video" in raw:
        return "video"
    if "speech" in raw or "audio" in raw:
        return "audio"
    return "text"


def _models():
    reg = yaml.safe_load((GATEWAY_DIR / "registry.yaml").read_text(encoding="utf-8"))
    return reg["models"]


def test_explicit_license_overrides_table():
    out = resolve_license("x", {"license": "mit"})
    assert out["license"] == "mit"
    assert out["commercial_use"] == "yes"


def test_non_commercial_license_marked_no():
    out = resolve_license("x", {"license": "cc-by-nc-4.0"})
    assert out["commercial_use"] == "no"


def test_explicit_commercial_use_flag_respected():
    out = resolve_license("x", {"license": "cogvideox-license", "commercial_use": "yes"})
    assert out["commercial_use"] == "yes"


def test_unknown_license_defaults_to_unknown():
    out = resolve_license("totally-unseen-model", {})
    assert out["license"] == "unknown"
    assert out["commercial_use"] == "unknown"


def test_text_prefix_heuristics():
    assert resolve_license("mistral:7b", {})["commercial_use"] == "yes"
    assert resolve_license("qwen2.5-coder:7b", {})["commercial_use"] == "yes"
    assert resolve_license("phi4:14b", {})["license"] == "mit"


def test_original_image_models_resolve_via_table():
    assert resolve_license("flux-1-schnell", {})["license"] == "apache-2.0"
    assert resolve_license("stable-diffusion-3.5-medium", {})["commercial_use"] == "yes"


def test_every_category_has_at_least_20_commercial():
    counts = defaultdict(lambda: [0, 0])
    for model_id, entry in _models().items():
        cat = _category(entry)
        counts[cat][1] += 1
        if resolve_license(model_id, entry)["commercial_use"] == "yes":
            counts[cat][0] += 1
    for cat in ["text", "image", "audio", "video"]:
        commercial, _total = counts[cat]
        assert commercial >= TARGET_PER_CATEGORY, f"{cat} only has {commercial} commercial models"


def test_commercial_models_carry_a_real_license():
    for model_id, entry in _models().items():
        info = resolve_license(model_id, entry)
        if info["commercial_use"] == "yes":
            assert info["license"] not in ("", "unknown"), f"{model_id} commercial but no license"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
            passed += 1
    print(f"\n{passed} tests passed")
