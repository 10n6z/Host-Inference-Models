#!/usr/bin/env python3
"""Generate the model license audit + commercial-coverage report.

Reads registry.yaml + licensing.py (the single source of truth) and emits a
Markdown report covering every model: category, name, license, commercial-use
status, and per-category commercial counts.

Usage:
    python3 model-gateway/audit_models.py            # print to stdout
    python3 model-gateway/audit_models.py --out FILE  # also write to FILE
    python3 model-gateway/audit_models.py --check     # exit 1 if a category < 20
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from licensing import resolve_license  # noqa: E402

REGISTRY_PATH = HERE / "registry.yaml"
TARGET_PER_CATEGORY = 20


def category(entry: dict) -> str:
    raw = f"{entry.get('modality') or entry.get('task') or ''}".lower()
    if "image" in raw:
        return "image"
    if "video" in raw:
        return "video"
    if "speech" in raw or "audio" in raw:
        return "audio"
    return "text"


def load_models() -> dict:
    reg = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    return reg.get("models") or {}


def build_report() -> tuple[str, bool]:
    models = load_models()
    by_cat: dict[str, list[dict]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [commercial, total]

    for model_id, entry in sorted(models.items()):
        lic = resolve_license(model_id, entry)
        cat = category(entry)
        commercial = lic["commercial_use"]
        counts[cat][1] += 1
        if commercial == "yes":
            counts[cat][0] += 1
        by_cat[cat].append(
            {
                "id": model_id,
                "name": entry.get("display_name", model_id),
                "license": lic["license"],
                "commercial": commercial,
                "implemented": entry.get("implemented", True) is not False,
                "source_repo": entry.get("source_repo"),
            }
        )

    lines: list[str] = []
    lines.append("# Server Model License Audit\n")
    lines.append(
        "Generated from `model-gateway/registry.yaml` + `model-gateway/licensing.py`. "
        "`commercial_use` is tri-state: `yes` (license permits commercial use), "
        "`no` (research/non-commercial), `unknown` (unverified, counted as non-commercial).\n"
    )

    lines.append("## Commercial coverage per category\n")
    lines.append("| Category | Commercial | Total | Meets >=20 |")
    lines.append("| --- | --- | --- | --- |")
    all_ok = True
    for cat in ["text", "image", "audio", "video"]:
        com, tot = counts[cat]
        ok = com >= TARGET_PER_CATEGORY
        all_ok = all_ok and ok
        lines.append(f"| {cat} | {com} | {tot} | {'yes' if ok else 'NO'} |")
    lines.append("")

    for cat in ["text", "image", "audio", "video"]:
        com, tot = counts[cat]
        lines.append(f"## {cat} ({com} commercial / {tot} total)\n")
        lines.append("| Model | Name | License | Commercial use | Runner |")
        lines.append("| --- | --- | --- | --- | --- |")
        for m in by_cat[cat]:
            runner = "wired" if m["implemented"] else "pending"
            lines.append(
                f"| `{m['id']}` | {m['name']} | {m['license']} | {m['commercial']} | {runner} |"
            )
        lines.append("")

    return "\n".join(lines), all_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    report, all_ok = build_report()
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"\n[written] {args.out}", file=sys.stderr)
    if args.check and not all_ok:
        print("[FAIL] at least one category has < 20 commercial models", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
