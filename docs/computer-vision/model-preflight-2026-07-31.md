# Computer Vision Task 4 preflight — 2026-07-31

Host: gpu-farmi-004.rd.tuni.fi. Shared conda env `host-models`: torch 2.12.0+cu130,
transformers 5.9.0, no yolox/vllm/ultralytics installed. GPUs: 4x L40S, 44-47GB free each
at time of check.

## Disk blocker (resolved)

Root filesystem was at 99% (6.9GB/438GB free) — `/var/lib/docker` lives on root, so any new
service image build would have failed. `docker image prune -f && docker builder prune -f`
reclaimed 135GB (dangling images + build cache only; no running containers or named volumes
touched). Root is now at 68% (136GB free). User-approved before running.

## yolox-s — verdict: NEEDS_ISOLATION (proceed as its own Docker service, matching

vision-tesseract/vision-ocr-ensemble pattern; do not touch host-models shared env)

- Source: github.com/Megvii-BaseDetection/YOLOX, tag 0.3.0
- Resolved commit (immutable): 419778480ab6ec0590e5d3831b3afb3b46ab2aa3
- License: apache-2.0 (verified via GitHub API repo license field)
- Weights: yolox_s.pth, release tag 0.1.1rc0, verified content-length 72,089,125 bytes
- Not yet installed anywhere on host; no container built yet — container_digest is
  genuinely unknown until Task 6 builds the service image.

## grounding-dino-tiny — verdict: NEEDS_ISOLATION

- Source: huggingface.co/IDEA-Research/grounding-dino-tiny
- Resolved revision: a2bb814dd30d776dcf7e30523b00659f4f141c71
- License: apache-2.0 (from HF cardData)
- Weights: model.safetensors, 689,359,096 bytes
- transformers 5.9.0 (already in host-models) supports GroundingDinoForObjectDetection;
  irrelevant either way since this ships as its own isolated container per the
  established pattern, not a host-models install.
- container_digest unknown until Task 6 builds the service image.

## Ministral-8B-Instruct-2410 (plan's vLLM planner) — verdict: BLOCKED (licensing, not infra)

- huggingface.co/mistralai/Ministral-8B-Instruct-2410 is gated, license field "other" /
  "mrl" = Mistral AI Research License (license_link: mistral.ai/licenses/MRL-0.1.md).
  Model card's own gated prompt: "If You want to use a Mistral Model ... for any purpose
  that is not expressly authorized under this Agreement, You must request a license from
  Mistral AI." This is non-commercial by default; SW4E Sandbox is a commercial multi-tenant
  product, so this specific model cannot be used without a separate paid/negotiated
  agreement with Mistral AI. This is a legal constraint, not something to route around.

### Substitution (per standing instruction: substitute when a plan model can't be used)

- mistralai/Mistral-7B-Instruct-v0.3 — license: apache-2.0 (verified, unambiguous
  commercial-use grant)
- Resolved revision: c170c708c41dac9275d15a8fff4eca08d52bab71
- Weights: 3-shard safetensors, 4949453792 + 4999819336 + 4546807800 bytes
  (~14.5GB total, bf16) — fits comfortably in disk (136GB free) and GPU memory
  (44-47GB free per L40S)
- vLLM version/CUDA compatibility with the target service's own container (not the shared
  host-models env, since vision-planner ships isolated per the plan's own file list) still
  needs to be pinned when that service is actually built.

## What's left before docs/computer-vision-model-lock.yaml can be generated for real

The existing `scripts/write-computer-vision-model-lock.py` (already built, 9/9 tests
passing, commit 30db93c) requires `container_digest` matching `^sha256:[0-9a-f]{64}$` per
report. That digest only exists once a service's Docker image is actually built — i.e.
after Task 6 (YOLOX-S + Grounding DINO services) and the vision-planner service (Mistral
substitute) are implemented. Writing a fabricated digest now would violate the
no-placeholders instruction. This is the concrete next dependency, not a stall.
