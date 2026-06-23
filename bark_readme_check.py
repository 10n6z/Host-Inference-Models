"""Verify implemented Bark runner against capabilities claimed in suno-ai/bark README."""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

REPO = Path("/home/long/Host-Inference-Models")
os.environ.setdefault("HF_HOME", str(REPO / "models" / "hf-cache"))
sys.path.insert(0, str(REPO / "servers"))

from runners.text_to_speech.bark_runner import BarkRunner  # noqa: E402

OUT = Path(os.environ.get("BARK_OUT", "/home/long/bark_readme_check"))
OUT.mkdir(parents=True, exist_ok=True)

# README capability matrix. Each item mirrors a README section/demo.
CASES = [
    # (id, README claim, text, voice_preset)
    ("01_basic_nonverbal",
     "Basics + nonverbal [laughs] (pizza.webm)",
     "Hello, my name is Suno. And, uh - and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.",
     "v2/en_speaker_6"),
    ("02_foreign_korean",
     "Foreign language auto-detect, Korean (suno_korean.webm)",
     "\ucd94\uc11d\uc740 \ub0b4\uac00 \uac00\uc7a5 \uc88b\uc544\ud558\ub294 \uba85\uc808\uc774\ub2e4. \ub098\ub294 \uba70\uce60 \ub3d9\uc548 \ud734\uc2dd\uc744 \ucde8\ud558\uace0 \uce5c\uad6c \ubc0f \uac00\uc871\uacfc \uc2dc\uac04\uc744 \ubcf4\ub0bc \uc218 \uc788\uc2b5\ub2c8\ub2e4.",
     "v2/en_speaker_6"),
    ("03_german_accent",
     "German-prompt english -> german accent (suno_german_accent.webm)",
     "Der Drei\u00dfigj\u00e4hrige Krieg (1618-1648) war ein verheerender Konflikt, der Europa stark gepr\u00e4gt hat. This is a beginning of the history. If you want to hear more, please continue.",
     "v2/en_speaker_6"),
    ("04_music",
     "Music generation with note markers (lion.webm)",
     "\u266a In the jungle, the mighty jungle, the lion barks tonight \u266a",
     "v2/en_speaker_6"),
    ("05_voice_preset",
     "Voice preset history_prompt=v2/en_speaker_1 (sloth.webm)",
     "I have a silky smooth voice, and today I will tell you about the exercise regimen of the common sloth.",
     "v2/en_speaker_1"),
    ("06_random_voice",
     "Random unique voice (empty preset)",
     "This sentence should be spoken by a random unique voice chosen by the model.",
     None),
    ("07_default_length",
     "~13s default generation length for normal sentence",
     "By default Bark works well with around thirteen seconds of spoken text, and this sentence is meant to test that the model produces audio of roughly that length.",
     "v2/en_speaker_6"),
]

results = []
variant = os.environ.get("BARK_VARIANT", "small")
runner = BarkRunner(variant=variant)
print(f"# Bark capability check  variant={variant}  model_id={runner.model_id}", flush=True)

for cid, claim, text, preset in CASES:
    out_path = str(OUT / f"{variant}_{cid}.wav")
    kw = dict(text=text, output_path=out_path,
              semantic_max_new_tokens=512, coarse_max_new_tokens=1024, fine_max_new_tokens=1024)
    if preset is not None:
        kw["voice_preset"] = preset
    t0 = time.time()
    err = None
    meta = None
    try:
        meta = runner.generate(**kw)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    dt = time.time() - t0
    rec = {
        "id": cid, "claim": claim, "preset": preset,
        "ok": err is None,
        "error": err,
        "wall_s": round(dt, 1),
        "duration_s": round(meta["duration_seconds"], 2) if meta else None,
        "sample_rate": meta["sample_rate"] if meta else None,
        "wav": out_path if err is None else None,
    }
    results.append(rec)
    print(f"[{'OK ' if err is None else 'FAIL'}] {cid}: {claim} "
          f"-> dur={rec['duration_s']}s wall={rec['wall_s']}s {err or ''}", flush=True)

print("\n=== JSON ===")
print(json.dumps({"variant": variant, "model_id": runner.model_id, "results": results}, indent=2))
