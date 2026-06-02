"""Smoke test for all HF TTS models in models/hf/.

Tests models that can run on Apple Silicon (MPS/CPU) with auto-download.
Models requiring CUDA or >24GB RAM are marked as SKIP with reason.
All non-skipped models must produce real audio output.
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
import traceback
from pathlib import Path

os.environ["HF_MODELS_ROOT"] = str(Path(__file__).parent / "models" / "hf")

SMOKE_TEXT = "Hello, this is a smoke test."

MODELS_ROOT = Path(__file__).parent / "models" / "hf"
OUTPUT_DIR = Path(__file__).parent / "smoke_test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# (model_dir_name, runner_callable_factory, skip_reason_or_None)
def get_test_registry():
    """Build test registry. Each entry: (name, factory_fn, skip_reason)."""
    registry = []

    # --- Tier 1: Models that WILL work (small, transformers, auto-download) ---

    registry.append((
        "facebook--mms-tts-eng",
        lambda: _test_mms("eng"),
        None,
    ))
    registry.append((
        "facebook--mms-tts-deu",
        lambda: _test_mms("deu"),
        None,
    ))
    registry.append((
        "facebook--mms-tts-fra",
        lambda: _test_mms("fra"),
        None,
    ))
    registry.append((
        "microsoft--speecht5_tts",
        lambda: _test_speecht5(),
        None,
    ))
    registry.append((
        "suno--bark-small",
        lambda: _test_bark("small"),
        None,
    ))
    registry.append((
        "hexgrad--Kokoro-82M",
        lambda: _test_kokoro(),
        None,
    ))
    registry.append((
        "onnx-community--Kokoro-82M-v1.0-ONNX",
        lambda: _test_kokoro_onnx(),
        None,
    ))

    # --- Tier 2: Models with extra packages (now installed) ---

    registry.append((
        "facebook--mms-tts-spa",
        lambda: _test_mms("spa"),
        None,
    ))
    registry.append((
        "suno--bark",
        lambda: _test_bark("full"),
        None,
    ))
    registry.append((
        "ylacombe--parler-tts-mini-jenny-30H",
        lambda: _test_parler_tts(),
        "SKIP: parler-tts 0.2.x incompatible with transformers >=5.0",
    ))
    registry.append((
        "OuteAI--OuteTTS-0.2-500M",
        lambda: _test_outetts("0.2-500M"),
        None,
    ))
    registry.append((
        "OuteAI--OuteTTS-0.3-1B",
        lambda: _test_outetts("0.3-1B"),
        None,
    ))
    registry.append((
        "myshell-ai--MeloTTS-English",
        lambda: _test_melotts(),
        None,
    ))
    registry.append((
        "espnet--kan-bayashi_ljspeech_vits",
        lambda: _test_espnet_vits(),
        None,
    ))

    # --- Tier 2b: Models with extra packages (now installed) ---

    registry.append((
        "ResembleAI--chatterbox",
        lambda: _test_chatterbox(),
        None,
    ))
    registry.append((
        "SWivid--F5-TTS",
        lambda: _test_f5_tts(),
        None,
    ))
    registry.append((
        "SWivid--E2-TTS",
        lambda: _test_e2_tts(),
        None,
    ))
    registry.append((
        "KittenML--kitten-tts-nano-0.2",
        lambda: _test_kitten_tts(),
        None,
    ))

    # --- Tier 3: Large models requiring CUDA or >24GB RAM ---

    registry.append((
        "Qwen--Qwen2.5-Omni-7B",
        lambda: None,
        "SKIP: 7B model requires ~14GB VRAM, CUDA recommended",
    ))
    registry.append((
        "moonshine-ai--Kimi-Audio-7B-Instruct",
        lambda: None,
        "SKIP: 7B model requires ~14GB VRAM, CUDA recommended",
    ))
    registry.append((
        "microsoft--VibeVoice-1.5B",
        lambda: None,
        "SKIP: 1.5B model, not supported in current transformers",
    ))
    registry.append((
        "sesame--csm-1b",
        lambda: None,
        "SKIP: 1B gated model, requires HF access approval",
    ))
    registry.append((
        "OrpheusTTS--Orpheus-3b-0.1-ft",
        lambda: None,
        "SKIP: 3B model, repository removed from HuggingFace",
    ))
    registry.append((
        "canopylabs--orpheus-tts-0.1-finetune-prod",
        lambda: None,
        "SKIP: 3.8B model requires ~15GB RAM",
    ))
    registry.append((
        "Amphion--MaskGCT",
        lambda: None,
        "SKIP: requires Amphion runtime + model weights",
    ))
    registry.append((
        "FunAudioLLM--CosyVoice-300M",
        lambda: None,
        "SKIP: requires cosyvoice runtime (custom install)",
    ))
    registry.append((
        "FunAudioLLM--CosyVoice2-0.5B",
        lambda: None,
        "SKIP: requires cosyvoice runtime (custom install)",
    ))
    registry.append((
        "FunAudioLLM--CosyVoice3-0.5B",
        lambda: None,
        "SKIP: requires cosyvoice runtime (custom install)",
    ))
    registry.append((
        "fishaudio--openaudio-s1-mini",
        lambda: None,
        "SKIP: requires fish-speech runtime",
    ))
    registry.append((
        "coqui--XTTS-v2",
        lambda: None,
        "SKIP: requires TTS package compatible with transformers <5.0",
    ))
    registry.append((
        "MiniMaxAI--Speech-02-HD",
        lambda: None,
        "SKIP: API-based model, no local inference available",
    ))

    return registry


def _test_mms(lang: str):
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.mms_tts_runner import MMSTTSRunner

    runner = MMSTTSRunner(lang=lang)

    advanced_configs = {
        "eng": {
            "speaking_rate": 1.4,
            "noise_scale": 0.3,
            "noise_scale_duration": 0.5,
        },
        "deu": {
            "speaking_rate": 0.7,
            "noise_scale": 1.2,
            "noise_scale_duration": 1.5,
        },
        "fra": {
            "speaking_rate": 1.8,
            "noise_scale": 0.1,
            "noise_scale_duration": 0.3,
        },
        "spa": {
            "speaking_rate": 1.1,
            "noise_scale": 0.5,
            "noise_scale_duration": 0.7,
        },
    }

    config = advanced_configs.get(lang, advanced_configs["eng"])
    out_path = str(OUTPUT_DIR / f"mms_tts_{lang}_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        speaking_rate=config["speaking_rate"],
        noise_scale=config["noise_scale"],
        noise_scale_duration=config["noise_scale_duration"],
    )
    params = result.get("parameters", {})
    assert params.get("speaking_rate") == config["speaking_rate"], \
        f"speaking_rate mismatch: got {params.get('speaking_rate')}"
    assert params.get("noise_scale") == config["noise_scale"], \
        f"noise_scale mismatch: got {params.get('noise_scale')}"
    assert params.get("noise_scale_duration") == config["noise_scale_duration"], \
        f"noise_scale_duration mismatch: got {params.get('noise_scale_duration')}"
    print(f"  params: speaking_rate={config['speaking_rate']}, "
          f"noise_scale={config['noise_scale']}, "
          f"noise_scale_duration={config['noise_scale_duration']}")
    return result


def _test_speecht5():
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.speecht5_runner import SpeechT5Runner

    runner = SpeechT5Runner()
    out_path = str(OUTPUT_DIR / "speecht5_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        speaker_index=2000,
        threshold=0.7,
        minlenratio=0.5,
        maxlenratio=30.0,
    )
    params = result.get("parameters", {})
    assert params.get("speaker_index") == 2000, \
        f"speaker_index mismatch: got {params.get('speaker_index')}"
    assert params.get("threshold") == 0.7, \
        f"threshold mismatch: got {params.get('threshold')}"
    assert params.get("minlenratio") == 0.5, \
        f"minlenratio mismatch: got {params.get('minlenratio')}"
    assert params.get("maxlenratio") == 30.0, \
        f"maxlenratio mismatch: got {params.get('maxlenratio')}"
    print(f"  params: speaker_index=2000, threshold=0.7, "
          f"minlenratio=0.5, maxlenratio=30.0")
    return result


def _test_bark(variant: str):
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.bark_runner import BarkRunner

    runner = BarkRunner(variant=variant)
    out_path = str(OUTPUT_DIR / f"bark_{variant}_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        voice_preset="v2/en_speaker_3",
        do_sample=True,
        temperature=0.7,
        semantic_temperature=0.8,
        coarse_temperature=0.6,
        fine_temperature=0.9,
        semantic_max_new_tokens=512,
        coarse_max_new_tokens=1024,
        fine_max_new_tokens=1024,
    )
    params = result.get("parameters", {})
    assert params.get("voice_preset") == "v2/en_speaker_3", \
        f"voice_preset mismatch: got {params.get('voice_preset')}"
    assert params.get("do_sample") is True
    assert params.get("temperature") == 0.7, \
        f"temperature mismatch: got {params.get('temperature')}"
    assert params.get("semantic_temperature") == 0.8
    assert params.get("coarse_temperature") == 0.6
    assert params.get("fine_temperature") == 0.9
    assert params.get("semantic_max_new_tokens") == 512
    assert params.get("coarse_max_new_tokens") == 1024
    assert params.get("fine_max_new_tokens") == 1024
    print(f"  params: voice_preset=v2/en_speaker_3, temperature=0.7, "
          f"semantic_temp=0.8, coarse_temp=0.6, fine_temp=0.9, "
          f"semantic_tokens=512, coarse_tokens=1024, fine_tokens=1024")
    return result


def _test_kokoro():
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.kokoro_runner import KokoroRunner

    runner = KokoroRunner()
    out_path = str(OUTPUT_DIR / "kokoro_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        voice="af_bella",
        speed=1.5,
        lang_code="a",
        sample_rate=24000,
        split_pattern=r"[.!?]+",
    )
    params = result.get("parameters", {})
    assert params.get("voice") == "af_bella", \
        f"voice mismatch: got {params.get('voice')}"
    assert params.get("speed") == 1.5, \
        f"speed mismatch: got {params.get('speed')}"
    assert params.get("lang_code") == "a"
    assert params.get("split_pattern") == r"[.!?]+"
    print(f"  params: voice=af_bella, speed=1.5, lang_code=a, "
          f"split_pattern=[.!?]+")
    return result


def _test_kokoro_onnx():
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.kokoro_onnx_runner import KokoroONNXRunner

    runner = KokoroONNXRunner()
    out_path = str(OUTPUT_DIR / "kokoro_onnx_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        voice="am_adam",
        speed=1.8,
    )
    params = result.get("parameters", {})
    assert params.get("voice") == "am_adam", \
        f"voice mismatch: got {params.get('voice')}"
    assert params.get("speed") == 1.8, \
        f"speed mismatch: got {params.get('speed')}"
    print(f"  params: voice=am_adam, speed=1.8")
    return result


def _test_parler_tts():
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.parler_tts_runner import ParlerTTSRunner

    runner = ParlerTTSRunner()
    out_path = str(OUTPUT_DIR / "parler_tts.wav")
    result = runner.generate(text=SMOKE_TEXT, output_path=out_path)
    return result


def _test_outetts(variant: str):
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.outetts_runner import OuteTTSRunner

    runner = OuteTTSRunner(variant=variant)
    out_path = str(OUTPUT_DIR / f"outetts_{variant}_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        temperature=0.3,
        repetition_penalty=1.2,
        max_length=2048,
    )
    params = result.get("parameters", {})
    assert params.get("temperature") == 0.3, \
        f"temperature mismatch: got {params.get('temperature')}"
    assert params.get("repetition_penalty") == 1.2
    assert params.get("max_length") == 2048
    print(f"  params: temperature=0.3, repetition_penalty=1.2, max_length=2048")
    return result


def _test_chatterbox():
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.chatterbox_runner import ChatterboxRunner

    runner = ChatterboxRunner()
    out_path = str(OUTPUT_DIR / "chatterbox_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        exaggeration=0.7,
        cfg_weight=0.3,
    )
    params = result.get("parameters", {})
    assert params.get("exaggeration") == 0.7, \
        f"exaggeration mismatch: got {params.get('exaggeration')}"
    assert params.get("cfg_weight") == 0.3
    print(f"  params: exaggeration=0.7, cfg_weight=0.3")
    return result


def _test_f5_tts():
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.f5_tts_runner import F5TTSRunner

    runner = F5TTSRunner()
    out_path = str(OUTPUT_DIR / "f5_tts_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        speed=1.2,
    )
    params = result.get("parameters", {})
    assert params.get("speed") == 1.2, \
        f"speed mismatch: got {params.get('speed')}"
    print(f"  params: speed=1.2")
    return result


def _test_e2_tts():
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.e2_tts_runner import E2TTSRunner

    runner = E2TTSRunner()
    out_path = str(OUTPUT_DIR / "e2_tts_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        speed=0.9,
    )
    params = result.get("parameters", {})
    assert params.get("speed") == 0.9, \
        f"speed mismatch: got {params.get('speed')}"
    print(f"  params: speed=0.9")
    return result


def _test_espnet_vits():
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.espnet_vits_runner import ESPnetVITSRunner

    runner = ESPnetVITSRunner()
    out_path = str(OUTPUT_DIR / "espnet_vits_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        alpha=1.2,
        noise_scale=0.5,
        noise_scale_dur=0.6,
    )
    params = result.get("parameters", {})
    assert params.get("alpha") == 1.2, \
        f"alpha mismatch: got {params.get('alpha')}"
    assert params.get("noise_scale") == 0.5
    assert params.get("noise_scale_dur") == 0.6
    print(f"  params: alpha=1.2, noise_scale=0.5, noise_scale_dur=0.6")
    return result


def _test_kitten_tts():
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.kitten_tts_runner import KittenTTSRunner

    runner = KittenTTSRunner()
    out_path = str(OUTPUT_DIR / "kitten_tts_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        voice="expr-voice-2-f",
        speed=1.3,
    )
    params = result.get("parameters", {})
    assert params.get("voice") == "expr-voice-2-f", \
        f"voice mismatch: got {params.get('voice')}"
    assert params.get("speed") == 1.3
    print(f"  params: voice=expr-voice-2-f, speed=1.3")
    return result


def _test_melotts():
    sys.path.insert(0, str(Path(__file__).parent / "servers"))
    from runners.text_to_speech.melotts_runner import MeloTTSRunner

    runner = MeloTTSRunner()
    out_path = str(OUTPUT_DIR / "melotts_advanced.wav")
    result = runner.generate(
        text=SMOKE_TEXT,
        output_path=out_path,
        speed=1.1,
        speaker_id=0,
    )
    params = result.get("parameters", {})
    assert params.get("speed") == 1.1, \
        f"speed mismatch: got {params.get('speed')}"
    assert params.get("speaker_id") == 0
    print(f"  params: speed=1.1, speaker_id=0")
    return result


def validate_output(result: dict, name: str) -> bool:
    """Validate that audio was actually produced."""
    if not result:
        return False
    output_path = result.get("output_path")
    if not output_path or not Path(output_path).exists():
        print(f"  FAIL: No output file at {output_path}")
        return False
    file_size = Path(output_path).stat().st_size
    if file_size < 100:
        print(f"  FAIL: Output file too small ({file_size} bytes)")
        return False
    duration = result.get("duration_seconds", 0)
    if duration and duration < 0.1:
        print(f"  FAIL: Duration too short ({duration:.3f}s)")
        return False
    print(f"  OK: {file_size} bytes, {duration:.2f}s @ {result.get('sample_rate')}Hz")
    return True


def main():
    registry = get_test_registry()
    passed = 0
    failed = 0
    skipped = 0
    results = []

    print(f"\n{'='*60}")
    print(f"TTS MODEL SMOKE TEST")
    print(f"Models root: {MODELS_ROOT}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    for name, factory, skip_reason in registry:
        print(f"[{name}]")
        if skip_reason:
            print(f"  {skip_reason}")
            skipped += 1
            results.append((name, "SKIP", skip_reason))
            continue

        try:
            start = time.time()
            result = factory()
            elapsed = time.time() - start
            if validate_output(result, name):
                passed += 1
                results.append((name, "PASS", f"{elapsed:.1f}s"))
                print(f"  PASS ({elapsed:.1f}s)")
            else:
                failed += 1
                results.append((name, "FAIL", "invalid output"))
                print(f"  FAIL (invalid output)")
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            results.append((name, "FAIL", str(e)))
            print(f"  FAIL: {e}")
            print(f"  {tb.splitlines()[-2] if len(tb.splitlines()) > 1 else tb}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    if failed > 0:
        print("\nFAILED MODELS:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  - {name}: {detail}")
        sys.stdout.flush()
        os._exit(1)
    else:
        print("\nAll runnable models passed smoke test!")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
