# Task 2 Report — OuteTTS Voice Cloning via ref_audio_path

## Status
DONE

## Commit
`fcbc514` — feat(outetts): add speaker-profile voice cloning via ref_audio_path

## Changes Made

### `servers/runners/text_to_speech/outetts_runner.py`
- Replaced `generate()` with new signature adding `ref_audio_path: str | None = None` and `ref_text: str | None = None`
- When `ref_audio_path` is provided and the file exists: calls `interface.create_speaker(ref_audio_path, ref_text)` (or without transcript if `ref_text` is falsy), stores result as `speaker`
- `speaker` is forwarded into both the `GenerationConfig` branch (new API) and the legacy kwargs branch
- `cloned: bool` added to returned `parameters` dict
- `os` and `sf` were already imported — no new imports added

### `services/audio-outetts/main.py`
- Added `ref_audio_path` and `ref_text` to `SUPPORTED_MODELS["outetts"]["fields"]`
- Added `ref_audio_path: str = Field("", max_length=500)` and `ref_text: str = Field("", max_length=TTS_TEXT_MAX_LENGTH)` to `OuteTTSParams` (required because `extra="forbid"`)
- Passed `ref_audio_path=req.ref_audio_path or None` and `ref_text=req.ref_text or None` to the runner `.generate()` call

### `model-gateway/registry.yaml`
- Added `ref_audio_path` and `ref_text` fields to the `outetts:` entry's `fields:` block (after `max_length`, before `format`)
- Kept YAML indentation consistent with sibling fields

### `tests/test_outetts_clone.py` (new file)
- TDD test that stubs `numpy`, `soundfile`, and `outetts` modules (none installed in test env)
- Verified `create_speaker` is called with correct `audio_path` and `transcript`
- Verified the returned speaker object is forwarded to `interface.generate()`

## Test Output

```
Pytest: 3 passed

tests/test_outetts_clone.py::test_generate_creates_speaker_when_ref_audio_present  PASSED
tests/test_uploads_audio.py::...  PASSED (2 tests)
```

Registry parse: `registry ok`

## Deviations / Concerns

- **Test stubs numpy/soundfile**: The test env does not have `numpy` or `soundfile` installed, so the test stubs them via `monkeypatch.setitem(sys.modules, ...)` before importing the runner. This is consistent with how other tests in this repo stub heavy ML deps. The stubs are minimal (only what `generate()` uses: `sf.info()` returning frames/samplerate).
- No save_speaker/load_speaker JSON caching implemented (explicitly out of scope per task spec).
- No format validation added for the reference WAV/FLAC (out of scope per task spec).
