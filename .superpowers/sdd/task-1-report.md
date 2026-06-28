# Task 1 Report — POST /uploads/audio

## Status
DONE

## Commit
`92680a4` feat(gateway): add POST /uploads/audio for voice-clone reference audio

## Test Summary
2 passed (`tests/test_uploads_audio.py`)

---

## What was changed

### `model-gateway/main.py`
- Added `import shutil` to stdlib imports.
- Added `File` and `UploadFile` to the fastapi import line (already had `FastAPI`, `Request`).
- Added `ALLOWED_REF_AUDIO_EXT = {".wav", ".flac"}` constant.
- Added `@app.post("/uploads/audio")` async handler immediately before `@app.get("/jobs/{job_id}")`.
  - Rejects non-.wav/.flac with HTTP 415 and a structured error body.
  - Creates `OUTPUT_ROOT / "uploads"` dir, writes file as `ref_<uuid><ext>`.
  - Returns `{"path": "<abs_dest>", "url": "<public_base>/outputs/uploads/<name>"}`.
  - Reuses existing `OUTPUT_ROOT` Path and `_gateway_public_base_from_request(request)` helper.

### `model-gateway/requirements.txt`
- Added `python-multipart==0.0.18` (required by FastAPI for `UploadFile` support).

### `tests/test_uploads_audio.py` (new file)
- `test_upload_audio_accepts_wav` — POSTs a .wav, asserts 200, checks `path` ends with `.wav`, `url` contains `/outputs/uploads/`, and file exists on disk.
- `test_upload_audio_rejects_mp3` — POSTs a .mp3, asserts 415.

---

## TDD Run Log

**Before implementation (confirm FAIL):**
```
FAILED tests/test_uploads_audio.py::test_upload_audio_accepts_wav — assert 404 == 200
FAILED tests/test_uploads_audio.py::test_upload_audio_rejects_mp3  — assert 404 == 415
```

**After implementation (confirm PASS):**
```
Pytest: 2 passed
```

---

## Deviations / Concerns
None. No features were added beyond the spec. No other handlers were modified.
One environmental note: pytest was not pre-installed in any conda environment; it was installed into the system Python 3.13 via `pip3 install --system` to run the tests. The repo's production code runs in Docker/conda on the GPU server, so this has no impact on deployment.

---

## Fix Report (reviewer follow-up — 2026-06-23)

### Status
DONE

### Commit
`fd9094b` fix(gateway): return 400 for nameless upload + assert ref_ prefix

### Test result
`python -m pytest tests/test_uploads_audio.py -v` — **2 passed**

### What was changed

**Fix 1 — `model-gateway/main.py`**
Added an explicit guard as the first statement inside `upload_audio`, before the `ext = ...` line:
- When `file.filename` is empty/None, returns HTTP 400 with error code `MISSING_FILENAME` instead of falling through to the misleading 415 branch.

**Fix 2 — `tests/test_uploads_audio.py`**
In `test_upload_audio_accepts_wav`, added after the existing `.endswith(".wav")` assertion:
```python
    assert Path(body["path"]).name.startswith("ref_")
```
Used the `Path` already imported at the top of the file (no local alias needed).
