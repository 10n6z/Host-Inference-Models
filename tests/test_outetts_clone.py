import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNERS_DIR = REPO_ROOT / "servers"


def _stub_heavy_deps(monkeypatch):
    """Stub out native deps that aren't installed in the test env."""
    if "numpy" not in sys.modules:
        monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    if "soundfile" not in sys.modules:
        fake_sf = types.ModuleType("soundfile")

        class _FakeSFInfo:
            frames = 24000
            samplerate = 24000

        fake_sf.info = lambda path: _FakeSFInfo()
        monkeypatch.setitem(sys.modules, "soundfile", fake_sf)


def _load_runner(monkeypatch):
    _stub_heavy_deps(monkeypatch)
    monkeypatch.syspath_prepend(str(RUNNERS_DIR))
    # Remove cached module so monkeypatched sys.modules take effect
    for key in list(sys.modules):
        if "outetts_runner" in key:
            del sys.modules[key]
    from runners.text_to_speech.outetts_runner import OuteTTSRunner
    return OuteTTSRunner


def test_generate_creates_speaker_when_ref_audio_present(monkeypatch, tmp_path):
    OuteTTSRunner = _load_runner(monkeypatch)
    runner = OuteTTSRunner(variant="0.2-500M")

    created = {}

    class _FakeOutput:
        def save(self, path):
            Path(path).write_bytes(b"RIFF0000WAVEfmt ")

    class _FakeInterface:
        def create_speaker(self, audio_path, *args):
            created["audio_path"] = audio_path
            created["transcript"] = args[0] if args else None
            return {"speaker": "fake"}

        def generate(self, *args, **kwargs):
            created["speaker_arg"] = kwargs.get("speaker")
            return _FakeOutput()

    runner.interface = _FakeInterface()
    runner.load = lambda: None  # already "loaded"

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF0000WAVEfmt ")
    out = tmp_path / "out.wav"

    fake_outetts = types.ModuleType("outetts")  # no GenerationConfig attr -> legacy kwargs branch
    monkeypatch.setitem(sys.modules, "outetts", fake_outetts)

    runner.generate(
        text="hello",
        output_path=str(out),
        ref_audio_path=str(ref),
        ref_text="hello there",
    )

    assert created["audio_path"] == str(ref)
    assert created["transcript"] == "hello there"
    assert created["speaker_arg"] == {"speaker": "fake"}
