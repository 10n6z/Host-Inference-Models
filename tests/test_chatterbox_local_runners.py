from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_chatterbox_multilingual_runner_is_local():
    source = (REPO_ROOT / "servers/runners/text_to_speech/chatterbox_multilingual_runner.py").read_text()

    assert "gradio_client" not in source
    assert "Client(" not in source
    assert "ChatterboxMultilingualTTS" in source
    assert "t3_model=\"v3\"" in source


def test_chatterbox_turbo_runner_is_local():
    source = (REPO_ROOT / "servers/runners/text_to_speech/chatterbox_turbo_runner.py").read_text()

    assert "gradio_client" not in source
    assert "Client(" not in source
    assert "ChatterboxTurboTTS" in source
    assert "top_k=int(top_k)" in source


def test_chatterbox_service_does_not_depend_on_gradio_client():
    requirements = (REPO_ROOT / "services/audio-chatterbox/requirements.txt").read_text()

    assert "gradio_client" not in requirements
    assert "github.com/resemble-ai/chatterbox.git" in requirements
