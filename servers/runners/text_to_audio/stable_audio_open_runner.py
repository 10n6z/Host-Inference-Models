from runners.text_to_audio.stable_audio_open import StableAudioOpenRunner as _StableAudioOpenRunner


class StableAudioOpenRunner(_StableAudioOpenRunner):
    available = True
    unavailable_reason = None

__all__ = ["StableAudioOpenRunner"]
