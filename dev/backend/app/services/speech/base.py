"""Speech provider interface (STT and TTS). See dev/docs/SPEECH_PLAN.md."""
from abc import ABC, abstractmethod


class SpeechNotConfigured(RuntimeError):
    """Raised when no speech provider is configured (the endpoints answer 501)."""


class SpeechProvider(ABC):
    name = "base"

    @abstractmethod
    def transcribe(self, audio: bytes, mime: str, lang: str) -> str:
        """Audio (as recorded by the browser) -> text in `lang` (en / mr / hi)."""

    @abstractmethod
    def synthesize(self, text: str, lang: str) -> tuple[bytes, str]:
        """Text -> (audio bytes, mime type)."""
