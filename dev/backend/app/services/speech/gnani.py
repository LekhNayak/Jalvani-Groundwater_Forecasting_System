"""Gnani Vachana speech adapter: speech-to-text and text-to-speech with ONE API key.

From Gnani's API quick start (https://docs.gnani.ai/api/introduction/quick-start.md, checked 2026-10-31):
- auth:  header `X-API-Key-ID: <key>` on every request
- STT:   POST https://api.vachana.ai/stt/v3, multipart: `audio_file`, `language_code` (hi-IN, mr-IN, en-IN, ...),
         optional `model` (prisma-v3.0) and `format`. WAV / MP3 / OGG / FLAC / AAC / M4A, 8-44.1 kHz, at most 30 s.
         -> JSON {success, transcript, ...}
- TTS:   POST https://api.vachana.ai/api/v1/tts/inference, JSON: `text`, `voice`, `model` (timbre-v2.5), `language`,
         `speed`, `audio_config` {sample_rate, num_channels, sample_width, encoding, container} -> audio/wav bytes
The browser sends WAV (converted from its recording in src/lib/wav.ts), so no server-side audio conversion is needed.
Endpoints, models and voices can be overridden in dev/backend/.env (GNANI_* in .env.example).
"""
import httpx

from ...config import settings
from .base import SpeechNotConfigured, SpeechProvider

LANG_CODES = {"en": "en-IN", "mr": "mr-IN", "hi": "hi-IN"}
TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class GnaniError(RuntimeError):
    pass


class GnaniProvider(SpeechProvider):
    name = "gnani"

    def __init__(self):
        if not settings.gnani_api_key:
            raise SpeechNotConfigured("set GNANI_API_KEY in dev/backend/.env")
        self.headers = {"X-API-Key-ID": settings.gnani_api_key}

    def transcribe(self, audio: bytes, mime: str, lang: str) -> str:
        ext = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3", "audio/ogg": "ogg",
               "audio/mp4": "m4a", "audio/flac": "flac"}.get(mime.split(";")[0], "wav")
        data = {"language_code": LANG_CODES.get(lang, "en-IN")}
        if settings.gnani_stt_model:
            data["model"] = settings.gnani_stt_model
        r = httpx.post(settings.gnani_stt_url, headers=self.headers, data=data,
                       files={"audio_file": (f"speech.{ext}", audio, mime.split(";")[0] or "audio/wav")}, timeout=TIMEOUT)
        if r.status_code != 200:
            raise GnaniError(f"STT HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        if body.get("success") is False:
            raise GnaniError(f"STT failed: {str(body)[:200]}")
        return str(body.get("transcript", "")).strip()

    def synthesize(self, text: str, lang: str) -> tuple[bytes, str]:
        voice = {"mr": settings.gnani_tts_voice_mr, "hi": settings.gnani_tts_voice_hi,
                 "en": settings.gnani_tts_voice_en}.get(lang) or settings.gnani_tts_voice
        payload = {"text": text, "voice": voice, "model": settings.gnani_tts_model,
                   "language": LANG_CODES.get(lang, "en-IN"), "speed": 1.0,
                   "audio_config": {"sample_rate": 24000, "num_channels": 1, "sample_width": 2,
                                    "encoding": "linear_pcm", "container": "wav"}}
        r = httpx.post(settings.gnani_tts_url, headers={**self.headers, "Content-Type": "application/json"},
                       json=payload, timeout=TIMEOUT)
        if r.status_code != 200 or not r.content:
            raise GnaniError(f"TTS HTTP {r.status_code}: {r.text[:200]}")
        return r.content, r.headers.get("content-type", "audio/wav").split(";")[0]


def get_provider():
    try:
        return GnaniProvider()
    except SpeechNotConfigured:
        return None
