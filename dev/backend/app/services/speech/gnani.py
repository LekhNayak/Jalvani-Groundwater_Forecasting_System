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
import hashlib
import threading
import time

import httpx

from ...config import settings
from .base import SpeechNotConfigured, SpeechProvider

LANG_CODES = {"en": "en-IN", "mr": "mr-IN", "hi": "hi-IN", "gu": "gu-IN"}
TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class GnaniError(RuntimeError):
    pass


# Gnani allows ONE active request per account ("Only 1 active request is allowed", HTTP 429), so every call waits its
# turn here, and a call that still meets a 429 (another process, another user) is retried a few times.
_lock = threading.Lock()
RETRIES = 4
_clips: dict[str, bytes] = {}     # clips already made, newest last (also on disk, see settings.tts_cache_dir)
MAX_CLIPS_IN_MEMORY = 200
_client = None


def _post(*args, **kwargs):
    for attempt in range(RETRIES):
        r = _http().post(*args, **kwargs)
        if r.status_code != 429 or attempt == RETRIES - 1:
            return r
        time.sleep(0.6 * (attempt + 1))


def _http() -> httpx.Client:
    """One shared client, so the secure connection to Gnani is opened once and reused (saves a few hundred ms per call)."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=TIMEOUT, limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=120))
    return _client


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
        with _lock:
            r = _post(settings.gnani_stt_url, headers=self.headers, data=data,
                      files={"audio_file": (f"speech.{ext}", audio, mime.split(";")[0] or "audio/wav")})
        if r.status_code != 200:
            raise GnaniError(f"STT HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        if body.get("success") is False:
            raise GnaniError(f"STT failed: {str(body)[:200]}")
        return str(body.get("transcript", "")).strip()

    def synthesize(self, text: str, lang: str) -> tuple[bytes, str]:
        """Text -> WAV. The same text in the same language and voice always gives the same clip, so clips are kept
        (in memory and on disk): a card read once is instant for everyone afterwards."""
        voice = {"mr": settings.gnani_tts_voice_mr, "hi": settings.gnani_tts_voice_hi, "en": settings.gnani_tts_voice_en,
                 "gu": settings.gnani_tts_voice_gu}.get(lang) or settings.gnani_tts_voice
        rate = settings.gnani_tts_sample_rate
        key = hashlib.sha256("|".join([text, LANG_CODES.get(lang, "en-IN"), voice, settings.gnani_tts_model, str(rate)]).encode()).hexdigest()
        clip = self._cached(key)
        if clip is not None:
            return clip, "audio/wav"
        payload = {"text": text, "voice": voice, "model": settings.gnani_tts_model,
                   "language": LANG_CODES.get(lang, "en-IN"), "speed": 1.0,
                   "audio_config": {"sample_rate": rate, "num_channels": 1, "sample_width": 2,
                                    "encoding": "linear_pcm", "container": "wav"}}
        with _lock:
            clip = self._cached(key)             # another request may have made it while this one waited
            if clip is not None:
                return clip, "audio/wav"
            r = _post(settings.gnani_tts_url, headers={**self.headers, "Content-Type": "application/json"}, json=payload)
        if r.status_code != 200 or not r.content:
            raise GnaniError(f"TTS HTTP {r.status_code}: {r.text[:200]}")
        self._store(key, r.content)
        return r.content, r.headers.get("content-type", "audio/wav").split(";")[0]

    def _cached(self, key: str) -> bytes | None:
        if key in _clips:
            return _clips[key]
        f = settings.tts_cache_dir / f"{key}.wav"
        try:
            if f.exists():
                _clips[key] = f.read_bytes()
                return _clips[key]
        except OSError:
            pass
        return None

    def _store(self, key: str, clip: bytes):
        _clips[key] = clip
        while len(_clips) > MAX_CLIPS_IN_MEMORY:
            _clips.pop(next(iter(_clips)))
        try:
            settings.tts_cache_dir.mkdir(parents=True, exist_ok=True)
            (settings.tts_cache_dir / f"{key}.wav").write_bytes(clip)
        except OSError:
            pass                                   # a read-only disk only costs the speed-up


def get_provider():
    try:
        return GnaniProvider()
    except SpeechNotConfigured:
        return None
