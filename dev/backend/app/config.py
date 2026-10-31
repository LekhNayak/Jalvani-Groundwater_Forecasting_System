"""Settings from environment / dev/backend/.env (keys stay on the server, never sent to the browser)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / ".env")


def env(name: str, default: str = "") -> str:
    """Environment value; an empty value counts as unset (so `KEY=` in .env falls back to the default)."""
    return (os.getenv(name) or "").strip() or default


def groq_keys() -> tuple:
    """GROQ_API_KEY, then GROQ_API_KEY_2, GROQ_API_KEY_3, ... (any that are set), in that order."""
    keys = [env("GROQ_API_KEY")] + [env(f"GROQ_API_KEY_{i}") for i in range(2, 10)]
    return tuple(k for k in keys if k)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(env("JALVAANI_DATA_DIR", str(BACKEND / "data")))
    frontend_dist: Path = BACKEND.parent / "frontend" / "dist"
    serve_frontend: bool = env("SERVE_FRONTEND", "0") == "1"
    # chat (Groq): several keys are tried in turn when one is rate-limited
    groq_api_keys: tuple = field(default_factory=groq_keys)
    groq_model: str = env("GROQ_MODEL", "qwen/qwen3.8-27b")
    groq_url: str = "https://api.groq.com/openai/v1/chat/completions"
    # speech (Gnani Vachana): one API key for STT and TTS (https://docs.gnani.ai)
    gnani_api_key: str = env("GNANI_API_KEY")
    gnani_stt_url: str = env("GNANI_STT_URL", "https://api.vachana.ai/stt/v3")
    gnani_tts_url: str = env("GNANI_TTS_URL", "https://api.vachana.ai/api/v1/tts/inference")
    gnani_stt_model: str = env("GNANI_STT_MODEL", "prisma-v3.0")
    gnani_tts_model: str = env("GNANI_TTS_MODEL", "timbre-v2.5")
    gnani_tts_voice: str = env("GNANI_TTS_VOICE", "Nalini")
    gnani_tts_voice_mr: str = env("GNANI_TTS_VOICE_MR")
    gnani_tts_voice_hi: str = env("GNANI_TTS_VOICE_HI")
    gnani_tts_voice_en: str = env("GNANI_TTS_VOICE_EN")
    gnani_tts_voice_gu: str = env("GNANI_TTS_VOICE_GU")
    # site languages (must match frontend src/lib/langs.ts); voice checked live for all four (Gujarati round trip 2026-11-01)
    languages: tuple = ("en", "mr", "hi", "gu")
    speech_languages: tuple = ("en", "mr", "hi", "gu")


settings = Settings()
