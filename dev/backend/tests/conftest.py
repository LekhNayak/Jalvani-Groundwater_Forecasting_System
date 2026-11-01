"""Tests never call Groq or Gnani: real keys in dev/backend/.env are switched off for every test, unless a test
puts in its own fake (see test_speech_and_keys.py)."""
from dataclasses import replace

import pytest

import app.main as main
import app.services.chat as chat
from app.services.speech import gnani


@pytest.fixture(autouse=True)
def no_external_services(monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "settings", replace(chat.settings, groq_api_keys=()))
    monkeypatch.setattr(main, "get_provider", lambda: None)
    monkeypatch.setattr(main, "settings", replace(main.settings, groq_api_keys=()))
    # speech clips made in a test never land in the real cache
    monkeypatch.setattr(gnani, "settings", replace(gnani.settings, tts_cache_dir=tmp_path / "tts"))
    monkeypatch.setattr(gnani, "_clips", {})
