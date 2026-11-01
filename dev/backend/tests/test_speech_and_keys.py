"""Groq key rotation and the Gnani adapter, with fakes (no network)."""
from dataclasses import replace

import types

import httpx
from fastapi.testclient import TestClient

import app.main as main
import app.services.chat as chat
import app.services.speech.gnani as gnani
from app.main import ChatRequest, app

client = TestClient(app)


class FakeResp:
    def __init__(self, status, body=None, content=b"", headers=None):
        self.status_code, self._body, self.content = status, body, content
        self.text = str(body)
        self.headers = headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


def test_groq_rotates_to_next_key_on_rate_limit(monkeypatch):
    used = []

    class FakeClient:
        def post(self, url, json, headers):
            used.append(headers["Authorization"])
            if headers["Authorization"].endswith("k1"):
                return FakeResp(429, {"error": "rate limited"})
            return FakeResp(200, {"choices": [{"message": {"content":
                '{"reply": "Here is the forecast card.", "actions": [{"type": "highlight", "element": "card-forecast"}]}'}}]})

    monkeypatch.setattr(chat, "settings", replace(chat.settings, groq_api_keys=("k1", "k2", "k3")))
    monkeypatch.setattr(chat, "get_client", lambda: FakeClient())
    monkeypatch.setattr(chat, "_next_key", iter([0]))
    out = chat.chat(ChatRequest(message="show forecast", lang="en"))
    assert used == ["Bearer k1", "Bearer k2"]
    assert out["mode"] == "groq" and out["actions"] == [{"type": "highlight", "element": "card-forecast"}]


def test_groq_failure_falls_back_to_guide(monkeypatch):
    class Boom:
        def post(self, *a, **k): raise httpx.ConnectError("offline")

    monkeypatch.setattr(chat, "settings", replace(chat.settings, groq_api_keys=("k1",)))
    monkeypatch.setattr(chat, "get_client", lambda: Boom())
    out = chat.chat(ChatRequest(message="show me the forecast", lang="en"))
    assert out["mode"] == "guide" and "note" in out


def test_gnani_requests_match_the_docs(monkeypatch):
    calls = []

    def fake_post(url, headers=None, data=None, files=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "data": data, "files": files, "json": json})
        if url.endswith("/stt/v3"):
            return FakeResp(200, {"success": True, "transcript": " पाऊस किती झाला "})
        return FakeResp(200, content=b"RIFFxxxxWAVE", headers={"content-type": "audio/wav"})

    monkeypatch.setattr(gnani, "settings", replace(gnani.settings, gnani_api_key="test-key"))
    monkeypatch.setattr(gnani, "_http", lambda: types.SimpleNamespace(post=fake_post))
    p = gnani.GnaniProvider()
    assert p.transcribe(b"RIFF....", "audio/wav", "mr") == "पाऊस किती झाला"
    audio, mime = p.synthesize("नमस्कार", "mr")
    stt, tts = calls
    assert stt["url"] == "https://api.vachana.ai/stt/v3" and stt["headers"]["X-API-Key-ID"] == "test-key"
    assert stt["data"]["language_code"] == "mr-IN" and "audio_file" in stt["files"]
    assert tts["url"] == "https://api.vachana.ai/api/v1/tts/inference" and tts["json"]["language"] == "mr-IN"
    assert tts["json"]["audio_config"]["container"] == "wav" and mime == "audio/wav" and audio.startswith(b"RIFF")


def test_speech_endpoints_with_a_fake_provider(monkeypatch):
    class Fake:
        name = "fake"
        def transcribe(self, audio, mime, lang): return f"heard {len(audio)} bytes in {lang}"
        def synthesize(self, text, lang): return b"RIFF", "audio/wav"

    monkeypatch.setattr(main, "get_provider", lambda: Fake())
    r = client.post("/api/speech/stt", files={"audio": ("s.wav", b"12345", "audio/wav")}, data={"lang": "hi"})
    assert r.status_code == 200 and r.json()["text"] == "heard 5 bytes in hi"
    r = client.post("/api/speech/tts", json={"text": "hello", "lang": "en"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert client.get("/api/meta").json()["features"]["speech"] is True


def test_tts_clip_is_made_once_then_served_from_the_cache(monkeypatch):
    calls = []

    def fake_post(url, headers=None, data=None, files=None, json=None, timeout=None):
        calls.append(json["text"])
        return FakeResp(200, content=b"RIFF" + json["text"].encode(), headers={"content-type": "audio/wav"})

    monkeypatch.setattr(gnani, "settings", replace(gnani.settings, gnani_api_key="k"))
    monkeypatch.setattr(gnani, "_http", lambda: types.SimpleNamespace(post=fake_post))
    p = gnani.GnaniProvider()
    a1, _ = p.synthesize("पाणी वर येण्याची शक्यता", "mr")
    a2, _ = p.synthesize("पाणी वर येण्याची शक्यता", "mr")
    assert a1 == a2 and calls == ["पाणी वर येण्याची शक्यता"]           # second call: no request
    gnani._clips.clear()                                                    # a restart: the disk copy is used
    assert p.synthesize("पाणी वर येण्याची शक्यता", "mr")[0] == a1 and len(calls) == 1
    p.synthesize("पाणी वर येण्याची शक्यता", "hi")                            # another language is another clip
    p.synthesize("पाणी खाली जाण्याची शक्यता", "mr")
    assert len(calls) == 3
    assert gnani.settings.tts_cache_dir.exists() and len(list(gnani.settings.tts_cache_dir.glob("*.wav"))) == 3


def test_tts_retries_when_the_service_says_busy(monkeypatch):
    seen = []

    def fake_post(url, headers=None, data=None, files=None, json=None, timeout=None):
        seen.append(1)
        if len(seen) < 3:
            return FakeResp(429, {"detail": {"error_code": "RATE_LIMITED"}})
        return FakeResp(200, content=b"RIFFok", headers={"content-type": "audio/wav"})

    monkeypatch.setattr(gnani, "settings", replace(gnani.settings, gnani_api_key="k"))
    monkeypatch.setattr(gnani, "_http", lambda: types.SimpleNamespace(post=fake_post))
    monkeypatch.setattr(gnani.time, "sleep", lambda s: None)
    assert gnani.GnaniProvider().synthesize("नमस्कार", "mr")[0] == b"RIFFok" and len(seen) == 3
