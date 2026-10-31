"""Groq key rotation and the Gnani adapter, with fakes (no network)."""
from dataclasses import replace

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
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def post(self, url, json, headers):
            used.append(headers["Authorization"])
            if headers["Authorization"].endswith("k1"):
                return FakeResp(429, {"error": "rate limited"})
            return FakeResp(200, {"choices": [{"message": {"content":
                '{"reply": "Here is the forecast card.", "actions": [{"type": "highlight", "element": "card-forecast"}]}'}}]})

    monkeypatch.setattr(chat, "settings", replace(chat.settings, groq_api_keys=("k1", "k2", "k3")))
    monkeypatch.setattr(chat.httpx, "Client", FakeClient)
    monkeypatch.setattr(chat, "_next_key", iter([0]))
    out = chat.chat(ChatRequest(message="show forecast", lang="en"))
    assert used == ["Bearer k1", "Bearer k2"]
    assert out["mode"] == "groq" and out["actions"] == [{"type": "highlight", "element": "card-forecast"}]


def test_groq_failure_falls_back_to_guide(monkeypatch):
    class Boom:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **k): raise httpx.ConnectError("offline")

    monkeypatch.setattr(chat, "settings", replace(chat.settings, groq_api_keys=("k1",)))
    monkeypatch.setattr(chat.httpx, "Client", Boom)
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
    monkeypatch.setattr(gnani.httpx, "post", fake_post)
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
