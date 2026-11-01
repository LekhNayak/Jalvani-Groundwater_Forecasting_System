"""JalVaani dashboard API (local).

Run from dev/backend:   uvicorn app.main:app --reload --port 8000
Data comes from dev/backend/data (built by dev/pipeline/build_dashboard_data.py); this app never writes data.
"""
import csv
import io

import httpx

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .services import chat as chat_service
from .services.advisories import advisories, forecast_direction
from .services.priorities import district_flags, overview, priorities
from .services.speech.gnani import GnaniError, get_provider
from .services.store import DataMissing, get_store

app = FastAPI(title="JalVaani dashboard API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.exception_handler(DataMissing)
def _missing(_, exc):
    return JSONResponse(status_code=503, content={"error": "data_not_built", "detail": str(exc)})


def _district_or_404(slug):
    rec = get_store().district(slug)
    if rec is None:
        raise HTTPException(404, f"unknown district '{slug}'")
    return rec


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta():
    m = get_store().manifest
    return {"built": m["built"], "model": m["model"], "freshness": m["freshness"], "counts": m["counts"],
            "languages": list(settings.languages),
            "features": {"chat": "groq" if settings.groq_api_keys else "guide", "speech": get_provider() is not None,
                         "speech_languages": list(settings.speech_languages) if get_provider() is not None else []}}


@app.get("/api/districts")
def districts():
    """Light list for the picker and the map layers."""
    out = []
    for d in get_store().districts:
        gw, fc, ext = d.get("groundwater") or {}, d.get("forecast") or {}, d.get("extraction") or {}
        h30 = (fc.get("horizons") or {}).get("t+30") or {}
        out.append({
            "district": d["district"], "slug": d["slug"], "status": d["status"],
            "tier": (d.get("monitoring") or {}).get("tier"), "wells": (d.get("monitoring") or {}).get("wells"),
            "layers": {"vs_usual_m": gw.get("vs_usual_m"), "forecast_change_t30_m": h30.get("change_m"),
                       "stage_pct": ext.get("stage_pct"), "category": ext.get("category"),
                       "wells": (d.get("monitoring") or {}).get("wells"),
                       "reliability_t30": ((d.get("reliability") or {}).get("t+30") or {}).get("label")}})
    return {"districts": out}


@app.get("/api/districts/{slug}")
def district(slug: str):
    rec = _district_or_404(slug)
    fc = (rec.get("forecast") or {}).get("horizons") or {}
    return {**rec, "advice": advisories(rec), "flags": district_flags(rec),
            "direction": {h: forecast_direction(v) for h, v in fc.items()}}


@app.get("/api/districts/{slug}/history")
def district_history(slug: str):
    _district_or_404(slug)
    h = get_store().history(slug)
    if h is None:
        raise HTTPException(404, "no history")
    return h


@app.get("/api/state/overview")
def state_overview():
    s = get_store()
    return overview(s.districts, s.manifest)


@app.get("/api/state/priorities")
def state_priorities():
    return {"rows": priorities(get_store().districts)}


@app.get("/api/export/priorities.csv")
def export_priorities():
    rows = priorities(get_store().districts)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["district", "status", "score", "actions", "reasons", "stage_of_extraction_pct", "cgwb_category",
                "depth_m_below_ground", "vs_usual_m", "forecast_change_t30_m", "wells", "reliability_t30",
                "season_rain_pct_of_2023_25_avg"])
    for r in rows:
        w.writerow([r["district"], r["status"], r["score"], ";".join(r["actions"]), ";".join(f["key"] for f in r["flags"]),
                    r["stage_pct"], r["category"], r["depth_m"], r["vs_usual_m"], r["forecast_change_t30_m"], r["wells"],
                    r["reliability_t30"], r["season_rain_pct"]])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=jalvaani_priorities.csv"})


@app.get("/api/geo/districts")
def geo_districts():
    return get_store().geo


@app.get("/api/geo/stations")
def geo_stations():
    return get_store().stations


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    lang: str = "en"
    role: str = "farmer"
    page: str | None = None
    district: str | None = None
    visible_context: dict | None = None
    history: list[dict] | None = None


@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.lang not in settings.languages:
        req.lang = "en"
    return chat_service.chat(req)


@app.post("/api/speech/stt")
def speech_stt(audio: UploadFile = File(...), lang: str = Form("en")):
    # a plain `def`: FastAPI runs it in a worker thread, so waiting for Gnani does not block the other requests
    provider = get_provider()
    if provider is None:
        raise HTTPException(501, "speech_not_configured")
    data = audio.file.read()
    if not data or len(data) > 5_000_000:
        raise HTTPException(400, "empty or too large recording (max 30 s)")
    try:
        text = provider.transcribe(data, audio.content_type or "audio/wav", lang if lang in settings.languages else "en")
    except (GnaniError, httpx.HTTPError) as exc:
        raise HTTPException(502, f"speech_to_text_failed: {exc}")
    return {"text": text, "provider": provider.name}


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1500)
    lang: str = "en"


@app.post("/api/speech/tts")
def speech_tts(req: TTSRequest):
    provider = get_provider()
    if provider is None:
        raise HTTPException(501, "speech_not_configured")
    try:
        audio, mime = provider.synthesize(req.text, req.lang if req.lang in settings.languages else "en")
    except (GnaniError, httpx.HTTPError) as exc:
        raise HTTPException(502, f"text_to_speech_failed: {exc}")
    return Response(content=audio, media_type=mime)


# ── built frontend (SERVE_FRONTEND=1): one URL for everything ────────────────
if settings.serve_frontend and settings.frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=settings.frontend_dist / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = settings.frontend_dist / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(settings.frontend_dist / "index.html")
