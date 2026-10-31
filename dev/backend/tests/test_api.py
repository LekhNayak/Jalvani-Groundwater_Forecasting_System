"""API tests on the real built data (dev/backend/data). Skipped if the data has not been built."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(not (settings.data_dir / "manifest.json").exists(),
                                reason="run dev/pipeline/build_dashboard_data.py first")
client = TestClient(app)


def test_meta():
    m = client.get("/api/meta").json()
    assert m["counts"]["districts"] == 36 and set(m["languages"]) == {"en", "mr", "hi"}
    assert m["features"]["chat"] in ("guide", "groq") and m["freshness"]["groundwater_last_reading"]


def test_districts_list_and_statuses():
    ds = client.get("/api/districts").json()["districts"]
    assert len(ds) == 36
    st = {d["district"]: d["status"] for d in ds}
    assert st["Pune"] == "forecast" and st["Satara"] == "no_data" and st["Latur"] == "insufficient_data"


def test_district_values_match_sources():
    d = client.get("/api/districts/pune").json()
    gw = pd.read_csv(ROOT / "data/clean/groundwater_clean.csv", parse_dates=["date"])
    p = gw[(gw.district == "Pune") & (gw.gwl_observed == 1)].sort_values("date")
    last = p.date.max()
    assert d["groundwater"]["last_reading"] == last.strftime("%Y-%m-%d")
    assert d["groundwater"]["level_m"] == round(p[p.date > last - pd.Timedelta(days=7)].gwl_mean.mean(), 2)
    cg = pd.read_csv(ROOT / "data/raw/cgwb_extraction_maharashtra.csv")
    row = cg[cg.district == "Pune"].sort_values("year").iloc[-1]
    assert d["extraction"]["stage_pct"] == round(row.stage_of_extraction_pct, 1)
    assert d["extraction"]["category"] == row.category
    assert isinstance(d["advice"], list) and "flags" in d


def test_history_and_unknown():
    h = client.get("/api/districts/nagpur/history").json()
    assert h["observed_weekly"] and set(h["backtest"]) == {"t+7", "t+14", "t+30"}
    assert client.get("/api/districts/atlantis").status_code == 404


def test_state_and_export():
    o = client.get("/api/state/overview").json()
    assert sum(o["extraction_categories"].values()) == 36
    rows = client.get("/api/state/priorities").json()["rows"]
    assert len(rows) == 36 and rows[0]["score"] >= rows[-1]["score"]
    r = client.get("/api/export/priorities.csv")
    assert r.status_code == 200 and r.text.startswith("district,status,score")


def test_geo():
    g = client.get("/api/geo/districts").json()
    assert len(g["features"]) == 36
    s = client.get("/api/geo/stations").json()
    assert s["n_with_coordinates"] > 100


def test_chat_guide_mode_three_languages():
    for msg, lang, el in (("show me the forecast", "en", "card-forecast"), ("पाऊस किती झाला", "mr", "card-rain-soil"),
                          ("जिला बदलो", "hi", "district-picker")):
        r = client.post("/api/chat", json={"message": msg, "lang": lang, "role": "farmer"}).json()
        assert r["reply"] and any(a.get("element") == el for a in r["actions"]), (msg, r)
    r = client.post("/api/chat", json={"message": "take a tour", "lang": "en", "role": "government"}).json()
    assert {"type": "start_tour", "tour": "government"} in r["actions"]


def test_chat_action_allowlist():
    from app.services.chat import clean_actions
    acts = clean_actions([{"type": "navigate", "route": "https://evil.example"}, {"type": "highlight", "element": "x"},
                          {"type": "navigate", "route": "/farmer/pune"}, {"type": "run_js"}], "farmer")
    assert acts == [{"type": "navigate", "route": "/farmer/pune"}]


def test_speech_not_configured():
    assert client.post("/api/speech/tts", json={"text": "hello", "lang": "en"}).status_code == 501


def test_latest_forecast_matches_the_model_export():
    m = client.get("/api/meta").json()
    if not m["model"]["operational_forecast"]:
        pytest.skip("no v4.3 export yet")
    f = pd.read_csv(ROOT / "04_results/tft_v4_3/dashboard_export/forecast_latest.csv")
    row = f[(f.district == "Nagpur") & (f.model == "Ensemble") & (f.horizon == "t+30")].iloc[0]
    d = client.get("/api/districts/nagpur").json()
    h = d["forecast"]["horizons"]["t+30"]
    assert d["forecast"]["issue_date"] == row.issue_date[:10] and h["target_date"] == row.target_date[:10]
    assert (h["q10"], h["q50"], h["q90"]) == (round(row.q10, 2), round(row.q50, 2), round(row.q90, 2))
    assert h["change_m"] == round(row.q50 - row.level_now, 2) and h["interval"] == row.interval
    nanded = client.get("/api/districts/nanded").json()
    assert nanded["forecast"] is None and nanded["forecast_unavailable"]["code"] == "too_few_readings"


def test_most_likely_range_is_median_past_error():
    """'Most likely' = q50 -/+ the median |actual - q50| of all back-tests for that district and horizon,
    clipped to the 80% range."""
    m = client.get("/api/meta").json()
    if not m["model"]["operational_forecast"]:
        pytest.skip("no v4.3 export yet")
    bt = pd.read_csv(ROOT / "04_results/tft_v4_3/dashboard_export/backtest_ensemble.csv")
    for slug, district in (("akola", "Akola"), ("pune", "Pune"), ("gondia", "Gondia")):
        d = client.get(f"/api/districts/{slug}").json()
        for hz, h in d["forecast"]["horizons"].items():
            b = bt[(bt.district == district) & (bt.horizon == hz)]
            margin = (b.actual - b.q50).abs().median()
            assert h["likely_margin_m"] == round(margin, 2)
            assert abs(h["likely_lo"] - max(h["q50"] - margin, h["q10"])) <= 0.011
            assert abs(h["likely_hi"] - min(h["q50"] + margin, h["q90"])) <= 0.011
            assert h["q10"] <= h["likely_lo"] <= h["q50"] <= h["likely_hi"] <= h["q90"] + 1e-9
    check = json.loads((settings.data_dir / "manifest.json").read_text(encoding="utf-8"))["most_likely"]["check_coverage_pct"]
    assert all(40 <= v <= 75 for v in check.values())
