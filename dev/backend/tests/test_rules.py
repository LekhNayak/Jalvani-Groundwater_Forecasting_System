"""Advice and priority rules on SYNTHETIC records (test-only; never loaded by the app)."""
from app.services.advisories import advisories, forecast_direction
from app.services.priorities import district_flags, priorities


def synthetic_district(**over):
    rec = {"district": "Synthetic", "slug": "synthetic", "status": "forecast",
           "monitoring": {"wells": 5, "tier": "A", "single_well": False},
           "groundwater": {"vs_usual": "about", "vs_usual_m": 0.1, "depth_m": 5.0},
           "forecast": {"horizons": {"t+30": {"change_m": -0.8, "change_q10_m": -1.4, "change_q90_m": -0.2}}},
           "rain": {"season": {"pct_of_usual": 100}},
           "extraction": {"stage_pct": 50.0, "category": "Safe"},
           "reliability": {"t+30": {"label": "good"}}}
    rec.update(over)
    return rec


def test_forecast_direction():
    assert forecast_direction({"change_m": -0.5, "change_q10_m": -1, "change_q90_m": -0.1}) == "falling"
    assert forecast_direction({"change_m": 0.5, "change_q10_m": 0.1, "change_q90_m": 1}) == "rising"
    assert forecast_direction({"change_m": 0.05, "change_q10_m": -0.3, "change_q90_m": 0.4}) == "steady"
    assert forecast_direction({"change_m": -0.3, "change_q10_m": -0.9, "change_q90_m": 0.2}) == "uncertain"
    assert forecast_direction(None) is None


def test_falling_advice_and_contact():
    keys = [t["key"] for t in advisories(synthetic_district())]
    assert keys[0] == "advice.falling" and keys[-1] == "advice.contact_local"


def test_no_forecast_and_deficit():
    rec = synthetic_district(status="no_data", forecast=None, groundwater=None, rain={"season": {"pct_of_usual": 60}})
    keys = [t["key"] for t in advisories(rec)]
    assert "advice.no_groundwater_data" in keys and "advice.rain_deficit" in keys


def test_flags_and_ranking():
    a = synthetic_district(extraction={"stage_pct": 95.0, "category": "Critical"})
    b = synthetic_district(slug="b", district="B", forecast=None)
    c = synthetic_district(slug="c", district="C", status="no_data", forecast=None, groundwater=None)
    fa = {f["key"] for f in district_flags(a)}
    assert {"extraction_critical", "forecast_decline"} <= fa
    assert {f["key"] for f in district_flags(c)} == {"no_monitoring"}
    order = [r["district"] for r in priorities([b, c, a])]
    assert order[0] == "Synthetic"
