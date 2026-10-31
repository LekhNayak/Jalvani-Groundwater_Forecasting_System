"""Farmer tips: transparent rules over the district's real numbers.

Each tip is an i18n key + the values it quotes; the frontend renders the text in the chosen language.
The tips are general water-use guidance, never crop or legal prescriptions (dev/docs/DATA_CONTRACT.md)."""

FALL_M = 0.3            # forecast t+30 median fall (m) that counts as "falling"
RAIN_LOW, RAIN_HIGH = 75, 125   # season rain, % of the 2023-25 average


def forecast_direction(h):
    """'falling' | 'rising' | 'uncertain' | 'steady' for one forecast horizon, from the median and the 80% range."""
    if h is None:
        return None
    if h["change_q90_m"] is not None and h["change_q90_m"] < 0:
        return "falling"
    if h["change_q10_m"] is not None and h["change_q10_m"] > 0:
        return "rising"
    if h["change_m"] is not None and abs(h["change_m"]) < 0.1:
        return "steady"
    return "uncertain"


def advisories(rec, limit=3):
    tips = []
    if rec["status"] != "forecast":
        tips.append({"key": "advice.no_groundwater_data", "level": "info", "params": {}})
    fc = (rec.get("forecast") or {}).get("horizons", {})
    h30 = fc.get("t+30")
    direction = forecast_direction(h30)
    if direction == "falling" and h30["change_m"] <= -FALL_M:
        tips.append({"key": "advice.falling", "level": "warn", "params": {"fall_m": round(-h30["change_m"], 1)}})
    elif direction == "rising":
        tips.append({"key": "advice.rising", "level": "good", "params": {"rise_m": round(h30["change_m"], 1)}})
    elif direction == "uncertain":
        tips.append({"key": "advice.uncertain", "level": "info", "params": {}})
    gw = rec.get("groundwater") or {}
    if gw.get("vs_usual") == "deeper":
        tips.append({"key": "advice.deeper_than_usual", "level": "warn", "params": {"diff_m": round(-gw["vs_usual_m"], 1)}})
    season = ((rec.get("rain") or {}).get("season") or {})
    pct = season.get("pct_of_usual")
    if pct is not None and pct < RAIN_LOW:
        tips.append({"key": "advice.rain_deficit", "level": "warn", "params": {"pct": int(pct)}})
    elif pct is not None and pct > RAIN_HIGH:
        tips.append({"key": "advice.rain_surplus", "level": "good", "params": {"pct": int(pct)}})
    ext = rec.get("extraction") or {}
    if ext.get("category") and ext["category"] != "Safe":
        tips.append({"key": "advice.extraction_high", "level": "warn", "params": {"stage_pct": ext["stage_pct"]}})
    rel = ((rec.get("reliability") or {}).get("t+30") or {}).get("label")
    if rel == "limited":
        tips.append({"key": "advice.reliability_limited", "level": "info", "params": {}})
    order = {"warn": 0, "good": 1, "info": 2}
    tips = sorted(tips, key=lambda t: order[t["level"]])[:limit]
    if any(t["level"] == "warn" for t in tips):
        tips.append({"key": "advice.contact_local", "level": "info", "params": {}})
    return tips
