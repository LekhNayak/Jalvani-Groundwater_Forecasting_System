"""Government view: per-district flags with reasons, a priority score, and the state overview.

Every flag is a transparent rule over real values; the weights are listed in FLAG_WEIGHTS and shown in the UI.
"""
from collections import Counter

from .advisories import RAIN_LOW, forecast_direction

FLAG_WEIGHTS = {
    # key: (weight, action)
    "no_monitoring": (3, "improve_data"),
    "insufficient_data": (2, "improve_data"),
    "single_well": (1, "improve_data"),
    "extraction_critical": (3, "manage_demand"),
    "extraction_semi_critical": (2, "manage_demand"),
    "forecast_decline": (2, "monitor"),
    "deeper_than_usual": (1, "monitor"),
    "rain_deficit": (1, "monitor"),
    "reliability_limited": (1, "improve_data"),
}
DECLINE_M = 0.5     # forecast t+30 median fall (m), with the whole 80% range below today's level


def district_flags(rec):
    flags = []
    st = rec["status"]
    if st == "no_data":
        flags.append(("no_monitoring", {}))
    elif st == "insufficient_data":
        flags.append(("insufficient_data", {"reason": rec.get("status_reason")}))
    mon = rec.get("monitoring") or {}
    if st == "forecast" and mon.get("single_well"):
        flags.append(("single_well", {}))
    ext = rec.get("extraction") or {}
    stage = ext.get("stage_pct")
    if stage is not None and stage >= 90:
        flags.append(("extraction_critical", {"stage_pct": stage, "category": ext.get("category")}))
    elif stage is not None and stage >= 70:
        flags.append(("extraction_semi_critical", {"stage_pct": stage, "category": ext.get("category")}))
    h30 = ((rec.get("forecast") or {}).get("horizons") or {}).get("t+30")
    if h30 and forecast_direction(h30) == "falling" and h30["change_m"] <= -DECLINE_M:
        flags.append(("forecast_decline", {"fall_m": round(-h30["change_m"], 2)}))
    gw = rec.get("groundwater") or {}
    if gw.get("vs_usual") == "deeper":
        flags.append(("deeper_than_usual", {"diff_m": round(-gw["vs_usual_m"], 2)}))
    pct = ((rec.get("rain") or {}).get("season") or {}).get("pct_of_usual")
    if pct is not None and pct < RAIN_LOW:
        flags.append(("rain_deficit", {"pct": int(pct)}))
    rel = ((rec.get("reliability") or {}).get("t+30") or {}).get("label")
    if rel == "limited":
        flags.append(("reliability_limited", {}))
    return [{"key": k, "weight": FLAG_WEIGHTS[k][0], "action": FLAG_WEIGHTS[k][1], "params": p} for k, p in flags]


def priorities(districts):
    rows = []
    for rec in districts:
        flags = district_flags(rec)
        ext = rec.get("extraction") or {}
        h30 = ((rec.get("forecast") or {}).get("horizons") or {}).get("t+30") or {}
        rows.append({
            "district": rec["district"], "slug": rec["slug"], "status": rec["status"],
            "score": sum(f["weight"] for f in flags), "actions": sorted({f["action"] for f in flags}), "flags": flags,
            "stage_pct": ext.get("stage_pct"), "category": ext.get("category"),
            "depth_m": (rec.get("groundwater") or {}).get("depth_m"),
            "vs_usual_m": (rec.get("groundwater") or {}).get("vs_usual_m"),
            "forecast_change_t30_m": h30.get("change_m"),
            "wells": (rec.get("monitoring") or {}).get("wells"),
            "reliability_t30": ((rec.get("reliability") or {}).get("t+30") or {}).get("label"),
            "season_rain_pct": ((rec.get("rain") or {}).get("season") or {}).get("pct_of_usual"),
        })
    return sorted(rows, key=lambda x: (-x["score"], x["district"]))


def overview(districts, manifest):
    cats = Counter((d.get("extraction") or {}).get("category") for d in districts if d.get("extraction"))
    pr = priorities(districts)
    flag_counts = Counter(f["key"] for row in pr for f in row["flags"])
    return {
        "counts": manifest["counts"],
        "freshness": manifest["freshness"],
        "model": manifest["model"],
        "extraction_categories": {k: cats.get(k, 0) for k in ("Safe", "Semi-Critical", "Critical", "Over-Exploited")},
        "flag_counts": dict(flag_counts),
        "weights": {k: {"weight": w, "action": a} for k, (w, a) in FLAG_WEIGHTS.items()},
    }
