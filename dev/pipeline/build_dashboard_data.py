"""
Build the dashboard data (dev/backend/data/) from the project's real data and model outputs.

Reads ONLY (never writes) repo files:
  data/clean/groundwater_clean.csv, rainfall_clean.csv, env_features_clean.csv, district_data_quality.csv,
  groundwater_station_daily.csv, groundwater_stations.geojson
  data/raw/cgwb_extraction_maharashtra.csv, mh_gw_2025.csv, maharashtra_district_geoms.geojson
  04_results/tft_v4_3/dashboard_export/   (the operational forecast; if missing, falls back to the v4.2 backtests
                                            and the dashboards say "latest forecast not yet produced")
Writes ONLY into dev/backend/data/. Nothing here is mock data: every value comes from the files above.

Run:  python dev/pipeline/build_dashboard_data.py
Test-only overrides: JALVAANI_MODEL_EXPORT (another dashboard_export folder) and JALVAANI_BUILD_OUT (another output
folder) - used by the tests to check the v4.3 branch without ever writing smoke-test numbers into dev/backend/data.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.getenv("JALVAANI_BUILD_OUT", ROOT / "dev" / "backend" / "data"))
CLEAN, RAW = ROOT / "data" / "clean", ROOT / "data" / "raw"
V43 = Path(os.getenv("JALVAANI_MODEL_EXPORT", ROOT / "04_results" / "tft_v4_3" / "dashboard_export"))
V42 = ROOT / "04_results" / "tft_v4_2" / "simple" / "ensemble"

HORIZONS = ["t+7", "t+14", "t+30"]
USUAL_WINDOW_DAYS = 15          # "compared with usual": same day of year +/- this many days, in other years
CGWB_CATEGORY_ORDER = ["Safe", "Semi-Critical", "Critical", "Over-Exploited"]
MH_GW_NAME_FIX = {"Yawatmal": "Yavatmal", "Mumbai": "Mumbai City", "Mumbai Sub": "Mumbai Suburban"}


# ─── helpers ──────────────────────────────────────────────────────────────────
def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def r(x, nd=3):
    """JSON-safe rounded float (None for NaN)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, nd)


def dstr(t):
    return None if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


SOURCES: dict = {}


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def read_csv(path: Path, **kw):
    df = pd.read_csv(path, **kw)
    SOURCES[relpath(path)] = {"md5": md5(path), "rows": int(len(df))}
    return df


# ─── groundwater ──────────────────────────────────────────────────────────────
def level_window_mean(g, end, days=7):
    w = g[(g.date > end - pd.Timedelta(days=days)) & (g.date <= end)]
    return float(w.gwl_mean.mean()) if len(w) else np.nan


def groundwater_facts(gw: pd.DataFrame, district: str):
    g = gw[(gw.district == district) & (gw.gwl_observed == 1)].sort_values("date")
    if g.empty:
        return None
    last = g.date.max()
    now = level_window_mean(g, last)
    m30 = level_window_mean(g, last - pd.Timedelta(days=30))
    y1 = level_window_mean(g, last - pd.DateOffset(years=1))
    doy = g.date.dt.dayofyear.values
    d = np.abs(doy - last.dayofyear)
    d = np.minimum(d, 365 - d)
    other = g[(d <= USUAL_WINDOW_DAYS) & (g.date.dt.year != last.year)]
    usual = float(other.gwl_mean.mean()) if len(other) else np.nan
    sd = float(other.gwl_mean.std()) if len(other) > 2 else np.nan
    diff = now - usual if np.isfinite(usual) else np.nan
    if np.isfinite(diff):
        band = max(0.5, 0.5 * sd) if np.isfinite(sd) else 0.5
        label = "about" if abs(diff) < band else ("shallower" if diff > 0 else "deeper")
    else:
        label = None
    return {
        "last_reading": dstr(last), "level_m": r(now, 2), "depth_m": r(-now, 2),
        "change_30d_m": r(now - m30, 2), "change_1y_m": r(now - y1, 2),
        "usual_level_m": r(usual, 2), "vs_usual_m": r(diff, 2), "vs_usual": label,
        "usual_years": sorted(int(y) for y in other.date.dt.year.unique()),
    }


def weekly_series(gw, district):
    g = gw[(gw.district == district) & (gw.gwl_observed == 1)].set_index("date").gwl_mean
    if g.empty:
        return []
    w = g.resample("W-SUN").mean().dropna()
    return [[dstr(t), r(v, 2)] for t, v in w.items()]


# ─── weather / soil ───────────────────────────────────────────────────────────
def window_vs_years(df, col, end, start, agg, years_back=(1, 2, 3)):
    """Value of `col` over [start, end] and its average over the same calendar window in earlier years."""
    cur = df[(df.date >= start) & (df.date <= end)]
    val = getattr(cur[col], agg)() if len(cur) else np.nan
    past = []
    for k in years_back:
        s, e = start - pd.DateOffset(years=k), end - pd.DateOffset(years=k)
        w = df[(df.date >= s) & (df.date <= e)]
        if len(w):
            past.append((getattr(w[col], agg)(), int(s.year if s.year == e.year else e.year)))
    avg = float(np.mean([p[0] for p in past])) if past else np.nan
    return val, avg, sorted({p[1] for p in past}), cur


def rain_facts(rain, district):
    g = rain[rain.district == district].sort_values("date")
    if g.empty:
        return None
    end = g.date.max()
    out = {"as_of": dstr(end)}
    prov = g[g.rainfall_observed == 0]
    out["provisional_from"] = dstr(prov.date.min()) if len(prov) else None
    season_start = pd.Timestamp(year=end.year, month=6, day=1) if end.month >= 6 else pd.Timestamp(year=end.year - 1, month=6, day=1)
    for key, start in (("last_7d", end - pd.Timedelta(days=6)), ("last_30d", end - pd.Timedelta(days=29)),
                       ("season", season_start)):
        val, avg, yrs, cur = window_vs_years(g, "rainfall_mm", end, start, "sum")
        out[key] = {"from": dstr(start), "to": dstr(end), "mm": r(val, 1), "usual_mm": r(avg, 1),
                    "pct_of_usual": r(100 * val / avg, 0) if avg and avg > 0 else None,
                    "provisional_days": int((cur.rainfall_observed == 0).sum()), "usual_years": yrs}
    return out


def soil_facts(env, district):
    g = env[(env.district == district) & (env.sm_l1_observed == 1)].sort_values("date")
    if g.empty:
        return None
    end = g.date.max()
    out = {"as_of": dstr(end)}
    for col in ("sm_l1", "sm_l2"):
        val, avg, yrs, _ = window_vs_years(g, col, end, end - pd.Timedelta(days=6), "mean")
        rel = (val - avg) / avg if avg else np.nan
        label = None if not np.isfinite(rel) else ("about" if abs(rel) < 0.10 else ("wetter" if rel > 0 else "drier"))
        out[col] = {"pct": r(val, 1), "usual_pct": r(avg, 1), "vs_usual": label, "usual_years": yrs}
    return out


def monthly_rain(rain, district):
    g = rain[rain.district == district].set_index("date").rainfall_mm
    m = g.resample("MS").sum()
    return [[t.strftime("%Y-%m"), r(v, 1)] for t, v in m.items()]


# ─── model outputs ────────────────────────────────────────────────────────────
def load_model_outputs():
    if (V43 / "forecast_latest.csv").exists():
        man = json.loads((V43 / "export_manifest.json").read_text(encoding="utf-8"))
        fc = read_csv(V43 / "forecast_latest.csv", parse_dates=["issue_date", "target_date"])
        path = read_csv(V43 / "forecast_latest_tft_path.csv", parse_dates=["issue_date", "target_date"])
        bt = read_csv(V43 / "backtest_ensemble.csv", parse_dates=["issue_date", "target_date"])
        met = read_csv(V43 / "metrics_ensemble_per_district.csv")
        met = met[met["test set"] == "all folds"].rename(columns={"Horizon": "horizon"})
        met = met.rename(columns={"rMAE_P": "mase", "rMAE_SN": "vs_sn", "Cov (%)": "coverage",
                                  "Interval score (m)": "interval_score"})
        sk = V43 / "forecast_latest_skipped.csv"
        skipped = read_csv(sk) if sk.exists() and sk.stat().st_size > 2 else pd.DataFrame(
            columns=["district", "observed_share_last_window"])
        return {"source": "v4.3 operational export", "run": man.get("run_tag"), "pipeline": man.get("pipeline_version"),
                "manifest": man, "forecast": fc[fc.model == "Ensemble"], "path": path, "backtest": bt, "metrics": met,
                "skipped": {d: {"code": "too_few_readings", "share_pct": int(round(100 * float(sh)))}
                            for d, sh in zip(skipped.district, skipped.observed_share_last_window)}}
    pred = read_csv(V42 / "ensemble_predictions_all_folds.csv", parse_dates=["date", "issue_date"])
    bt = pred.rename(columns={"date": "target_date"})
    met = read_csv(V42 / "ensemble_metrics_per_district.csv").rename(
        columns={"Horizon": "horizon", "MASE (vs no-change)": "mase", "vs seasonal-naive": "vs_sn",
                 "Coverage (%)": "coverage"})
    return {"source": "v4.2 backtests (no operational forecast yet: run the v4.3 notebook)", "run": "FINAL_v4_2",
            "pipeline": "4.6", "manifest": None, "forecast": None, "path": None, "backtest": bt, "metrics": met,
            "skipped": {}}


def reliability(mase, coverage, tier, single_well):
    """Good: MASE <= 0.85, coverage 70-90%, data tier A/B, more than one well. Fair: MASE < 1 and more than one
    well. Limited: otherwise (incl. single-well districts). Thresholds are documented in dev/docs/DATA_CONTRACT.md."""
    if mase is None or not np.isfinite(mase):
        return None
    if single_well or mase >= 1.0:
        return "limited"
    if mase <= 0.85 and coverage is not None and 70 <= coverage <= 90 and tier in ("A", "B"):
        return "good"
    return "fair"


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    started = datetime.now().isoformat(timespec="seconds")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "history").mkdir(parents=True)

    gw = read_csv(CLEAN / "groundwater_clean.csv", parse_dates=["date"])
    rain = read_csv(CLEAN / "rainfall_clean.csv", parse_dates=["date"])
    env = read_csv(CLEAN / "env_features_clean.csv", parse_dates=["date"])
    dq = read_csv(CLEAN / "district_data_quality.csv")
    st_daily = read_csv(CLEAN / "groundwater_station_daily.csv", parse_dates=["date"])
    cg = read_csv(RAW / "cgwb_extraction_maharashtra.csv")
    mh = read_csv(RAW / "mh_gw_2025.csv")
    geo_path = RAW / "maharashtra_district_geoms.geojson"
    st_geo_path = CLEAN / "groundwater_stations.geojson"
    for pth in (geo_path, st_geo_path):
        SOURCES[relpath(pth)] = {"md5": md5(pth)}
    model = load_model_outputs()
    excluded = set(pd.read_csv(CLEAN / "excluded_districts.csv").district) if (CLEAN / "excluded_districts.csv").exists() else set()

    # district boundaries (simplified for the web)
    import geopandas as gpd
    geo = gpd.read_file(geo_path)
    geo["geometry"] = geo.geometry.simplify(0.004, preserve_topology=True)
    districts = sorted(geo["district"].tolist())
    gj = json.loads(geo.to_json(to_wgs84=False))
    for f in gj["features"]:
        f.pop("id", None)
        f["properties"] = {"district": f["properties"]["district"]}

        def rnd(c):
            return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 4), round(c[1], 4)]
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
    write_json(OUT / "geo_districts.json", gj)

    # stations actually used (telemetric, in the cleaned station-level file) with coordinates
    st_geo = json.loads(st_geo_path.read_text(encoding="utf-8"))
    coords = {f["properties"]["stationCode"]: f["geometry"]["coordinates"] for f in st_geo["features"]}
    st = st_daily.groupby(["stationCode", "district"]).agg(first=("date", "min"), last=("date", "max"),
                                                           n_days=("date", "nunique")).reset_index()
    stations = [{"code": s.stationCode, "district": s.district, "lon": r(coords[s.stationCode][0], 4),
                 "lat": r(coords[s.stationCode][1], 4), "first": dstr(s.first), "last": dstr(s.last), "n_days": int(s.n_days)}
                for s in st.itertuples() if s.stationCode in coords]
    write_json(OUT / "stations.json", {"stations": stations, "n_used": int(st.stationCode.nunique()),
                                       "n_with_coordinates": len(stations)})

    # CGWB extraction (2023-2025, all 36) + the 2025 use split (ha·m)
    mh = mh.rename(columns={"Name of District": "district"})
    mh["district"] = mh["district"].astype(str).str.strip().replace(MH_GW_NAME_FIX)
    use_cols = {"irrigation": "Irrigation - Annual extraction", "industrial": "Industrial - Annual extraction",
                "domestic": "Domestic - Annual Extraction"}
    unmatched_mh = sorted(set(mh.district) - set(districts) - {"nan", "Total"})

    q = dq.set_index("district")
    bt = model["backtest"]
    met = model["metrics"]
    fc = model["forecast"]
    out_districts, last_dates = [], []
    for d in districts:
        rec = {"district": d, "slug": slug(d)}
        in_model = d in set(bt.district)
        has_gw = d in set(gw.district)
        if in_model:
            rec["status"] = "forecast"
        elif has_gw:
            rec["status"] = "insufficient_data"
            rec["status_reason"] = ("excluded: groundwater missing on most days" if d in excluded
                                    else "single well, too few readings (data tier D)")
        else:
            rec["status"] = "no_data"
        if d in q.index:
            rec["monitoring"] = {"wells": int(q.loc[d, "n_stations"]), "completeness_pct": r(100 * q.loc[d, "coverage_all"], 0),
                                 "tier": q.loc[d, "data_tier"], "single_well": bool(q.loc[d, "n_stations"] <= 1)}
        elif d in excluded:
            rec["monitoring"] = {"wells": int(st_daily[st_daily.district == d].stationCode.nunique()), "completeness_pct": None,
                                 "tier": None, "single_well": True}
        else:
            rec["monitoring"] = {"wells": 0, "completeness_pct": None, "tier": None, "single_well": False}

        gwf = groundwater_facts(gw, d) if in_model else None
        rec["groundwater"] = gwf
        if gwf:
            last_dates.append(gwf["last_reading"])

        # forecast (operational) + reliability (backtests)
        rel = {}
        if in_model:
            for h in HORIZONS:
                m = met[(met.district == d) & (met.horizon == h)]
                if len(m):
                    mase, cov = float(m["mase"].iloc[0]), float(m["coverage"].iloc[0])
                    rel[h] = {"mase": r(mase), "vs_sn": r(m["vs_sn"].iloc[0]), "coverage_pct": r(cov, 1),
                              "label": reliability(mase, cov, rec["monitoring"]["tier"], rec["monitoring"]["single_well"])}
        rec["reliability"] = rel or None
        if fc is not None and in_model and d in set(fc.district):
            f = fc[fc.district == d]
            rec["forecast"] = {
                "issue_date": dstr(f.issue_date.iloc[0]), "level_now_m": r(f.level_now.iloc[0], 2),
                "horizons": {row.horizon: {"target_date": dstr(row.target_date), "q10": r(row.q10, 2), "q50": r(row.q50, 2),
                                           "q90": r(row.q90, 2), "change_m": r(row.q50 - row.level_now, 2),
                                           "change_q10_m": r(row.q10 - row.level_now, 2),
                                           "change_q90_m": r(row.q90 - row.level_now, 2), "interval": row.interval}
                             for row in f.itertuples()}}
        else:
            rec["forecast"] = None
            if in_model:
                rec["forecast_unavailable"] = (model["skipped"].get(d, {"code": "no_window"}) if fc is not None
                                               else {"code": "not_produced"})

        rec["rain"] = rain_facts(rain, d)
        rec["soil"] = soil_facts(env, d)
        c = cg[cg.district == d].sort_values("year")
        rec["extraction"] = None if c.empty else {
            "latest_year": int(c.year.iloc[-1]), "stage_pct": r(c.stage_of_extraction_pct.iloc[-1], 1),
            "category": c.category.iloc[-1], "recharge_bcm": r(c.annual_recharge_bcm.iloc[-1]),
            "extraction_bcm": r(c.annual_extraction_bcm.iloc[-1]), "extractable_bcm": r(c.extractable_resource_bcm.iloc[-1]),
            "history": [{"year": int(x.year), "stage_pct": r(x.stage_of_extraction_pct, 1), "category": x.category} for x in c.itertuples()]}
        mrow = mh[mh.district == d]
        if rec["extraction"] and len(mrow):
            vals = {k: float(pd.to_numeric(mrow[v], errors="coerce").iloc[0]) for k, v in use_cols.items()}
            tot = sum(v for v in vals.values() if np.isfinite(v))
            rec["extraction"]["use_split_pct"] = {k: r(100 * v / tot, 1) for k, v in vals.items()} if tot > 0 else None
        out_districts.append(rec)

        # per-district history file (charts)
        hist = {"district": d, "observed_weekly": weekly_series(gw, d) if in_model else [],
                "rain_monthly": monthly_rain(rain, d), "backtest": {}, "latest_path": None}
        if in_model:
            b = bt[bt.district == d]
            for h in HORIZONS:
                bh = b[b.horizon == h].sort_values("target_date")
                hist["backtest"][h] = {"cols": ["issue_date", "target_date", "actual", "q10", "q50", "q90"],
                                       "rows": [[dstr(x.issue_date), dstr(x.target_date), r(x.actual, 2), r(x.q10, 2),
                                                 r(x.q50, 2), r(x.q90, 2)] for x in bh.itertuples()]}
            if model["path"] is not None and d in set(model["path"].district):
                pth = model["path"][model["path"].district == d].sort_values("day")
                hist["latest_path"] = {"cols": ["target_date", "q10", "q50", "q90"],
                                       "rows": [[dstr(x.target_date), r(x.q10, 2), r(x.q50, 2), r(x.q90, 2)] for x in pth.itertuples()]}
        write_json(OUT / "history" / f"{slug(d)}.json", hist)

    write_json(OUT / "districts.json", {"districts": out_districts})
    manifest = {
        "built": started, "builder": "dev/pipeline/build_dashboard_data.py",
        "model": {"source": model["source"], "run": model["run"], "pipeline": model["pipeline"],
                  "operational_forecast": fc is not None,
                  "last_issue_date": dstr(fc.issue_date.max()) if fc is not None else None,
                  "backtest_period": [dstr(bt.target_date.min()), dstr(bt.target_date.max())]},
        "freshness": {
            "groundwater_last_reading": max(last_dates) if last_dates else None,
            "rain_as_of": dstr(rain.date.max()),
            "rain_provisional_from": dstr(rain[rain.rainfall_observed == 0].date.min()) if (rain.rainfall_observed == 0).any() else None,
            "soil_as_of": dstr(env[env.sm_l1_observed == 1].date.max()),
            "extraction_year": int(cg.year.max()),
        },
        "counts": {"districts": len(districts), "with_forecast": sum(x["status"] == "forecast" for x in out_districts),
                   "insufficient_data": sum(x["status"] == "insufficient_data" for x in out_districts),
                   "no_data": sum(x["status"] == "no_data" for x in out_districts), "wells_used": int(st.stationCode.nunique())},
        "sources": SOURCES, "warnings": ([f"mh_gw_2025 names not matched: {unmatched_mh}"] if unmatched_mh else []),
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("model", "freshness", "counts", "warnings")}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
