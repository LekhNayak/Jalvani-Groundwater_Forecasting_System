"""
Audit of the groundwater TARGET: old (v3) district series vs the rebuilt one.

Model-free checks, so the choice of cleaning/aggregation is not tuned on forecast
results:
  1. jumps   : |change| > 2 m between consecutive observed days, and the share that
               coincide with a change in the number of reporting stations
  2. LOSO    : leave-one-station-out — rebuild the district from the other stations and
               correlate the held-out station's 30-day change with the district's.
               Higher = the district series better represents its individual wells.
  3. coverage: observed-day rate per district (no district may lose > 10%)
  4. persistence MAE at h = 7 / 30 on observed pairs (for reference — a cleaner target
               makes persistence stronger, which makes the forecasting task harder)
Writes audit/target_audit_*.csv and before/after plots for the worst districts.

Run from `02_preprocessing/`:  python audit_groundwater_target.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import preprocess_groundwater as pg

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "audit"
OUT.mkdir(exist_ok=True)
OLD = HERE.parent / "archive" / "backups" / "2026-10_v3_target" / "groundwater_clean__v3_target.csv"
METHODS = ["median_anom", "offset", "factor", "factor_med"]
PLOT = ["Akola", "Nagpur", "Dhule", "Aurangabad", "Nashik", "Jalgaon"]


def jump_stats(gw):
    rows = []
    for d, g in gw.groupby("district"):
        g = g.sort_values("date")
        v = g["gwl_mean"].where(g["gwl_observed"] == 1)
        sc = g["station_count"]
        dv = v.diff()
        big = dv.abs() > 2.0
        set_change = sc.diff().fillna(0) != 0
        rows.append({"district": d, "jumps_gt2": int(big.sum()), "jumps_gt5": int((dv.abs() > 5).sum()),
                     "jumps_gt2_on_set_change": int((big & set_change).sum()),
                     "observed": round(float(g["gwl_observed"].mean()), 3),
                     "std": round(float(v.std()), 2)})
    return pd.DataFrame(rows).set_index("district")


def persistence_mae(gw, h):
    out = {}
    for d, g in gw.groupby("district"):
        v = g.sort_values("date")["gwl_mean"].where(g.sort_values("date")["gwl_observed"] == 1).values
        e = np.abs(v[h:] - v[:-h])
        out[d] = float(np.nanmean(e)) if np.isfinite(e).any() else np.nan
    return pd.Series(out)


def loso(segs, dist_of, cal, method):
    """median corr of held-out station's 30-day change vs district (others only)."""
    rows = []
    by_dist = {}
    for c, k in segs:
        by_dist.setdefault(dist_of[c], set()).add(c)
    for dist, stations in by_dist.items():
        if len(stations) < 3:
            continue
        X = pd.DataFrame({f"{c}#{k}": segs[(c, k)].reindex(cal) for c, k in segs if dist_of[c] == dist})
        for s in stations:
            own = [col for col in X.columns if col.startswith(s + "#")]
            D, _, _ = pg.aggregate(X.drop(columns=own), method)
            d30 = D - D.shift(30)
            # held-out station's 30-day change, never across a segment boundary
            parts = [X[col] - X[col].shift(30) for col in own]
            s30 = pd.concat(parts, axis=1).mean(axis=1) if parts else None
            both = d30.notna() & s30.notna()
            if both.sum() >= 60:
                rows.append({"district": dist, "station": s, "method": method,
                             "r30": float(np.corrcoef(d30[both], s30[both])[0, 1]), "n": int(both.sum())})
    return pd.DataFrame(rows)


def main():
    raw, segs, dist_of, log = pg.build_segments(verbose=True)
    cal = pd.date_range(pg.START, pg.END, freq="D")
    old = pd.read_csv(OLD, parse_dates=["date"])
    results = {"v3 (old target)": old}
    for m in METHODS:
        gw, _ = pg.build_district(segs, dist_of, cal, method=m, verbose=False)
        results[m] = gw

    # 1 + 3: jumps, coverage, std
    summ, per = [], {}
    for name, gw in results.items():
        js = jump_stats(gw)
        per[name] = js
        p7, p30 = persistence_mae(gw, 7), persistence_mae(gw, 30)
        summ.append({"target": name, "jumps>2m": int(js["jumps_gt2"].sum()), "jumps>5m": int(js["jumps_gt5"].sum()),
                     "% of >2m on set change": round(100 * js["jumps_gt2_on_set_change"].sum()
                                                       / max(js["jumps_gt2"].sum(), 1), 1),
                     "observed": round(float(gw["gwl_observed"].mean()), 3),
                     "persist MAE h7 (median)": round(float(p7.median()), 3),
                     "persist MAE h30 (median)": round(float(p30.median()), 3)})
    summ = pd.DataFrame(summ).set_index("target")

    # 2: LOSO
    lo = pd.concat([loso(segs, dist_of, cal, m) for m in METHODS], ignore_index=True)
    lo_s = lo.groupby("method")["r30"].agg(["median", "mean", "count"]).round(3)
    summ["LOSO r30 (median)"] = summ.index.map(lambda t: lo_s["median"].get(t, np.nan))

    pd.set_option("display.width", 220)
    print("\n" + "=" * 110)
    print("TARGET AUDIT")
    print("=" * 110)
    print(summ.to_string())
    print("\nLOSO by method (districts with >= 3 stations):")
    print(lo_s.to_string())

    cmp = pd.concat({k: v[["jumps_gt2", "observed", "std"]] for k, v in per.items()}, axis=1)
    print("\nPer district (jumps>2m / observed / std):")
    print(cmp.to_string())
    summ.to_csv(OUT / "target_audit_summary.csv")
    cmp.to_csv(OUT / "target_audit_per_district.csv")
    lo.to_csv(OUT / "target_audit_loso.csv", index=False)

    # plots
    for d in PLOT:
        fig, ax = plt.subplots(figsize=(14, 4))
        for name, gw in results.items():
            g = gw[gw.district == d].sort_values("date")
            ax.plot(g["date"], g["gwl_mean"].where(g["gwl_observed"] == 1), lw=1,
                    label=name, alpha=0.9 if name != "v3 (old target)" else 0.6,
                    color={"v3 (old target)": "grey"}.get(name))
        ax.set_title(f"{d} — district GWL (observed days only)")
        ax.set_ylabel("m"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(OUT / f"target_{d.lower()}.png", dpi=110); plt.close()
    print(f"\nWrote audit/*.csv and audit/target_<district>.png")


if __name__ == "__main__":
    main()
