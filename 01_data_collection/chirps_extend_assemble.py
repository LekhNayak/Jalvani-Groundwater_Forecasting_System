"""Assemble the extended rainfall series to 2026-10-01.

  <= 2026-08-31   CHIRPS v2.0 final (GEE)                         rainfall_observed = 1
  09-01 .. 09-25  CHIRPS v2.0 preliminary (CHC), per-district       rainfall_observed = 0
                  ratio-calibrated to final over 2026-07-01..08-31
  09-26 .. 10-01  Open-Meteo precipitation (no CHIRPS exists yet),  rainfall_observed = 0
                  per-district ratio-calibrated the same way
Re-run once CHIRPS final publishes (use chirps incremental + this script) to replace the provisional days."""
import pandas as pd, numpy as np
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

OUT = "../data/raw/rainfall_chirps_maharashtra_23_25.csv"
START, END = "2023-01-01", "2026-10-01"
CAL = ("2026-07-01", "2026-08-31")

old = pd.read_csv("../archive/backups/2026-10_pre_extend/rainfall_chirps_maharashtra_23_25__pre_2026_10_extend.csv", parse_dates=["date"])
gee = pd.read_csv("../data/raw/chirps_gee_increment_2026.csv", parse_dates=["date"])
pre = pd.read_csv("../data/raw/chirps_prelim_2026.csv", parse_dates=["date"])
om  = pd.read_csv("../data/raw/openmeteo_precip_2026.csv", parse_dates=["date"])
last_final = gee["date"].max()
print("last CHIRPS final day:", last_final.date())

# refresh the tail: GEE rows (from 06-15) replace the old file's rows for the same days
base = old[["district", "date", "rainfall_mm"]]
base = base[base["date"] < gee["date"].min()]
final = pd.concat([base, gee]).drop_duplicates(["district", "date"], keep="last")
chg = old.merge(gee, on=["district", "date"], suffixes=("_old", "_new"))
print("refresh check, re-fetched vs old file, days <= 05-31 (should match): max abs diff = %.4f over %d rows" % ((chg[chg.date<="2026-05-31"].rainfall_mm_old - chg[chg.date<="2026-05-31"].rainfall_mm_new).abs().max(), (chg.date<="2026-05-31").sum()))

def ratios(src, col):
    src = src.rename(columns={col: "src"})[["district", "date", "src"]]
    j = final.merge(src, on=["district", "date"], suffixes=("", "_s"))
    j = j[(j["date"] >= CAL[0]) & (j["date"] <= CAL[1])]
    s = j.groupby("district").agg(f=("rainfall_mm", "sum"), s=("src", "sum"))
    r = (s.f / s.s).clip(0.5, 4.0)
    print(f"  {col}: calibration ratio final/source  median %.2f  min %.2f  max %.2f" % (r.median(), r.min(), r.max()))
    return r

r_pre, r_om = ratios(pre, "rainfall_mm"), ratios(om, "precip_om")

p = pre[pre["date"] > last_final].copy()
p["rainfall_mm"] = (p["rainfall_mm"] * p["district"].map(r_pre)).round(3)
p_last = p["date"].max()
o = om[om["date"] > p_last].copy()
o["rainfall_mm"] = (o["precip_om"] * o["district"].map(r_om)).round(3)
print("prelim days:", p["date"].min().date(), "->", p_last.date(), "| open-meteo days:", o["date"].min().date(), "->", o["date"].max().date())

allrows = pd.concat([final.assign(rainfall_observed=1),
                     p[["district", "date", "rainfall_mm"]].assign(rainfall_observed=0),
                     o[["district", "date", "rainfall_mm"]].assign(rainfall_observed=0)])
idx = pd.date_range(START, END, freq="D")
parts = []
for d, g in allrows.groupby("district"):
    g = g.drop_duplicates("date", keep="last").set_index("date").reindex(idx)
    g["district"] = d
    g["rainfall_mm"] = g["rainfall_mm"].fillna(0.0)               # CHIRPS' occasional dropped days (as before)
    g["rainfall_observed"] = g["rainfall_observed"].fillna(1).astype(int)
    parts.append(g.rename_axis("date").reset_index())
df = pd.concat(parts).sort_values(["district", "date"]).reset_index(drop=True)
df["rainfall_7d"]  = df.groupby("district")["rainfall_mm"].transform(lambda s: s.rolling(7,  min_periods=1).sum())
df["rainfall_30d"] = df.groupby("district")["rainfall_mm"].transform(lambda s: s.rolling(30, min_periods=1).sum())
df["station_count"] = 0
df = df[["district", "date", "rainfall_mm", "rainfall_7d", "rainfall_30d", "station_count", "rainfall_observed"]]
df.to_csv(OUT, index=False)
print(f"Saved {OUT}: {len(df):,} rows, {df.district.nunique()} districts, {df.date.min().date()} -> {df.date.max().date()}")
print("provisional (rainfall_observed=0) days per district:", df.loc[df.rainfall_observed == 0].groupby("district").size().unique())
print(df[df.date >= "2026-08-27"].groupby("date").rainfall_mm.mean().round(2).to_string())
