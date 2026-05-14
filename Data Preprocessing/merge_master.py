"""
Merges all clean datasets into one master training CSV.
Run this AFTER all preprocess_*.py scripts have completed.

Inputs  (from clean datasets/):
    groundwater_clean.csv
    rainfall_clean.csv
    env_features_clean.csv

Static input:
    ../Data Collection/datasets/soil_texture_maharashtra.csv

Sentinel inputs (monthly, interpolated to daily):
    ../Data Collection/datasets/sentinel2_indices_maharashtra_23_25.csv
    ../Data Collection/datasets/sentinel1_backscatter_maharashtra_23_25.csv

Output:
    clean datasets/master_dataset.csv

Schema:
    district, district_id, date, day_of_year,
    gwl_mean, gwl_min, gwl_max, station_count,
    rainfall_mm, rainfall_7d, rainfall_30d,
    et_mm, soil_moisture, net_water_balance,
    temp_max, temp_min, temp_mean, humidity_mean, vpd_approx,
    month, season, is_monsoon,
    gwl_lag_7, gwl_lag_14, gwl_lag_30,
    gwl_7d_change, gwl_30d_change,
    sand_0cm, clay_0cm, silt_0cm,
    sand_10cm, clay_10cm, silt_10cm,
    sand_30cm, clay_30cm, silt_30cm,
    ndvi_mean, ndwi_mean, ndmi_mean,
    vv_db, vh_db, vh_vv_ratio
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CLEAN        = Path("clean datasets")
SOIL_TEXTURE = Path("../Data Collection/datasets/soil_texture_maharashtra.csv")
SENTINEL2    = Path("../Data Collection/datasets/sentinel2_indices_maharashtra_23_25.csv")
SENTINEL1    = Path("../Data Collection/datasets/sentinel1_backscatter_maharashtra_23_25.csv")

# ─── LOAD CLEAN FILES ─────────────────────────────────────────────────────────
print("Loading clean datasets ...")
gw   = pd.read_csv(CLEAN / "groundwater_clean.csv",  parse_dates=["date"])
rf   = pd.read_csv(CLEAN / "rainfall_clean.csv",     parse_dates=["date"])
ev   = pd.read_csv(CLEAN / "env_features_clean.csv", parse_dates=["date"])
soil = pd.read_csv(SOIL_TEXTURE)
s2   = pd.read_csv(SENTINEL2)
s1   = pd.read_csv(SENTINEL1)

print(f"  Groundwater  : {len(gw):,} rows, {gw['district'].nunique()} districts")
print(f"  Rainfall     : {len(rf):,} rows, {rf['district'].nunique()} districts")
print(f"  Env features : {len(ev):,} rows, {ev['district'].nunique()} districts")
print(f"  Soil texture : {len(soil):,} rows (static, one per district)")
print(f"  Sentinel-2   : {len(s2):,} rows, {s2['district'].nunique()} districts")
print(f"  Sentinel-1   : {len(s1):,} rows, {s1['district'].nunique()} districts")

# ─── MERGE ────────────────────────────────────────────────────────────────────
print("\nMerging ...")
master = gw.merge(rf, on=["district", "date"], how="inner", suffixes=("", "_rf"))
master = master.merge(ev, on=["district", "date"], how="left")

print(f"  After merge  : {len(master):,} rows, {master['district'].nunique()} districts")

# ─── TIME FEATURES ────────────────────────────────────────────────────────────
master["month"]       = master["date"].dt.month
master["day_of_year"] = master["date"].dt.dayofyear
master["is_monsoon"]  = master["month"].isin([6, 7, 8, 9]).astype(int)
master["season"]      = master["month"].map({
    12: "winter", 1: "winter", 2: "winter",
    3:  "summer", 4: "summer", 5: "summer",
    6:  "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
})

# ─── LAG FEATURES ─────────────────────────────────────────────────────────────
master = master.sort_values(["district", "date"])
master["gwl_lag_7"]  = master.groupby("district")["gwl_mean"].shift(7)
master["gwl_lag_14"] = master.groupby("district")["gwl_mean"].shift(14)
master["gwl_lag_30"] = master.groupby("district")["gwl_mean"].shift(30)

# ─── TREND FEATURES ───────────────────────────────────────────────────────────
master["gwl_7d_change"]  = master["gwl_mean"] - master["gwl_lag_7"]
master["gwl_30d_change"] = master["gwl_mean"] - master["gwl_lag_30"]

# ─── HYDRO BALANCE FEATURES ───────────────────────────────────────────────────
master["net_water_balance"] = master["rainfall_mm"] - master["et_mm"]
# VPD proxy: higher temp × lower humidity → atmosphere is "thirstier" → more ET
master["vpd_approx"] = master["temp_mean"] * (1.0 - master["humidity_mean"] / 100.0)

# ─── DISTRICT ID ──────────────────────────────────────────────────────────────
# Integer 0..N-1 sorted alphabetically — used as embedding index in LSTM/TFT
sorted_districts = sorted(master["district"].unique())
district_id_map  = {d: i for i, d in enumerate(sorted_districts)}
master["district_id"] = master["district"].map(district_id_map)
print(f"  district_id  : {len(sorted_districts)} unique districts mapped to 0..{len(sorted_districts)-1}")

# ─── STATIC SOIL TEXTURE JOIN ─────────────────────────────────────────────────
soil_cols = [
    "sand_0cm", "clay_0cm", "silt_0cm",
    "sand_10cm", "clay_10cm", "silt_10cm",
    "sand_30cm", "clay_30cm", "silt_30cm",
]
# Keep only the district + soil columns that exist in the file
available_soil_cols = ["district"] + [c for c in soil_cols if c in soil.columns]
soil_clean = soil[available_soil_cols].drop_duplicates(subset=["district"])

master = master.merge(soil_clean, on="district", how="left")
soil_missing = master[[c for c in soil_cols if c in master.columns]].isnull().any(axis=1).sum()
print(f"  Soil texture : {soil_missing:,} rows with missing soil data (districts not in soil file)")

# ─── SENTINEL MERGE (monthly → daily via linear interpolation) ───────────────
def expand_sentinel_to_daily(sent_df, value_cols, master_dates):
    """
    Sentinel data is monthly (year, month). Expand to daily by:
      1. Convert (year, month) to the 1st of each month as a date
      2. Reindex to every day in master_dates per district
      3. Linear interpolate interior gaps, then ffill/bfill edges
    """
    sent_df = sent_df.copy()
    sent_df["date"] = pd.to_datetime(
        sent_df["year"].astype(str) + "-" + sent_df["month"].astype(str).str.zfill(2) + "-01"
    )
    sent_df = sent_df[["district", "date"] + value_cols]

    daily_parts = []
    for dist, grp in sent_df.groupby("district"):
        grp = grp.set_index("date").sort_index()
        # Reindex to full daily range
        full_idx = pd.date_range(master_dates.min(), master_dates.max(), freq="D")
        grp = grp.reindex(full_idx)
        # Linear interpolation for interior NaNs, ffill/bfill for edges
        for col in value_cols:
            grp[col] = grp[col].interpolate("linear").ffill().bfill()
        grp["district"] = dist
        grp.index.name = "date"
        daily_parts.append(grp.reset_index())

    return pd.concat(daily_parts, ignore_index=True)

s2_cols = ["ndvi_mean", "ndwi_mean", "ndmi_mean"]
s1_cols = ["vv_db", "vh_db", "vh_vv_ratio"]

s2_daily = expand_sentinel_to_daily(s2, s2_cols, master["date"])
s1_daily = expand_sentinel_to_daily(s1, s1_cols, master["date"])

master = master.merge(s2_daily[["district", "date"] + s2_cols], on=["district", "date"], how="left")
master = master.merge(s1_daily[["district", "date"] + s1_cols], on=["district", "date"], how="left")

s2_missing = master["ndvi_mean"].isnull().sum()
s1_missing = master["vv_db"].isnull().sum()
print(f"  Sentinel-2   : {s2_missing:,} rows with no ndvi (districts not in Sentinel coverage)")
print(f"  Sentinel-1   : {s1_missing:,} rows with no vv_db (districts not in Sentinel coverage)")

# ─── DROP ROWS WITH NO TARGET ─────────────────────────────────────────────────
before = len(master)
master = master.dropna(subset=["gwl_mean"])
print(f"  Dropped {before - len(master):,} rows with no groundwater reading")

# ─── COLUMN ORDER ─────────────────────────────────────────────────────────────
ordered_cols = [
    "district", "district_id", "date", "day_of_year",
    "gwl_mean", "gwl_min", "gwl_max", "station_count",
    "rainfall_mm", "rainfall_7d", "rainfall_30d",
    "et_mm", "soil_moisture", "net_water_balance",
    "temp_max", "temp_min", "temp_mean", "humidity_mean", "vpd_approx",
    "month", "season", "is_monsoon",
    "gwl_lag_7", "gwl_lag_14", "gwl_lag_30",
    "gwl_7d_change", "gwl_30d_change",
    "sand_0cm", "clay_0cm", "silt_0cm",
    "sand_10cm", "clay_10cm", "silt_10cm",
    "sand_30cm", "clay_30cm", "silt_30cm",
    "ndvi_mean", "ndwi_mean", "ndmi_mean",
    "vv_db", "vh_db", "vh_vv_ratio",
]
# Only keep columns that actually ended up in master (soil cols may vary)
final_cols = [c for c in ordered_cols if c in master.columns]
master = master[final_cols]

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\nFinal master dataset:")
print(f"  Rows       : {len(master):,}")
print(f"  Districts  : {master['district'].nunique()}")
print(f"  Date range : {master['date'].min().date()} -> {master['date'].max().date()}")
print(f"  Columns ({len(final_cols)}) : {final_cols}")
print(f"\nMissing values:")
missing = master.isnull().sum()
print(missing[missing > 0].to_string())

# ─── SAVE ─────────────────────────────────────────────────────────────────────
master.to_csv(CLEAN / "master_dataset.csv", index=False)
print(f"\nSaved -> clean datasets/master_dataset.csv")
print(master.head(3).to_string(index=False))
