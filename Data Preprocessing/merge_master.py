"""
Merges all clean datasets into one master training CSV.
Run this AFTER all preprocess_*.py scripts have completed.

Inputs  (from clean datasets/):
    groundwater_clean.csv
    rainfall_clean.csv
    env_features_clean.csv

Static input:
    ../Data Collection/datasets/soil_texture_maharashtra.csv

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
    sand_30cm, clay_30cm, silt_30cm
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CLEAN        = Path("clean datasets")
SOIL_TEXTURE = Path("../Data Collection/datasets/soil_texture_maharashtra.csv")

# ─── LOAD CLEAN FILES ─────────────────────────────────────────────────────────
print("Loading clean datasets ...")
gw   = pd.read_csv(CLEAN / "groundwater_clean.csv",  parse_dates=["date"])
rf   = pd.read_csv(CLEAN / "rainfall_clean.csv",     parse_dates=["date"])
ev   = pd.read_csv(CLEAN / "env_features_clean.csv", parse_dates=["date"])
soil = pd.read_csv(SOIL_TEXTURE)

print(f"  Groundwater  : {len(gw):,} rows, {gw['district'].nunique()} districts")
print(f"  Rainfall     : {len(rf):,} rows, {rf['district'].nunique()} districts")
print(f"  Env features : {len(ev):,} rows, {ev['district'].nunique()} districts")
print(f"  Soil texture : {len(soil):,} rows (static, one per district)")

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
