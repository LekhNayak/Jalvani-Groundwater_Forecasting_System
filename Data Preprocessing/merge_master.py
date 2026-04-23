"""
Merges all clean datasets into one master training CSV.
Run this AFTER all preprocess_*.py scripts have completed.

Inputs  (from clean datasets/):
    groundwater_clean.csv
    rainfall_clean.csv
    env_features_clean.csv

Output:
    clean datasets/master_dataset.csv

Schema:
    district, date,
    gwl_mean, gwl_min, gwl_max,          ← target + context
    rainfall_mm, rainfall_7d, rainfall_30d,
    et_mm, soil_moisture,
    temp_max, temp_min, temp_mean, humidity_mean,
    month, season,                         ← engineered time features
    gwl_lag_7, gwl_lag_30                  ← engineered lag features
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CLEAN = Path("clean datasets")

# ─── LOAD CLEAN FILES ─────────────────────────────────────────────────────────
print("Loading clean datasets ...")
gw = pd.read_csv(CLEAN / "groundwater_clean.csv",  parse_dates=["date"])
rf = pd.read_csv(CLEAN / "rainfall_clean.csv",     parse_dates=["date"])
ev = pd.read_csv(CLEAN / "env_features_clean.csv", parse_dates=["date"])

print(f"  Groundwater  : {len(gw):,} rows, {gw['district'].nunique()} districts")
print(f"  Rainfall     : {len(rf):,} rows, {rf['district'].nunique()} districts")
print(f"  Env features : {len(ev):,} rows, {ev['district'].nunique()} districts")

# ─── MERGE ────────────────────────────────────────────────────────────────────
print("\nMerging ...")
master = gw.merge(rf, on=["district", "date"], how="inner")
master = master.merge(ev, on=["district", "date"], how="left")

print(f"  After merge  : {len(master):,} rows, {master['district'].nunique()} districts")

# ─── TIME FEATURES ────────────────────────────────────────────────────────────
master["month"]  = master["date"].dt.month
master["season"] = master["month"].map({
    12: "winter", 1: "winter", 2: "winter",
    3:  "summer", 4: "summer", 5: "summer",
    6:  "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
})

# ─── LAG FEATURES ─────────────────────────────────────────────────────────────
# Groundwater has strong temporal autocorrelation — lags are critical for ARIMA/LSTM
master = master.sort_values(["district", "date"])
master["gwl_lag_7"]  = master.groupby("district")["gwl_mean"].shift(7)
master["gwl_lag_30"] = master.groupby("district")["gwl_mean"].shift(30)

# ─── DROP ROWS WITH NO TARGET ─────────────────────────────────────────────────
before = len(master)
master = master.dropna(subset=["gwl_mean"])
print(f"  Dropped {before - len(master):,} rows with no groundwater reading")

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\nFinal master dataset:")
print(f"  Rows       : {len(master):,}")
print(f"  Districts  : {master['district'].nunique()}")
print(f"  Date range : {master['date'].min().date()} -> {master['date'].max().date()}")
print(f"  Columns    : {list(master.columns)}")
print(f"\nMissing values:")
missing = master.isnull().sum()
print(missing[missing > 0].to_string())

# ─── SAVE ─────────────────────────────────────────────────────────────────────
master.to_csv(CLEAN / "master_dataset.csv", index=False)
print(f"\nSaved -> clean datasets/master_dataset.csv")
print(master.head(3).to_string(index=False))
