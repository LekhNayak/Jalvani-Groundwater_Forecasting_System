"""
Preprocesses raw rainfall data:
  - Removes duplicates
  - Aggregates sub-daily readings to daily district total
  - Fills missing days with 0 (no rain is valid)
  - Computes 7-day and 30-day rolling cumulative rainfall (features for LSTM)

Input  : ../../data/raw/rainfall_maharashtra_23_25_uncleaned.csv
Output : data/clean/rainfall_clean.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

INPUT  = Path("../../data/raw/rainfall_maharashtra_23_25_uncleaned.csv")
OUTPUT = Path("../../data/clean/rainfall_clean.csv")
Path("../../data/clean").mkdir(exist_ok=True)

# ─── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading rainfall data ...")
df = pd.read_csv(INPUT, low_memory=False, parse_dates=["dataTime"])

print(f"  Raw rows  : {len(df):,}")
print(f"  Stations  : {df['stationCode'].nunique()}")
print(f"  Districts : {df['district'].nunique()}")

# ─── BASIC CLEANING ───────────────────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates(subset=["stationCode", "dataTime"], keep="last")
print(f"  Duplicates removed : {before - len(df):,}")

df = df.dropna(subset=["dataValue"])

# Rainfall can't be negative
df = df[df["dataValue"] >= 0]

# Cap extreme outliers (>500mm/day is unrealistic for a single reading)
df = df[df["dataValue"] <= 500]

print(f"  After cleaning : {len(df):,} rows")

# ─── DAILY AGGREGATION ────────────────────────────────────────────────────────
df["date"] = df["dataTime"].dt.date

# Sum all sub-daily readings per station per day
station_daily = (
    df.groupby(["stationCode", "district", "date"])["dataValue"]
    .sum()
    .reset_index()
    .rename(columns={"dataValue": "rainfall_mm"})
)

# District daily: mean across stations (avoids bias from station density)
district_daily = (
    station_daily.groupby(["district", "date"])
    .agg(
        rainfall_mm   = ("rainfall_mm", "mean"),
        station_count = ("stationCode", "count"),
    )
    .reset_index()
)

district_daily["date"] = pd.to_datetime(district_daily["date"])
district_daily = district_daily.sort_values(["district", "date"])

# ─── FILL MISSING DAYS WITH 0 ─────────────────────────────────────────────────
# Missing rainfall days = no rain (stations only report when active)
full_idx   = pd.date_range("2023-01-01", "2025-04-22", freq="D")
filled_dfs = []

for district, grp in district_daily.groupby("district"):
    grp = grp.set_index("date").reindex(full_idx)
    grp["district"]     = district
    grp["rainfall_mm"]  = grp["rainfall_mm"].fillna(0)
    grp["station_count"]= grp["station_count"].fillna(0)
    filled_dfs.append(grp.reset_index().rename(columns={"index": "date"}))

district_daily = pd.concat(filled_dfs, ignore_index=True)

# ─── ROLLING FEATURES ─────────────────────────────────────────────────────────
district_daily = district_daily.sort_values(["district", "date"])
district_daily["rainfall_7d"]  = (
    district_daily.groupby("district")["rainfall_mm"]
    .transform(lambda x: x.rolling(7,  min_periods=1).sum())
)
district_daily["rainfall_30d"] = (
    district_daily.groupby("district")["rainfall_mm"]
    .transform(lambda x: x.rolling(30, min_periods=1).sum())
)

print(f"\nAfter aggregation:")
print(f"  Districts  : {district_daily['district'].nunique()}")
print(f"  Date range : {district_daily['date'].min().date()} -> {district_daily['date'].max().date()}")
print(f"  Total rows : {len(district_daily):,}")

# ─── SAVE ─────────────────────────────────────────────────────────────────────
cols = ["district", "date", "rainfall_mm", "rainfall_7d", "rainfall_30d", "station_count"]
district_daily[cols].to_csv(OUTPUT, index=False)
print(f"\nSaved -> {OUTPUT}")
print(district_daily[cols].head(3).to_string(index=False))
