"""
Preprocesses raw groundwater data:
  - Keeps active stations (both telemetric and manual)
  - Removes duplicates and outliers
  - Aggregates to daily district mean
  - Forward + backward fills gaps <= 7 days
  - Reports missing % per district clearly

Input  : unclean datasets/groundwater_maharashtra_23_25_uncleaned.csv
Output : clean datasets/groundwater_clean.csv
"""

import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

INPUT  = Path("unclean datasets/groundwater_maharashtra_23_25_uncleaned.csv")
OUTPUT = Path("clean datasets/groundwater_clean.csv")
Path("clean datasets").mkdir(exist_ok=True)

START = "2023-01-01"
END   = "2025-04-22"

# ─── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading groundwater data ...")
df = pd.read_csv(INPUT, low_memory=False, parse_dates=["dataTime"])

print(f"  Raw rows       : {len(df):,}")
print(f"  Stations       : {df['stationCode'].nunique()}")
print(f"  Districts      : {df['district'].nunique()}")

# ─── BASIC CLEANING ───────────────────────────────────────────────────────────
# Keep active stations — both telemetric AND manual (dropping mode filter recovers 5 districts)
df = df[df["stationStatus"] == "Active"].copy()

before = len(df)
df = df.drop_duplicates(subset=["stationCode", "dataTime"], keep="last")
print(f"  Duplicates removed     : {before - len(df):,}")

df = df.dropna(subset=["dataValue"])
df = df[df["dataValue"] != 0]

# Sanity range: Maharashtra wells are 0 to -100m (negative = below surface)
df = df[(df["dataValue"] >= -100) & (df["dataValue"] <= 0)]

print(f"  Stations after cleaning: {df['stationCode'].nunique()}")
print(f"  Districts after cleaning: {df['district'].nunique()}")
print(f"  Rows after cleaning    : {len(df):,}")

# ─── DAILY AGGREGATION ────────────────────────────────────────────────────────
df["date"] = df["dataTime"].dt.normalize()

station_daily = (
    df.groupby(["stationCode", "district", "date"])["dataValue"]
    .mean()
    .reset_index()
    .rename(columns={"dataValue": "gwl_station_daily_mean"})
)

district_daily = (
    station_daily.groupby(["district", "date"])
    .agg(
        gwl_mean      = ("gwl_station_daily_mean", "mean"),
        gwl_min       = ("gwl_station_daily_mean", "min"),
        gwl_max       = ("gwl_station_daily_mean", "max"),
        station_count = ("stationCode", "count"),
    )
    .reset_index()
)

district_daily["date"] = pd.to_datetime(district_daily["date"])
district_daily = district_daily.sort_values(["district", "date"]).reset_index(drop=True)

print(f"\nAfter aggregation:")
print(f"  Districts  : {district_daily['district'].nunique()}")
print(f"  Date range : {district_daily['date'].min().date()} -> {district_daily['date'].max().date()}")
print(f"  Rows       : {len(district_daily):,}")

# ─── REINDEX TO FULL CALENDAR + FILL GAPS ─────────────────────────────────────
full_idx   = pd.date_range(START, END, freq="D")
filled_dfs = []

print("\nGap-fill summary per district:")
print(f"  {'District':<20} {'Data starts':<14} {'Missing %':>10} {'Status'}")
print(f"  {'-'*60}")

for district, grp in district_daily.groupby("district"):
    grp = grp.set_index("date").reindex(full_idx)
    grp["district"] = district

    # Forward-fill then backward-fill — both limited to 7 days
    grp["gwl_mean"] = grp["gwl_mean"].ffill(limit=7).bfill(limit=7)
    grp["gwl_min"]  = grp["gwl_min"].ffill(limit=7).bfill(limit=7)
    grp["gwl_max"]  = grp["gwl_max"].ffill(limit=7).bfill(limit=7)
    grp["station_count"] = grp["station_count"].ffill(limit=7).bfill(limit=7)

    missing_pct = grp["gwl_mean"].isna().mean() * 100
    first_valid = grp["gwl_mean"].first_valid_index()
    status = "OK" if missing_pct < 10 else ("GAPS" if missing_pct < 40 else "SPARSE")

    first_str = first_valid.strftime("%Y-%m-%d") if first_valid else "NO DATA"
    print(f"  {district:<20} {first_str:<14} {missing_pct:>9.1f}%  {status}")

    filled_dfs.append(grp.reset_index().rename(columns={"index": "date"}))

district_daily = pd.concat(filled_dfs, ignore_index=True)

overall_missing = district_daily["gwl_mean"].isna().mean() * 100
print(f"\nOverall missing gwl_mean : {overall_missing:.1f}%")
print(f"Total rows               : {len(district_daily):,}")
print(f"Districts                : {district_daily['district'].nunique()}")

# ─── SAVE ─────────────────────────────────────────────────────────────────────
out_cols = ["district", "date", "gwl_mean", "gwl_min", "gwl_max", "station_count"]
district_daily[out_cols].to_csv(OUTPUT, index=False)
print(f"\nSaved -> {OUTPUT}")
