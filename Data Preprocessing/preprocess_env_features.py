"""
Preprocesses ET, Soil Moisture, Temperature, and Humidity into one
clean environmental features CSV at (district, date) daily granularity.

Inputs:
    unclean datasets/et_maharashtra_23_25_uncleaned.csv
    unclean datasets/soil_moisture_maharashtra_23_25_uncleaned.csv
    unclean datasets/temperature_humidity_maharashtra_23_25.csv

Output:
    clean datasets/env_features_clean.csv

Columns: district, date, et_mm, soil_moisture, temp_max, temp_min, temp_mean, humidity_mean
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

Path("clean datasets").mkdir(exist_ok=True)

FULL_IDX   = pd.date_range("2023-01-01", "2025-04-22", freq="D")
DISTRICTS  = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
]

def fill_gaps(df: pd.DataFrame, value_cols: list, district_col: str, date_col: str) -> pd.DataFrame:
    """Reindex each district to full daily calendar, forward-fill gaps <= 7 days."""
    filled = []
    for district, grp in df.groupby(district_col):
        grp = grp.set_index(date_col).reindex(FULL_IDX)
        grp[district_col] = district
        for col in value_cols:
            # Forward-fill gaps up to 7 days, leave longer gaps as NaN
            grp[col] = grp[col].ffill(limit=7)
        filled.append(grp.reset_index().rename(columns={"index": "date"}))
    return pd.concat(filled, ignore_index=True)


# ─── ET ───────────────────────────────────────────────────────────────────────
print("Processing ET ...")
et = pd.read_csv(
    "unclean datasets/et_maharashtra_23_25_uncleaned.csv",
    parse_dates=["date"]
)
et = et.rename(columns={"districtName": "district", "dataValue": "et_mm"})
et = et[["district", "date", "et_mm"]].drop_duplicates(subset=["district", "date"], keep="last")
et = et[et["et_mm"] >= 0]
et = fill_gaps(et, ["et_mm"], "district", "date")
print(f"  Rows: {len(et):,} | Missing: {et['et_mm'].isna().sum()}")


# ─── SOIL MOISTURE ────────────────────────────────────────────────────────────
print("Processing Soil Moisture ...")
sm = pd.read_csv(
    "unclean datasets/soil_moisture_maharashtra_23_25_uncleaned.csv",
    parse_dates=["date"]
)
sm = sm.rename(columns={"districtName": "district", "dataValue": "soil_moisture"})
sm = sm[["district", "date", "soil_moisture"]].drop_duplicates(subset=["district", "date"], keep="last")
sm = sm[sm["soil_moisture"] >= 0]
sm = fill_gaps(sm, ["soil_moisture"], "district", "date")
print(f"  Rows: {len(sm):,} | Missing: {sm['soil_moisture'].isna().sum()}")


# ─── TEMPERATURE + HUMIDITY (Open-Meteo — already clean, no gaps) ─────────────
print("Processing Temperature + Humidity ...")
th = pd.read_csv(
    "unclean datasets/temperature_humidity_maharashtra_23_25.csv",
    parse_dates=["date"]
)
th = th.drop_duplicates(subset=["district", "date"], keep="last")

# Sanity bounds
th = th[(th["temp_mean"] > -5) & (th["temp_mean"] < 55)]
th = th[(th["humidity_mean"] >= 0) & (th["humidity_mean"] <= 100)]
print(f"  Rows: {len(th):,} | Missing temp: {th['temp_mean'].isna().sum()} | Missing humidity: {th['humidity_mean'].isna().sum()}")


# ─── MERGE ALL ────────────────────────────────────────────────────────────────
print("\nMerging ET + Soil Moisture + Temperature + Humidity ...")
env = et.merge(sm, on=["district", "date"], how="outer")
env = env.merge(th, on=["district", "date"], how="outer")
env = env.sort_values(["district", "date"]).reset_index(drop=True)

print(f"  Total rows : {len(env):,}")
print(f"  Districts  : {env['district'].nunique()}")
print(f"  Date range : {env['date'].min().date()} -> {env['date'].max().date()}")
print(f"\nMissing values per column:")
print(env.isnull().sum().to_string())

# ─── SAVE ─────────────────────────────────────────────────────────────────────
cols = ["district", "date", "et_mm", "soil_moisture", "temp_max", "temp_min", "temp_mean", "humidity_mean"]
cols = [c for c in cols if c in env.columns]
env[cols].to_csv("clean datasets/env_features_clean.csv", index=False)
print(f"\nSaved -> clean datasets/env_features_clean.csv")
print(env[cols].head(3).to_string(index=False))
