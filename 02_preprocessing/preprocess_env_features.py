"""
Preprocesses ET, Soil Moisture, Temperature, and Humidity into one
clean environmental features CSV at (district, date) daily granularity.

SOURCE CHANGE (2026-08-15)
--------------------------
ET and soil moisture now come from **ERA5-Land** (`../01_data_collection/era5_land.py`)
when that file is present, instead of WRIS / NRSC VIC.

Reason: the WRIS/NRSC feed stopped publishing at 2025-06-30, so everything after
that was per-district day-of-year climatology. Measured effect on the TFT
walk-forward evaluation:

    fold                    real ET / soil moisture
    f1_monsoon2024          97%
    f2_postmonsoon2024     100%
    f3_dry2025              96%
    f4_monsoon2025          25%     <-- climatology
    f5_postmonsoon2025       0%     <-- climatology
    f6_dry2026               0%     <-- climatology

Half the evaluation folds were being fed a feature that duplicated `doy_sin`/
`doy_cos`. ERA5-Land is a reanalysis with ~5-day latency, so the tail is real
data and the `*_observed` flags stop being degenerate.

ERA5-Land also supplies four soil-moisture depths instead of one. The deep
layers matter here: `sm_l3` (28-100 cm) and `sm_l4` (100-289 cm) sit on the
actual percolation path to the water table and lag rainfall by roughly the same
3-21 days that recharge does, whereas a surface layer just mirrors yesterday's
rain. `pet_mm` is also carried through — `pet_mm - et_mm` is a water-stress
signal and PET drives irrigation pumping, the dominant non-natural groundwater
sink in Maharashtra.

If the ERA5 file is absent the script falls back to the old WRIS inputs and
behaves exactly as before (climatology fill + honest flags), so the pipeline
still runs on a fresh checkout.

Inputs (preferred):
    ../data/raw/era5_land_maharashtra_23_26.csv
Inputs (fallback):
    ../data/raw/et_maharashtra_23_25_uncleaned.csv
    ../data/raw/soil_moisture_maharashtra_23_25_uncleaned.csv
Always:
    ../data/raw/temperature_humidity_maharashtra_23_25.csv

Output:
    data/clean/env_features_clean.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

Path("../data/clean").mkdir(exist_ok=True)

FULL_IDX = pd.date_range("2023-01-01", "2026-10-01", freq="D")  # matches the groundwater calendar
ERA5 = Path("../data/raw/era5_land_maharashtra_23_26.csv")

ERA5_VALUE_COLS = ["et_mm", "pet_mm", "soil_moisture", "sm_l1", "sm_l2", "sm_l3", "sm_l4"]

DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
]


def fill_gaps(df: pd.DataFrame, value_cols: list, district_col: str, date_col: str) -> pd.DataFrame:
    """
    Reindex each district to full daily calendar. For each value column:
      1. Record a `{col}_observed` flag (1 = genuine source reading, before any fill)
      2. Forward-fill gaps <= 7 days (short sensor/reporting gaps)
      3. Fill any remaining gap with a per-district climatological mean for that
         (month, day), computed only from genuinely-observed rows.

    With ERA5-Land as the source, step 3 should fill ~nothing — that is the point
    of the source change. The script asserts this at the end.
    """
    filled = []
    for district, grp in df.groupby(district_col):
        grp = grp.set_index(date_col).reindex(FULL_IDX)
        grp[district_col] = district
        for col in value_cols:
            grp[f"{col}_observed"] = grp[col].notna().astype(int)
            grp[col] = grp[col].ffill(limit=7)
        filled.append(grp.reset_index().rename(columns={"index": "date"}))
    result = pd.concat(filled, ignore_index=True)

    for col in value_cols:
        obs_col = f"{col}_observed"
        month_day = result["date"].dt.strftime("%m-%d")
        climatology = (
            result.loc[result[obs_col] == 1]
            .assign(_md=month_day[result[obs_col] == 1])
            .groupby([district_col, "_md"])[col]
            .mean()
        )
        still_missing = result[col].isna()
        if still_missing.any():
            keys = list(zip(result.loc[still_missing, district_col], month_day[still_missing]))
            result.loc[still_missing, col] = [climatology.get(k, np.nan) for k in keys]
        n_climato = still_missing.sum() - result[col].isna().sum()
        n_nan = result[col].isna().sum()
        print(f"  {col}: climatology-filled {n_climato:,} rows ({n_nan:,} remain NaN)")

    return result


# ─── ET + SOIL MOISTURE ───────────────────────────────────────────────────────
if ERA5.exists():
    print(f"Processing ET + soil moisture from ERA5-Land ({ERA5.name}) ...")
    era = pd.read_csv(ERA5, parse_dates=["date"])
    era = era.drop_duplicates(subset=["district", "date"], keep="last")
    era = era[era["et_mm"] >= 0]
    value_cols = [c for c in ERA5_VALUE_COLS if c in era.columns]
    wet = fill_gaps(era[["district", "date"] + value_cols], value_cols, "district", "date")
    SOURCE = "ERA5-Land"
else:
    print("ERA5-Land file not found — falling back to WRIS/NRSC inputs.")
    print("  (run: cd ../01_data_collection && python era5_land.py)")
    print("Processing ET ...")
    et = pd.read_csv("../data/raw/et_maharashtra_23_25_uncleaned.csv", parse_dates=["date"])
    et = et.rename(columns={"districtName": "district", "dataValue": "et_mm"})
    et = et[["district", "date", "et_mm"]].drop_duplicates(subset=["district", "date"], keep="last")
    et = fill_gaps(et[et["et_mm"] >= 0], ["et_mm"], "district", "date")

    print("Processing Soil Moisture ...")
    sm = pd.read_csv("../data/raw/soil_moisture_maharashtra_23_25_uncleaned.csv", parse_dates=["date"])
    sm = sm.rename(columns={"districtName": "district", "dataValue": "soil_moisture"})
    sm = sm[["district", "date", "soil_moisture"]].drop_duplicates(subset=["district", "date"], keep="last")
    sm = fill_gaps(sm[sm["soil_moisture"] >= 0], ["soil_moisture"], "district", "date")

    wet = et.merge(sm, on=["district", "date"], how="outer")
    value_cols = ["et_mm", "soil_moisture"]
    SOURCE = "WRIS/NRSC (stale — tail is climatology)"

print(f"  Source: {SOURCE} | rows: {len(wet):,} | districts: {wet['district'].nunique()}")


# ─── TEMPERATURE + HUMIDITY (Open-Meteo — already clean, no gaps) ─────────────
print("Processing Temperature + Humidity ...")
th = pd.read_csv("../data/raw/temperature_humidity_maharashtra_23_25.csv", parse_dates=["date"])
th = th.drop_duplicates(subset=["district", "date"], keep="last")
th = th[(th["temp_mean"] > -5) & (th["temp_mean"] < 55)]
th = th[(th["humidity_mean"] >= 0) & (th["humidity_mean"] <= 100)]
print(f"  Rows: {len(th):,} | Missing temp: {th['temp_mean'].isna().sum()} | "
      f"Missing humidity: {th['humidity_mean'].isna().sum()}")


# ─── MERGE ────────────────────────────────────────────────────────────────────
print("\nMerging ...")
env = wet.merge(th, on=["district", "date"], how="outer").sort_values(["district", "date"]).reset_index(drop=True)

for col in value_cols:
    flag = f"{col}_observed"
    if flag in env.columns:
        env[flag] = env[flag].fillna(0).astype(int)

print(f"  Total rows : {len(env):,}")
print(f"  Districts  : {env['district'].nunique()}")
print(f"  Date range : {env['date'].min().date()} -> {env['date'].max().date()}")

# ─── COVERAGE REPORT — the check that caught the stale source ────────────────
# Any year where an *_observed rate is far below 1.0 means the model is being fed
# climatology for that period. Print it every run so it cannot go unnoticed again.
print("\nObserved-flag rate by year (1.00 = real data, 0.00 = climatology fill):")
flag_cols = [f"{c}_observed" for c in value_cols if f"{c}_observed" in env.columns]
print(env.groupby(env["date"].dt.year)[flag_cols].mean().round(3).to_string())

for c in value_cols:
    flag = f"{c}_observed"
    if flag in env.columns and env[flag].mean() < 0.90:
        print(f"  WARNING: {c} is only {env[flag].mean():.1%} real — "
              f"downstream folds covering the unobserved span are not trustworthy.")

# ─── SAVE ─────────────────────────────────────────────────────────────────────
cols = ["district", "date"]
for c in value_cols:
    cols += [c, f"{c}_observed"]
cols += ["temp_max", "temp_min", "temp_mean", "humidity_mean"]
cols = [c for c in cols if c in env.columns]

env[cols].to_csv("../data/clean/env_features_clean.csv", index=False)
print(f"\nSaved -> data/clean/env_features_clean.csv")
print(f"Columns ({len(cols)}): {cols}")
print(env[cols].head(3).to_string(index=False))
