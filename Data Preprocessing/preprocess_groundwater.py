"""
Preprocesses raw groundwater data:
  - Normalises sign convention (WRIS mixes +ve/-ve depth conventions across stations)
  - Drops mixed-sign stations (data-quality issue)
  - Drops sensor-fault outliers (|value| > 100 m)
  - Applies per-station Hampel filter to remove spikes (pump cycling, sensor glitches)
  - Caps daily district-level changes at 5 m/day (physically realistic)
  - Aggregates to daily district median (robust to outlier stations)
  - Forward + backward fills gaps <= 7 days
  - Passes through static per-district metadata: aquifer_type, well_depth_m, n_stations
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

# Outlier thresholds
MAX_ABS_DEPTH      = 100.0   # |value| above this is sensor fault (raw had max +172m)
HAMPEL_WINDOW      = 7       # days, per-station rolling window for spike detection
HAMPEL_K           = 3.0     # MAD multiplier; values beyond this are spikes
MAX_DAILY_CHANGE_M = 5.0     # max physically plausible 1-day change at district scale

# ─── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading groundwater data ...")
df = pd.read_csv(INPUT, low_memory=False, parse_dates=["dataTime"])

print(f"  Raw rows       : {len(df):,}")
print(f"  Stations       : {df['stationCode'].nunique()}")
print(f"  Districts      : {df['district'].nunique()}")
print(f"  Value range    : [{df['dataValue'].min():.2f}, {df['dataValue'].max():.2f}]")

# ─── BASIC CLEANING ───────────────────────────────────────────────────────────
df = df[df["stationStatus"] == "Active"].copy()
df = df.dropna(subset=["dataValue", "district"])
df = df.drop_duplicates(subset=["stationCode", "dataTime"], keep="last")
df = df[df["dataValue"] != 0]

# Drop sensor faults — any reading > MAX_ABS_DEPTH metres in absolute value
before = len(df)
df = df[df["dataValue"].abs() <= MAX_ABS_DEPTH]
print(f"  Sensor-fault rows dropped (|v| > {MAX_ABS_DEPTH}m): {before - len(df):,}")

# ─── SIGN-CONVENTION NORMALISATION ────────────────────────────────────────────
# WRIS mixes conventions: some stations report depth as negative ("-5m below surface"),
# others as positive ("5m below surface"). Determine per-station, then normalise to
# negative convention (depth below surface = negative metres).
station_sign = df.groupby("stationCode")["dataValue"].agg(["min", "max"]).reset_index()
station_sign["all_negative"] = station_sign["max"] <= 0
station_sign["all_positive"] = station_sign["min"] >= 0
station_sign["mixed"]        = ~(station_sign["all_negative"] | station_sign["all_positive"])

n_neg   = station_sign["all_negative"].sum()
n_pos   = station_sign["all_positive"].sum()
n_mixed = station_sign["mixed"].sum()
print(f"\nSign convention split:")
print(f"  Negative-only : {n_neg:>5} stations (already correct sign)")
print(f"  Positive-only : {n_pos:>5} stations (sign-flipped to negative)")
print(f"  Mixed-sign    : {n_mixed:>5} stations (dropped — data quality issue)")

# Drop mixed-sign stations
mixed_codes = set(station_sign.loc[station_sign["mixed"], "stationCode"])
df = df[~df["stationCode"].isin(mixed_codes)]

# Normalise: every reading becomes -|value| → uniformly negative
df["dataValue"] = -df["dataValue"].abs()

print(f"  Rows after sign normalisation: {len(df):,}")
print(f"  Stations after sign normalisation: {df['stationCode'].nunique()}")
print(f"  Districts after sign normalisation: {df['district'].nunique()}")
print(f"  New value range: [{df['dataValue'].min():.2f}, {df['dataValue'].max():.2f}]")

# ─── HAMPEL FILTER PER STATION (removes pump-cycling spikes & glitches) ──────
# For each station, rolling 7-day median; flag any reading whose deviation from the
# median exceeds K × MAD as a spike and drop it.
print("\nApplying Hampel spike filter per station ...")
df = df.sort_values(["stationCode", "dataTime"]).reset_index(drop=True)

def hampel_mask(s, window=HAMPEL_WINDOW, k=HAMPEL_K):
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    # 1.4826 makes MAD a consistent estimator of std for normal data
    threshold = k * 1.4826 * mad
    # keep where deviation within threshold OR mad is zero/NaN (stable)
    return ((s - med).abs() <= threshold) | mad.isna() | (mad == 0)

keep_mask = df.groupby("stationCode")["dataValue"].transform(hampel_mask).astype(bool)
spikes_dropped = (~keep_mask).sum()
df = df[keep_mask]
print(f"  Spike readings dropped: {spikes_dropped:,}")
print(f"  Rows after Hampel    : {len(df):,}")

# ─── STATIC PER-DISTRICT METADATA ────────────────────────────────────────────
# Derive once from the full (post-sign-normalisation) DataFrame before aggregation.
# wellAquiferType: most common value per district (mode)
# well_depth_m   : mean wellDepth across all readings for that district
# n_stations     : count of unique station codes contributing data

district_meta = (
    df.groupby("district")
    .agg(
        aquifer_type = ("wellAquiferType", lambda s: s.mode().iloc[0] if s.notna().any() else "Unknown"),
        well_depth_m = ("wellDepth",       "mean"),
        n_stations   = ("stationCode",     "nunique"),
    )
    .reset_index()
)
district_meta["well_depth_m"] = district_meta["well_depth_m"].round(1)

print("\nStatic district metadata (sample):")
print(district_meta.sort_values("district").to_string(index=False))

# ─── DAILY AGGREGATION (per station, then per district) ───────────────────────
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
        gwl_mean      = ("gwl_station_daily_mean", "median"),  # median — robust to outlier stations
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
print(f"  Mean stations/day per district:")
sc = district_daily.groupby("district")["station_count"].mean().round(1).sort_values()
print(sc.to_string())

# ─── CAP IMPLAUSIBLE DAILY CHANGES AT DISTRICT LEVEL ─────────────────────────
# A district mean changing > 5m in one day is almost certainly a station mix issue
# (stations entering/leaving the average). Replace those gwl_mean values with NaN
# so the forward-fill below smooths them out.
print(f"\nCapping district-level daily changes > {MAX_DAILY_CHANGE_M}m ...")
district_daily["prev_gwl"] = district_daily.groupby("district")["gwl_mean"].shift(1)
big_jump = (district_daily["gwl_mean"] - district_daily["prev_gwl"]).abs() > MAX_DAILY_CHANGE_M
n_jumps = big_jump.sum()
district_daily.loc[big_jump, ["gwl_mean", "gwl_min", "gwl_max"]] = np.nan
district_daily = district_daily.drop(columns=["prev_gwl"])
print(f"  Implausible jumps NaN'd: {n_jumps:,}")

# ─── REINDEX TO FULL CALENDAR + FILL GAPS ─────────────────────────────────────
full_idx   = pd.date_range(START, END, freq="D")
filled_dfs = []

print("\nGap-fill summary per district:")
print(f"  {'District':<20} {'Stations':>9} {'Data starts':<14} {'Missing %':>10} {'Status'}")
print(f"  {'-'*70}")

for district, grp in district_daily.groupby("district"):
    grp = grp.set_index("date").reindex(full_idx)
    grp["district"] = district

    # 1 = a real (post-cleaning) reading exists for this day; 0 = gap-filled or missing.
    # Downstream models use this to mask the loss / know which inputs are real.
    grp["gwl_observed"] = grp["gwl_mean"].notna().astype(int)

    grp["gwl_mean"]      = grp["gwl_mean"].ffill(limit=7).bfill(limit=7)
    grp["gwl_min"]       = grp["gwl_min"].ffill(limit=7).bfill(limit=7)
    grp["gwl_max"]       = grp["gwl_max"].ffill(limit=7).bfill(limit=7)
    grp["station_count"] = grp["station_count"].ffill(limit=7).bfill(limit=7)

    # Attach static metadata — constant across all rows for this district
    meta_row = district_meta.loc[district_meta["district"] == district]
    if not meta_row.empty:
        grp["aquifer_type"] = meta_row["aquifer_type"].iloc[0]
        grp["well_depth_m"] = meta_row["well_depth_m"].iloc[0]
        grp["n_stations"]   = meta_row["n_stations"].iloc[0]
    else:
        grp["aquifer_type"] = "Unknown"
        grp["well_depth_m"] = np.nan
        grp["n_stations"]   = 0

    missing_pct = grp["gwl_mean"].isna().mean() * 100
    first_valid = grp["gwl_mean"].first_valid_index()
    mean_stations = grp["station_count"].mean()
    status = "OK" if missing_pct < 10 else ("GAPS" if missing_pct < 40 else "SPARSE")

    first_str = first_valid.strftime("%Y-%m-%d") if first_valid else "NO DATA"
    print(f"  {district:<20} {mean_stations:>9.1f} {first_str:<14} {missing_pct:>9.1f}%  {status}")

    filled_dfs.append(grp.reset_index().rename(columns={"index": "date"}))

district_daily = pd.concat(filled_dfs, ignore_index=True)

overall_missing = district_daily["gwl_mean"].isna().mean() * 100
print(f"\nOverall missing gwl_mean : {overall_missing:.1f}%")
print(f"Total rows               : {len(district_daily):,}")
print(f"Districts                : {district_daily['district'].nunique()}")

# ─── SAVE ─────────────────────────────────────────────────────────────────────
out_cols = [
    "district", "date",
    "gwl_mean", "gwl_min", "gwl_max", "station_count", "gwl_observed",
    "aquifer_type", "well_depth_m", "n_stations",
]
district_daily[out_cols].to_csv(OUTPUT, index=False)
print(f"\nSaved -> {OUTPUT}")
print(f"Columns: {out_cols}")
print(f"Aquifer type distribution:\n{district_daily.drop_duplicates('district')['aquifer_type'].value_counts().to_string()}")
