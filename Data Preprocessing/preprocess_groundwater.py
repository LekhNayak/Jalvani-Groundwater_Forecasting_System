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
END   = "2026-06-04"   # actual max CGWB coverage as of the 2026-08 re-collection

# Outlier thresholds
MAX_ABS_DEPTH      = 100.0   # |value| above this is sensor fault (raw had max +172m)
HAMPEL_WINDOW      = 7       # days, per-station rolling window for spike detection
HAMPEL_K           = 3.0     # MAD multiplier; values beyond this are spikes
MAX_DAILY_CHANGE_M = 5.0     # max physically plausible 1-day change at district scale

# ─── STATION PANEL FILTER ────────────────────────────────────────────────────
# The raw CGWB feed mixes two completely different measurement programmes:
#   1. ~125 TELEMETRIC stations  — automatic loggers, 20-90% daily coverage, run
#      continuously across the whole period. These are the real monitoring network.
#   2. ~1250 MANUAL-SURVEY stations — exactly 2 readings each (Jan + May 2023),
#      mostly shallow dug wells.
# Coverage is sharply bimodal (nothing between ~1% and ~20%), so a 10% threshold
# cleanly separates them.
#
# Mixing the two was corrupting the district series: e.g. Aurangabad showed
# -4.94 m in Jan 2023 (40 manual dug wells) then -98.9 m by Jun 2023 (1 deep
# telemetric borewell) — a ~94 m "change" that is purely a change of *which wells
# were measured*, not of the water table. Same in Thane (36 -> 1 station).
MIN_STATION_COVERAGE = 0.10  # fraction of the full period a station must report on

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
#
# A station is "mixed" if it has both positive and negative readings. Most mixed
# stations are actually clean single-convention stations with a tiny number of
# spurious readings (observed: exact value 1.0 scattered among thousands of
# otherwise-consistent negative readings — a sensor/telemetry placeholder, not a
# real measurement). Dropping the whole station over a few bad rows was throwing
# away good data — e.g. Gadchiroli's only two working stations (CGWNAG2184/2185,
# 99.5%+ and 99.97%+ negative respectively) were being discarded entirely this way.
# Fix: keep a mixed station if one sign dominates (>=95% of its readings); drop
# only the minority-sign rows as spurious. Only drop the whole station if the sign
# is genuinely ambiguous (<95% dominance) — a real data-quality issue.
MIXED_DOMINANCE_THRESHOLD = 0.95

station_sign = df.groupby("stationCode")["dataValue"].agg(["min", "max"]).reset_index()
station_sign["all_negative"] = station_sign["max"] <= 0
station_sign["all_positive"] = station_sign["min"] >= 0
station_sign["mixed"]        = ~(station_sign["all_negative"] | station_sign["all_positive"])

neg_frac = df.groupby("stationCode")["dataValue"].apply(lambda s: (s < 0).mean())
station_sign = station_sign.set_index("stationCode")
station_sign["neg_frac"] = neg_frac
station_sign["dominant_frac"] = station_sign["neg_frac"].clip(upper=1).combine(
    1 - station_sign["neg_frac"], max
)
station_sign = station_sign.reset_index()

salvageable_mask = station_sign["mixed"] & (station_sign["dominant_frac"] >= MIXED_DOMINANCE_THRESHOLD)
ambiguous_mask   = station_sign["mixed"] & (station_sign["dominant_frac"] < MIXED_DOMINANCE_THRESHOLD)

n_neg        = station_sign["all_negative"].sum()
n_pos        = station_sign["all_positive"].sum()
n_salvaged   = salvageable_mask.sum()
n_ambiguous  = ambiguous_mask.sum()
print(f"\nSign convention split:")
print(f"  Negative-only     : {n_neg:>5} stations (already correct sign)")
print(f"  Positive-only     : {n_pos:>5} stations (sign-flipped to negative)")
print(f"  Mixed, salvaged   : {n_salvaged:>5} stations (>={MIXED_DOMINANCE_THRESHOLD:.0%} one-signed — kept, minority-sign rows dropped as spurious)")
print(f"  Mixed, ambiguous  : {n_ambiguous:>5} stations (dropped — genuine data-quality issue)")

# Drop only the genuinely ambiguous mixed-sign stations entirely
ambiguous_codes = set(station_sign.loc[ambiguous_mask, "stationCode"])
df = df[~df["stationCode"].isin(ambiguous_codes)]

# For salvaged stations, drop only their minority-sign (spurious) rows
salvaged_codes = set(station_sign.loc[salvageable_mask, "stationCode"])
dominant_is_neg = station_sign.set_index("stationCode")["neg_frac"] >= 0.5
is_salvaged_row = df["stationCode"].isin(salvaged_codes)
row_dominant_neg = df["stationCode"].map(dominant_is_neg).fillna(True)
spurious_row = is_salvaged_row & (
    (row_dominant_neg & (df["dataValue"] >= 0)) | (~row_dominant_neg & (df["dataValue"] < 0))
)
n_spurious_dropped = spurious_row.sum()
df = df[~spurious_row]
print(f"  Spurious minority-sign rows dropped from salvaged stations: {n_spurious_dropped:,}")

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

# ─── STATION PANEL FILTER ────────────────────────────────────────────────────
# Keep only continuously-reporting (telemetric) stations. See the note at the top
# of this file: the one-off manual-survey stations create a large artificial jump
# at the start of each district's series when they drop out.
df["date"] = df["dataTime"].dt.normalize()
total_days = (pd.Timestamp(END) - pd.Timestamp(START)).days + 1

station_days = df.groupby("stationCode")["date"].nunique()
station_cov  = station_days / total_days
panel_codes  = set(station_cov[station_cov >= MIN_STATION_COVERAGE].index)

print(f"\nStation panel filter (>= {MIN_STATION_COVERAGE:.0%} daily coverage):")
print(f"  Stations before : {df['stationCode'].nunique():,}")
print(f"  Stations in panel: {len(panel_codes):,}  (dropped {df['stationCode'].nunique()-len(panel_codes):,} "
      f"sparse/manual-survey stations)")
df = df[df["stationCode"].isin(panel_codes)]
print(f"  Rows after panel filter: {len(df):,}")
print(f"  Districts remaining    : {df['district'].nunique()}")

# ─── DRIFT-ARTIFACT STATION FILTER ───────────────────────────────────────────
# Some panel wells are newly-drilled/re-commissioned borewells whose level is
# still equilibrating: they drift monotonically over the whole record instead of
# oscillating seasonally. e.g. Aurangabad's CGWBNG0515 (100 m deep) starts at
# -98.8 m — essentially the bottom of the borehole — and climbs steadily to -5 m
# over three years (102 m of pure trend). That is the well filling up, not the
# district water table moving, and it dominated Aurangabad's district series.
#
# NOTE: large range alone is NOT a valid test — Dhule's and Nandurbar's wells
# genuinely swing 30-50 m *seasonally* (deep hard-rock aquifers). The
# discriminator is how much of the variance is a single monotonic trend:
#   drift artifact  -> high trend R^2 (0.6+) AND large range
#   real deep well  -> large range but low trend R^2 (Dhule: 0.004-0.06)
DRIFT_R2_THRESHOLD    = 0.60
DRIFT_RANGE_THRESHOLD = 20.0   # metres

_sd = (df.groupby(["stationCode", "date"])["dataValue"].mean().reset_index())
drift_codes = []
for code, g in _sd.groupby("stationCode"):
    g = g.sort_values("date")
    v = g["dataValue"].values.astype(float)
    if len(v) < 50:
        continue
    t = (g["date"] - g["date"].min()).dt.days.values.astype(float)
    ss_tot = ((v - v.mean()) ** 2).sum()
    if ss_tot <= 0:
        continue
    coef, *_ = np.linalg.lstsq(np.vstack([t, np.ones_like(t)]).T, v, rcond=None)
    r2 = 1.0 - (((v - (coef[0] * t + coef[1])) ** 2).sum() / ss_tot)
    if r2 >= DRIFT_R2_THRESHOLD and (v.max() - v.min()) > DRIFT_RANGE_THRESHOLD:
        drift_codes.append((code, round(r2, 3), round(v.max() - v.min(), 1),
                            round(abs(coef[0]) * t.max(), 1)))

print(f"\nDrift-artifact filter (trend R^2 >= {DRIFT_R2_THRESHOLD}, range > {DRIFT_RANGE_THRESHOLD} m):")
if drift_codes:
    for code, r2, rng, drift in drift_codes:
        dist = df.loc[df["stationCode"] == code, "district"].iloc[0]
        print(f"  DROPPED {code} ({dist}): trend R^2={r2}, range={rng} m, total drift={drift} m")
    df = df[~df["stationCode"].isin([c[0] for c in drift_codes])]
else:
    print("  (none)")
print(f"  Rows after drift filter: {len(df):,}  |  Stations: {df['stationCode'].nunique()}  "
      f"|  Districts: {df['district'].nunique()}")

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

# ─── DAILY AGGREGATION VIA PER-STATION ANOMALIES ─────────────────────────────
# Aggregating RAW depths across stations makes the district series jump whenever
# the set of reporting stations changes, because wells sit at very different
# depths (e.g. Aurangabad's panel: one well averaging 28.6 m, another 3.4 m — the
# shallow one only starts reporting in Mar 2024, which alone shifted the district
# median by tens of metres).
#
# Instead each station is converted to its OWN anomaly (deviation from its own
# long-run mean), the anomalies are aggregated, and the result is re-referenced to
# a fixed per-district baseline. A station joining or leaving now contributes
# "how far above/below its own normal it is" rather than its absolute depth, so
# composition changes no longer masquerade as water-table movement.
station_daily = (
    df.groupby(["stationCode", "district", "date"])["dataValue"]
    .mean()
    .reset_index()
    .rename(columns={"dataValue": "gwl_station_daily_mean"})
)

# Each station's own long-run mean level, and the district's fixed reference level
station_mean   = station_daily.groupby("stationCode")["gwl_station_daily_mean"].mean()
station_daily["station_mean"] = station_daily["stationCode"].map(station_mean)
station_daily["anomaly"]      = station_daily["gwl_station_daily_mean"] - station_daily["station_mean"]

# District reference = median of its panel stations' own mean levels (fixed in time)
district_ref = (
    station_daily.drop_duplicates("stationCode")
    .groupby("district")["station_mean"].median()
)

district_daily = (
    station_daily.groupby(["district", "date"])
    .agg(
        anom_med      = ("anomaly", "median"),   # median — robust to a single bad station
        anom_min      = ("anomaly", "min"),
        anom_max      = ("anomaly", "max"),
        station_count = ("stationCode", "count"),
    )
    .reset_index()
)
ref = district_daily["district"].map(district_ref)
district_daily["gwl_mean"] = ref + district_daily["anom_med"]
district_daily["gwl_min"]  = ref + district_daily["anom_min"]
district_daily["gwl_max"]  = ref + district_daily["anom_max"]
district_daily = district_daily.drop(columns=["anom_med", "anom_min", "anom_max"])

print("\nPer-district reference levels (median of panel stations' own mean depths):")
print(district_ref.round(2).sort_values().to_string())

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
