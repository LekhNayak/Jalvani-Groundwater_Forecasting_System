"""
Merges all clean datasets into one master training CSV.
Run this AFTER all preprocess_*.py scripts have completed.

v2 (2026-07) — full-calendar rebuild. Key changes vs v1:
  - Rows with missing gwl_mean are KEPT (full daily calendar per district).
    `gwl_observed` (from groundwater_clean) marks real readings; models must
    mask the loss on unobserved target days instead of training on fake rows.
  - Because every district is a complete daily calendar, shift(7) etc. are now
    date-true lags (v1 dropped NaN rows first, corrupting lags across gaps).
  - District gate: districts with >= 60% missing groundwater are excluded and
    logged to excluded_districts.csv (they produce zero valid windows anyway).
  - Districts with < 5 unique stations are kept but flagged (low_station_flag)
    so they can be reported separately — their "district mean" is 1-4 wells.
  - Static per-district metadata from groundwater_clean is passed through:
    aquifer_type_id, well_depth_m, n_stations.
  - CGWB extraction/recharge yearly features are folded in here (previously a
    separate merge_cgwb_into_master.py step that overwrote the master).
  - District names normalised via CANON_NAME map ("MUMBAI CITY" -> "Mumbai City").

Inputs  (from data/clean/):
    groundwater_clean.csv        (full calendar, 30 districts, observed mask, statics)
    rainfall_clean.csv           (CHIRPS gridded, full calendar)
    env_features_clean.csv       (ET, soil moisture, temp, humidity)

Static inputs:
    ../data/raw/soil_texture_maharashtra.csv
    ../data/raw/cgwb_extraction_maharashtra.csv

Sentinel inputs (monthly, interpolated to daily):
    ../data/raw/sentinel2_indices_maharashtra_23_25.csv
    ../data/raw/sentinel1_backscatter_maharashtra_23_25.csv

Outputs:
    data/clean/master_dataset.csv
    data/clean/excluded_districts.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CLEAN        = Path("../data/clean")
SOIL_TEXTURE = Path("../data/raw/soil_texture_maharashtra.csv")
CGWB         = Path("../data/raw/cgwb_extraction_maharashtra.csv")
SENTINEL2    = Path("../data/raw/sentinel2_indices_maharashtra_23_25.csv")
SENTINEL1    = Path("../data/raw/sentinel1_backscatter_maharashtra_23_25.csv")

MAX_GWL_MISSING = 0.60   # exclude districts missing >= this fraction of gwl days
# Flag districts whose series comes from a single well. Since preprocess_groundwater.py
# now restricts to the continuously-reporting telemetric panel (~109 stations state-wide
# instead of 1379 mostly-one-off manual wells), typical per-district counts are 1-11, so
# the old threshold of 5 would have flagged most of the state. A district with 2+ wells
# is a small but genuine sample; a district with exactly 1 is literally one borehole and
# is reported separately rather than being blended into the headline.
MIN_STATIONS    = 2      # below this -> low_station_flag = 1 (kept, reported separately)

# Canonical district names — every input file is mapped through this before merging
CANON_NAME = {
    "MUMBAI CITY": "Mumbai City",
}

AQUIFER_TYPE_MAP = {"Unconfined": 0, "Semi-Confined": 1, "Confined": 2}

CGWB_CATEGORY_MAP = {
    "Safe": 0, "Semi-Critical": 1, "Critical": 2, "Over-Exploited": 3, "Saline": 4,
}

# ─── LOAD CLEAN FILES ─────────────────────────────────────────────────────────
print("Loading clean datasets ...")
gw   = pd.read_csv(CLEAN / "groundwater_clean.csv",  parse_dates=["date"])
rf   = pd.read_csv(CLEAN / "rainfall_clean.csv",     parse_dates=["date"])
ev   = pd.read_csv(CLEAN / "env_features_clean.csv", parse_dates=["date"])
soil = pd.read_csv(SOIL_TEXTURE)
cgwb = pd.read_csv(CGWB)
s2   = pd.read_csv(SENTINEL2)
s1   = pd.read_csv(SENTINEL1)

for frame in (gw, rf, ev, soil, cgwb, s2, s1):
    frame["district"] = frame["district"].replace(CANON_NAME)

print(f"  Groundwater  : {len(gw):,} rows, {gw['district'].nunique()} districts")
print(f"  Rainfall     : {len(rf):,} rows, {rf['district'].nunique()} districts")
print(f"  Env features : {len(ev):,} rows, {ev['district'].nunique()} districts")
print(f"  Soil texture : {len(soil):,} rows (static, one per district)")
print(f"  CGWB extract : {len(cgwb):,} rows, {cgwb['district'].nunique()} districts")
print(f"  Sentinel-2   : {len(s2):,} rows, {s2['district'].nunique()} districts")
print(f"  Sentinel-1   : {len(s1):,} rows, {s1['district'].nunique()} districts")

# ─── DISTRICT GATE ────────────────────────────────────────────────────────────
# groundwater_clean is a full daily calendar per district, so isna() over it is
# the true missing fraction. Districts above the threshold cannot produce a
# single valid training window and only pollute per-district statistics.
print(f"\nDistrict gate (exclude if gwl missing >= {MAX_GWL_MISSING:.0%}):")
# Only judge coverage over the span CGWB actually published. The calendar extends past the last
# real reading (covariates are newer than the target), and that empty tail must not count as
# "missing" or it would push borderline districts over the exclusion threshold.
GW_LAST_OBS = gw.loc[gw["gwl_observed"] == 1, "date"].max()
print(f"  Last real groundwater reading: {GW_LAST_OBS.date()} (calendar ends {gw['date'].max().date()})")
gw_span = gw[gw["date"] <= GW_LAST_OBS]
coverage = gw_span.groupby("district").agg(
    missing_frac=("gwl_mean", lambda s: s.isna().mean()),
    n_stations  =("n_stations", "first"),
)
excluded = coverage[coverage["missing_frac"] >= MAX_GWL_MISSING].copy()
excluded["reason"] = excluded["missing_frac"].map(lambda f: f"gwl missing {f:.0%}")
kept = coverage.drop(excluded.index)

for d, row in excluded.sort_index().iterrows():
    print(f"  EXCLUDED  {d:<20} {row['reason']}")
excluded.reset_index().to_csv(CLEAN / "excluded_districts.csv", index=False)

gw = gw[~gw["district"].isin(excluded.index)]
print(f"  Kept {gw['district'].nunique()} districts "
      f"(low-station flagged: {sorted(kept[kept['n_stations'] < MIN_STATIONS].index)})")

# ─── DATA TIER (new, 2026-08-15) ─────────────────────────────────────────────
# A district's forecastability is driven by how many wells it has and how much of
# the record is genuinely observed — NOT by how much its water table moves.
# Measured on this dataset (Spearman vs. how well any model does at t+30):
#     2023 observed rate   -0.54
#     overall observed rate -0.54
#     n_stations            -0.51
#     median 30-day movement -0.10   (i.e. nothing)
# 2023 is the discriminating year: it is only 57% observed state-wide, and the
# districts missing most of it (Aurangabad 25%, Beed 25%, Kolhapur 26%, Dhule
# 27%, Gadchiroli 27%, Bhandara 29%) are the ones no model can forecast.
#
# The tier is a statement about DATA ONLY. It is deliberately not fitted to model
# results, so it can be quoted as an independent explanation of them.
#   A  dense panel + full history   -> headline results
#   B  usable
#   C  thin panel AND missing most of 2023 -> report as a data finding, not a model failure
#   D  unusable                     -> excluded from training
TIER_A_STATIONS, TIER_A_COV23 = 6, 0.65
TIER_C_STATIONS, TIER_C_COV23 = 4, 0.55
TIER_D_OVERALL_COV = 0.55

gw_2023 = gw[gw["date"].dt.year == 2023]
cov23 = gw_2023.groupby("district")["gwl_observed"].mean()
cov_all = gw[gw["date"] <= GW_LAST_OBS].groupby("district")["gwl_observed"].mean()
n_st = gw.groupby("district")["n_stations"].first()


def _tier(d):
    c23, call, ns = cov23.get(d, 0.0), cov_all.get(d, 0.0), n_st.get(d, 0)
    if call < TIER_D_OVERALL_COV:
        return "D"
    if ns >= TIER_A_STATIONS and c23 >= TIER_A_COV23:
        return "A"
    if ns <= TIER_C_STATIONS and c23 < TIER_C_COV23:
        return "C"
    return "B"


tiers = pd.Series({d: _tier(d) for d in sorted(gw["district"].unique())}, name="data_tier")
# Tiers as they were before the v4 target rebuild (recovered wells change n_stations).
# Kept alongside so results can be compared on the frozen v3 grouping.
_V3_SCORECARD = CLEAN.parent / "master" / "district_data_quality_v3target.csv"
tiers_v3 = (pd.read_csv(_V3_SCORECARD).set_index("district")["data_tier"]
            if _V3_SCORECARD.exists() else pd.Series(dtype=object))
scorecard = pd.DataFrame({
    "n_stations": n_st, "coverage_2023": cov23.round(3),
    "coverage_all": cov_all.round(3), "data_tier": tiers,
    "data_tier_v3": tiers_v3.reindex(tiers.index),
}).sort_values(["data_tier", "coverage_2023"], ascending=[True, False])
scorecard.index.name = "district"
scorecard.to_csv(CLEAN / "district_data_quality.csv")

print("\nData tier (A dense+complete / B usable / C thin+gappy / D unusable):")
print(scorecard.to_string())
for t in ["A", "B", "C", "D"]:
    members = sorted(scorecard.index[scorecard["data_tier"] == t])
    print(f"  Tier {t} ({len(members)}): {members}")
print(f"  -> data/clean/district_data_quality.csv")

# ─── MERGE (left joins on the full groundwater calendar — rows are NOT dropped) ─
print("\nMerging ...")
master = gw.merge(rf, on=["district", "date"], how="left", suffixes=("", "_rf"))
master = master.merge(ev, on=["district", "date"], how="left")
if "station_count_rf" in master.columns:            # CHIRPS carries a dummy station_count
    master = master.drop(columns=["station_count_rf"])

print(f"  After merge  : {len(master):,} rows, {master['district'].nunique()} districts")
rain_null = master.groupby("district")["rainfall_mm"].apply(lambda s: s.isna().mean())
rain_null = rain_null[rain_null > 0]
if len(rain_null):
    print("  WARNING — districts with missing rainfall days:")
    print(rain_null.map("{:.1%}".format).to_string())

# Daily station_count: NaN means no station reported that day
master["station_count"] = master["station_count"].fillna(0)

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

# ─── LAG FEATURES (date-true: each district is a complete daily calendar) ────
master = master.sort_values(["district", "date"]).reset_index(drop=True)

# Safety: assert the calendar really is gap-free per district, otherwise the
# shift()-based lags below silently stop being date-true again.
step = master.groupby("district")["date"].diff().dropna()
assert (step == pd.Timedelta(days=1)).all(), "calendar grid has holes — lags would be corrupted"

master["gwl_lag_7"]  = master.groupby("district")["gwl_mean"].shift(7)
master["gwl_lag_14"] = master.groupby("district")["gwl_mean"].shift(14)
master["gwl_lag_30"] = master.groupby("district")["gwl_mean"].shift(30)

# NaN wherever either side is missing — no imputation here (that is a modelling
# decision; median-filling lags injects fake flat history into gappy districts)
master["gwl_7d_change"]  = master["gwl_mean"] - master["gwl_lag_7"]
master["gwl_30d_change"] = master["gwl_mean"] - master["gwl_lag_30"]

# ─── HYDRO BALANCE FEATURES ───────────────────────────────────────────────────
master["net_water_balance"] = master["rainfall_mm"] - master["et_mm"]
# VPD proxy: higher temp × lower humidity → atmosphere is "thirstier" → more ET
master["vpd_approx"] = master["temp_mean"] * (1.0 - master["humidity_mean"] / 100.0)

# Water stress = unmet atmospheric demand. High stress drives irrigation pumping,
# which is the dominant non-natural groundwater sink in Maharashtra. Only present
# when ET comes from ERA5-Land (the old WRIS feed had no PET band).
if "pet_mm" in master.columns:
    master["water_stress"] = master["pet_mm"] - master["et_mm"]

# ─── STATIC METADATA FROM GROUNDWATER CLEANING ───────────────────────────────
master["aquifer_type_id"]  = master["aquifer_type"].map(AQUIFER_TYPE_MAP).fillna(-1).astype(int)
master["low_station_flag"] = (master["n_stations"] < MIN_STATIONS).astype(int)
master["data_tier"]        = master["district"].map(tiers)
if len(tiers_v3):
    master["data_tier_v3"] = master["district"].map(tiers_v3)
master = master.drop(columns=["aquifer_type"])

# ERA5-Land's `soil_moisture` is the same field as sm_l2 (r = 1.000) — keep one copy.
if "sm_l2" in master.columns and "soil_moisture" in master.columns:
    master = master.drop(columns=["soil_moisture", "soil_moisture_observed"], errors="ignore")

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
available_soil_cols = ["district"] + [c for c in soil_cols if c in soil.columns]
soil_clean = soil[available_soil_cols].drop_duplicates(subset=["district"])

master = master.merge(soil_clean, on="district", how="left")
soil_missing_districts = sorted(
    master.loc[master[[c for c in soil_cols if c in master.columns]].isnull().any(axis=1), "district"].unique()
)
if soil_missing_districts:
    print(f"  Soil texture : missing for {soil_missing_districts}")

# ─── CGWB EXTRACTION / RECHARGE (yearly static, ffill to unpublished years) ──
cgwb["aquifer_category_id"] = cgwb["category"].map(CGWB_CATEGORY_MAP).fillna(-1).astype(int)
cgwb_cols = ["annual_recharge_bcm", "annual_extraction_bcm",
             "extractable_resource_bcm", "stage_of_extraction_pct",
             "aquifer_category_id"]
cgwb_cols = [c for c in cgwb_cols if c in cgwb.columns]

years = sorted(master["date"].dt.year.unique())
all_years = pd.DataFrame(
    [(d, y) for d in cgwb["district"].unique() for y in years],
    columns=["district", "year"],
)
cgwb_full = all_years.merge(cgwb.sort_values(["district", "year"]),
                            on=["district", "year"], how="left")
cgwb_full[cgwb_cols] = cgwb_full.groupby("district")[cgwb_cols].ffill().bfill()

master["year"] = master["date"].dt.year
master = master.merge(cgwb_full[["district", "year"] + cgwb_cols],
                      on=["district", "year"], how="left").drop(columns=["year"])

cgwb_unmatched = sorted(master.loc[master["annual_recharge_bcm"].isna(), "district"].unique())
if cgwb_unmatched:
    print(f"  CGWB extract : no data for {cgwb_unmatched} — filled with column medians")
    for c in cgwb_cols:
        master[c] = master[c].fillna(master[c].median())
master["aquifer_category_id"] = master["aquifer_category_id"].astype(int)

# ─── SENTINEL MERGE (monthly → daily via linear interpolation) ───────────────
def expand_sentinel_to_daily(sent_df, value_cols, master_dates):
    """
    Sentinel data is monthly (year, month). Expand to daily, CAUSALLY.

    v3 fix (2026-08-15) — the previous version stamped each monthly composite at
    the **1st** of its month and then linearly interpolated to daily. That meant
    NDVI on 15 January was a 50/50 blend of the January and *February*
    composites: up to half a month of future information in every daily encoder
    value, in a model whose whole point is forecasting 7-30 days ahead.

    Now: a monthly composite summarises a month that has already finished, so it
    is stamped at month-END and held constant (step / ffill) until the next
    composite lands. A value on any given day therefore depends only on months
    that were already complete. The cost is a lag of up to ~30 days, which is
    honest — that is genuinely when the information becomes available.
    """
    sent_df = sent_df.copy()
    # month-END: the composite for YYYY-MM is only knowable once YYYY-MM is over
    sent_df["date"] = (
        pd.to_datetime(sent_df["year"].astype(str) + "-" + sent_df["month"].astype(str).str.zfill(2) + "-01")
        + pd.offsets.MonthEnd(0)
    )
    sent_df = sent_df[["district", "date"] + value_cols]

    daily_parts = []
    full_idx = pd.date_range(master_dates.min(), master_dates.max(), freq="D")
    for dist, grp in sent_df.groupby("district"):
        grp = grp.set_index("date").sort_index()
        grp = grp[~grp.index.duplicated(keep="last")]
        grp = grp.reindex(grp.index.union(full_idx)).sort_index()
        for col in value_cols:
            # ffill only — never interpolate forwards from a future composite.
            # bfill covers only the warm-up before the first composite exists.
            grp[col] = grp[col].ffill().bfill()
        grp = grp.reindex(full_idx)
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

sent_missing_districts = sorted(master.loc[master["ndvi_mean"].isna(), "district"].unique())
if sent_missing_districts:
    print(f"  Sentinel     : no coverage for {sent_missing_districts} (left NaN — impute per-district at training time)")

# ─── COLUMN ORDER ─────────────────────────────────────────────────────────────
ordered_cols = [
    "district", "district_id", "date", "day_of_year",
    "gwl_mean", "gwl_min", "gwl_max", "station_count", "gwl_observed",
    "n_active", "gwl_frac_real",          # v4 target: panel wells active / share really reporting
    "rainfall_mm", "rainfall_7d", "rainfall_30d", "rainfall_observed",
    "et_mm", "et_mm_observed", "soil_moisture", "soil_moisture_observed", "net_water_balance",
    # ERA5-Land extras (absent when falling back to the old WRIS feed)
    "pet_mm", "pet_mm_observed", "water_stress",
    "sm_l1", "sm_l2", "sm_l3", "sm_l4",
    "temp_max", "temp_min", "temp_mean", "humidity_mean", "vpd_approx",
    "month", "season", "is_monsoon",
    "gwl_lag_7", "gwl_lag_14", "gwl_lag_30",
    "gwl_7d_change", "gwl_30d_change",
    "aquifer_type_id", "well_depth_m", "n_stations", "low_station_flag", "data_tier", "data_tier_v3",
    "annual_recharge_bcm", "annual_extraction_bcm",
    "extractable_resource_bcm", "stage_of_extraction_pct", "aquifer_category_id",
    "sand_0cm", "clay_0cm", "silt_0cm",
    "sand_10cm", "clay_10cm", "silt_10cm",
    "sand_30cm", "clay_30cm", "silt_30cm",
    "ndvi_mean", "ndwi_mean", "ndmi_mean",
    "vv_db", "vh_db", "vh_vv_ratio",
]
final_cols = [c for c in ordered_cols if c in master.columns]
master = master[final_cols]

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\nFinal master dataset:")
print(f"  Rows        : {len(master):,} (full calendar — includes unobserved-gwl days)")
print(f"  Observed gwl: {master['gwl_observed'].sum():,} rows ({master['gwl_observed'].mean():.1%})")
print(f"  Districts   : {master['district'].nunique()}")
print(f"  Date range  : {master['date'].min().date()} -> {master['date'].max().date()}")
print(f"  Columns ({len(final_cols)}) : {final_cols}")
print(f"\nMissing values:")
missing = master.isnull().sum()
print(missing[missing > 0].to_string())

# ─── SAVE ─────────────────────────────────────────────────────────────────────
master.to_csv(CLEAN / "master_dataset.csv", index=False)
print(f"\nSaved -> data/clean/master_dataset.csv")
print(master.head(3).to_string(index=False))
