"""
Collects daily evapotranspiration and multi-layer soil moisture from ERA5-Land
(via Google Earth Engine) for all Maharashtra districts.

WHY THIS EXISTS
---------------
The previous ET / soil-moisture source (WRIS / NRSC VIC) stopped publishing at
**2025-06-30**. `preprocess_env_features.py` filled everything after that with a
per-district day-of-year climatology and honestly flagged it via
`et_mm_observed` / `soil_moisture_observed` — but the effect on the TFT
walk-forward evaluation was severe:

    fold                    real ET / soil-moisture
    f1_monsoon2024          97%
    f2_postmonsoon2024     100%
    f3_dry2025              96%
    f4_monsoon2025          25%
    f5_postmonsoon2025       0%
    f6_dry2026               0%

i.e. in half the evaluation folds `et_mm`, `soil_moisture` and
`net_water_balance` carried no information the model didn't already have from
`doy_sin`/`doy_cos`. ERA5-Land has a ~5-day latency instead of a ~14-month one,
so this problem does not come back.

WHAT IT ADDS BEYOND A LIKE-FOR-LIKE REPLACEMENT
-----------------------------------------------
The old source gave a single surface soil-moisture number. ERA5-Land gives four
depth layers, and the deep ones are the actual recharge pathway to the water
table:

    volumetric_soil_water_layer_1   0-7 cm      surface, fast, rain-following
    volumetric_soil_water_layer_2   7-28 cm     root zone
    volumetric_soil_water_layer_3   28-100 cm   drainage / percolation
    volumetric_soil_water_layer_4   100-289 cm  closest proxy to recharge flux

`sm_l3` / `sm_l4` are expected to be far better groundwater predictors than a
surface layer, because they lag rainfall by roughly the same 3-21 days that
recharge does.

Potential evaporation is also collected: `pet_mm - et_mm` is a direct water-stress
signal, and PET is what drives irrigation pumping (the main non-natural
groundwater sink in Maharashtra).

DATASET
-------
    ECMWF/ERA5_LAND/DAILY_AGGR
    0.1 deg (~11 km), global, 1950-present, ~5 day latency.

    total_evaporation_sum       m of water equivalent, NEGATIVE for evaporation
    potential_evaporation_sum   m of water equivalent, NEGATIVE for evaporation
    volumetric_soil_water_*     m^3/m^3 (0-1 fraction)

Both evaporation bands are converted to positive mm/day; soil water is converted
to percent so it stays on the same scale as the old NRSC column.

SETUP (one-time, same as chirps_rainfall.py / sentinel_indices.py)
    earthengine authenticate
    earthengine set_project major-project-495508

RUN
    python era5_land.py

OUTPUT
    datasets/era5_land_maharashtra_23_26.csv
        district, date, et_mm, pet_mm, soil_moisture,
        sm_l1, sm_l2, sm_l3, sm_l4

    Resumable: (district, year) pairs already fetched are recorded in
    txt files/era5_progress.txt and skipped on re-run.

NEXT
    python "../Data Preprocessing/preprocess_env_features.py"   # picks this up automatically
    python "../Data Preprocessing/merge_master.py"
"""

import time
from pathlib import Path

import ee
import pandas as pd

# ─── CONFIG ───────────────────────────────────────────────────────────────────
START_DATE = "2023-01-01"
END_DATE   = "2026-06-30"
PROJECT    = "major-project-495508"
YEARS      = [2023, 2024, 2025, 2026]

COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
SCALE_M    = 11132          # ERA5-Land native ~0.1 deg

BANDS = [
    "total_evaporation_sum",
    "potential_evaporation_sum",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]

DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
]

# FAO GAUL spellings — identical map to chirps_rainfall.py / sentinel_indices.py.
# Gadchiroli in particular was silently failing for months under the typo
# "Gadtchiroli"; the verified GAUL name is "Garhchiroli".
GAUL_NAME_MAP = {
    "Mumbai City":     ["Mumbai city", "Mumbai", "Mumbai City"],
    "Mumbai Suburban": ["Mumbai Suburban", "Mumbai (Suburban)"],
    "Osmanabad":       ["Osmanabad", "Usmanabad"],
    "Buldhana":        ["Buldhana", "Buldana"],
    "Ahmednagar":      ["Ahmednagar", "Ahmadnagar"],
    "Beed":            ["Beed", "Bid"],
    "Gondia":          ["Gondia", "Gondiya"],
    "Gadchiroli":      ["Gadchiroli", "Garhchiroli"],
    "Raigad":          ["Raigad", "Raigarh"],
    "Palghar":         ["Palghar", "Thane"],
}

OUTPUT     = Path("datasets/era5_land_maharashtra_23_26.csv")
PROGRESS_F = Path("txt files/era5_progress.txt")
OUTPUT.parent.mkdir(exist_ok=True)
PROGRESS_F.parent.mkdir(exist_ok=True)

OUT_COLS = ["district", "date", "et_mm", "pet_mm", "soil_moisture",
            "sm_l1", "sm_l2", "sm_l3", "sm_l4"]

# ─── GEE INIT ─────────────────────────────────────────────────────────────────
print("Initialising Earth Engine ...")
ee.Initialize(project=PROJECT)
ee.data.setDeadline(180000)

india = ee.FeatureCollection("FAO/GAUL/2015/level2")
mh    = india.filter(ee.Filter.eq("ADM1_NAME", "Maharashtra"))


def get_district_geometry(district_name: str):
    """Return GEE geometry, trying alternate GAUL spellings."""
    for variant in GAUL_NAME_MAP.get(district_name, [district_name]):
        match = mh.filter(ee.Filter.eq("ADM2_NAME", variant))
        if match.size().getInfo() > 0:
            return match.geometry(), variant
    return None, None


def fetch_era5_year(geometry, district_name: str, year: int):
    """One server-side reduction per (district, year) — same pattern as CHIRPS."""
    y_start = max(f"{year}-01-01", START_DATE)
    y_end   = min(f"{year + 1}-01-01",
                  (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    if y_start >= y_end:
        return []

    coll = (ee.ImageCollection(COLLECTION)
            .filterDate(y_start, y_end)
            .filterBounds(geometry)
            .select(BANDS))

    def reduce_image(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        stats = img.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = geometry,
            scale     = SCALE_M,
            maxPixels = 1e9,
        )
        return ee.Feature(None, stats.set("date", date))

    features = coll.map(reduce_image).getInfo()["features"]

    rows = []
    for f in features:
        p = f["properties"]
        if p.get("total_evaporation_sum") is None:
            continue

        def mm(band):
            """m of water equivalent (negative = flux away from surface) -> positive mm/day."""
            v = p.get(band)
            return None if v is None else round(abs(float(v)) * 1000.0, 4)

        def pct(band):
            """m^3/m^3 -> percent, matching the old NRSC soil-moisture scale."""
            v = p.get(band)
            return None if v is None else round(float(v) * 100.0, 4)

        rows.append({
            "district":      district_name,
            "date":          p["date"],
            "et_mm":         mm("total_evaporation_sum"),
            "pet_mm":        mm("potential_evaporation_sum"),
            # `soil_moisture` keeps the old column name/scale so nothing downstream
            # breaks; the root zone (7-28 cm) is the closest analogue to what the
            # NRSC VIC product reported.
            "soil_moisture": pct("volumetric_soil_water_layer_2"),
            "sm_l1":         pct("volumetric_soil_water_layer_1"),
            "sm_l2":         pct("volumetric_soil_water_layer_2"),
            "sm_l3":         pct("volumetric_soil_water_layer_3"),
            "sm_l4":         pct("volumetric_soil_water_layer_4"),
        })
    return rows


# ─── PROGRESS TRACKING ────────────────────────────────────────────────────────
def load_progress() -> set:
    return set(PROGRESS_F.read_text().splitlines()) if PROGRESS_F.exists() else set()


def mark_done(key: str):
    with open(PROGRESS_F, "a") as f:
        f.write(key + "\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
done = load_progress()
print(f"Resuming — {len(done)} (district, year) pairs already done.\n")

if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
    pd.DataFrame(columns=OUT_COLS).to_csv(OUTPUT, index=False)

total, done_count, skipped = len(DISTRICTS) * len(YEARS), 0, []

for district in DISTRICTS:
    geom, matched = get_district_geometry(district)
    if geom is None:
        print(f"  {district:<20} -> NOT in FAO GAUL, skipping")
        skipped.append(district)
        continue

    note = f" (matched '{matched}')" if matched != district else ""
    print(f"  {district:<20}{note}")

    t0 = time.time()
    for year in YEARS:
        key = f"{district}|{year}"
        if key in done:
            done_count += 1
            print(f"    {year}: cached, skipping")
            continue
        try:
            rows = fetch_era5_year(geom, district, year)
            if rows:
                pd.DataFrame(rows)[OUT_COLS].to_csv(OUTPUT, mode="a", header=False, index=False)
            mark_done(key)
            done_count += 1
            print(f"    {year}: {len(rows):>3} days   "
                  f"({time.time()-t0:.1f}s elapsed, {done_count}/{total} overall)")
        except Exception as e:
            print(f"    {year}: ERROR — {e}")

# ─── POST-PROCESS ─────────────────────────────────────────────────────────────
print("\nConsolidating ...")
df = pd.read_csv(OUTPUT, parse_dates=["date"])
df = df.drop_duplicates(subset=["district", "date"], keep="last")
df = df.sort_values(["district", "date"]).reset_index(drop=True)
df[OUT_COLS].to_csv(OUTPUT, index=False)

print(f"\nSaved -> {OUTPUT}")
print(f"  Rows       : {len(df):,}")
print(f"  Districts  : {df['district'].nunique()}")
print(f"  Date range : {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"  Missing    :\n{df[OUT_COLS[2:]].isna().sum().to_string()}")

# Per-year coverage — this is the table that must NOT show a collapsing tail again.
print("\n  Coverage by year (days per district, 365/366 = complete):")
cov = df.groupby([df['date'].dt.year, 'district']).size().groupby(level=0).median()
print(cov.to_string())

if skipped:
    print(f"\n  Skipped (not in FAO GAUL): {skipped}")

print("\nNext steps:")
print('  1. python "../Data Preprocessing/preprocess_env_features.py"')
print('  2. python "../Data Preprocessing/merge_master.py"')
