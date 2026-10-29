"""
Collects daily rainfall via CHIRPS (Climate Hazards Group InfraRed Precipitation
with Stations) from Google Earth Engine for ALL Maharashtra districts.

This REPLACES the WRIS/CWC rainfall data with a consistent satellite-based source.

CHIRPS specs:
    - 0.05° (~5 km) gridded daily precipitation in mm/day
    - Combines satellite IR + station data (best-available product over India)
    - Coverage: 1981 - present, all of India
    - Asset: UCSB-CHG/CHIRPS/DAILY
    - Free, no rate limits when used in batch reductions

Setup (one-time, same as sentinel_indices.py):
    earthengine authenticate
    earthengine set_project major-project-495508

Run:
    python chirps_rainfall.py

Output:
    datasets/rainfall_chirps_maharashtra_23_25.csv
        Same schema as ../data/clean/rainfall_clean.csv:
        district, date, rainfall_mm, rainfall_7d, rainfall_30d, station_count

After running, replace data/clean/rainfall_clean.csv with this file, then
re-run merge_master.py.
"""

import ee
import pandas as pd
import time
from pathlib import Path
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

# ─── CONFIG ───────────────────────────────────────────────────────────────────
START_DATE = "2023-01-01"
END_DATE   = "2026-06-30"
PROJECT    = "major-project-495508"

# All 36 Maharashtra districts (same list as sentinel_indices.py)
DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
]

# FAO GAUL uses slightly different spellings for some districts
GAUL_NAME_MAP = {
    "Mumbai City":     ["Mumbai city", "Mumbai", "Mumbai City"],
    "Mumbai Suburban": ["Mumbai Suburban", "Mumbai (Suburban)"],
    "Osmanabad":       ["Osmanabad", "Usmanabad"],
    "Buldhana":        ["Buldhana", "Buldana"],
    "Ahmednagar":      ["Ahmednagar", "Ahmadnagar"],
    "Beed":            ["Beed", "Bid"],
    "Gondia":          ["Gondia", "Gondiya"],
    "Gadchiroli":      ["Gadchiroli", "Garhchiroli"],  # GAUL spelling (verified via ADM2_NAME listing)
    "Raigad":          ["Raigad", "Raigarh"],
    "Palghar":         ["Palghar", "Thane"],   # Palghar split from Thane in 2014, may not exist in GAUL 2015
}

OUTPUT = Path("../data/raw/rainfall_chirps_maharashtra_23_25.csv")
PROGRESS_F = Path("progress/chirps_progress.txt")
OUTPUT.parent.mkdir(exist_ok=True)
PROGRESS_F.parent.mkdir(exist_ok=True)

# ─── GEE INIT ─────────────────────────────────────────────────────────────────
print("Initialising Earth Engine ...")
ee.Initialize(project=PROJECT)
ee.data.setDeadline(180000)   # 3-minute timeout per call

india = ee.FeatureCollection("FAO/GAUL/2015/level2")
mh    = india.filter(ee.Filter.eq("ADM1_NAME", "Maharashtra"))


def get_district_geometry(district_name: str):
    """Return GEE geometry, trying alternate GAUL spellings."""
    variants = GAUL_NAME_MAP.get(district_name, [district_name])
    for variant in variants:
        match = mh.filter(ee.Filter.eq("ADM2_NAME", variant))
        if match.size().getInfo() > 0:
            return match.geometry(), variant
    return None, None


def fetch_chirps_year(geometry, district_name: str, year: int):
    """One server-side reduction per year per district — fastest pattern."""
    y_start = f"{year}-01-01"
    y_end   = f"{year + 1}-01-01"
    # Clip to overall range
    y_start = max(y_start, START_DATE)
    y_end   = min(y_end,   (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    if y_start >= y_end:
        return []

    chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterDate(y_start, y_end)
              .filterBounds(geometry))

    def reduce_image(img):
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        mean = img.reduceRegion(
            reducer  = ee.Reducer.mean(),
            geometry = geometry,
            scale    = 5500,           # ~CHIRPS native resolution
            maxPixels= 1e9,
        ).get("precipitation")
        return ee.Feature(None, {"date": date, "rainfall_mm": mean})

    features = chirps.map(reduce_image).getInfo()["features"]
    rows = []
    for f in features:
        p = f["properties"]
        if p.get("rainfall_mm") is not None:
            rows.append({
                "district":    district_name,
                "date":        p["date"],
                "rainfall_mm": round(float(p["rainfall_mm"]), 3),
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
print(f"Resuming from progress file — {len(done)} (district, year) pairs already done.\n")

# Write header if file doesn't exist
write_header = not OUTPUT.exists() or OUTPUT.stat().st_size == 0
if write_header:
    pd.DataFrame(columns=["district","date","rainfall_mm"]).to_csv(OUTPUT, index=False)

total = len(DISTRICTS) * 3   # 3 years per district
done_count = 0
skipped = []

for district in DISTRICTS:
    geom, matched_name = get_district_geometry(district)
    if geom is None:
        print(f"  {district:<20} -> NOT in FAO GAUL, skipping")
        skipped.append(district)
        continue

    note = f" (matched '{matched_name}')" if matched_name != district else ""
    print(f"  {district:<20}{note}")

    t0 = time.time()
    for year in [2023, 2024, 2025, 2026]:
        key = f"{district}|{year}"
        if key in done:
            print(f"    {year}: cached, skipping")
            done_count += 1
            continue
        try:
            rows = fetch_chirps_year(geom, district, year)
            if rows:
                pd.DataFrame(rows).to_csv(OUTPUT, mode="a", header=False, index=False)
            mark_done(key)
            done_count += 1
            print(f"    {year}: {len(rows):>3} days   ({time.time()-t0:.1f}s elapsed, {done_count}/{total} overall)")
        except Exception as e:
            print(f"    {year}: ERROR — {e}")

# ─── POST-PROCESS: build rolling totals and final clean file ─────────────────
print("\nBuilding rolling features and final cleaned file ...")
df = pd.read_csv(OUTPUT, parse_dates=["date"])
df = df.drop_duplicates(subset=["district","date"], keep="last")
df = df.sort_values(["district", "date"]).reset_index(drop=True)

# Reindex each district to full daily calendar (CHIRPS sometimes drops occasional days)
full_idx = pd.date_range(START_DATE, END_DATE, freq="D")
parts = []
for d, grp in df.groupby("district"):
    grp = grp.set_index("date").reindex(full_idx)
    grp["district"]    = d
    grp["rainfall_mm"] = grp["rainfall_mm"].fillna(0.0)
    parts.append(grp.reset_index().rename(columns={"index": "date"}))
df = pd.concat(parts, ignore_index=True)

# Rolling 7d / 30d totals (same calc as preprocess_rainfall.py)
df = df.sort_values(["district","date"]).reset_index(drop=True)
df["rainfall_7d"]  = df.groupby("district")["rainfall_mm"].transform(
    lambda s: s.rolling(7,  min_periods=1).sum())
df["rainfall_30d"] = df.groupby("district")["rainfall_mm"].transform(
    lambda s: s.rolling(30, min_periods=1).sum())
df["station_count"] = 0   # gridded source, not station-based

out_cols = ["district","date","rainfall_mm","rainfall_7d","rainfall_30d","station_count"]
df[out_cols].to_csv(OUTPUT, index=False)

print(f"\nSaved -> {OUTPUT}")
print(f"  Rows      : {len(df):,}")
print(f"  Districts : {df['district'].nunique()}")
print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
if skipped:
    print(f"  Skipped   : {skipped}  (not in FAO GAUL — may need OSM boundaries)")

print(f"\nNext steps:")
print(f"  1. Copy {OUTPUT} -> ../data/clean/rainfall_clean.csv")
print(f"  2. Re-run: python ../02_preprocessing/merge_master.py")
