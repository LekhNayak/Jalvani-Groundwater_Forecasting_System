"""
Extracts all unique groundwater station locations from the raw groundwater CSV
and saves as a GeoJSON FeatureCollection.

Each feature contains:
    stationCode, stationName, district, tehsil, block, village,
    wellType, wellDepth, wellAquiferType, majorBasin, tributary,
    agencyName, dataAcquisitionMode, stationStatus

Output : data/clean/groundwater_stations.geojson
"""

import sys
import warnings
import pandas as pd
import json
from pathlib import Path
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

INPUT  = Path("../data/raw/groundwater_maharashtra_23_25_uncleaned.csv")
OUTPUT = Path("../data/clean/groundwater_stations.geojson")
Path("../data/clean").mkdir(exist_ok=True)

# ─── LOAD & DEDUPLICATE STATIONS ──────────────────────────────────────────────
print("Loading groundwater data ...")
df = pd.read_csv(INPUT, low_memory=False)

print(f"  Raw rows        : {len(df):,}")
print(f"  Unique stations : {df['stationCode'].nunique()}")

# Keep one row per station — all metadata columns are the same across time rows
STATION_COLS = [
    "stationCode", "stationName", "stationType",
    "latitude", "longitude",
    "agencyName", "state", "district", "tehsil",
    "block", "village",
    "wellType", "wellDepth", "wellAquiferType",
    "majorBasin", "tributary",
    "dataAcquisitionMode", "stationStatus",
]

# Only keep columns that actually exist in the file
STATION_COLS = [c for c in STATION_COLS if c in df.columns]

stations = df[STATION_COLS].drop_duplicates(subset=["stationCode"]).copy()

# Drop stations with no coordinates
before = len(stations)
stations = stations.dropna(subset=["latitude", "longitude"])
print(f"  Dropped {before - len(stations)} stations with missing coordinates")
print(f"  Stations with coordinates : {len(stations)}")

# ─── BUILD GEOJSON ────────────────────────────────────────────────────────────
def make_feature(row: pd.Series) -> dict:
    props = {}
    for col in STATION_COLS:
        if col in ("latitude", "longitude"):
            continue
        val = row.get(col)
        # Convert NaN to None for valid JSON
        props[col] = None if (isinstance(val, float) and pd.isna(val)) else val

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [row["longitude"], row["latitude"]],  # GeoJSON is [lon, lat]
        },
        "properties": props,
    }

features = [make_feature(row) for _, row in stations.iterrows()]

geojson = {
    "type": "FeatureCollection",
    "crs": {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
    },
    "features": features,
}

# ─── SAVE ─────────────────────────────────────────────────────────────────────
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(geojson, f, indent=2, ensure_ascii=False)

print(f"\nSaved -> {OUTPUT}")
print(f"  Total features : {len(features)}")
print(f"\nSample station:")
if features:
    s = features[0]
    print(f"  Code      : {s['properties']['stationCode']}")
    print(f"  Name      : {s['properties']['stationName']}")
    print(f"  District  : {s['properties']['district']}")
    print(f"  Tehsil    : {s['properties'].get('tehsil')}")
    print(f"  Aquifer   : {s['properties'].get('wellAquiferType')}")
    print(f"  Coords    : {s['geometry']['coordinates']}")
