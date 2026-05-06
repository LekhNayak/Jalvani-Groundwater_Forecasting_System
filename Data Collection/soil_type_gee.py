"""
Extracts district-level soil texture (sand %, clay %, silt %) for all
Maharashtra districts using OpenLandMap soil data in Google Earth Engine.

Why OpenLandMap instead of ISRIC REST API:
  The ISRIC SoilGrids REST API returns null for most inland Maharashtra
  districts (Deccan Plateau coverage gap). OpenLandMap in GEE uses the
  same underlying SoilGrids data but with full global raster coverage —
  no null values for Maharashtra.

Data source : OpenLandMap/SOL — publicly available in GEE, no extra permissions.
Resolution  : 250m, aggregated to district mean.
Depths      : 0cm, 10cm, 30cm (surface + root zone).

Output:
    datasets/soil_texture_maharashtra.csv

Columns:
    district,
    sand_0cm,  clay_0cm,  silt_0cm,
    sand_10cm, clay_10cm, silt_10cm,
    sand_30cm, clay_30cm, silt_30cm

All values in % (0–100). Static — run once, reuse for all model training.

Setup (same as sentinel_indices.py):
    earthengine authenticate
    earthengine set_project <your-project-id>
"""

import ee
import pandas as pd
from pathlib import Path

ee.Initialize()

Path("datasets").mkdir(exist_ok=True)
OUTPUT = Path("datasets/soil_texture_maharashtra.csv")

DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
]

# ─── DISTRICT BOUNDARIES ─────────────────────────────────────────────────────
india_districts = ee.FeatureCollection("FAO/GAUL/2015/level2")
maharashtra     = india_districts.filter(ee.Filter.eq("ADM1_NAME", "Maharashtra"))

def get_geometry(district_name: str):
    return maharashtra.filter(ee.Filter.eq("ADM2_NAME", district_name)).geometry()


# ─── OPENLANDMAP SOIL DATASETS ────────────────────────────────────────────────
# Values already in % (0–100). Band names = depth in cm: b0, b10, b30, b60, b100, b200.
SAND_ASSET = "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"
CLAY_ASSET = "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
# Silt is not a separate asset — compute as: silt = 100 - sand - clay

DEPTH_BANDS = {
    "0cm":  "b0",
    "10cm": "b10",
    "30cm": "b30",
}


def build_soil_image() -> tuple[ee.Image, list[str]]:
    """
    Builds a single multi-band image with sand, clay, silt at 3 depths.
    Silt = 100 - sand - clay (derived, not a separate asset).
    Returns (image, list_of_band_names).
    """
    sand_img = ee.Image(SAND_ASSET)
    clay_img = ee.Image(CLAY_ASSET)

    bands      = []
    band_names = []

    for depth_label, band_id in DEPTH_BANDS.items():
        sand = sand_img.select(band_id).rename(f"sand_{depth_label}")
        clay = clay_img.select(band_id).rename(f"clay_{depth_label}")
        silt = sand.add(clay).multiply(-1).add(100).rename(f"silt_{depth_label}")

        bands.extend([sand, clay, silt])
        band_names.extend([
            f"sand_{depth_label}",
            f"clay_{depth_label}",
            f"silt_{depth_label}",
        ])

    return ee.Image(bands), band_names


def extract_district(district_name: str, soil_image: ee.Image, band_names: list) -> dict | None:
    geometry = get_geometry(district_name)

    stats = soil_image.reduceRegion(
        reducer  = ee.Reducer.mean(),
        geometry = geometry,
        scale    = 250,
        maxPixels= 1e9,
    ).getInfo()

    if not stats:
        print(f"  [{district_name}] No stats returned from GEE")
        return None

    row = {"district": district_name}
    for band in band_names:
        val = stats.get(band)
        row[band] = round(val, 2) if val is not None else None

    return row


def main():
    print("Building OpenLandMap soil image ...")
    soil_image, band_names = build_soil_image()
    print(f"  Bands: {band_names}\n")

    rows = []

    for i, district in enumerate(DISTRICTS, 1):
        print(f"[{i}/{len(DISTRICTS)}] {district} ...")
        row = extract_district(district, soil_image, band_names)

        if row:
            s  = row.get("sand_0cm", "?")
            c  = row.get("clay_0cm", "?")
            sl = row.get("silt_0cm", "?")

            # Sanity check: values should sum to ~100
            if all(v is not None for v in [s, c, sl]):
                total = s + c + sl
                status = "OK" if 95 <= total <= 105 else f"SUM={total:.1f} CHECK"
            else:
                status = "MISSING"

            print(f"  0cm → sand={s}%  clay={c}%  silt={sl}%  [{status}]")
            rows.append(row)
        else:
            print(f"  [{district}] FAILED")

    df = pd.DataFrame(rows)

    print(f"\nNaN summary:")
    print(df.isnull().sum().to_string())

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved → {OUTPUT}")
    print(f"Districts with data : {df['sand_0cm'].notna().sum()} / {len(df)}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
