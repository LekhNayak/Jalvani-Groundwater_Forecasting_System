"""
Collects monthly district-level NDVI, NDWI, NDMI from Sentinel-2
and VV/VH backscatter from Sentinel-1 for all Maharashtra districts (2023-2025).

Setup (one-time):
    pip install earthengine-api geemap
    earthengine authenticate          # opens browser, log in with Google account
    earthengine set_project <your-project-id>

Run:
    python sentinel_indices.py

Output:
    datasets/sentinel2_indices_maharashtra_23_25.csv   — NDVI, NDWI, NDMI
    datasets/sentinel1_backscatter_maharashtra_23_25.csv — VV, VH (soil moisture proxy)

Both files are at (district, year, month) granularity — ready to merge with
your groundwater/rainfall data on (district, date) after resampling to monthly.
"""

import ee
import pandas as pd
import time
from pathlib import Path
from datetime import date
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
START_YEAR  = 2023
END_YEAR    = 2026
END_DATE    = date(2026, 6, 30)   # explicit cutoff — do not rely on date.today(),
                                   # which would include a partial/incomplete current month
CLOUD_COVER = 20          # skip Sentinel-2 scenes with >20% cloud cover
SCALE       = 500         # spatial resolution for aggregation in metres
                          # 500m is fine for district-level means (much faster than 10m)

DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
]

Path("datasets").mkdir(exist_ok=True)

# ─── GEE INIT ─────────────────────────────────────────────────────────────────
ee.Initialize(project="major-project-495508")
ee.data.setDeadline(60000)   # 60-second timeout per GEE call — prevents silent hangs

# ─── MAHARASHTRA DISTRICT BOUNDARIES (from GEE public asset) ──────────────────
# FAO GAUL level-2 boundaries — contains all Indian districts with ADM2_NAME field.
# GAUL uses older / alternative spellings for several Maharashtra districts.
# This map converts our canonical project names → actual GAUL ADM2_NAME strings.
GAUL_NAME_MAP = {
    "Beed":            "Bid",         # GAUL uses pre-2006 spelling
    "Buldhana":        "Buldana",     # GAUL alternate spelling
    "Gadchiroli":      "Garhchiroli", # GAUL spelling (verified via ADM2_NAME listing)
    "Gondia":          "Gondiya",     # GAUL uses 'iya' suffix
    "Raigad":          "Raigarh",     # GAUL uses old name
    "Palghar":         "Palghar",     # new district (carved from Thane 2014); may be absent
    "Mumbai City":     None,          # intentionally skipped — no significant aquifer
    "Mumbai Suburban": "Mumbai Suburban",
}

india_districts = ee.FeatureCollection("FAO/GAUL/2015/level2")
maharashtra     = india_districts.filter(ee.Filter.eq("ADM1_NAME", "Maharashtra"))


def list_gaul_names():
    """Print all ADM2_NAME values GEE has for Maharashtra — useful for debugging."""
    names = maharashtra.aggregate_array("ADM2_NAME").getInfo()
    print("GAUL ADM2_NAME values for Maharashtra:")
    for n in sorted(names):
        print(f"  {n}")


def get_district_geometry(district_name: str):
    """Returns the GEE geometry for a single Maharashtra district.
    Applies GAUL_NAME_MAP translation; returns None for intentionally skipped districts.
    """
    gaul_name = GAUL_NAME_MAP.get(district_name, district_name)
    if gaul_name is None:
        return None  # intentionally skipped (e.g. Mumbai City — no aquifer)
    geom = maharashtra.filter(ee.Filter.eq("ADM2_NAME", gaul_name)).geometry()
    # Verify geometry is non-empty; if empty, the GAUL name is wrong — log it
    size = geom.coordinates().length().getInfo()
    if size == 0:
        tqdm.write(f"  WARNING: GAUL name '{gaul_name}' returned empty geometry for '{district_name}' — update GAUL_NAME_MAP")
        return None
    return geom


# ─── SENTINEL-2: NDVI, NDWI, NDMI ─────────────────────────────────────────────
def add_indices(image):
    """Add NDVI, NDWI, NDMI bands to a Sentinel-2 image."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")   # surface water
    ndmi = image.normalizedDifference(["B8A", "B11"]).rename("NDMI") # vegetation moisture
    return image.addBands([ndvi, ndwi, ndmi])


def _retry_getinfo(ee_obj, retries: int = 4, backoff: float = 5.0):
    """Call .getInfo() with exponential backoff on transient GEE errors."""
    for attempt in range(retries):
        try:
            return ee_obj.getInfo()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            tqdm.write(f"    GEE error (attempt {attempt+1}/{retries}): {e} — retrying in {wait:.0f}s")
            time.sleep(wait)


def get_s2_monthly_mean(district_name: str, year: int, month: int):
    """
    Returns mean NDVI, NDWI, NDMI for one district in one month.
    Returns None if the district is skipped or no cloud-free images are available.
    """
    geometry = get_district_geometry(district_name)
    if geometry is None:
        return None  # intentionally skipped district
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date   = start_date.advance(1, "month")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER))
        .map(add_indices)
    )

    count = _retry_getinfo(collection.size())
    if count == 0:
        return None

    composite = collection.select(["NDVI", "NDWI", "NDMI"]).mean()

    stats = _retry_getinfo(composite.reduceRegion(
        reducer  = ee.Reducer.mean(),
        geometry = geometry,
        scale    = SCALE,
        maxPixels= 1e9,
    ))

    return {
        "district":  district_name,
        "year":      year,
        "month":     month,
        "ndvi_mean": round(stats.get("NDVI") or float("nan"), 4),
        "ndwi_mean": round(stats.get("NDWI") or float("nan"), 4),
        "ndmi_mean": round(stats.get("NDMI") or float("nan"), 4),
        "s2_scene_count": count,
    }


# ─── SENTINEL-1: VV / VH BACKSCATTER (soil moisture proxy) ───────────────────
def get_s1_monthly_mean(district_name: str, year: int, month: int):
    """
    Returns mean VV and VH backscatter (dB) for one district in one month.
    Sentinel-1 works through clouds — important for monsoon months.
    Returns None if the district is skipped.
    """
    geometry = get_district_geometry(district_name)
    if geometry is None:
        return None  # intentionally skipped district
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date   = start_date.advance(1, "month")

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))          # Interferometric Wide swath
        .select(["VV", "VH"])
    )

    count = _retry_getinfo(collection.size())
    if count == 0:
        return None

    composite = collection.mean()

    stats = _retry_getinfo(composite.reduceRegion(
        reducer  = ee.Reducer.mean(),
        geometry = geometry,
        scale    = SCALE,
        maxPixels= 1e9,
    ))

    vv = stats.get("VV")
    vh = stats.get("VH")
    if vv is None and vh is None:
        return None

    return {
        "district":  district_name,
        "year":      year,
        "month":     month,
        "vv_db":     round(vv, 4) if vv is not None else float("nan"),
        "vh_db":     round(vh, 4) if vh is not None else float("nan"),
        "vh_vv_ratio": round(vh - vv, 4) if (vv is not None and vh is not None) else float("nan"),
        "s1_scene_count": count,
    }


# ─── PROGRESS TRACKING ────────────────────────────────────────────────────────
def load_done_s2() -> set:
    p = Path("txt files/sentinel2_progress.txt")
    return set(p.read_text().splitlines()) if p.exists() else set()

def mark_done_s2(key: str):
    with open("txt files/sentinel2_progress.txt", "a") as f:
        f.write(key + "\n")

def load_done_s1() -> set:
    p = Path("txt files/sentinel1_progress.txt")
    return set(p.read_text().splitlines()) if p.exists() else set()

def mark_done_s1(key: str):
    with open("txt files/sentinel1_progress.txt", "a") as f:
        f.write(key + "\n")

def progress_key(district: str, year: int, month: int) -> str:
    return f"{district}|{year}|{month:02d}"


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    Path("txt files").mkdir(exist_ok=True)

    s2_out = Path("datasets/sentinel2_indices_maharashtra_23_25.csv")
    s1_out = Path("datasets/sentinel1_backscatter_maharashtra_23_25.csv")

    # Uncomment to debug GAUL district names available for Maharashtra:
    # list_gaul_names()

    done_s2 = load_done_s2()
    done_s1 = load_done_s1()

    # Build full job list: all (district, year, month) combinations
    jobs = [
        (d, y, m)
        for d in DISTRICTS
        for y in range(START_YEAR, END_YEAR + 1)
        for m in range(1, 13)
        if date(y, m, 1) <= END_DATE   # skip months beyond the target cutoff
    ]

    total = len(jobs)
    print(f"Total (district × month) combinations : {total}")
    print(f"S2 already done : {len(done_s2)}")
    print(f"S1 already done : {len(done_s1)}\n")

    pbar = tqdm(jobs, desc="Sentinel collection", unit="job")
    for district, year, month in pbar:
        key   = progress_key(district, year, month)
        label = f"{district} {year}-{month:02d}"
        pbar.set_postfix_str(label)

        # ── Sentinel-2 ──────────────────────────────────────────────────────
        if key not in done_s2:
            try:
                row = get_s2_monthly_mean(district, year, month)
                if row:
                    df = pd.DataFrame([row])
                    df.to_csv(s2_out, mode="a", header=not s2_out.exists(), index=False)
                    tqdm.write(f"  S2 {label} | NDVI={row['ndvi_mean']:.3f} NDWI={row['ndwi_mean']:.3f} ({row['s2_scene_count']} scenes)")
                else:
                    tqdm.write(f"  S2 {label} | no cloud-free scenes (skipped)")
                mark_done_s2(key)
            except Exception as e:
                tqdm.write(f"  S2 {label} | ERROR: {e}")
        # ── Sentinel-1 ──────────────────────────────────────────────────────
        if key not in done_s1:
            try:
                row = get_s1_monthly_mean(district, year, month)
                if row:
                    df = pd.DataFrame([row])
                    df.to_csv(s1_out, mode="a", header=not s1_out.exists(), index=False)
                    tqdm.write(f"  S1 {label} | VV={row['vv_db']:.2f} dB  VH={row['vh_db']:.2f} dB ({row['s1_scene_count']} scenes)")
                else:
                    tqdm.write(f"  S1 {label} | no scenes (skipped)")
                mark_done_s1(key)
            except Exception as e:
                tqdm.write(f"  S1 {label} | ERROR: {e}")

    print("\nDone.")
    if s2_out.exists():
        df = pd.read_csv(s2_out)
        print(f"Sentinel-2 CSV : {len(df):,} rows, {df['district'].nunique()} districts")
    if s1_out.exists():
        df = pd.read_csv(s1_out)
        print(f"Sentinel-1 CSV : {len(df):,} rows, {df['district'].nunique()} districts")


if __name__ == "__main__":
    main()
