"""
Combines the three per-year CGWB CSV files (mh_gw_2023.csv, mh_gw_2024.csv,
mh_gw_2025.csv) into one canonical file that matches the schema expected by
../Data Preprocessing/merge_cgwb_into_master.py.

Input  (in datasets/):
    mh_gw_2023.csv
    mh_gw_2024.csv
    mh_gw_2025.csv

Output:
    datasets/cgwb_extraction_maharashtra.csv

Schema (matches merge_cgwb_into_master.py expectations):
    district, year,
    annual_recharge_bcm, extractable_resource_bcm, annual_extraction_bcm,
    stage_of_extraction_pct, category

Conversions:
    All CGWB numbers are in hectare-metres (Ham). 100,000 Ham = 1 BCM.
    Category derived from Stage % using official CGWB thresholds:
        ≤ 70    -> Safe
        70-90   -> Semi-Critical
        90-100  -> Critical
        > 100   -> Over-Exploited
"""

import sys
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASETS_DIR = Path("datasets")
OUTPUT       = DATASETS_DIR / "cgwb_extraction_maharashtra.csv"

YEAR_FILES = {
    2023: DATASETS_DIR / "mh_gw_2023.csv",
    2024: DATASETS_DIR / "mh_gw_2024.csv",
    2025: DATASETS_DIR / "mh_gw_2025.csv",
}

# Map CGWB district names to our canonical project names
NAME_FIX = {
    "Yawatmal":   "Yavatmal",
    "Mumbai":     "Mumbai City",
    "Mumbai Sub": "Mumbai Suburban",
}

def categorise(stage_pct: float) -> str:
    if pd.isna(stage_pct):
        return None
    if stage_pct <= 70:
        return "Safe"
    if stage_pct <= 90:
        return "Semi-Critical"
    if stage_pct <= 100:
        return "Critical"
    return "Over-Exploited"


all_rows = []
for year, path in YEAR_FILES.items():
    if not path.exists():
        print(f"WARNING: {path} not found, skipping {year}")
        continue

    df = pd.read_csv(path, encoding="utf-8-sig")  # strip BOM if present
    df.columns = [c.strip() for c in df.columns]

    # Drop summary rows (totals) which have blank Sl. No
    df = df[df["Name of District"].notna()].copy()
    df = df[~df["Name of District"].str.contains("Total", case=False, na=False)].copy()

    # Normalise district names
    df["district"] = df["Name of District"].str.strip().replace(NAME_FIX)

    # Convert Ham -> BCM
    df["annual_recharge_bcm"]      = df["Total annual groundwater recharge"]   / 100_000
    df["extractable_resource_bcm"] = df["Annual Extractable Groundwater Resource"] / 100_000
    df["annual_extraction_bcm"]    = df["Total Annual Extraction"]             / 100_000
    df["stage_of_extraction_pct"]  = df["Stage of GW extraction (%)"]
    df["category"]                 = df["stage_of_extraction_pct"].apply(categorise)
    df["year"]                     = year

    keep = ["district", "year", "annual_recharge_bcm", "extractable_resource_bcm",
            "annual_extraction_bcm", "stage_of_extraction_pct", "category"]
    all_rows.append(df[keep])
    print(f"  {year}: {len(df):>2} districts loaded from {path.name}")

if not all_rows:
    print("ERROR: No input files found.")
    sys.exit(1)

combined = pd.concat(all_rows, ignore_index=True)
combined = combined.sort_values(["district", "year"]).reset_index(drop=True)

# Round to sensible precision
for c in ["annual_recharge_bcm", "extractable_resource_bcm", "annual_extraction_bcm"]:
    combined[c] = combined[c].round(4)
combined["stage_of_extraction_pct"] = combined["stage_of_extraction_pct"].round(2)

combined.to_csv(OUTPUT, index=False)

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\nSaved -> {OUTPUT}")
print(f"  Total rows : {len(combined)}")
print(f"  Districts  : {combined['district'].nunique()}")
print(f"  Years      : {sorted(combined['year'].unique())}")
print()
print("Category distribution per year:")
print(combined.groupby(["year", "category"]).size().unstack(fill_value=0))
print()
print("Most stressed districts (mean stage % across years):")
top_stress = (combined.groupby("district")["stage_of_extraction_pct"]
              .mean().round(1).sort_values(ascending=False).head(10))
print(top_stress.to_string())
print()
print("Next step: python ../Data Preprocessing/merge_cgwb_into_master.py")
