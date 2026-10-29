"""
Joins CGWB extraction data into master_dataset.csv as static yearly features.

Adds these columns (joined on district + year-of-date):
    annual_recharge_bcm        - natural input
    annual_extraction_bcm      - human pumping
    extractable_resource_bcm   - physically pumpable max
    stage_of_extraction_pct    - main stress indicator (>100 = depleting)
    aquifer_category_id        - 0=Safe, 1=Semi-Critical, 2=Critical, 3=Over-Exploited
                                  (integer, for static categorical embedding in TFT)

For years where CGWB hasn't published yet (e.g. 2025), the most recent
available year is used — standard practice in groundwater modelling.

Input:
    ../../data/raw/cgwb_extraction_maharashtra.csv
    data/clean/master_dataset.csv

Output:
    data/clean/master_dataset.csv  (overwritten, with new columns appended)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CGWB   = Path("../../data/raw/cgwb_extraction_maharashtra.csv")
MASTER = Path("../../data/clean/master_dataset.csv")

if not CGWB.exists():
    print(f"ERROR: {CGWB} not found.")
    print("Run `python ../01_data_collection/cgwb_extraction.py` first")
    print("(then edit the template CSV by hand from the PDFs).")
    sys.exit(1)

cgwb   = pd.read_csv(CGWB)
master = pd.read_csv(MASTER, parse_dates=["date"])

print(f"CGWB rows           : {len(cgwb):,}")
print(f"Master rows         : {len(master):,}")
print(f"CGWB districts      : {cgwb['district'].nunique()}")
print(f"CGWB years          : {sorted(cgwb['year'].unique())}")

# ─── CATEGORY → INTEGER ──────────────────────────────────────────────────────
CAT_MAP = {
    "Safe":            0,
    "Semi-Critical":   1,
    "Critical":        2,
    "Over-Exploited":  3,
    "Saline":          4,
}
if "category" in cgwb.columns:
    cgwb["aquifer_category_id"] = cgwb["category"].map(CAT_MAP).fillna(-1).astype(int)
else:
    cgwb["aquifer_category_id"] = -1

# ─── FILL MISSING YEARS PER DISTRICT (forward-fill, then back-fill) ──────────
# If CGWB has 2023 + 2024 but not 2025, use 2024 for 2025.
# If only 2024 exists for a district, use it for all three years.
yearly_cols = ["annual_recharge_bcm", "annual_extraction_bcm",
               "extractable_resource_bcm", "stage_of_extraction_pct",
               "aquifer_category_id"]
yearly_cols = [c for c in yearly_cols if c in cgwb.columns]

cgwb = cgwb.sort_values(["district", "year"])
all_years = pd.DataFrame(
    [(d, y) for d in cgwb["district"].unique() for y in [2023, 2024, 2025]],
    columns=["district", "year"],
)
cgwb_full = all_years.merge(cgwb, on=["district", "year"], how="left")
cgwb_full[yearly_cols] = (
    cgwb_full.groupby("district")[yearly_cols]
             .ffill()
             .bfill()
)

# ─── JOIN INTO MASTER ────────────────────────────────────────────────────────
master["year"] = master["date"].dt.year
before_cols = set(master.columns)

master = master.merge(
    cgwb_full[["district", "year"] + yearly_cols],
    on=["district", "year"],
    how="left",
)
master = master.drop(columns=["year"])

new_cols = sorted(set(master.columns) - before_cols)
print(f"\nNew columns added: {new_cols}")

# Report null rates per column
print("\nMissing values in new columns:")
for c in new_cols:
    n_null = master[c].isnull().sum()
    pct = n_null / len(master) * 100
    print(f"  {c:<30} {n_null:>6,} ({pct:.1f}%)")

# Fill any remaining NaN with global median (for districts CGWB has no data for)
for c in new_cols:
    if master[c].dtype.kind in "fi":
        master[c] = master[c].fillna(master[c].median())

# ─── SAVE ────────────────────────────────────────────────────────────────────
master.to_csv(MASTER, index=False)
print(f"\nSaved -> {MASTER}")
print(f"Final master shape: {len(master):,} rows × {len(master.columns)} cols")
print("\nNext step: re-upload master_dataset.csv to Colab and retrain")
print("Also update the TFT notebook PAST_CONT / STATIC_CONT lists to use new features:")
print("  Recommended placement:")
print("    STATIC_CONT += ['annual_recharge_bcm', 'annual_extraction_bcm',")
print("                    'extractable_resource_bcm', 'stage_of_extraction_pct']")
print("    STATIC_CAT  += ['aquifer_category_id']  (new embedding)")
