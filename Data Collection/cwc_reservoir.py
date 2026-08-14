"""
Collects daily reservoir storage for the top Maharashtra dams (2023-2025).

Data source: data.gov.in Open Government Data Platform
  - Dataset: "Daily Data on Reservoir Level - Central Water Commission (CWC)"
  - URL    : https://data.gov.in/resource/daily-data-reservoir-level-central-water-commission-cwc
  - API key: FREE — register at https://data.gov.in/user/register, then go to
             My Account → API Keys → Generate Key

Setup (one-time):
    1. Register at https://data.gov.in/user/register
    2. Generate an API key under My Account → API Keys
    3. Either:
       a. Set env variable:  set DATA_GOV_IN_API_KEY=your_key_here   (Windows)
       b. Or paste key into  API_KEY = "..."  in the CONFIG section below

Run:
    python cwc_reservoir.py

Output:
    datasets/reservoir_storage_maharashtra_23_25.csv

Schema:
    date, dam_name, district, storage_mcm, storage_pct, storage_bcm

Notes:
  - storage_mcm: live storage in Million Cubic Metres
  - storage_pct: % of total capacity
  - storage_bcm: live storage in Billion Cubic Metres (= storage_mcm / 1000)
  - Missing days are forward-filled per dam (storage doesn't change overnight)
  - Weekly-resolution gaps (CWC publishes Thursdays) are forward-filled to daily
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
API_KEY = os.getenv("DATA_GOV_IN_API_KEY", "")   # set env var or paste here

# data.gov.in resource ID for CWC daily reservoir data
# Find it at: https://data.gov.in/resource/daily-data-reservoir-level-central-water-commission-cwc
# The ID is in the URL after /resource/ — looks like: 6176d65e-9d70-4ce3-9b77-...
RESOURCE_ID = os.getenv("DATA_GOV_IN_RESOURCE_ID",
                        "6176d65e-9d70-4ce3-9b77-7f27d21cbfa7")  # verify this on the site

BASE_URL    = "https://api.data.gov.in/resource"
PAGE_SIZE   = 500       # max records per request
SLEEP_S     = 0.4       # be polite — data.gov.in has rate limits
START_DATE  = date(2023, 1, 1)
END_DATE    = date(2025, 4, 22)

OUTPUT      = Path("datasets/reservoir_storage_maharashtra_23_25.csv")
PROGRESS_F  = Path("txt files/reservoir_progress.txt")
Path("datasets").mkdir(exist_ok=True)
Path("txt files").mkdir(exist_ok=True)

# ─── TOP MAHARASHTRA DAMS → DISTRICT MAPPING ──────────────────────────────────
# Covers the districts most affected by irrigation drawdown.
# dam_name must match the "Project_Name" or "Dam_Name" field returned by the API
# (run with --list-dams to see actual names from the API).
DAM_DISTRICT = {
    # Godavari basin
    "Jayakwadi"          : "Aurangabad",
    "Nathsagar"          : "Aurangabad",    # alternate name for Jayakwadi
    "Yeldari"            : "Parbhani",
    "Siddheshwar"        : "Hingoli",
    "Manjara"            : "Latur",
    "Vishnupuri"         : "Nanded",
    # Krishna / Bhima basin
    "Ujjani"             : "Solapur",
    "Koyna"              : "Satara",
    "Dhom"               : "Satara",
    "Warna"              : "Kolhapur",
    # Tapi basin
    "Upper Tapi"         : "Dhule",
    "Hatnur"             : "Jalgaon",
    "Girna"              : "Nashik",
    "Gangapur"           : "Nashik",
    # Vidarbha / east
    "Irai"               : "Chandrapur",
    "Pench (Totladoh)"   : "Nagpur",
    "Totladoh"           : "Nagpur",
    "Upper Wardha"       : "Amravati",
    "Lower Wardha"       : "Wardha",
    # Ahmednagar
    "Bhandardara"        : "Ahmednagar",
    "Mula"               : "Ahmednagar",
    "Nilwande"           : "Ahmednagar",
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def load_done() -> set:
    return set(PROGRESS_F.read_text().splitlines()) if PROGRESS_F.exists() else set()

def mark_done(key: str):
    with open(PROGRESS_F, "a") as f:
        f.write(key + "\n")

def progress_key(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def fetch_month(year: int, month: int) -> pd.DataFrame:
    """Fetch all CWC reservoir records for one month from data.gov.in."""
    start_str = f"{year}-{month:02d}-01"
    # last day of month
    if month == 12:
        end_str = f"{year}-12-31"
    else:
        import calendar
        last = calendar.monthrange(year, month)[1]
        end_str = f"{year}-{month:02d}-{last:02d}"

    rows = []
    offset = 0
    while True:
        params = {
            "api-key"  : API_KEY,
            "format"   : "json",
            "limit"    : PAGE_SIZE,
            "offset"   : offset,
            "filters[State_Name]" : "MAHARASHTRA",
            # date field name may vary — the API supports filter on column names
            # Common alternatives: "Date_of_Observation", "date", "obs_date"
        }
        # Add date range filter — field name may differ; try both common formats
        params["filters[Date_of_Observation][gte]"] = start_str
        params["filters[Date_of_Observation][lte]"] = end_str

        try:
            resp = requests.get(f"{BASE_URL}/{RESOURCE_ID}", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            tqdm.write(f"  API error for {year}-{month:02d}: {e}")
            return pd.DataFrame()

        records = data.get("records", [])
        if not records:
            break
        rows.extend(records)
        if len(records) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(SLEEP_S)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def normalise_df(raw: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Normalise raw API response to our schema.
    Column names from data.gov.in CWC dataset (verify against actual response):
      Date_of_Observation, State_Name, Project_Name, Live_Storage_Today_(MCM),
      Total_Live_Storage_Capacity_(MCM), Percentage_of_Capacity_Today
    These may differ — update the mapping below if needed.
    """
    if raw.empty:
        return pd.DataFrame()

    # Print available columns once (first call) to help debugging
    if not hasattr(normalise_df, "_cols_printed"):
        tqdm.write(f"  API columns: {list(raw.columns)}")
        normalise_df._cols_printed = True

    # Flexible column mapping — try multiple likely names
    col_map = {
        "date"        : ["Date_of_Observation", "date", "obs_date", "Date"],
        "dam_name"    : ["Project_Name", "Dam_Name", "project_name", "Reservoir_Name"],
        "storage_mcm" : ["Live_Storage_Today_(MCM)", "Live_Storage_MCM",
                         "live_storage_today_mcm", "Storage_MCM"],
        "capacity_mcm": ["Total_Live_Storage_Capacity_(MCM)", "Total_Capacity_MCM",
                         "total_live_storage_capacity_mcm"],
        "storage_pct" : ["Percentage_of_Capacity_Today", "Storage_Percentage",
                         "percentage_of_capacity_today"],
    }

    rename = {}
    for target, candidates in col_map.items():
        for c in candidates:
            if c in raw.columns:
                rename[c] = target
                break

    df = raw.rename(columns=rename)

    # Keep only columns we managed to map
    keep = [c for c in ["date", "dam_name", "storage_mcm", "capacity_mcm", "storage_pct"]
            if c in df.columns]
    if "date" not in keep or "dam_name" not in keep:
        tqdm.write(f"  WARNING: could not find date/dam_name columns in API response")
        tqdm.write(f"  Available: {list(raw.columns)}")
        return pd.DataFrame()

    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Normalise dam names: strip whitespace, title-case
    df["dam_name"] = df["dam_name"].str.strip().str.title()

    # Map to district
    df["district"] = df["dam_name"].map(
        {k.title(): v for k, v in DAM_DISTRICT.items()}
    )
    # Only keep dams we have a district mapping for
    df = df[df["district"].notna()].copy()

    if "storage_mcm" in df.columns:
        df["storage_mcm"] = pd.to_numeric(df["storage_mcm"], errors="coerce")
        df["storage_bcm"] = (df["storage_mcm"] / 1000).round(4)
    if "storage_pct" not in df.columns and "storage_mcm" in df.columns and "capacity_mcm" in df.columns:
        df["capacity_mcm"] = pd.to_numeric(df["capacity_mcm"], errors="coerce")
        df["storage_pct"] = (df["storage_mcm"] / df["capacity_mcm"] * 100).round(2)

    out_cols = [c for c in ["date", "dam_name", "district", "storage_mcm", "storage_pct", "storage_bcm"]
                if c in df.columns]
    return df[out_cols].sort_values(["dam_name", "date"]).reset_index(drop=True)


def list_maharashtra_dams():
    """Helper: print all Maharashtra dam names returned by the API (use to update DAM_DISTRICT)."""
    print("Fetching Maharashtra dam list from API ...")
    raw = fetch_month(2024, 6)
    if raw.empty:
        print("No data returned — check API key and RESOURCE_ID.")
        return
    name_cols = [c for c in raw.columns if "name" in c.lower() or "project" in c.lower()]
    for col in name_cols:
        print(f"\n{col} values:")
        for v in sorted(raw[col].unique()):
            print(f"  {v}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if "--list-dams" in sys.argv:
        list_maharashtra_dams()
        return

    if not API_KEY:
        print("ERROR: No API key set.")
        print("  1. Register free at https://data.gov.in/user/register")
        print("  2. Generate key at My Account → API Keys")
        print("  3. Run:  set DATA_GOV_IN_API_KEY=your_key_here")
        print("     then: python cwc_reservoir.py")
        sys.exit(1)

    done = load_done()

    # Build month list
    months = []
    d = START_DATE.replace(day=1)
    while d <= END_DATE:
        months.append((d.year, d.month))
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1)
        else:
            d = d.replace(month=d.month + 1)

    todo = [(y, m) for y, m in months if progress_key(y, m) not in done]
    print(f"Months total   : {len(months)}")
    print(f"Already done   : {len(done)}")
    print(f"To fetch       : {len(todo)}")
    print(f"Dams tracked   : {len(DAM_DISTRICT)}")
    print()

    all_rows = []

    for year, month in tqdm(todo, desc="Fetching reservoir data", unit="month"):
        raw = fetch_month(year, month)
        if raw.empty:
            tqdm.write(f"  {year}-{month:02d}: no data")
            mark_done(progress_key(year, month))
            time.sleep(SLEEP_S)
            continue

        df = normalise_df(raw, year, month)
        if not df.empty:
            all_rows.append(df)
            tqdm.write(f"  {year}-{month:02d}: {len(df)} rows, "
                       f"{df['dam_name'].nunique()} dams, "
                       f"{df['district'].nunique()} districts")
            df.to_csv(OUTPUT, mode="a", header=not OUTPUT.exists(), index=False)

        mark_done(progress_key(year, month))
        time.sleep(SLEEP_S)

    # ── FORWARD-FILL GAPS TO DAILY ────────────────────────────────────────────
    if not OUTPUT.exists():
        print("\nNo data collected — check API key, RESOURCE_ID, and column names.")
        print("Run with --list-dams to inspect what the API returns.")
        return

    print("\nForward-filling gaps to daily resolution ...")
    combined = pd.read_csv(OUTPUT, parse_dates=["date"])
    combined = combined.drop_duplicates(["date", "dam_name"])

    full_idx = pd.date_range(START_DATE, END_DATE, freq="D")
    filled = []
    for dam, grp in combined.groupby("dam_name"):
        grp = grp.set_index("date").reindex(full_idx)
        grp["dam_name"] = dam
        grp["district"] = grp["district"].ffill().bfill()
        for col in ["storage_mcm", "storage_pct", "storage_bcm"]:
            if col in grp.columns:
                grp[col] = grp[col].ffill(limit=10)   # don't fill more than 10 days
        filled.append(grp.reset_index().rename(columns={"index": "date"}))

    final = pd.concat(filled, ignore_index=True)
    final["date"] = pd.to_datetime(final["date"])
    final = final.sort_values(["district", "dam_name", "date"]).reset_index(drop=True)
    final.to_csv(OUTPUT, index=False)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\nSaved -> {OUTPUT}")
    print(f"  Rows      : {len(final):,}")
    print(f"  Dams      : {final['dam_name'].nunique()}")
    print(f"  Districts : {final['district'].nunique()}")
    print(f"  Date range: {final['date'].min().date()} → {final['date'].max().date()}")
    print()
    print("Storage summary by district (mean % capacity):")
    print(
        final.groupby("district")["storage_pct"]
        .mean().round(1).sort_values()
        .to_string()
    )
    print()
    print("Next step: update merge_master.py to join reservoir storage on (district, date)")


if __name__ == "__main__":
    main()
