"""
Universal WRIS dataset collector.
Usage:
    python collect_dataset.py --dataset Temperature
    python collect_dataset.py --dataset "Relative Humidity"
    python collect_dataset.py --dataset "Soil Moisture"
    python collect_dataset.py --dataset "River Water Level"
    python collect_dataset.py --dataset Reservoir
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
BASE      = "https://indiawris.gov.in/Dataset"
STATE     = "Maharashtra"
AGENCY    = "CWC"
START     = "2023-01-01"
END       = "2025-12-31"
PAGE_SIZE = 1000

DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
]

# Columns we always want if the API returns them
CORE_FIELDS = [
    "stationCode", "stationName", "stationType",
    "latitude", "longitude",
    "agencyName", "state", "district",
    "majorBasin", "tributary",
    "dataAcquisitionMode", "stationStatus",
    "tehsil", "datatypeCode", "description",
    "dataValue", "dataTime", "unit",
]

HEADERS = {"accept": "application/json"}

# ─── LOGGING ──────────────────────────────────────────────────────────────────
def _setup_logging(log_path: Path) -> logging.Logger:
    Path("../logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


# ─── FETCH ────────────────────────────────────────────────────────────────────
def fetch_page(url: str, district: str, page: int, log) -> list:
    params = {
        "stateName":    STATE,
        "districtName": district,
        "agencyName":   AGENCY,
        "startdate":    START,
        "enddate":      END,
        "download":     "false",
        "page":         page,
        "size":         PAGE_SIZE,
    }
    for attempt in range(5):
        try:
            r = requests.post(url, params=params, headers=HEADERS, timeout=30)
            body = r.json()
            status = body.get("statusCode")
            records = body.get("data", [])
            message = body.get("message", "")

            if r.status_code == 200 and status == 200:
                return records
            if "No data found" in message:
                return []
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning(f"Rate limited. Sleeping {wait}s ...")
                time.sleep(wait)
            else:
                log.error(f"[{district}] p{page} HTTP {r.status_code} | API {status} | {message}")
                return []
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"[{district}] attempt {attempt+1} failed: {e}. retry in {wait}s")
            time.sleep(wait)
    log.error(f"[{district}] all retries exhausted on page {page}")
    return []


def fetch_district(url: str, district: str, log) -> list:
    all_records, page = [], 0
    while True:
        records = fetch_page(url, district, page, log)
        if not records:
            break
        all_records.extend(records)
        log.info(f"  [{district}] p{page:3d} +{len(records):,} | total {len(all_records):,}")
        if len(records) < PAGE_SIZE:
            break
        page += 1
        time.sleep(0.5)
    return all_records


# ─── IO ───────────────────────────────────────────────────────────────────────
def append_csv(records: list, csv_path: Path, log):
    df = pd.DataFrame(records)
    cols = [c for c in CORE_FIELDS if c in df.columns]
    missing = [c for c in CORE_FIELDS if c not in df.columns]
    if missing:
        log.warning(f"  API didn't return these fields (skipped): {missing}")
    df = df[cols]
    df["dataTime"] = pd.to_datetime(df["dataTime"])
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def load_done(progress_path: Path) -> set:
    if progress_path.exists():
        return set(progress_path.read_text(encoding="utf-8").splitlines())
    return set()


def mark_done(district: str, progress_path: Path):
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(district + "\n")


# ─── SUMMARY ──────────────────────────────────────────────────────────────────
def print_summary(csv_path: Path, total_fetched: int, log):
    log.info("=" * 60)
    log.info(f"Done. New records fetched : {total_fetched:,}")
    if not csv_path.exists():
        log.warning("CSV not found — nothing written.")
        return
    df = pd.read_csv(csv_path)
    log.info(f"CSV rows       : {len(df):,}")
    log.info(f"Stations       : {df['stationCode'].nunique() if 'stationCode' in df.columns else 'n/a'}")
    log.info(f"Districts      : {df['district'].nunique() if 'district' in df.columns else 'n/a'}")
    log.info(f"Date range     : {df['dataTime'].min()} → {df['dataTime'].max()}")

    before = len(df)
    key = ["stationCode", "dataTime"] if "stationCode" in df.columns else ["district", "dataTime"]
    df = df.drop_duplicates(subset=key, keep="last")
    if len(df) < before:
        df.to_csv(csv_path, index=False)
        log.info(f"Dedup removed  : {before - len(df):,} rows")
    else:
        log.info("Dedup          : none found [OK]")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Dataset name as it appears in the API path, e.g. 'Temperature'")
    args = parser.parse_args()

    name = args.dataset
    slug = name.lower().replace(" ", "_")

    Path("../../data/raw").mkdir(exist_ok=True)
    Path("../progress").mkdir(exist_ok=True)

    url          = f"{BASE}/{name}"
    csv_path     = Path(f"../../data/raw/{slug}_maharashtra_23_25_uncleaned.csv")
    progress_path = Path(f"../progress/{slug}_progress.txt")
    log_path     = Path(f"../logs/{slug}_fetch.log")

    log = _setup_logging(log_path)

    done      = load_done(progress_path)
    remaining = [d for d in DISTRICTS if d not in done]

    log.info(f"Dataset        : {name}")
    log.info(f"URL            : {url}")
    log.info(f"Agency         : {AGENCY}")
    log.info(f"Date range     : {START} → {END}")
    log.info(f"Districts done : {len(done)} / {len(DISTRICTS)}")
    log.info(f"Remaining      : {len(remaining)}\n")

    total_fetched = 0

    for i, district in enumerate(remaining, 1):
        log.info(f"[{i}/{len(remaining)}] {district} ...")
        records = fetch_district(url, district, log)
        if records:
            append_csv(records, csv_path, log)
            mark_done(district, progress_path)
            total_fetched += len(records)
            log.info(f"  [{district}] done — {len(records):,} records\n")
        else:
            log.warning(f"  [{district}] no data returned (not marking done)\n")
        time.sleep(1)

    print_summary(csv_path, total_fetched, log)


if __name__ == "__main__":
    main()
