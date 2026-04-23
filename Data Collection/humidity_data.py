import requests
import pandas as pd
import time
import logging
import urllib.parse
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://indiawris.gov.in/Dataset/" + urllib.parse.quote("Relative Humidity")
STATE      = "Maharashtra"
AGENCY     = "CWC"
START_DATE = "2023-01-01"
END_DATE   = "2025-12-31"
PAGE_SIZE  = 1000
OUTPUT_CSV = "datasets/humidity_maharashtra_23_25_uncleaned.csv"
PROGRESS_F = "txt files/humidity_progress.txt"
ISSUES_F   = "log/humidity_validation_issues.log"

DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"
]

EXPECTED_FIELDS = [
    "stationCode", "stationName", "stationType",
    "latitude", "longitude",
    "agencyName", "state", "district",
    "majorBasin", "tributary",
    "dataAcquisitionMode", "stationStatus",
    "tehsil", "datatypeCode", "description",
    "dataValue", "dataTime", "unit",
]

Path("log").mkdir(exist_ok=True)
Path("datasets").mkdir(exist_ok=True)
Path("txt files").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("log/humidity_fetch.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

HEADERS = {"accept": "application/json"}

# ─── FETCH ONE PAGE ────────────────────────────────────────────────────────────
def fetch_page(district: str, page: int):
    params = {
        "stateName":    STATE,
        "districtName": district,
        "agencyName":   AGENCY,
        "startdate":    START_DATE,
        "enddate":      END_DATE,
        "download":     "false",
        "page":         page,
        "size":         PAGE_SIZE,
    }

    for attempt in range(5):
        try:
            resp = requests.post(BASE_URL, params=params, headers=HEADERS, timeout=30)
            body = resp.json()

            api_status = body.get("statusCode")
            records    = body.get("data", [])
            message    = body.get("message", "")

            if resp.status_code == 200 and api_status == 200:
                return records

            elif "No data found" in message:
                return []

            elif resp.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning(f"Rate limited. Sleeping {wait}s...")
                time.sleep(wait)

            else:
                log.error(f"[{district}] p{page} -> HTTP {resp.status_code} | API {api_status} | {message}")
                return []

        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"[{district}] Attempt {attempt+1} failed: {e}. Retry in {wait}s...")
            time.sleep(wait)

    log.error(f"[{district}] All retries exhausted on page {page}.")
    return []

# ─── PAGINATE THROUGH ALL PAGES FOR A DISTRICT ────────────────────────────────
def fetch_district(district: str) -> list:
    all_records = []
    page = 0

    while True:
        records = fetch_page(district, page)

        if not records:
            log.info(f"  [{district}] No records on page {page}. Done.")
            break

        all_records.extend(records)
        log.info(f"  [{district}] Page {page:3d} -> +{len(records):,} | Total: {len(all_records):,}")

        if len(records) < PAGE_SIZE:
            log.info(f"  [{district}] Last page reached.")
            break

        page += 1
        time.sleep(0.5)

    return all_records

# ─── VALIDATE DISTRICT DATA ────────────────────────────────────────────────────
def validate_district(district: str, records: list):
    if not records:
        return

    df = pd.DataFrame(records)
    df["dataTime"] = pd.to_datetime(df["dataTime"])

    issues = []
    expected_start = pd.Timestamp(START_DATE)

    for station, grp in df.groupby("stationCode"):
        grp = grp.sort_values("dataTime")

        earliest = grp["dataTime"].min()
        gap_days = (earliest - expected_start).days
        if gap_days > 30:
            issues.append(
                f"[{district}] {station}: data starts {earliest.date()} "
                f"({gap_days}d after {START_DATE})"
            )

        gaps = grp["dataTime"].diff().dt.total_seconds() / 3600
        big_gaps = gaps[gaps > 26]
        if len(big_gaps) > 0:
            issues.append(
                f"[{district}] {station}: {len(big_gaps)} gap(s) > 26 hrs between readings"
            )

    if issues:
        log.warning(f"  [{district}] {len(issues)} validation issue(s) - see {ISSUES_F}")
        with open(ISSUES_F, "a", encoding="utf-8") as f:
            for issue in issues:
                f.write(issue + "\n")
    else:
        log.info(f"  [{district}] Validation passed")

# ─── APPEND DISTRICT DATA TO CSV ──────────────────────────────────────────────
def append_to_csv(records: list):
    if not records:
        return

    df = pd.DataFrame(records)

    cols = [c for c in EXPECTED_FIELDS if c in df.columns]
    missing = [c for c in EXPECTED_FIELDS if c not in df.columns]
    if missing:
        log.warning(f"  Columns absent in API response (skipped): {missing}")

    df = df[cols]
    df["dataTime"] = pd.to_datetime(df["dataTime"])

    write_header = not Path(OUTPUT_CSV).exists()
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)

# ─── PROGRESS TRACKING ────────────────────────────────────────────────────────
def load_done() -> set:
    if Path(PROGRESS_F).exists():
        return set(Path(PROGRESS_F).read_text(encoding="utf-8").splitlines())
    return set()

def mark_done(district: str):
    with open(PROGRESS_F, "a", encoding="utf-8") as f:
        f.write(district + "\n")

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
def print_summary(total_fetched: int):
    log.info("=" * 60)
    log.info(f"DONE. Total new records fetched : {total_fetched:,}")

    if not Path(OUTPUT_CSV).exists():
        log.warning("Output CSV not found - nothing was written.")
        return

    df = pd.read_csv(OUTPUT_CSV)
    log.info(f"CSV total rows                 : {len(df):,}")
    log.info(f"Unique stations                : {df['stationCode'].nunique()}")
    log.info(f"Date range in data             : {df['dataTime'].min()} -> {df['dataTime'].max()}")
    log.info(f"Districts covered              : {df['district'].nunique()}")

    before = len(df)
    df = df.drop_duplicates(subset=["stationCode", "dataTime"], keep="last")
    after = len(df)
    if before != after:
        df.to_csv(OUTPUT_CSV, index=False)
        log.info(f"Dedup removed                  : {before - after:,} duplicate rows")
    else:
        log.info("Dedup                          : No duplicates found [OK]")

    if Path(ISSUES_F).exists():
        n = len(Path(ISSUES_F).read_text(encoding="utf-8").splitlines())
        log.warning(f"Validation issues logged       : {n} (see {ISSUES_F})")
    else:
        log.info("Validation                     : No issues [OK]")

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def probe():
    """Fetch 1 record from Pune to verify URL + agency are correct before full run."""
    log.info(f"Probing API ... URL: {BASE_URL}")
    params = {
        "stateName": STATE, "districtName": "Pune", "agencyName": AGENCY,
        "startdate": START_DATE, "enddate": "2023-01-07",
        "download": "false", "page": 0, "size": 1,
    }
    try:
        r = requests.post(BASE_URL, params=params, headers=HEADERS, timeout=30)
        log.info(f"  HTTP {r.status_code} | body: {r.text[:300]}")
        body = r.json()
        if body.get("data"):
            log.info("  Probe OK — API is reachable and returning data.")
            return True
        else:
            log.error(f"  Probe FAILED — no data. message: {body.get('message')}")
            return False
    except Exception as e:
        log.error(f"  Probe FAILED — exception: {e}")
        return False


def main():
    if not probe():
        log.error("Aborting — fix the URL or agency before running the full fetch.")
        return

    done      = load_done()
    remaining = [d for d in DISTRICTS if d not in done]

    log.info(f"Agency              : {AGENCY}")
    log.info(f"Districts remaining : {len(remaining)} / {len(DISTRICTS)}")
    log.info(f"Already completed   : {len(done)}")
    log.info(f"Date range          : {START_DATE} -> {END_DATE}")
    log.info(f"Output              : {OUTPUT_CSV}\n")

    total_fetched = 0

    for i, district in enumerate(remaining, 1):
        log.info(f"[{i}/{len(remaining)}] Fetching {district} ...")

        records = fetch_district(district)

        if records:
            validate_district(district, records)
            append_to_csv(records)
            mark_done(district)
            total_fetched += len(records)
            log.info(f"  [{district}] DONE - {len(records):,} records written\n")
        else:
            log.warning(f"  [{district}] [X] No data returned. Skipping.\n")

        time.sleep(1)

    print_summary(total_fetched)

if __name__ == "__main__":
    main()
