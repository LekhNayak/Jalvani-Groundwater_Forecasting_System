import requests
import pandas as pd
import time
import logging
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://indiawris.gov.in/Dataset/Evapo Transpiration"
STATE      = "Maharashtra"
AGENCY     = "NRSC VIC MODEL"
START_DATE = "2023-01-01"
END_DATE   = "2026-06-30"
PAGE_SIZE  = 1000
OUTPUT_CSV = "datasets/et_maharashtra_23_25_uncleaned.csv"
PROGRESS_F = "txt files/et_progress.txt"
ISSUES_F   = "log/et_validation_issues.log"

DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"
]

# Fields returned by the ET endpoint
EXPECTED_FIELDS = ["stateName", "stateCode", "districtName", "date", "dataValue"]

Path("log").mkdir(exist_ok=True)
Path("datasets").mkdir(exist_ok=True)
Path("txt files").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("log/et_fetch.log", encoding="utf-8"),
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
    df["date"] = pd.to_datetime(df["date"])

    issues = []
    expected_start = pd.Timestamp(START_DATE)

    earliest = df["date"].min()
    gap_days = (earliest - expected_start).days
    if gap_days > 30:
        issues.append(
            f"[{district}]: data starts {earliest.date()} ({gap_days}d after {START_DATE})"
        )

    # ET is daily — flag gaps > 1 day
    gaps = df.sort_values("date")["date"].diff().dt.days
    big_gaps = gaps[gaps > 1]
    if len(big_gaps) > 0:
        issues.append(f"[{district}]: {len(big_gaps)} gap(s) > 1 day between readings")

    if issues:
        log.warning(f"  [{district}] {len(issues)} validation issue(s) - see {ISSUES_F}")
        with open(ISSUES_F, "a") as f:
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
        log.warning(f"  Columns absent in response (skipped): {missing}")

    df = df[cols]
    df["date"] = pd.to_datetime(df["date"])

    write_header = not Path(OUTPUT_CSV).exists()
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)

# ─── PROGRESS TRACKING ────────────────────────────────────────────────────────
def load_done() -> set:
    if Path(PROGRESS_F).exists():
        return set(Path(PROGRESS_F).read_text().splitlines())
    return set()

def mark_done(district: str):
    with open(PROGRESS_F, "a") as f:
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
    log.info(f"Date range in data             : {df['date'].min()} -> {df['date'].max()}")
    log.info(f"Districts covered              : {df['districtName'].nunique()}")

    before = len(df)
    df = df.drop_duplicates(subset=["districtName", "date"], keep="last")
    after = len(df)
    if before != after:
        df.to_csv(OUTPUT_CSV, index=False)
        log.info(f"Dedup removed                  : {before - after:,} duplicate rows")
    else:
        log.info("Dedup                          : No duplicates found [OK]")

    if Path(ISSUES_F).exists():
        n = len(Path(ISSUES_F).read_text().splitlines())
        log.warning(f"Validation issues logged       : {n} (see {ISSUES_F})")
    else:
        log.info("Validation                     : No issues [OK]")

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
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
