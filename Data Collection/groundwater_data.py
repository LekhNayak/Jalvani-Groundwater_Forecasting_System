import requests
import pandas as pd
import time
import logging
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://indiawris.gov.in/Dataset/Ground Water Level"
STATE      = "Maharashtra"
AGENCIES   = ["CGWB"]   # CGWB already fetched; GSDA is Maharashtra's state agency
START_DATE = "2023-01-01"
END_DATE   = "2026-06-30"
PAGE_SIZE  = 1000
OUTPUT_CSV = "datasets/groundwater_maharashtra_23_25_uncleaned.csv"
PROGRESS_F = "txt files/progress.txt"     # keys: "district|agency"
ISSUES_F   = "log/validation_issues.log"

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
    "dataAcquisitionMode", "stationStatus",
    "tehsil", "datatypeCode", "description",
    "dataValue", "dataTime", "unit",
    "wellDepth", "wellAquiferType",
    "majorBasin", "tributary", "block", "village", "wellType"
]

Path("log").mkdir(exist_ok=True)
Path("datasets").mkdir(exist_ok=True)
Path("txt files").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("log/fetch.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

HEADERS = {"accept": "application/json"}

# ─── FETCH ONE PAGE ────────────────────────────────────────────────────────────
def fetch_page(district: str, agency: str, page: int):
    params = {
        "stateName":    STATE,
        "districtName": district,
        "agencyName":   agency,
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
                log.error(f"[{agency}/{district}] p{page} -> HTTP {resp.status_code} | API {api_status} | {message}")
                return []

        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"[{agency}/{district}] Attempt {attempt+1} failed: {e}. Retry in {wait}s...")
            time.sleep(wait)

    log.error(f"[{agency}/{district}] All retries exhausted on page {page}.")
    return []

# ─── PAGINATE THROUGH ALL PAGES FOR A DISTRICT ────────────────────────────────
def fetch_district(district: str, agency: str) -> list:
    all_records = []
    page = 0

    while True:
        records = fetch_page(district, agency, page)

        if not records:
            log.info(f"  [{agency}/{district}] No records on page {page}. Done.")
            break

        all_records.extend(records)
        log.info(f"  [{agency}/{district}] Page {page:3d} -> +{len(records):,} | Total: {len(all_records):,}")

        if len(records) < PAGE_SIZE:
            log.info(f"  [{agency}/{district}] Last page reached.")
            break

        page += 1
        time.sleep(0.5)

    return all_records

# ─── VALIDATE DISTRICT DATA ────────────────────────────────────────────────────
def validate_district(district: str, agency: str, records: list):
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
                f"[{agency}/{district}] {station}: data starts {earliest.date()} "
                f"({gap_days}d after {START_DATE})"
            )

        gaps = grp["dataTime"].diff().dt.total_seconds() / 3600
        big_gaps = gaps[gaps > 12]
        if len(big_gaps) > 0:
            issues.append(
                f"[{agency}/{district}] {station}: {len(big_gaps)} gap(s) > 12 hrs between readings"
            )

    if issues:
        log.warning(f"  [{agency}/{district}] {len(issues)} validation issue(s) found - see {ISSUES_F}")
        with open(ISSUES_F, "a") as f:
            for issue in issues:
                f.write(issue + "\n")
    else:
        log.info(f"  [{agency}/{district}] Validation passed - no gaps or missing history")

# ─── APPEND DISTRICT DATA TO CSV ──────────────────────────────────────────────
def append_to_csv(records: list):
    if not records:
        return

    df = pd.DataFrame(records)

    cols = [c for c in EXPECTED_FIELDS if c in df.columns]
    missing = [c for c in EXPECTED_FIELDS if c not in df.columns]
    if missing:
        log.warning(f"  Columns absent in API response (will be skipped): {missing}")

    df = df[cols]
    df["dataTime"] = pd.to_datetime(df["dataTime"])

    write_header = not Path(OUTPUT_CSV).exists()
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)

# ─── PROGRESS TRACKING ────────────────────────────────────────────────────────
def _progress_key(district: str, agency: str) -> str:
    return f"{district}|{agency}"

def load_done() -> set:
    if not Path(PROGRESS_F).exists():
        return set()

    entries = set(Path(PROGRESS_F).read_text().splitlines())

    # Migrate old format (plain district name = CGWB) to new "district|agency" format.
    # Rewrites the file once so future runs use the new format throughout.
    old_style = {e for e in entries if "|" not in e}
    if old_style:
        log.info(f"Migrating {len(old_style)} legacy progress entries to 'district|agency' format...")
        new_entries = {e for e in entries if "|" in e}
        new_entries |= {_progress_key(d, "CGWB") for d in old_style}
        Path(PROGRESS_F).write_text("\n".join(sorted(new_entries)) + "\n")
        return new_entries

    return entries

def mark_done(district: str, agency: str):
    with open(PROGRESS_F, "a") as f:
        f.write(_progress_key(district, agency) + "\n")

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
    if "agencyName" in df.columns:
        log.info(f"Agencies in data               : {sorted(df['agencyName'].dropna().unique().tolist())}")

    before = len(df)
    df = df.drop_duplicates(subset=["stationCode", "dataTime"], keep="last")
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
    done = load_done()

    # Build full work list: all (district, agency) combos not yet completed
    todo = [
        (d, a)
        for a in AGENCIES
        for d in DISTRICTS
        if _progress_key(d, a) not in done
    ]

    total_combos = len(AGENCIES) * len(DISTRICTS)
    log.info(f"Agencies            : {AGENCIES}")
    log.info(f"Combinations done   : {total_combos - len(todo)} / {total_combos}")
    log.info(f"Remaining           : {len(todo)}")
    log.info(f"Date range          : {START_DATE} -> {END_DATE}")
    log.info(f"Output              : {OUTPUT_CSV}\n")

    total_fetched = 0

    for i, (district, agency) in enumerate(todo, 1):
        log.info(f"[{i}/{len(todo)}] Fetching {agency} / {district} ...")

        records = fetch_district(district, agency)

        if records:
            validate_district(district, agency, records)
            append_to_csv(records)
            mark_done(district, agency)
            total_fetched += len(records)
            log.info(f"  [{agency}/{district}] DONE - {len(records):,} records written\n")
        else:
            log.warning(f"  [{agency}/{district}] [X] No data returned. Skipping (not marking done).\n")

        time.sleep(1)

    print_summary(total_fetched)

if __name__ == "__main__":
    main()
