import requests
import pandas as pd
import logging

HEADERS = {"accept": "application/json"}
STATE   = "Maharashtra"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── SECTION 1: Groundwater diagnostics (CGWB) ────────────────────────────────
GW_URL     = "https://indiawris.gov.in/Dataset/Ground Water Level"
GW_AGENCY  = "CGWB"
START_DATE = "2023-01-01"
END_DATE   = "2025-12-31"

def check_page_size():
    log.info("=" * 60)
    log.info("TEST 1: Does the API respect size=1000?")
    log.info("=" * 60)

    params = {
        "stateName":    STATE,
        "districtName": "Pune",
        "agencyName":   GW_AGENCY,
        "startdate":    START_DATE,
        "enddate":      END_DATE,
        "download":     "false",
        "page":         0,
        "size":         1000,
    }

    resp = requests.post(GW_URL, params=params, headers=HEADERS, timeout=30)
    body = resp.json()

    log.info(f"HTTP status      : {resp.status_code}")
    log.info(f"API status       : {body.get('statusCode')}")
    log.info(f"Response keys    : {list(body.keys())}")
    log.info(f"Records returned : {len(body.get('data', []))}")

    for key in ["totalCount", "totalRecords", "total", "totalElements", "count", "totalPages"]:
        if key in body:
            log.info(f"  -> Found '{key}' = {body[key]}")

    records = body.get("data", [])
    if len(records) == 1000:
        log.info("[OK] PASS - size=1000 is respected. Pagination should be reliable.")
    elif 0 < len(records) < 1000:
        log.warning(f"[!] WARN - Got only {len(records)} records for a 2-year window on page 0.")
        log.warning("   Either data is sparse OR server silently caps at this number.")
        log.warning("   Check 'totalCount' above. If total > returned, you have a cap problem.")
    else:
        log.error("[X] FAIL - Got 0 records. Check URL, params, or date range.")

    return records

def check_pagination(page0_records: list):
    log.info("")
    log.info("=" * 60)
    log.info("TEST 2: Does page 1 return DIFFERENT records from page 0?")
    log.info("=" * 60)

    if len(page0_records) < 1000:
        log.info("  Skipping - page 0 already returned < 1000 records (only one page exists).")
        return

    params = {
        "stateName":    STATE,
        "districtName": "Pune",
        "agencyName":   GW_AGENCY,
        "startdate":    START_DATE,
        "enddate":      END_DATE,
        "download":     "false",
        "page":         1,
        "size":         1000,
    }

    resp = requests.post(GW_URL, params=params, headers=HEADERS, timeout=30)
    body = resp.json()
    page1_records = body.get("data", [])

    log.info(f"Page 1 records   : {len(page1_records)}")

    if not page1_records:
        log.warning("[!] Page 1 returned 0 records - API may be ignoring the page param.")
    else:
        t0 = page0_records[0].get("dataTime")
        t1 = page1_records[0].get("dataTime")
        if t0 == t1:
            log.error(f"[X] FAIL - Page 0 and Page 1 start with same dataTime ({t0}). Pagination broken!")
        else:
            log.info(f"  Page 0 starts at: {t0}")
            log.info(f"  Page 1 starts at: {t1}")
            log.info("[OK] PASS - Pagination works correctly.")

def check_date_coverage(records: list):
    log.info("")
    log.info("=" * 60)
    log.info("TEST 3: What date range does the data actually cover?")
    log.info("=" * 60)

    if not records:
        log.warning("  No records to analyse.")
        return

    df = pd.DataFrame(records)
    df["dataTime"] = pd.to_datetime(df["dataTime"])

    log.info(f"Earliest record  : {df['dataTime'].min()}")
    log.info(f"Latest record    : {df['dataTime'].max()}")
    log.info(f"Unique stations  : {df['stationCode'].nunique()}")
    log.info(f"Sample stations  : {df['stationCode'].unique()[:5].tolist()}")

    gap_days = (df["dataTime"].min() - pd.Timestamp(START_DATE)).days
    if gap_days > 30:
        log.warning(f"[!] Data only starts {gap_days} days after requested start ({START_DATE}).")
    else:
        log.info(f"[OK] Data starts within {gap_days} days of requested start.")

def check_fields(records: list):
    log.info("")
    log.info("=" * 60)
    log.info("TEST 4: Sample record - confirming all field names exist")
    log.info("=" * 60)

    if not records:
        log.warning("  No records.")
        return

    sample = records[0]
    log.info("All fields in a record:")
    for k, v in sample.items():
        log.info(f"  {k:30s} = {v}")

    required = [
        "stationCode", "stationName", "stationType",
        "latitude", "longitude",
        "agencyName", "state", "district",
        "dataAcquisitionMode", "stationStatus",
        "tehsil", "datatypeCode", "description",
        "dataValue", "dataTime", "unit",
        "wellDepth", "wellAquiferType",
        "majorBasin", "tributary", "block", "village", "wellType"
    ]
    missing = [f for f in required if f not in sample]
    if missing:
        log.error(f"[X] Missing fields that main script expects: {missing}")
    else:
        log.info("[OK] All expected fields are present.")

# ─── SECTION 2: ET agency discovery ───────────────────────────────────────────
ET_URL = "https://indiawris.gov.in/Dataset/Evapo Transpiration"

ET_AGENCY_CANDIDATES = [
    "NRSC VIC MODEL",       # known working
    "NRSC",
    "IMD",
    "CWC",
    "CGWB",
    "SAC",
    "ISRO",
    "ICAR",
    "State",
    "",                     # no filter - reveals all agencies in response
]

DISTRICTS_SAMPLE = ["Pune", "Nashik", "Nagpur", "Aurangabad", "Amravati"]

def probe_et_agencies():
    log.info("")
    log.info("=" * 60)
    log.info("TEST 5: ET endpoint - probing agency name candidates (Pune, Jan 2023)")
    log.info("=" * 60)

    hits = []
    for agency in ET_AGENCY_CANDIDATES:
        params = {
            "stateName":    STATE,
            "districtName": "Pune",
            "startdate":    "2023-01-01",
            "enddate":      "2023-01-31",
            "download":     "false",
            "page":         0,
            "size":         5,
        }
        if agency:
            params["agencyName"] = agency

        resp = requests.post(ET_URL, params=params, headers=HEADERS, timeout=30)
        body = resp.json()
        data = body.get("data", [])
        label = f'"{agency}"' if agency else "(no agencyName)"
        log.info(f"  {label:<25} API={body.get('statusCode')}  records={len(data)}")
        if data:
            hits.append((agency, data))

    return hits

def probe_et_districts(working_agencies: list):
    if not working_agencies:
        log.warning("No working ET agencies found - nothing to scan across districts.")
        return

    log.info("")
    log.info("=" * 60)
    log.info("TEST 6: ET endpoint - scanning districts for each working agency")
    log.info("=" * 60)

    for agency, _ in working_agencies:
        log.info(f"\n  Agency: '{agency}'")
        for district in DISTRICTS_SAMPLE:
            params = {
                "stateName":    STATE,
                "districtName": district,
                "agencyName":   agency,
                "startdate":    "2023-01-01",
                "enddate":      "2023-06-30",
                "download":     "false",
                "page":         0,
                "size":         5,
            }
            try:
                resp = requests.post(ET_URL, params=params, headers=HEADERS, timeout=60)
                body = resp.json()
                data = body.get("data", [])
                total = body.get("totalElements", "?")
                log.info(f"    {district:<20} records={len(data)}  totalElements={total}")
                if data:
                    log.info(f"      sample: {data[0]}")
            except Exception as e:
                log.warning(f"    {district:<20} TIMEOUT/ERROR: {e}")

# ─── SECTION 3: Rainfall agency discovery ─────────────────────────────────────
RF_URL = "https://indiawris.gov.in/Dataset/RainFall"

RF_AGENCY_CANDIDATES = [
    "CWC",              # known working
    "CGWB",
    "IMD",
    "NRSC",
    "NRSC VIC MODEL",
    "SAC",
    "ISRO",
    "ICAR",
    "State",
    "BBMB",
    "NHP",
    "MWRRA",
    "",                 # no filter - reveals all agencies in response
]

def probe_rainfall_agencies():
    log.info("")
    log.info("=" * 60)
    log.info("TEST 7: Rainfall endpoint - probing agency candidates (Pune, Jan 2023)")
    log.info("=" * 60)

    hits = []
    for agency in RF_AGENCY_CANDIDATES:
        params = {
            "stateName":    STATE,
            "districtName": "Pune",
            "startdate":    "2023-01-01",
            "enddate":      "2023-01-31",
            "download":     "false",
            "page":         0,
            "size":         5,
        }
        if agency:
            params["agencyName"] = agency

        resp = requests.post(RF_URL, params=params, headers=HEADERS, timeout=30)
        body = resp.json()
        data = body.get("data", [])
        label = f'"{agency}"' if agency else "(no agencyName)"
        log.info(f"  {label:<25} API={body.get('statusCode')}  records={len(data)}")
        if data:
            seen_agencies = {r.get("agencyName") for r in data}
            log.info(f"    -> agencyName in response: {seen_agencies}")
            hits.append((agency, data))

    return hits

def probe_rainfall_districts(working_agencies: list):
    if not working_agencies:
        log.warning("No working Rainfall agencies found beyond CWC.")
        return

    log.info("")
    log.info("=" * 60)
    log.info("TEST 8: Rainfall endpoint - scanning districts for each working agency")
    log.info("=" * 60)

    for agency, _ in working_agencies:
        log.info(f"\n  Agency: '{agency}'")
        for district in DISTRICTS_SAMPLE:
            params = {
                "stateName":    STATE,
                "districtName": district,
                "agencyName":   agency,
                "startdate":    "2023-01-01",
                "enddate":      "2023-06-30",
                "download":     "false",
                "page":         0,
                "size":         5,
            }
            try:
                resp = requests.post(RF_URL, params=params, headers=HEADERS, timeout=60)
                body = resp.json()
                data = body.get("data", [])
                log.info(f"    {district:<20} records={len(data)}")
                if data:
                    log.info(f"      sample: {data[0]}")
            except Exception as e:
                log.warning(f"    {district:<20} TIMEOUT/ERROR: {e}")

# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Running groundwater + ET + Rainfall diagnostics...\n")

    records = check_page_size()
    check_pagination(records)
    check_date_coverage(records)
    check_fields(records)

    hits = probe_et_agencies()
    probe_et_districts(hits)

    rf_hits = probe_rainfall_agencies()
    probe_rainfall_districts(rf_hits)

    log.info("")
    log.info("=" * 60)
    log.info("DIAGNOSTICS COMPLETE")
    log.info("=" * 60)
