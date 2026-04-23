"""
Fetches daily temperature + relative humidity for all Maharashtra districts
from the Open-Meteo Historical Archive API (ERA5 reanalysis data).

Why Open-Meteo over WRIS:
  - ERA5 reanalysis: complete coverage, no station gaps, no missing days
  - Uses district centroid coordinates instead of sparse gauge stations
  - Also provides relative humidity — can replace WRIS humidity too
  - Free, no API key, no rate limits for reasonable usage

Output:
    datasets/temperature_humidity_maharashtra_23_25.csv

Columns: district, date, temp_max, temp_min, temp_mean, humidity_mean
"""

import requests
import pandas as pd
import time
import logging
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
START_DATE = "2023-01-01"
END_DATE   = "2025-04-22"
OUTPUT_CSV = "datasets/temperature_humidity_maharashtra_23_25.csv"
PROGRESS_F = "txt files/temperature_openmeteo_progress.txt"

# District centroids for all 36 Maharashtra districts
DISTRICT_COORDS = {
    "Ahmednagar":     (19.09, 74.74),
    "Akola":          (20.70, 77.00),
    "Amravati":       (20.93, 77.75),
    "Aurangabad":     (19.88, 75.34),
    "Beed":           (18.99, 75.76),
    "Bhandara":       (21.17, 79.65),
    "Buldhana":       (20.53, 76.18),
    "Chandrapur":     (19.95, 79.30),
    "Dhule":          (20.90, 74.78),
    "Gadchiroli":     (20.18, 80.00),
    "Gondia":         (21.46, 80.20),
    "Hingoli":        (19.72, 77.15),
    "Jalgaon":        (21.00, 75.57),
    "Jalna":          (19.84, 75.88),
    "Kolhapur":       (16.70, 74.24),
    "Latur":          (18.40, 76.56),
    "Mumbai City":    (18.96, 72.82),
    "Mumbai Suburban":(19.17, 72.97),
    "Nagpur":         (21.15, 79.09),
    "Nanded":         (19.15, 77.32),
    "Nandurbar":      (21.37, 74.24),
    "Nashik":         (20.01, 73.79),
    "Osmanabad":      (18.18, 76.04),
    "Palghar":        (19.70, 72.77),
    "Parbhani":       (19.27, 76.78),
    "Pune":           (18.52, 73.85),
    "Raigad":         (18.52, 73.18),
    "Ratnagiri":      (17.00, 73.33),
    "Sangli":         (16.86, 74.57),
    "Satara":         (17.68, 74.00),
    "Sindhudurg":     (16.35, 73.74),
    "Solapur":        (17.69, 75.90),
    "Thane":          (19.22, 73.00),
    "Wardha":         (20.75, 78.60),
    "Washim":         (20.11, 77.13),
    "Yavatmal":       (20.39, 78.12),
}

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

Path("log").mkdir(exist_ok=True)
Path("datasets").mkdir(exist_ok=True)
Path("txt files").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("log/temperature_openmeteo_fetch.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ─── FETCH ONE DISTRICT ────────────────────────────────────────────────────────
def fetch_district(district: str, lat: float, lon: float) -> pd.DataFrame | None:
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
        ],
        "timezone": "Asia/Kolkata",
    }

    for attempt in range(5):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)

            if resp.status_code == 200:
                body  = resp.json()
                daily = body.get("daily", {})

                df = pd.DataFrame({
                    "district":      district,
                    "date":          daily["time"],
                    "temp_max":      daily["temperature_2m_max"],
                    "temp_min":      daily["temperature_2m_min"],
                    "temp_mean":     daily["temperature_2m_mean"],
                    "humidity_mean": daily["relative_humidity_2m_mean"],
                })
                df["date"] = pd.to_datetime(df["date"])
                return df

            elif resp.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning(f"[{district}] Rate limited. Sleeping {wait}s...")
                time.sleep(wait)

            else:
                log.error(f"[{district}] HTTP {resp.status_code}: {resp.text[:200]}")
                return None

        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"[{district}] Attempt {attempt+1} failed: {e}. Retry in {wait}s...")
            time.sleep(wait)

    log.error(f"[{district}] All retries exhausted.")
    return None

# ─── PROGRESS TRACKING ────────────────────────────────────────────────────────
def load_done() -> set:
    if Path(PROGRESS_F).exists():
        return set(Path(PROGRESS_F).read_text(encoding="utf-8").splitlines())
    return set()

def mark_done(district: str):
    with open(PROGRESS_F, "a", encoding="utf-8") as f:
        f.write(district + "\n")

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
def print_summary():
    if not Path(OUTPUT_CSV).exists():
        log.warning("Output CSV not found.")
        return

    df = pd.read_csv(OUTPUT_CSV)
    log.info("=" * 60)
    log.info(f"CSV rows         : {len(df):,}")
    log.info(f"Districts        : {df['district'].nunique()}")
    log.info(f"Date range       : {df['date'].min()} -> {df['date'].max()}")
    log.info(f"Missing temp_mean: {df['temp_mean'].isna().sum()}")
    log.info(f"Missing humidity : {df['humidity_mean'].isna().sum()}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    done      = load_done()
    remaining = [(d, *c) for d, c in DISTRICT_COORDS.items() if d not in done]

    log.info(f"Districts remaining : {len(remaining)} / {len(DISTRICT_COORDS)}")
    log.info(f"Already completed   : {len(done)}")
    log.info(f"Date range          : {START_DATE} -> {END_DATE}")
    log.info(f"Output              : {OUTPUT_CSV}\n")

    for i, (district, lat, lon) in enumerate(remaining, 1):
        log.info(f"[{i}/{len(remaining)}] {district} ({lat}, {lon}) ...")

        df = fetch_district(district, lat, lon)

        if df is not None:
            write_header = not Path(OUTPUT_CSV).exists()
            df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
            mark_done(district)
            log.info(f"  [{district}] DONE — {len(df):,} days written\n")
        else:
            log.warning(f"  [{district}] [X] Failed. Skipping.\n")

        time.sleep(0.5)   # gentle pacing — Open-Meteo has no strict limit but be polite

    print_summary()

if __name__ == "__main__":
    main()
