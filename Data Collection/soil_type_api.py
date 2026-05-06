"""
Fetches district-level soil texture from the ISRIC SoilGrids REST API
using district centroid coordinates (same centroids as temperature_openmeteo.py).

No GEE required. No API key required.
Queries one point per district — returns sand, clay, silt fractions
at depths 0-5cm, 5-15cm, 15-30cm.

Output:
    datasets/soil_texture_maharashtra.csv

Use this if the GEE SoilGrids assets are unavailable or require extra permissions.
"""

import requests
import pandas as pd
import time
from pathlib import Path

Path("datasets").mkdir(exist_ok=True)
OUTPUT = Path("datasets/soil_texture_maharashtra.csv")

# Same centroids as temperature_openmeteo.py
DISTRICT_COORDS = {
    "Ahmednagar":      (19.09, 74.74),
    "Akola":           (20.70, 77.00),
    "Amravati":        (20.93, 77.75),
    "Aurangabad":      (19.88, 75.34),
    "Beed":            (18.99, 75.76),
    "Bhandara":        (21.17, 79.65),
    "Buldhana":        (20.53, 76.18),
    "Chandrapur":      (19.95, 79.30),
    "Dhule":           (20.90, 74.78),
    "Gadchiroli":      (20.18, 80.00),
    "Gondia":          (21.46, 80.20),
    "Hingoli":         (19.72, 77.15),
    "Jalgaon":         (21.00, 75.57),
    "Jalna":           (19.84, 75.88),
    "Kolhapur":        (16.70, 74.24),
    "Latur":           (18.40, 76.56),
    "Mumbai City":     (18.96, 72.82),
    "Mumbai Suburban": (19.17, 72.97),
    "Nagpur":          (21.15, 79.09),
    "Nanded":          (19.15, 77.32),
    "Nandurbar":       (21.37, 74.24),
    "Nashik":          (20.01, 73.79),
    "Osmanabad":       (18.18, 76.04),
    "Palghar":         (19.70, 72.77),
    "Parbhani":        (19.27, 76.78),
    "Pune":            (18.52, 73.85),
    "Raigad":          (18.52, 73.18),
    "Ratnagiri":       (17.00, 73.33),
    "Sangli":          (16.86, 74.57),
    "Satara":          (17.68, 74.00),
    "Sindhudurg":      (16.35, 73.74),
    "Solapur":         (17.69, 75.90),
    "Thane":           (19.22, 73.00),
    "Wardha":          (20.75, 78.60),
    "Washim":          (20.11, 77.13),
    "Yavatmal":        (20.39, 78.12),
}

# Depth labels returned by the API
DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]
PROPERTIES = ["sand", "clay", "silt"]

BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"


def fetch_district(district: str, lat: float, lon: float) -> dict | None:
    params = {
        "lon":        lon,
        "lat":        lat,
        "property":   PROPERTIES,
        "depth":      DEPTHS,
        "value":      "mean",
    }

    for attempt in range(4):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)

            if resp.status_code == 200:
                body = resp.json()
                row  = {"district": district}

                for prop_block in body.get("properties", {}).get("layers", []):
                    prop_name = prop_block["name"]           # "sand", "clay", "silt"
                    for depth_block in prop_block["depths"]:
                        depth_label = depth_block["label"]   # "0-5cm" etc.
                        mean_val    = depth_block["values"].get("mean")

                        # SoilGrids returns values in g/kg — convert to %
                        col = f"{prop_name}_{depth_label.replace('-', '_')}"
                        row[col] = round(mean_val / 10, 2) if mean_val is not None else None

                return row

            elif resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  [{district}] Rate limited. Sleeping {wait}s ...")
                time.sleep(wait)

            else:
                print(f"  [{district}] HTTP {resp.status_code}: {resp.text[:200]}")
                return None

        except Exception as e:
            wait = 2 ** attempt
            print(f"  [{district}] Attempt {attempt+1} failed: {e}. Retry in {wait}s ...")
            time.sleep(wait)

    return None


def main():
    rows = []

    for i, (district, (lat, lon)) in enumerate(DISTRICT_COORDS.items(), 1):
        print(f"[{i}/{len(DISTRICT_COORDS)}] {district} ({lat}, {lon}) ...")
        row = fetch_district(district, lat, lon)

        if row:
            # Print a quick summary
            s  = row.get("sand_0_5cm", "?")
            c  = row.get("clay_0_5cm", "?")
            sl = row.get("silt_0_5cm", "?")
            print(f"  Surface (0-5cm): sand={s}%  clay={c}%  silt={sl}%")
            rows.append(row)
        else:
            print(f"  [{district}] FAILED — skipping")

        time.sleep(1)   # polite pacing

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved -> {OUTPUT}")
    print(f"Districts : {len(df)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
