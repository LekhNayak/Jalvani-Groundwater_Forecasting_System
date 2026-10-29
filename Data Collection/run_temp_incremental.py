"""Incremental Open-Meteo temp/humidity: re-fetch from 2026-06-15 so the previously
collected tail is refreshed, then merge into the main CSV (new rows win)."""
import time, pandas as pd
import temperature_openmeteo as t
t.START_DATE, t.END_DATE = "2026-06-15", "2026-10-01"
parts = []
for i, (d, (lat, lon)) in enumerate(t.DISTRICT_COORDS.items(), 1):
    df = t.fetch_district(d, lat, lon)
    if df is None:
        raise SystemExit(f"FAILED {d}")
    parts.append(df); print(i, d, len(df), flush=True); time.sleep(0.5)
new = pd.concat(parts)
old = pd.read_csv(t.OUTPUT_CSV, parse_dates=["date"])
out = (pd.concat([old, new]).drop_duplicates(["district", "date"], keep="last")
       .sort_values(["district", "date"]))
out.to_csv(t.OUTPUT_CSV, index=False)
print(out["date"].min(), out["date"].max(), len(out), out["district"].nunique())
print(out.groupby("district").size().describe())
print("nulls:", out[["temp_mean","humidity_mean"]].isna().sum().to_dict())
