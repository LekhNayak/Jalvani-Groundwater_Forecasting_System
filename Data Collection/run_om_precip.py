"""Open-Meteo daily precipitation at district centroids, 2026-07-01..2026-10-01 -- only used to
bridge the days after the last CHIRPS release (calibrated against CHIRPS final in the assembler)."""
import time, requests, pandas as pd
import temperature_openmeteo as t
rows = []
for d, (lat, lon) in t.DISTRICT_COORDS.items():
    for a in range(5):
        r = requests.get(t.BASE_URL, params={"latitude": lat, "longitude": lon, "start_date": "2026-07-01", "end_date": "2026-10-01",
                         "daily": "precipitation_sum", "timezone": "Asia/Kolkata"}, timeout=30)
        if r.status_code == 200: break
        time.sleep(30 * (a + 1))
    r.raise_for_status(); dd = r.json()["daily"]
    rows += [{"district": d, "date": x, "precip_om": v} for x, v in zip(dd["time"], dd["precipitation_sum"])]
    time.sleep(0.5)
df = pd.DataFrame(rows); df.to_csv("datasets/openmeteo_precip_2026.csv", index=False)
print(len(df), df.date.min(), df.date.max(), df.precip_om.isna().sum())
