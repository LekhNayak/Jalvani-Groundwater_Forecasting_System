"""District-mean daily rainfall from CHIRPS v2.0 *preliminary* rasters (CHC server), for the days
after the GEE 'final' collection ends. Same 0.05 deg product family, same district polygons.
Overlap days (<= GEE last day) are used to validate the reduction against GEE."""
import gzip, io, json, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from rasterio.io import MemoryFile
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import shape

URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/prelim/global_daily/tifs/p05/{y}/chirps-v2.0.{y}.{m:02d}.{d:02d}.tif.gz"
CACHE = Path("datasets/chirps_prelim_cache"); CACHE.mkdir(exist_ok=True)
FIRST, LAST = sys.argv[1], sys.argv[2]
geoms = {f["properties"]["district"]: shape(f["geometry"]) for f in json.load(open("datasets/maharashtra_district_geoms.geojson"))["features"]}
BOUNDS = (72.0, 15.0, 81.5, 22.5)

def get_tif(day):
    p = CACHE / f"{day:%Y%m%d}.tif"
    if not p.exists():
        raw = urllib.request.urlopen(URL.format(y=day.year, m=day.month, d=day.day), timeout=120).read()
        p.write_bytes(gzip.decompress(raw))
    return p

rows, missing = [], []
for day in pd.date_range(FIRST, LAST):
    try: p = get_tif(day)
    except Exception as e:
        missing.append(str(day.date())); print("missing", day.date(), repr(e)[:80], flush=True); continue
    with rasterio.open(p) as src:
        win = from_bounds(*BOUNDS, transform=src.transform)
        arr = src.read(1, window=win).astype("float64"); tr = src.window_transform(win)
        nod = src.nodata
    arr[(arr == nod) | (arr < 0)] = np.nan
    for d, g in geoms.items():
        m = ~geometry_mask([g], out_shape=arr.shape, transform=tr, all_touched=False)
        if not m.any(): m = ~geometry_mask([g], out_shape=arr.shape, transform=tr, all_touched=True)
        v = np.nanmean(arr[m]) if np.isfinite(arr[m]).any() else np.nan
        rows.append({"district": d, "date": day.strftime("%Y-%m-%d"), "rainfall_mm": round(float(v), 3) if np.isfinite(v) else None})
    print(day.date(), flush=True)
pd.DataFrame(rows).to_csv("datasets/chirps_prelim_2026.csv", index=False)
print("missing days:", missing)
