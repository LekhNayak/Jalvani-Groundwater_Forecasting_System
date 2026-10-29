"""Incremental CHIRPS (GEE, v2.0 final) for 2026-06-15 .. last available day, plus
the district polygons (saved so the CHC preliminary rasters can be reduced over
the *same* geometries). Writes scratch files only; assembly is in chirps_extend_assemble.py."""
import json, time
import ee, pandas as pd

ee.Initialize(project="major-project-495508")
ee.data.setDeadline(180000)

START, END_EXCL = "2026-06-01", "2026-10-02"
DISTRICTS = ["Ahmednagar","Akola","Amravati","Aurangabad","Beed","Bhandara","Buldhana","Chandrapur","Dhule","Gadchiroli","Gondia","Hingoli","Jalgaon","Jalna","Kolhapur","Latur","Mumbai City","Mumbai Suburban","Nagpur","Nanded","Nandurbar","Nashik","Osmanabad","Palghar","Parbhani","Pune","Raigad","Ratnagiri","Sangli","Satara","Sindhudurg","Solapur","Thane","Wardha","Washim","Yavatmal"]
GAUL = {"Mumbai City":["Mumbai city","Mumbai","Mumbai City"],"Mumbai Suburban":["Mumbai Suburban","Mumbai (Suburban)"],
        "Osmanabad":["Osmanabad","Usmanabad"],"Buldhana":["Buldhana","Buldana"],"Ahmednagar":["Ahmednagar","Ahmadnagar"],
        "Beed":["Beed","Bid"],"Gondia":["Gondia","Gondiya"],"Gadchiroli":["Gadchiroli","Garhchiroli"],
        "Raigad":["Raigad","Raigarh"],"Palghar":["Palghar","Thane"]}
mh = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM1_NAME", "Maharashtra"))

rows, feats = [], []
for d in DISTRICTS:
    geom = None
    for v in GAUL.get(d, [d]):
        m = mh.filter(ee.Filter.eq("ADM2_NAME", v))
        if m.size().getInfo() > 0:
            geom = m.geometry(); break
    assert geom is not None, d
    feats.append({"type": "Feature", "properties": {"district": d}, "geometry": geom.getInfo()})

    def red(img):
        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=5500, maxPixels=1e9).get("precipitation")
        return ee.Feature(None, {"date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"), "rainfall_mm": mean})
    fs = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(START, END_EXCL).filterBounds(geom)
          .map(red).getInfo()["features"])
    for f in fs:
        p = f["properties"]
        if p.get("rainfall_mm") is not None:
            rows.append({"district": d, "date": p["date"], "rainfall_mm": round(float(p["rainfall_mm"]), 3)})
    print(d, len(fs), flush=True)

pd.DataFrame(rows).to_csv("datasets/chirps_gee_increment_2026.csv", index=False)
json.dump({"type": "FeatureCollection", "features": feats}, open("datasets/maharashtra_district_geoms.geojson", "w"))
df = pd.DataFrame(rows); print(df["date"].min(), df["date"].max(), len(df))
