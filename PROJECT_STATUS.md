# Jalvani — Project Status

**Last updated:** 2026-05-06
**Scope:** Maharashtra groundwater level prediction using environmental time-series data
**Time range:** 2023-01-01 to 2025-04-22
**Geography:** All 36 districts of Maharashtra

---

## Table of Contents
1. [Data Collected](#1-data-collected)
2. [Data Sources](#2-data-sources)
3. [Data Cleaning](#3-data-cleaning)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Output Files](#5-output-files)
6. [Pipeline Order](#6-pipeline-order)

---

## 1. Data Collected

| # | Dataset | Granularity | Date Range | Agency / Source |
|---|---------|-------------|------------|-----------------|
| 1 | **Groundwater Level** | Sub-daily per station | 2023–2025 | CGWB via India WRIS |
| 2 | **Rainfall** | Twice-daily per station (08:30 & 17:30) | 2023–2025 | CWC via India WRIS |
| 3 | **Evapotranspiration (ET)** | Daily per district | 2023–2025 | NRSC VIC MODEL via India WRIS |
| 4 | **Soil Moisture** | Daily per district | 2023–2025 | NRSC VIC MODEL via India WRIS |
| 5 | **Relative Humidity** | Sub-daily per station | 2023–2025 | CWC via India WRIS |
| 6 | **Temperature + Humidity (ERA5)** | Daily per district centroid | 2023–2025-04-22 | Open-Meteo Historical Archive |
| 7 | **Sentinel-2 Indices** (NDVI, NDWI, NDMI) | Monthly per district | 2023–2025 | Google Earth Engine (Copernicus S2) |
| 8 | **Sentinel-1 SAR Backscatter** (VV, VH) | Monthly per district | 2023–2025 | Google Earth Engine (Copernicus S1) |

### Why Open-Meteo for Temperature instead of WRIS?
WRIS temperature/humidity comes from sparse CWC gauge stations with coverage gaps. Open-Meteo provides ERA5 reanalysis data — continuous daily values at every district centroid, no missing days, no station gaps. It also returns relative humidity in the same call, so it replaces the WRIS humidity dataset as well.

### Why Sentinel-1 alongside Sentinel-2?
Sentinel-2 is optical — it fails during heavy monsoon cloud cover. Sentinel-1 is SAR (radar), which works through clouds. The VH/VV backscatter ratio from S1 serves as an independent soil moisture proxy that remains valid during monsoon months when S2 data is unavailable.

---

## 2. Data Sources

### India WRIS API
- **Base URL:** `https://indiawris.gov.in/Dataset/<dataset-name>`
- **Method:** POST with query parameters (`stateName`, `districtName`, `agencyName`, `startdate`, `enddate`, `page`, `size`)
- **Page size:** 1,000 records per request; paginated until empty page
- **Rate limiting:** 5-attempt retry with exponential backoff; 429 → sleep 60s × attempt
- **Progress tracking:** Per-district `.txt` progress files; resumable if script is interrupted
- **Datasets collected:**
  - `RainFall` — agency: CWC
  - `Ground Water Level` — agency: CGWB
  - `Evapo Transpiration` — agency: NRSC VIC MODEL
  - `Soil Moisture` — agency: NRSC VIC MODEL
  - `Relative Humidity` — agency: CWC (collected but superseded by Open-Meteo)

### Open-Meteo Historical Archive API
- **Base URL:** `https://archive-api.open-meteo.com/v1/archive`
- **Method:** GET with `latitude`, `longitude`, `start_date`, `end_date`, `daily` variables
- **Variables fetched:** `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `relative_humidity_2m_mean`
- **Timezone:** Asia/Kolkata
- **Coordinates:** District centroids (hardcoded lat/lon for all 36 districts)
- **No API key required**

### Google Earth Engine
- **Sentinel-2:** `COPERNICUS/S2_SR_HARMONIZED` — harmonised surface reflectance
  - Cloud filter: scenes with >20% cloud cover excluded
  - Bands used: B8 (NIR), B4 (Red), B3 (Green), B8A (Red-Edge), B11 (SWIR)
  - Indices: NDVI = (B8−B4)/(B8+B4), NDWI = (B3−B8)/(B3+B8), NDMI = (B8A−B11)/(B8A+B11)
- **Sentinel-1:** `COPERNICUS/S1_GRD` — IW (Interferometric Wide) mode
  - Polarisations: VV and VH
  - VH/VV ratio computed as additional feature
- **District boundaries:** FAO GAUL 2015 Level-2 (`FAO/GAUL/2015/level2`)
- **Spatial aggregation:** Mean reducer at 500m scale over district geometry

---

## 3. Data Cleaning

### 3.1 Groundwater Level
**Script:** [Data Preprocessing/preprocess_groundwater.py](Data%20Preprocessing/preprocess_groundwater.py)

| Step | Action |
|------|--------|
| Station filter | Kept only `stationStatus == "Active"` stations |
| Duplicates | Dropped exact `(stationCode, dataTime)` duplicates, kept last |
| Null values | Dropped rows where `dataValue` is null |
| Zero filter | Dropped rows where `dataValue == 0` (sensor/reporting artefact) |
| Range filter | Kept only values in range `[-100, 0]` metres (Maharashtra well depths) |

### 3.2 Rainfall
**Script:** [Data Preprocessing/preprocess_rainfall.py](Data%20Preprocessing/preprocess_rainfall.py)

| Step | Action |
|------|--------|
| Duplicates | Dropped exact `(stationCode, dataTime)` duplicates, kept last |
| Null values | Dropped rows where `dataValue` is null |
| Negative filter | Dropped rows where `dataValue < 0` |
| Outlier cap | Dropped rows where `dataValue > 500` mm (physically implausible for a single reading) |

### 3.3 Evapotranspiration (ET)
**Script:** [Data Preprocessing/preprocess_env_features.py](Data%20Preprocessing/preprocess_env_features.py)

| Step | Action |
|------|--------|
| Column rename | `districtName → district`, `dataValue → et_mm` |
| Duplicates | Dropped exact `(district, date)` duplicates, kept last |
| Negative filter | Dropped rows where `et_mm < 0` |

### 3.4 Soil Moisture
**Script:** [Data Preprocessing/preprocess_env_features.py](Data%20Preprocessing/preprocess_env_features.py)

| Step | Action |
|------|--------|
| Column rename | `districtName → district`, `dataValue → soil_moisture` |
| Duplicates | Dropped exact `(district, date)` duplicates, kept last |
| Negative filter | Dropped rows where `soil_moisture < 0` |

### 3.5 Temperature + Humidity (Open-Meteo / ERA5)
**Script:** [Data Preprocessing/preprocess_env_features.py](Data%20Preprocessing/preprocess_env_features.py)

| Step | Action |
|------|--------|
| Duplicates | Dropped exact `(district, date)` duplicates, kept last |
| Temp bounds | Kept only rows where `temp_mean` is in `(-5°C, 55°C)` |
| Humidity bounds | Kept only rows where `humidity_mean` is in `[0%, 100%]` |
| Note | ERA5 data has no station gaps — no gap-fill needed |

### 3.6 Validation During Collection (All WRIS Datasets)
Each collection script ran inline validation per station/district after fetching:
- Flagged stations whose data started >30 days after 2023-01-01
- Flagged stations with gaps >12 hours (rainfall, humidity) or >1 day (ET, soil moisture)
- Issues were logged to per-dataset `.log` files in `log/`

---

## 4. Data Preprocessing

### 4.1 Groundwater — Aggregation & Gap-Fill
**Script:** [Data Preprocessing/preprocess_groundwater.py](Data%20Preprocessing/preprocess_groundwater.py)

1. **Station → daily:** mean `dataValue` per `(stationCode, date)` (averaging sub-daily readings)
2. **Station → district:** per `(district, date)` compute `gwl_mean`, `gwl_min`, `gwl_max`, and `station_count` across all active stations
3. **Reindex to full calendar:** each district reindexed to 2023-01-01 → 2025-04-22
4. **Gap-fill:** forward-fill then backward-fill gaps ≤ 7 days; gaps longer than 7 days remain `NaN`
5. **Missing % report:** each district classified as `OK` (<10% missing), `GAPS` (<40%), or `SPARSE` (≥40%)

### 4.2 Rainfall — Aggregation & Rolling Features
**Script:** [Data Preprocessing/preprocess_rainfall.py](Data%20Preprocessing/preprocess_rainfall.py)

1. **Sub-daily → station daily:** sum all readings per `(stationCode, date)`
2. **Station → district daily:** mean across stations per `(district, date)` with `station_count` recorded (mean avoids bias from unequal station density between districts)
3. **Fill missing days with 0:** reindex to full calendar; missing days → 0 mm (no rain is a valid reading, stations only report when active)
4. **Rolling features engineered:**
   - `rainfall_7d`: 7-day cumulative rainfall (rolling sum, min_periods=1)
   - `rainfall_30d`: 30-day cumulative rainfall (rolling sum, min_periods=1)

### 4.3 ET & Soil Moisture — Gap-Fill
**Script:** [Data Preprocessing/preprocess_env_features.py](Data%20Preprocessing/preprocess_env_features.py)

1. **Reindex to full calendar:** each district reindexed to 2023-01-01 → 2025-04-22
2. **Gap-fill:** forward-fill gaps ≤ 7 days; gaps longer than 7 days remain `NaN`

### 4.4 Environmental Features — Merge
**Script:** [Data Preprocessing/preprocess_env_features.py](Data%20Preprocessing/preprocess_env_features.py)

All four environmental streams (ET, Soil Moisture, Temperature, Humidity) merged on `(district, date)` using an outer join, producing `env_features_clean.csv` with columns:

```
district, date, et_mm, soil_moisture, temp_max, temp_min, temp_mean, humidity_mean
```

### 4.5 Master Dataset — Full Merge & Feature Engineering
**Script:** [Data Preprocessing/merge_master.py](Data%20Preprocessing/merge_master.py)

1. **Join order:**
   - Groundwater INNER JOIN Rainfall on `(district, date)` — keeps only districts with groundwater readings
   - LEFT JOIN environmental features on `(district, date)`
2. **Time features engineered:**
   - `month`: integer month (1–12)
   - `season`: categorical — `winter` (Dec–Feb), `summer` (Mar–May), `monsoon` (Jun–Sep), `post_monsoon` (Oct–Nov)
3. **Lag features engineered:**
   - `gwl_lag_7`: groundwater level 7 days prior (captures short-term recharge lag)
   - `gwl_lag_30`: groundwater level 30 days prior (captures monthly-scale trends)
4. **Final filter:** rows where `gwl_mean` is `NaN` are dropped (no target = unusable for training)

### 4.6 Station GeoJSON Export
**Script:** [Data Preprocessing/extract_station_geojson.py](Data%20Preprocessing/extract_station_geojson.py)

Extracted unique groundwater station metadata to a GeoJSON FeatureCollection for spatial visualisation. Stations with missing coordinates were excluded. Properties per feature: `stationCode`, `stationName`, `district`, `tehsil`, `block`, `village`, `wellType`, `wellDepth`, `wellAquiferType`, `majorBasin`, `tributary`, `agencyName`, `dataAcquisitionMode`, `stationStatus`.

---

## 5. Output Files

### Raw / Unclean (collected, not transformed)
| File | Contents |
|------|----------|
| `unclean datasets/groundwater_maharashtra_23_25_uncleaned.csv` | Raw CGWB groundwater readings, all 36 districts |
| `unclean datasets/rainfall_maharashtra_23_25_uncleaned.csv` | Raw CWC rainfall readings, all 36 districts |
| `unclean datasets/et_maharashtra_23_25_uncleaned.csv` | Raw NRSC ET readings, district-level daily |
| `unclean datasets/soil_moisture_maharashtra_23_25_uncleaned.csv` | Raw NRSC soil moisture readings, district-level daily |
| `unclean datasets/temperature_humidity_maharashtra_23_25.csv` | ERA5 temperature & humidity from Open-Meteo (already clean) |
| `datasets/sentinel2_indices_maharashtra_23_25.csv` | Monthly NDVI, NDWI, NDMI per district from GEE |
| `datasets/sentinel1_backscatter_maharashtra_23_25.csv` | Monthly VV, VH, VH/VV per district from GEE |

### Clean / Processed
| File | Contents |
|------|----------|
| `clean datasets/groundwater_clean.csv` | Daily district-level gwl_mean, gwl_min, gwl_max, station_count |
| `clean datasets/rainfall_clean.csv` | Daily district-level rainfall_mm, rainfall_7d, rainfall_30d, station_count |
| `clean datasets/env_features_clean.csv` | Daily district-level et_mm, soil_moisture, temp_max, temp_min, temp_mean, humidity_mean |
| `clean datasets/master_dataset.csv` | Full merged training dataset with all features + engineered columns |
| `clean datasets/groundwater_stations.geojson` | GeoJSON of all groundwater monitoring stations with metadata |

---

## 6. Pipeline Order

Run scripts in this order to reproduce all clean datasets from scratch:

```
# Step 1 — Data Collection (run from Data Collection/)
python groundwater_data.py
python CWC_Rainfall.py
python ET_data.py
python soil_moisture_data.py
python temperature_openmeteo.py
python sentinel_indices.py          # requires GEE authentication

# Step 2 — Data Preprocessing (run from Data Preprocessing/)
python extract_station_geojson.py   # exports station map (optional)
python preprocess_groundwater.py
python preprocess_rainfall.py
python preprocess_env_features.py
python merge_master.py              # must run last
```

All collection scripts are resumable — they track completed districts in `txt files/*.txt` and skip already-fetched districts on re-run.
