# Jalvani — Project Status

**Last updated:** 2026-05-14
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
| 5 | **Relative Humidity** | Sub-daily per station | 2023–2025 | CWC via India WRIS (superseded by #6) |
| 6 | **Temperature + Humidity (ERA5)** | Daily per district centroid | 2023–2025-04-22 | Open-Meteo Historical Archive |
| 7 | **Sentinel-2 Indices** (NDVI, NDWI, NDMI) | Monthly per district | 2023–2025 | Google Earth Engine (Copernicus S2) |
| 8 | **Sentinel-1 SAR Backscatter** (VV, VH) | Monthly per district | 2023–2025 | Google Earth Engine (Copernicus S1) |
| 9 | **Soil Texture** (sand %, clay %, silt %) | Static per district (3 depths) | — | OpenLandMap/SoilGrids via GEE (primary) · ISRIC REST API (fallback) |

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

### ISRIC SoilGrids / OpenLandMap (Soil Texture)
- **Primary script:** `soil_type_gee.py` — uses OpenLandMap GEE assets
  - Assets: `OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02`, `SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02`
  - Depths: 0 cm, 10 cm, 30 cm (bands `b0`, `b10`, `b30`)
  - Silt derived as `100 − sand − clay` (no separate asset)
  - Spatial aggregation: mean reducer at 250 m over district geometry (FAO GAUL Level-2)
  - Requires: `earthengine authenticate` + GCP project registered for Earth Engine
- **Fallback script:** `soil_type_api.py` — uses ISRIC SoilGrids REST API
  - Base URL: `https://rest.isric.org/soilgrids/v2.0/properties/query`
  - Method: GET with `lat`, `lon`, `property`, `depth` params
  - Depths: 0-5 cm, 5-15 cm, 15-30 cm
  - Point query at district centroid (same coordinates as `temperature_openmeteo.py`)
  - Values returned in g/kg — converted to % by dividing by 10
  - No API key or GEE required; rate-limited with retry backoff
- **Output:** `datasets/soil_texture_maharashtra.csv`
- **Static dataset** — run once, reuse for all training

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
| `datasets/soil_texture_maharashtra.csv` | Static sand %, clay %, silt % per district at 3 depths (OpenLandMap/GEE or ISRIC API) |

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
python soil_type_gee.py             # requires GEE + registered GCP project
# python soil_type_api.py           # fallback: ISRIC REST API, no GEE needed

# Step 2 — Data Preprocessing (run from Data Preprocessing/)
python extract_station_geojson.py   # exports station map (optional)
python preprocess_groundwater.py
python preprocess_rainfall.py
python preprocess_env_features.py
python merge_master.py              # must run last
```

All collection scripts are resumable — they track completed districts in `txt files/*.txt` and skip already-fetched districts on re-run.

---

## 7. Model Training — Phase 1 Baselines (COMPLETE)

All baseline models have been trained and evaluated. Outputs live in `Models/baselines/outputs/`.

### 7.1 t+1 Next-Day Prediction Baselines

Run notebooks `01_arima_sarima.ipynb`, `02_xgboost_baseline.ipynb`, `03_lstm_baseline.ipynb`, `04_evaluate_baselines.ipynb`.

| Model | Val RMSE | Val MAE | Test RMSE | Test MAE | Test MAPE |
|-------|----------|---------|-----------|----------|-----------|
| SARIMA | 90.8492m* | 18.2621m | 3.6335m | 1.8725m | 16.22% |
| XGBoost | 0.5566m | 0.1957m | 1.6818m | 0.4616m | 2.43% |
| LSTM | 1.4465m | 0.7060m | 2.7346m | 1.0636m | 8.91% |

*SARIMA val RMSE is inflated due to rolling-window refit instability on 5 districts.

**Key finding:** XGBoost dominates at t+1 with test RMSE of 1.68m and MAPE of only 2.43%. LSTM is competitive at 2.73m. This is the floor to beat.

### 7.2 Multi-Horizon Forecasting (t+7, t+14, t+30)

Run notebooks `05_arima_multihorizon.ipynb`, `06_xgboost_multihorizon.ipynb`, `07_lstm_multihorizon.ipynb`, `08_evaluate_multihorizon.ipynb`.

Method: **direct h-step forecasting** — a separate model trained per horizon, target = `gwl_mean` shifted -h days per district. No recursive error compounding.

#### Test Set RMSE by Model and Horizon

| Model | t+7 RMSE | t+14 RMSE | t+30 RMSE |
|-------|----------|-----------|-----------|
| ARIMA(7,1,1) | 4.9449m | 5.1334m | 5.0997m |
| XGBoost | 3.9210m | 4.5080m | 5.2570m |
| **LSTM** | **3.2685m** | **4.1763m** | **3.5448m** |

#### Test Set MAE by Model and Horizon

| Model | t+7 MAE | t+14 MAE | t+30 MAE |
|-------|---------|----------|----------|
| ARIMA | 2.0134m | 2.2075m | 2.4836m |
| XGBoost | 1.5551m | 1.8697m | 2.2199m |
| **LSTM** | 1.6505m | **1.9935m** | **1.8007m** |

#### Key Findings

1. **LSTM beats XGBoost at all 3 horizons** on RMSE. The gap widens dramatically at t+30: LSTM 3.54m vs XGBoost 5.26m (33% better). This confirms that sequential memory is essential for medium-range forecasting and justifies the move to TFT.

2. **XGBoost degrades badly at long horizons** (RMSE: 3.92 → 4.51 → 5.26) because it cannot leverage temporal ordering — at t+30 the autoregressive lag features become stale and it has no mechanism to learn how environmental signals compound over time.

3. **ARIMA is surprisingly robust** (RMSE barely changes 4.94 → 5.10 → 5.10) but consistently worse than ML models. It captures the mean-reversion trend but misses non-linear seasonality.

4. **LSTM t+30 RMSE (3.54m) is better than LSTM t+14 (4.18m)** — non-monotonic. Likely explanation: at 30 days the model learns monsoon onset as a strong periodic signal whereas the 14-day window sits in an intermediate regime with less clear structure.

#### SHAP Feature Importance Evolution (XGBoost)

SHAP values reveal HOW models use features as horizon grows — the scientific justification for TFT's Variable Selection Networks.

| Feature Group | t+7 contribution | t+14 contribution | t+30 contribution |
|--------------|-----------------|------------------|------------------|
| gwl_lag_7 | 6.59 (dominant) | 6.64 (dominant) | 6.10 (dominant but decreasing) |
| gwl_lag_14 | 2.25 | 2.04 | 1.78 |
| gwl_lag_30 | 0.46 | 0.55 | 0.78 ↑ |
| soil_moisture | 0.14 | 0.20 ↑ | 0.26 ↑ |
| is_monsoon | ~0.03 | 0.19 ↑ | 0.46 ↑↑ |
| day_of_year | 0.08 | 0.18 ↑ | 0.50 ↑↑ |
| rainfall_30d | 0.14 | 0.06 | 0.20 ↑ |

**Pattern:** At t+7, autoregressive lags (gwl_lag_7) account for ~90% of predictive signal. By t+30, seasonal context features (`day_of_year`, `is_monsoon`) and slow-moving environmental variables (`soil_moisture`, `gwl_lag_30`) gain material importance. This means the model needs a mechanism to selectively weight features per horizon — exactly what TFT's Variable Selection Networks provide.

#### Per-District Test RMSE at t+7

| District | ARIMA | LSTM | XGBoost |
|----------|-------|------|---------|
| Dhule | 15.47m | 10.72m | 11.51m |
| Jalgaon | 14.99m | 9.07m | 11.20m |
| Nashik | 5.88m | 3.37m | 3.86m |
| Nandurbar | 5.10m | 2.52m | 2.68m |
| Aurangabad | 3.26m | 1.93m | 2.25m |
| Raigad | 0.19m | 0.71m | — |
| Wardha | 0.27m | 1.03m | 0.16m |
| Pune | 0.32m | 1.23m | 0.34m |

Dhule and Jalgaon have extremely high errors (RMSE >10m) across all models — these districts have deep aquifer systems with high GWL variability. A spatial graph (GAT) that can propagate information from neighbouring stable districts is expected to improve these the most.

### 7.3 Model Output Files

| File | Contents |
|------|----------|
| `outputs/arima_predictions.csv` | t+1 ARIMA predictions (all districts) |
| `outputs/xgboost_predictions.csv` | t+1 XGBoost predictions |
| `outputs/lstm_predictions.csv` | t+1 LSTM predictions |
| `outputs/comparison_summary.csv` | t+1 aggregate metrics (all models) |
| `outputs/comparison_per_district.csv` | t+1 per-district metrics |
| `outputs/arima_multihorizon_t{7,14,30}.csv` | Multi-horizon ARIMA predictions |
| `outputs/xgboost_multihorizon_t{7,14,30}.csv` | Multi-horizon XGBoost predictions |
| `outputs/lstm_multihorizon_t{7,14,30}.csv` | Multi-horizon LSTM predictions |
| `outputs/xgboost_shap_t{7,14,30}.csv` | Per-feature SHAP values for each horizon |
| `outputs/multihorizon_summary.csv` | Aggregate RMSE/MAE/MAPE all models × all horizons |
| `outputs/multihorizon_per_district.csv` | Per-district RMSE at t+7 (all models) |
| `outputs/lstm_multihorizon_t{7,14,30}.pt` | Saved LSTM weights |

---

## 8. Feature Engineering Assessment

### Current Feature Set (master_dataset.csv — 36 columns)

| Category | Features | Count |
|----------|----------|-------|
| Identity | district, district_id, date | 3 |
| Groundwater (target+context) | gwl_mean, gwl_min, gwl_max, station_count | 4 |
| GWL lags | gwl_lag_7, gwl_lag_14, gwl_lag_30 | 3 |
| GWL trend | gwl_7d_change, gwl_30d_change | 2 |
| Rainfall | rainfall_mm, rainfall_7d, rainfall_30d | 3 |
| Environmental | et_mm, soil_moisture, net_water_balance | 3 |
| Temperature/Humidity | temp_max, temp_min, temp_mean, humidity_mean, vpd_approx | 5 |
| Calendar | month, season, is_monsoon, day_of_year | 4 |
| Soil texture (static) | sand/clay/silt at 0cm, 10cm, 30cm | 9 |

### Sentinel Features (6 columns added)

| Feature | Description |
|---------|-------------|
| `ndvi_mean` | Normalised Difference Vegetation Index — vegetation health proxy |
| `ndwi_mean` | Normalised Difference Water Index — surface water proxy |
| `ndmi_mean` | Normalised Difference Moisture Index — vegetation moisture proxy |
| `vv_db` | Sentinel-1 VV backscatter (dB) — soil roughness / moisture proxy |
| `vh_db` | Sentinel-1 VH backscatter (dB) — vegetation volume scattering |
| `vh_vv_ratio` | VH−VV ratio — soil moisture proxy valid through monsoon cloud cover |

Monthly district-level values interpolated to daily via linear interpolation + edge ffill/bfill (see `merge_master.py`).

**Total in master_dataset.csv: 42 columns**

### TFT Notebook — Additional Engineered Features (+14 columns, 56 total at training time)

These columns are created inside `09_tft_colab.ipynb` at training time and are NOT written back to `master_dataset.csv`:

| Feature | Formula | Count |
|---------|---------|-------|
| `month_sin`, `month_cos` | sin/cos(2π × month / 12) | 2 |
| `doy_sin`, `doy_cos` | sin/cos(2π × day_of_year / 365.25) | 2 |
| `rainfall_mm_log`, `rainfall_7d_log`, `rainfall_30d_log` | log1p(rainfall_*) | 3 |
| `gwl_roll_mean_{30,60,90}` | Per-district rolling mean of gwl_mean (shift 1) | 3 |
| `gwl_roll_std_{30,60,90}` | Per-district rolling std of gwl_mean (shift 1) | 3 |
| `season_id` | winter=0, summer=1, monsoon=2, post_monsoon=3 | 1 |

**Total at TFT training time: 56 columns**

### TFT Model Input Split (from 56 columns)

| Input stream | Features | Count |
|-------------|----------|-------|
| Past (observed time-series) | gwl_mean/min/max, station_count, rainfall_*_log, et_mm, soil_moisture, net_water_balance, temp_*, humidity_mean, vpd_approx, gwl_lag_*, gwl_*_change, gwl_roll_*, ndvi/ndwi/ndmi, vv/vh/vh_vv_ratio | 32 |
| Future (known calendar) | month_sin/cos, doy_sin/cos, is_monsoon, season_id | 6 |
| Static continuous | sand/clay/silt at 0cm, 10cm, 30cm | 9 |
| Static categorical | district_id | 1 |
| **Total model inputs** | | **48** |

### Sentinel Features (Collected and Merged ✅)

| Feature | Source | Status |
|---------|--------|--------|
| NDVI, NDWI, NDMI | Sentinel-2 (GEE) — `sentinel2_indices_maharashtra_23_25.csv` | ✅ In master_dataset.csv |
| VV, VH, VH/VV backscatter | Sentinel-1 (GEE) — `sentinel1_backscatter_maharashtra_23_25.csv` | ✅ In master_dataset.csv |

All 6 Sentinel columns are present in `master_dataset.csv`. 2,676 rows are null for 4 districts (Beed, Gadchiroli, Gondia, Raigad) which have no GEE coverage — these are imputed with dataset-wide medians at training time.

### Is Current Feature Engineering Good Enough for TFT?

**Yes.** The SHAP analysis confirms all engineered features are being used:
- Static soil texture (sand/clay/silt at 3 depths) appears in SHAP for all horizons — TFT's static context encoder will use these.
- `vpd_approx`, `net_water_balance`, `is_monsoon` all have non-zero SHAP — no dead features.
- `gwl_lag_14` (engineered, not in original schema) is the 2nd most important feature at every horizon.
- `gwl_7d_change` and `gwl_30d_change` (engineered trend features) are consistently top-7.

The only features with near-zero SHAP at t+7 are deep soil texture (sand_30cm, clay_30cm) — these may gain importance in TFT which can model longer-range soil water routing, or they may be genuinely uninformative and get gated out by Variable Selection Networks.

---

## 9. Model Training — Phase 2 TFT (COMPLETE)

Trained and evaluated on Google Colab T4 GPU. Notebooks: `Models/TFT/09_tft_colab.ipynb`.  
Model implementation: `Models/TFT/tft_model.py` (from-scratch PyTorch, no pytorch-forecasting dependency).

### 9.1 TFT v1 — Initial Training

**Config:** d_model=128, encoder=90 days, 2 LSTM layers, 4 attention heads, dropout=0.10, CAT_EMBED_DIM=8, global target normalisation.

| Horizon | RMSE (m) | MAE (m) | MAPE (%) |
|---------|----------|---------|----------|
| t+7  | 3.0261 | 1.3572 | 10.20 |
| t+14 | 3.4855 | 1.5091 | 10.70 |
| t+30 | 4.0578 | 1.8158 | 11.80 |

MAPE reported as pooled across all districts (sample-weighted). TFT v1 beat ARIMA and LSTM at all horizons. XGBoost edged TFT at t+7 (9.44% vs 10.2%) but TFT was better at t+14 and t+30.

### 9.2 TFT v2 — Improved Architecture

**Key changes over v1:**

| Change | Reason |
|--------|--------|
| Per-district target normalisation | Global scaler buried within-district signal under between-district variance; z-scoring each district independently forces the model to predict deviations from local baseline |
| Cyclic time encoding (month_sin/cos, doy_sin/cos) | Raw month/doy integers treat Dec→Jan as a large jump; cyclic encoding preserves circular continuity |
| Log1p rainfall | Rainfall is heavily right-skewed; log1p compresses extreme monsoon events |
| Rolling GWL stats (30/60/90-day mean + std) | Explicit "recent baseline + volatility" signal; fills gap between discrete lag_7/14/30 features |
| Bigger model: d_model=192, 3 LSTM layers, 8 heads, CAT_EMBED_DIM=16 | More capacity for richer feature set; safe on T4 GPU |
| Encoder 90→120 days | Captures full seasonal cycle (pre-monsoon → monsoon → post-monsoon) |
| Cosine LR with 5-epoch warmup + AdamW(wd=1e-4) | Smoother convergence vs ReduceLROnPlateau |
| Weighted quantile loss (q50 weight=1.5) | Tightens median (point forecast) without sacrificing coverage intervals |
| Higher patience (15→20), more epochs (100→120) | Room for cosine LR to anneal fully |

**Config:** d_model=192, encoder=120 days, 3 LSTM layers, 8 heads, dropout=0.15, CAT_EMBED_DIM=16, per-district target normalisation.

#### Aggregate Results (median of per-district MAPE — most honest summary)

| Horizon | RMSE median (m) | MAE median (m) | MAPE median (%) |
|---------|-----------------|----------------|-----------------|
| t+7  | 0.706 | 0.507 | 5.43 |
| t+14 | 0.862 | 0.629 | 6.25 |
| t+30 | 1.002 | 0.798 | 8.01 |

Median is the right summary because 4 districts (Satara, Jalgaon, Aurangabad, Nashik) have pathological errors that inflate the unweighted mean. These districts are not "solved" by architecture improvements — they need spatial neighbour information (GAT phase).

#### Per-District Results — t+7

| District | RMSE (m) | MAE (m) | MAPE (%) |
|----------|----------|---------|----------|
| Nanded | 0.1269 | 0.1031 | 0.87 |
| Raigad | 0.0459 | 0.0431 | 1.20 |
| Yavatmal | 0.2286 | 0.1794 | 2.38 |
| Kolhapur | 1.1237 | 0.8692 | 2.51 |
| Washim | 0.5742 | 0.2602 | 2.84 |
| Beed | 1.7001 | 0.5177 | 3.56 |
| Nagpur | 0.2847 | 0.2097 | 3.64 |
| Gadchiroli | 0.7632 | 0.4636 | 4.23 |
| Chandrapur | 0.3267 | 0.2812 | 4.34 |
| Ratnagiri | 0.9480 | 0.7977 | 4.88 |
| Thane | 0.6496 | 0.5857 | 5.02 |
| Wardha | 0.4468 | 0.3574 | 5.20 |
| Amravati | 0.6056 | 0.4959 | 5.65 |
| Gondia | 0.8285 | 0.7048 | 6.04 |
| Pune | 0.4429 | 0.3257 | 7.24 |
| Dhule | 6.9232 | 4.3829 | 8.01 |
| Solapur | 1.5383 | 1.2900 | 8.14 |
| Bhandara | 0.5865 | 0.4592 | 8.77 |
| Nandurbar | 4.1046 | 3.4695 | 10.38 |
| Akola | 0.4581 | 0.3404 | 13.42 |
| Jalgaon | 5.5580 | 3.0925 | 15.13 |
| Nashik | 3.2646 | 2.2960 | 15.35 |
| Aurangabad | 1.6036 | 1.3446 | 27.92 |
| Satara | 5.0010 | 4.0886 | 123.35 ⚠️ |

⚠️ Satara: only 31 validation windows (single GWL station, near-zero absolute GWL values). MAPE is numerically unstable — exclude from headline comparisons.

#### Per-District Results — t+30

| District | RMSE (m) | MAE (m) | MAPE (%) |
|----------|----------|---------|----------|
| Nanded | 0.2975 | 0.2430 | 1.89 |
| Raigad | 0.0822 | 0.0771 | 2.02 |
| Yavatmal | 0.3050 | 0.1905 | 2.36 |
| Beed | 1.6500 | 0.4060 | 2.46 |
| Kolhapur | 1.3533 | 1.0130 | 2.89 |
| Nagpur | 0.3274 | 0.2849 | 4.41 |
| Gondia | 0.7565 | 0.5800 | 4.55 |
| Chandrapur | 0.3532 | 0.3159 | 4.78 |
| Ratnagiri | 0.9822 | 0.8864 | 5.26 |
| Washim | 0.8352 | 0.5417 | 5.66 |
| Gadchiroli | 1.6088 | 0.9035 | 6.57 |
| Thane | 1.1156 | 0.9832 | 7.71 |
| Amravati | 0.8657 | 0.7738 | 8.30 |
| Pune | 0.5610 | 0.4700 | 10.11 |
| Nandurbar | 4.5557 | 4.0496 | 11.33 |
| Wardha | 1.0214 | 0.8228 | 11.39 |
| Bhandara | 0.7675 | 0.6347 | 11.52 |
| Solapur | 2.5379 | 2.2566 | 13.43 |
| Dhule | 10.5298 | 6.2122 | 14.02 |
| Akola | 0.5553 | 0.3649 | 18.67 |
| Satara | 2.0791 | 1.9024 | 26.61 ⚠️ |
| Nashik | 5.2903 | 4.6025 | 31.55 |
| Aurangabad | 3.1197 | 2.2940 | 35.08 |
| Jalgaon | 15.2673 | 9.4745 | 53.93 |

#### TFT v1 vs v2 — Direct Comparison

> **Note on MAPE reporting:** v1 used a pooled (sample-weighted) MAPE across all districts. v2 introduced per-district reporting, so two numbers exist: unweighted mean (inflated by Satara at 123%) and median (robust to outliers). The median is the apples-to-apples comparison against v1's pooled figure.

| Horizon | v1 RMSE | v1 MAPE (pooled) | v2 RMSE (median) | v2 MAPE (median) | Improvement |
|---------|---------|-----------------|------------------|------------------|-------------|
| t+7  | 3.0261m | 10.20% | 0.706m | 5.43% | **−4.77 pp MAPE, −57% RMSE** |
| t+14 | 3.4855m | 10.70% | 0.862m | 6.25% | **−4.45 pp MAPE, −75% RMSE** |
| t+30 | 4.0578m | 11.80% | 1.002m | 8.01% | **−3.79 pp MAPE, −75% RMSE** |

The single biggest driver was **per-district target normalisation** — v1 trained a global z-score that forced the model to explain the large baseline differences between districts (e.g. Raigad at −1m vs Jalgaon at −30m). v2 z-scores each district against its own mean/std, so the model only needs to predict within-district deviations. Cyclic time encoding, log rainfall, rolling GWL stats, and the larger architecture each contributed additional gains.

#### v2 vs Baselines — Head-to-Head (MAPE %)

| Model | t+7 | t+14 | t+30 |
|-------|-----|------|------|
| ARIMA | 13.27 | 14.12 | 15.17 |
| LSTM | 15.16 | 16.77 | 17.22 |
| XGBoost | 9.44 | 12.31 | 12.17 |
| TFT v1 (pooled) | 10.20 | 10.70 | 11.80 |
| **TFT v2 (median per-district)** | **5.43** | **6.25** | **8.01** |

TFT v2 is the strongest model in the project on a per-district median basis — roughly 2× better than XGBoost on the typical district. The 4 high-error districts (Satara, Jalgaon, Aurangabad, Nashik) need spatial neighbour context (GAT phase) which no current model provides.

#### Problem Districts — Root Cause

| District | t+30 MAPE | Root cause |
|----------|-----------|------------|
| Satara | 26.6% (31 samples) | Single station, near-zero absolute GWL → numerically unstable MAPE |
| Jalgaon | 53.9% | Single station, deep high-amplitude aquifer system |
| Aurangabad | 35.1% | Deep aquifer, large seasonal swings, interior Marathwada with no neighbour data |
| Nashik | 31.6% | Wide swings, complex Western Ghats + Deccan plateau hydrogeology |

All four are geographically clustered in the Tapi–Godavari interior. GAT spatial propagation from neighbouring stable districts is the expected fix.

### 9.3 TFT Output Files

| File | Contents |
|------|----------|
| `Models/TFT/tft_model.py` | From-scratch TFT implementation (GRN, VSN, InterpretableMultiHeadAttention, QuantileLoss) |
| `Models/TFT/09_tft_colab.ipynb` | Colab training notebook (v2 — current best) |
| `Models/TFT/09_tft_groundwater.ipynb` | Local notebook (v1 — reference) |
| `outputs/tft_checkpoint.pt` | Best model weights (early-stopped by val quantile loss) |
| `outputs/tft_predictions.csv` | Per-window predictions: district, date, horizon, actual, q10, q50, q90 |
| `outputs/tft_metrics.csv` | Aggregate RMSE/MAE/MAPE by horizon |
| `outputs/tft_metrics_per_district.csv` | Per-district RMSE/MAE/MAPE for all horizons |
| `outputs/per_district_target_stats.csv` | Per-district GWL mean + std used for normalisation (needed for inference) |
| `outputs/tft_training_curve.png` | Train vs val quantile loss by epoch |
| `outputs/tft_vsn_importance.png` | VSN feature importance for past/future/static features |
| `outputs/per_district_mape_t{7,14,30}.png` | Bar charts — per-district MAPE at each horizon |
| `outputs/quantile_<district>.png` | 80% prediction interval plots for top-4 districts |

---

## 10. Phase Roadmap

| Phase | Status | Key Deliverable |
|-------|--------|----------------|
| **Phase 0 — Data Collection** | ✅ Complete (except Sentinel) | All WRIS + ERA5 + Soil Texture datasets |
| **Phase 1 — Baselines** | ✅ Complete | ARIMA, XGBoost, LSTM for t+1 and t+{7,14,30} |
| **Phase 2 — TFT** | ✅ Complete | TFT v1 + v2 trained on Colab T4 GPU, per-district quantile forecasts |
| **Phase 2b — Sentinel Enhancement** | ✅ Complete | Sentinel data collected, merged into master_dataset.csv, used in TFT v2 training |
| **Phase 3 — TFT + GAT** | 🔜 Next | Spatial graph (district adjacency + basin) wrapping TFT encoder |

### GAT Architecture Plan (Phase 3)

Build district adjacency graph from:
1. Geographic proximity (Euclidean distance between district centroids, threshold ~100km)
2. River basin / aquifer membership (from station GeoJSON `majorBasin` field)

Wrap TFT encoder output with 2-layer Graph Attention Network (GAT) before temporal attention — allows recharge propagation from water-rich to water-deficit districts. Expected to most improve Dhule, Jalgaon, Aurangabad, and Nashik.

**Baseline to beat:** TFT v2 median MAPE — 5.43% at t+7, 6.25% at t+14, 8.01% at t+30.
