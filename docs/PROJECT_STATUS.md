# Jalvani — Project Status

**Last updated:** 2026-10-31. The current status is at the end of §10: v4.2 run done; v4.3 and the dashboards built. Several older sections are superseded; see the notices below.
**Scope:** Maharashtra groundwater level prediction using environmental time-series data
**Time range:** 2023-01-01 to 2026-06-04 (raw groundwater/rainfall) — see §1 for per-source ranges
**Geography:** All 36 districts of Maharashtra (24–26 usable for training, see district gate in §4.5)

> ## ⚠️ Naming — two unrelated "TFT v1/v2" version lines exist in this repo
> **§9 below** ("TFT v1", "TFT v2") refers to the *original* pre-walk-forward prototype runs
> (`09_tft_groundwater.ipynb`, `09_tft_colab.ipynb`) from before the July 2026 rebuild — global vs.
> per-district normalisation, no baseline comparison, validation-only evaluation. That work is
> superseded and kept for historical reference only.
>
> **The current, actively-developed line is "production v1 → v2 → v3"**, documented in
> `docs/history/TFT_VERSION_HISTORY.md`: full walk-forward evaluation, persistence-baseline MASE, delta-target
> framing, data tiers, and (as of v3) statistical significance testing. **If you are looking for
> current TFT results, go to `docs/history/TFT_VERSION_HISTORY.md`, not §9 of this document.**

> ## 🟢 2026-08-15 update — v3: data re-sourced, seven "unexplained" districts fixed
> Full audit of the v2 pipeline and a real 6-fold × 3-seed GPU run. Summary (full detail and numbers
> in `docs/history/TFT_VERSION_HISTORY.md` §7–§13):
> - **ET and soil moisture re-sourced to ERA5-Land** (`01_data_collection/era5_land.py`) — the old
>   WRIS/NRSC feed stopped publishing 2025-06-30, so folds 4–6 of the walk-forward evaluation were
>   training on climatology fill dressed up as a live feature (0% real data in 2 of 6 folds). ERA5-Land
>   also adds potential evapotranspiration and 4 soil-moisture depths instead of 1.
> - **Sentinel monthly composites were leaking up to 15 days of the future** into every daily
>   NDVI/NDWI/NDMI/VV/VH value (stamped at month-start, then linearly interpolated). Now stamped at
>   month-end and held constant — strictly causal.
> - **`data_tier` column added** to the master (`merge_master.py`), computed from station count and
>   record completeness only — never from model results — so per-district accuracy can be explained
>   independently of the model. See §4.5 below.
> - **v2's reported TFT parameter count was wrong by ~6×** (205,173 claimed, 1,231,159 actual — the
>   variable-selection network's first layer alone was larger than the rest of the network). Fixed in
>   a rewritten, vectorised model (`03_models/tft/tft_model_v3.py`) that genuinely builds ~222,000
>   parameters and runs a full 6-fold × 3-seed training sweep in **6.5 minutes** on a T4.
> - **All seven districts that lost to persistence at every horizon in v2 now tie or beat it in v3**
>   (Aurangabad, Bhandara, Nashik, Jalgaon, Dhule, Kolhapur, Ratnagiri) — 6 of the 7 turned out to be a
>   data-completeness ceiling (confirmed by an independent ridge probe finding the same 6 districts
>   unpredictable for a linear model too), and 1 (Ratnagiri) was a genuine v2 model bug, now fixed.
>
> **Implemented in:** `03_models/tft/notebooks/archive/09_tft_colab_production_v3.ipynb`. Full writeup:
> `docs/history/TFT_VERSION_HISTORY.md`, `docs/history/V3_PLAN.md`.

> ## ⚠️ 2026-07-04 update — master dataset rebuilt, TFT v2 numbers superseded
> A data audit found the master dataset was stale and several sections below no longer match
> what's on disk. Full diagnosis and fix plan: **`docs/history/TFT_IMPROVEMENT_PLAN.md`** (repo root).
>
> **What changed:**
> - Rainfall is now **CHIRPS satellite gridded** (`rainfall_clean.csv`), not CWC station data — all 36 districts, through 2025-12-31.
> - Groundwater cleaning rewritten: sign normalisation, mixed-sign station drop, Hampel spike filter, district **median** (not mean), plus `gwl_observed` (0/1 real-reading mask), `aquifer_type`, `well_depth_m`, `n_stations` passed through.
> - `merge_master.py` rewritten: full daily calendar per district (no dropped rows — this was the root cause of the per-district accuracy gap, see §9 below and `docs/history/TFT_IMPROVEMENT_PLAN.md` §2), date-true lags, CGWB extraction merge folded in, explicit district exclusion gate.
> - Fixed silent GAUL-name bugs in `soil_type_gee.py` (had no name mapping at all), `sentinel_indices.py` and `chirps_rainfall.py` (wrong Gadchiroli spelling) — Beed/Gondia/Raigad/Gadchiroli previously had null soil texture and/or no Sentinel data.
> - **New master: 24 districts (down from the old 28, which included 6 near-empty ones), 843 days each, 20,232 rows, 52 columns.** Excluded: Bhandara, Hingoli, Mumbai City, Mumbai Suburban, Osmanabad, Sangli (all ≥95% GWL missing — see `excluded_districts.csv`). Flagged (kept, reported separately): Jalgaon (1 station), Akola (2 stations).
> - Training data range is unchanged (2023-01-01 → 2025-04-22) — the fix was to *use* that range correctly (full calendar grid, masked loss, walk-forward eval with a monsoon test fold), not to extend it.
> - **All Phase 2/2b TFT v1/v2 results in §9 below were trained on the old master and are not reproducible from current files** — the old master included districts (Satara, Gadchiroli) not present in the new one, and its windowing had gap-corruption bugs (see `docs/history/TFT_IMPROVEMENT_PLAN.md` §2.1). A new notebook (`archive/duplicates/09_tft_colab_production__dup_09_tft_colab.ipynb`, v3) fixes this; retraining on Colab is the next step (v2 notebook kept as `09_tft_colab_v2_backup.ipynb`).

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
| 1 | **Groundwater Level** | Sub-daily per station | 2023–2026-06-04 | CGWB via India WRIS |
| 2 | **Rainfall** | Daily gridded | 2023–2025-12-31 | CHIRPS via Google Earth Engine (superseded #2, see §2) |
| 2b | ~~Rainfall (station)~~ | Twice-daily per station | 2023–2025 | ~~CWC via India WRIS~~ — superseded by CHIRPS |
| 3 | **ET + Potential ET + Soil Moisture (4 depths)** | Daily per district | 2023–present, ~5-day latency | **ERA5-Land via Google Earth Engine** (`era5_land.py`) — replaces #3b, see notice above |
| 3b | ~~ET + Soil Moisture (old)~~ | Daily per district | 2023–2025-06-30 | ~~NRSC VIC MODEL via India WRIS~~ — stopped publishing, superseded |
| 5 | **Relative Humidity** | Sub-daily per station | 2023–2025 | CWC via India WRIS (superseded by #6) |
| 6 | **Temperature + Humidity (ERA5)** | Daily per district centroid | 2023–2025-04-22 | Open-Meteo Historical Archive |
| 7 | **Sentinel-2 Indices** (NDVI, NDWI, NDMI) | Monthly per district, causal (month-end stamped) | 2023–2025 | Google Earth Engine (Copernicus S2) |
| 8 | **Sentinel-1 SAR Backscatter** (VV, VH) | Monthly per district, causal (month-end stamped) | 2023–2025 | Google Earth Engine (Copernicus S1) |
| 9 | **Soil Texture** (sand %, clay %, silt %) | Static per district (3 depths) | — | OpenLandMap/SoilGrids via GEE (primary) · ISRIC REST API (fallback) |

### Why Open-Meteo for Temperature instead of WRIS?
WRIS temperature/humidity comes from sparse CWC gauge stations with coverage gaps. Open-Meteo provides ERA5 reanalysis data — continuous daily values at every district centroid, no missing days, no station gaps. It also returns relative humidity in the same call, so it replaces the WRIS humidity dataset as well.

### Why ERA5-Land for ET/soil moisture instead of WRIS/NRSC? (2026-08-15)
The NRSC VIC feed stopped publishing after **2025-06-30** — 14+ months before the last training fold
ends. Every day after that in `et_mm`/`soil_moisture` was a per-district day-of-year climatology fill
(honestly flagged via `*_observed=0`, but still fed to the model as an ordinary feature), which left
half the walk-forward evaluation training on a feature indistinguishable from `doy_sin`/`doy_cos`.
ERA5-Land is a reanalysis product with ~5-day latency — no publication-lag cliff — and additionally
provides **4 soil-moisture depths** (0–7, 7–28, 28–100, 100–289 cm) instead of one surface reading.
The two deepest layers sit on the actual percolation pathway to the water table and lag rainfall by
roughly the same 3–21 days that recharge does, which a single surface value cannot represent.
`preprocess_env_features.py` uses ERA5-Land automatically when `era5_land_maharashtra_23_26.csv` is
present, and falls back to the old WRIS files otherwise (with a printed warning) so the pipeline still
runs on a fresh checkout without Earth Engine credentials.

### Why Sentinel-1 alongside Sentinel-2?
Sentinel-2 is optical — it fails during heavy monsoon cloud cover. Sentinel-1 is SAR (radar), which works through clouds. The VH/VV backscatter ratio from S1 serves as an independent soil moisture proxy that remains valid during monsoon months when S2 data is unavailable.

### Why are Sentinel composites now stamped at month-end? (2026-08-15)
They were previously stamped at the **1st** of their month and linearly interpolated to daily, which
meant a value on, say, 15 January was a 50/50 blend of the January **and February** composites — up
to 15 days of future information leaking into a model whose entire job is forecasting the future.
Composites are now stamped at month-**end** (the point at which that month's composite is actually
knowable) and held constant via `ffill` until the next one lands. Strictly causal; the cost is a lag
of up to ~30 days, which is the genuinely honest availability delay.

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

### ERA5-Land (Google Earth Engine) — ET, PET, Soil Moisture
- **Script:** `01_data_collection/era5_land.py`
- **Asset:** `ECMWF/ERA5_LAND/DAILY_AGGR` — 0.1° (~11 km), global, 1950–present, ~5-day latency
- **Bands used:** `total_evaporation_sum`, `potential_evaporation_sum`,
  `volumetric_soil_water_layer_{1,2,3,4}` (0–7, 7–28, 28–100, 100–289 cm)
- **Unit conversion:** evaporation bands are metres of water equivalent, negative-signed — converted
  to positive mm/day; soil water fractions (m³/m³) converted to percent to match the old NRSC scale
- **Output:** `et_mm`, `pet_mm`, `soil_moisture` (= layer 2, the closest analogue to the old NRSC
  reading), `sm_l1`–`sm_l4`
- District boundaries and GAUL name mapping identical to `chirps_rainfall.py`/`sentinel_indices.py`
- Resumable via `01_data_collection/progress/era5_progress.txt`, same pattern as the other GEE collectors

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
**Script:** [02_preprocessing/preprocess_groundwater.py](../02_preprocessing/preprocess_groundwater.py)

| Step | Action |
|------|--------|
| Station filter | Kept only `stationStatus == "Active"` stations |
| Duplicates | Dropped exact `(stationCode, dataTime)` duplicates, kept last |
| Null values | Dropped rows where `dataValue` is null |
| Zero filter | Dropped rows where `dataValue == 0` (sensor/reporting artefact) |
| Range filter | Kept only values in range `[-100, 0]` metres (Maharashtra well depths) |

### 3.2 Rainfall
**Script:** [02_preprocessing/legacy/preprocess_rainfall.py](../02_preprocessing/legacy/preprocess_rainfall.py)

| Step | Action |
|------|--------|
| Duplicates | Dropped exact `(stationCode, dataTime)` duplicates, kept last |
| Null values | Dropped rows where `dataValue` is null |
| Negative filter | Dropped rows where `dataValue < 0` |
| Outlier cap | Dropped rows where `dataValue > 500` mm (physically implausible for a single reading) |

### 3.3 Evapotranspiration (ET)
**Script:** [02_preprocessing/preprocess_env_features.py](../02_preprocessing/preprocess_env_features.py)

| Step | Action |
|------|--------|
| Column rename | `districtName → district`, `dataValue → et_mm` |
| Duplicates | Dropped exact `(district, date)` duplicates, kept last |
| Negative filter | Dropped rows where `et_mm < 0` |

### 3.4 Soil Moisture
**Script:** [02_preprocessing/preprocess_env_features.py](../02_preprocessing/preprocess_env_features.py)

| Step | Action |
|------|--------|
| Column rename | `districtName → district`, `dataValue → soil_moisture` |
| Duplicates | Dropped exact `(district, date)` duplicates, kept last |
| Negative filter | Dropped rows where `soil_moisture < 0` |

### 3.5 Temperature + Humidity (Open-Meteo / ERA5)
**Script:** [02_preprocessing/preprocess_env_features.py](../02_preprocessing/preprocess_env_features.py)

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
**Script:** [02_preprocessing/preprocess_groundwater.py](../02_preprocessing/preprocess_groundwater.py)

1. **Station → daily:** mean `dataValue` per `(stationCode, date)` (averaging sub-daily readings)
2. **Station → district:** per `(district, date)` compute `gwl_mean`, `gwl_min`, `gwl_max`, and `station_count` across all active stations
3. **Reindex to full calendar:** each district reindexed to 2023-01-01 → 2025-04-22
4. **Gap-fill:** forward-fill then backward-fill gaps ≤ 7 days; gaps longer than 7 days remain `NaN`
5. **Missing % report:** each district classified as `OK` (<10% missing), `GAPS` (<40%), or `SPARSE` (≥40%)

### 4.2 Rainfall — Aggregation & Rolling Features
**Script:** [02_preprocessing/legacy/preprocess_rainfall.py](../02_preprocessing/legacy/preprocess_rainfall.py)

1. **Sub-daily → station daily:** sum all readings per `(stationCode, date)`
2. **Station → district daily:** mean across stations per `(district, date)` with `station_count` recorded (mean avoids bias from unequal station density between districts)
3. **Fill missing days with 0:** reindex to full calendar; missing days → 0 mm (no rain is a valid reading, stations only report when active)
4. **Rolling features engineered:**
   - `rainfall_7d`: 7-day cumulative rainfall (rolling sum, min_periods=1)
   - `rainfall_30d`: 30-day cumulative rainfall (rolling sum, min_periods=1)

### 4.3 ET & Soil Moisture — Gap-Fill
**Script:** [02_preprocessing/preprocess_env_features.py](../02_preprocessing/preprocess_env_features.py)

1. **Reindex to full calendar:** each district reindexed to 2023-01-01 → 2025-04-22
2. **Gap-fill:** forward-fill gaps ≤ 7 days; gaps longer than 7 days remain `NaN`

### 4.4 Environmental Features — Merge
**Script:** [02_preprocessing/preprocess_env_features.py](../02_preprocessing/preprocess_env_features.py)

All four environmental streams (ET, Soil Moisture, Temperature, Humidity) merged on `(district, date)` using an outer join, producing `env_features_clean.csv` with columns:

```
district, date, et_mm, soil_moisture, temp_max, temp_min, temp_mean, humidity_mean
```

### 4.5 Master Dataset — Full Merge & Feature Engineering
**Script:** [02_preprocessing/merge_master.py](../02_preprocessing/merge_master.py)

1. **Join order:**
   - Groundwater LEFT JOIN Rainfall, Environmental features on `(district, date)` on the full
     calendar grid (no rows dropped for missing `gwl_mean` — see the note in the 2026-07-04 notice
     at the top of this document; this behaviour changed from the original inner-join version)
2. **Time features engineered:**
   - `month`: integer month (1–12)
   - `season`: categorical — `winter` (Dec–Feb), `summer` (Mar–May), `monsoon` (Jun–Sep), `post_monsoon` (Oct–Nov)
3. **Lag features engineered:**
   - `gwl_lag_7` / `_14` / `_30`: date-true shifts on the gap-free calendar
4. **District gate:** districts ≥60% missing groundwater excluded, logged to `excluded_districts.csv`
5. **Data tier (added 2026-08-15):** every kept district is assigned a `data_tier` (A/B/C) from
   station count and 2023 record completeness **only — never from model results** — so per-district
   model accuracy can be explained independently rather than post-hoc rationalised:
   ```
   D  overall observed < 55%                          -> labelled tier D (still written to master; the training
                                                           notebook excludes tier-D districts at runtime)
   A  >= 6 stations AND >= 65% of 2023 observed        -> dense, complete — headline results
   C  <= 4 stations AND <  55% of 2023 observed        -> thin panel + gappy 2023 — data-collection finding, not a model failure
   B  everything else                                  -> usable, reported with caveat
   ```
   All 26 gated districts (including tier D) are written to `master_dataset.csv` with their
   `data_tier` label; only the training notebook (`EXCLUDE_TIERS = ['D']`) drops tier-D districts
   before building windows — `merge_master.py` itself does not remove them. Full scorecard in
   `district_data_quality.csv`. Rationale, the correlation evidence behind the thresholds, and the
   current tier assignments: `docs/history/TFT_VERSION_HISTORY.md` §8.1 and `docs/history/V3_PLAN.md` §2.

### 4.6 Station GeoJSON Export
**Script:** [02_preprocessing/extract_station_geojson.py](../02_preprocessing/extract_station_geojson.py)

Extracted unique groundwater station metadata to a GeoJSON FeatureCollection for spatial visualisation. Stations with missing coordinates were excluded. Properties per feature: `stationCode`, `stationName`, `district`, `tehsil`, `block`, `village`, `wellType`, `wellDepth`, `wellAquiferType`, `majorBasin`, `tributary`, `agencyName`, `dataAcquisitionMode`, `stationStatus`.

---

## 5. Output Files

### Raw / Unclean (collected, not transformed)
| File | Contents |
|------|----------|
| `data/raw/groundwater_maharashtra_23_25_uncleaned.csv` | Raw CGWB groundwater readings, all 36 districts |
| `data/raw/rainfall_maharashtra_23_25_uncleaned.csv` | Raw CWC rainfall readings, all 36 districts |
| `data/raw/et_maharashtra_23_25_uncleaned.csv` | Raw NRSC ET readings, district-level daily |
| `data/raw/soil_moisture_maharashtra_23_25_uncleaned.csv` | Raw NRSC soil moisture readings, district-level daily |
| `data/raw/temperature_humidity_maharashtra_23_25.csv` | ERA5 temperature & humidity from Open-Meteo (already clean) |
| `datasets/sentinel2_indices_maharashtra_23_25.csv` | Monthly NDVI, NDWI, NDMI per district from GEE |
| `datasets/sentinel1_backscatter_maharashtra_23_25.csv` | Monthly VV, VH, VH/VV per district from GEE |
| `datasets/soil_texture_maharashtra.csv` | Static sand %, clay %, silt % per district at 3 depths (OpenLandMap/GEE or ISRIC API) |

### Clean / Processed
| File | Contents |
|------|----------|
| `data/clean/groundwater_clean.csv` | Daily district-level gwl_mean, gwl_min, gwl_max, station_count |
| `data/clean/rainfall_clean.csv` | Daily district-level rainfall_mm, rainfall_7d, rainfall_30d, station_count |
| `data/clean/env_features_clean.csv` | Daily district-level et_mm, soil_moisture, temp_max, temp_min, temp_mean, humidity_mean |
| `data/clean/master_dataset.csv` | Full merged training dataset with all features + engineered columns |
| `data/clean/groundwater_stations.geojson` | GeoJSON of all groundwater monitoring stations with metadata |

---

## 6. Pipeline Order

Run scripts in this order to reproduce all clean datasets from scratch:

```
# Step 1 — Data Collection (run from 01_data_collection/)
python groundwater_data.py
python CWC_Rainfall.py
python ET_data.py
python soil_moisture_data.py
python temperature_openmeteo.py
python sentinel_indices.py          # requires GEE authentication
python soil_type_gee.py             # requires GEE + registered GCP project
# python soil_type_api.py           # fallback: ISRIC REST API, no GEE needed

# Step 2 — Data Preprocessing (run from 02_preprocessing/)
python extract_station_geojson.py   # exports station map (optional)
python preprocess_groundwater.py
python preprocess_rainfall.py
python preprocess_env_features.py
python merge_master.py              # must run last
```

All collection scripts are resumable — they track completed districts in `01_data_collection/progress/*.txt` and skip already-fetched districts on re-run.

---

## 7. Model Training — Phase 1 Baselines (COMPLETE)

All baseline models have been trained and evaluated. Outputs live in `04_results/phase1_baselines/`.

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

## 9. Model Training — Phase 2 TFT, original prototype (SUPERSEDED, historical only)

> ⚠️ **"TFT v1"/"TFT v2" in this section are NOT the same as "production v1/v2/v3" elsewhere in this
> repo.** This section documents the original pre-walk-forward prototype (single validation split,
> no persistence baseline, no MASE). The current, actively-developed results are the
> **production v1 → v2 → v3** line in `docs/history/TFT_VERSION_HISTORY.md` — go there for anything current.
> This section is retained only so the numbers below aren't orphaned; do not cite them as current.

> **⚠️ All results in this section (§9.1–§9.3) were trained on the old master dataset and are not
> reproducible from current files.** The old master included 28 districts (some near-empty; some,
> like Satara and Gadchiroli, absent from the new 24-district master) and its window builder had a
> calendar-gap bug that silently corrupted training windows for the lowest-coverage districts —
> exactly the districts these results call out as worst-performing (Dhule, Jalgaon, Aurangabad,
> Nashik). See `docs/history/TFT_IMPROVEMENT_PLAN.md` for the full diagnosis. The notebook has been rewritten
> (`archive/duplicates/09_tft_colab_production__dup_09_tft_colab.ipynb`, v3 — calendar-true windows, masked loss, walk-forward evaluation
> with a monsoon test fold); the old version is kept as `09_tft_colab_v2_backup.ipynb`. Retraining
> on the new master is the next step — this section is left below for historical reference only.

Trained and evaluated on Google Colab T4 GPU. Notebooks: `archive/duplicates/09_tft_colab_production__dup_09_tft_colab.ipynb`.  
Model implementation: `03_models/tft/legacy/tft_model.py` (from-scratch PyTorch, no pytorch-forecasting dependency).

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
| `03_models/tft/legacy/tft_model.py` | From-scratch TFT implementation (GRN, VSN, InterpretableMultiHeadAttention, QuantileLoss) |
| `archive/duplicates/09_tft_colab_production__dup_09_tft_colab.ipynb` | Colab training notebook (v2 — current best) |
| `03_models/tft/notebooks/archive/09_tft_groundwater.ipynb` | Local notebook (v1 — reference) |
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
| **Phase 0 — Data Collection** | ✅ Complete | WRIS + ERA5-Land + Open-Meteo + CHIRPS + Sentinel + Soil Texture datasets |
| **Phase 1 — Baselines** | ✅ Complete | ARIMA, XGBoost, LSTM for t+1 and t+{7,14,30} (see §7, original prototype) |
| **Phase 2 — TFT prototype** | ✅ Complete, superseded | Original TFT v1/v2 (§9) — no walk-forward eval, no baseline comparison |
| **Phase 2b — Windowing/eval rewrite** | ✅ Complete | Calendar-true windows, masked loss, walk-forward evaluation — see `docs/history/TFT_IMPROVEMENT_PLAN.md` |
| **Phase 2c — Production v1 → v2 → v3** | ✅ Complete | Persistence-baseline MASE, delta-target framing, data tiers, statistical significance testing. **Current results — see `docs/history/TFT_VERSION_HISTORY.md`** |
| **Phase 3 — TFT + GAT** | ⏸️ Held | Spatial graph (district adjacency + basin) wrapping TFT encoder — see note below |

### GAT Architecture Plan (Phase 3) — on hold

Build district adjacency graph from:
1. Geographic proximity (Euclidean distance between district centroids, threshold ~100km)
2. River basin / aquifer membership (from station GeoJSON `majorBasin` field)

Wrap TFT encoder output with 2-layer Graph Attention Network (GAT) before temporal attention — allows recharge propagation from water-rich to water-deficit districts.

**Held deliberately (2026-08-15):** an independent 2026 study over 2,951 German groundwater
monitoring wells found globally-trained models had **no systematic advantage** over per-well models
(global better at 45.4% of wells, worse at 48.9%), and that dynamic similarity between wells mattered
far more than training-set volume — filtering for correlated wells moved median NSE 0.47→0.72, while
randomly removing 85% of wells changed almost nothing. With only 9–16 stations in the best-covered
Maharashtra districts and several tier-C districts still thin, a spatial graph layer risks adding
capacity the data can't support yet. Revisit after more stations are collected for tier-C districts,
or after a cheaper first step — partitioning districts by dynamic similarity (correlation of their
Δ-series) rather than pure geographic adjacency. Full citation: `docs/history/V3_PLAN.md` §1 (M13).

**Current model: production v4.2 (Colab T4, 2026-10-30)** = v4.1's forecasts with v4.2's reported intervals. Results are in `04_results/tft_v4_2/`; v4.1's run is in `04_results/tft_v4_1/`.

Locked test (folds 5–6), Ensemble, at t+7 / t+14 / t+30:
- error ÷ persistence error: 0.839 / 0.763 / 0.616;
- error ÷ seasonal-naive error: 0.938 / 0.922 / 0.877;
- 80% interval coverage 79–82% (v4.2 reported intervals); the TFT alone covers 77–80%;
- all significant (district-cluster bootstrap and Diebold–Mariano, Holm-adjusted, p = 0.003).

All 6 folds: 0.903 / 0.830 / 0.712.

Chosen set-up `s1_tab2`:
- LightGBM with seasonal features, stride 1, trained up to the day before each test period;
- stride-1 network sampling;
- shrinkage toward seasonal-naive;
- 10 seeds.

The refit pass and d_model 48 did not help. The TFT alone beats seasonal-naive on the locked test by 2–4% (significant only at t+7) and ties it over all folds. Most of the gain is from LightGBM.

Open issues:
- the best-vs-worst district gap widened again;
- Pune still loses to persistence at t+14 and t+30;
- folds 5–6 contain no monsoon;
- intervals are honest on average but uneven across districts (v4.2, below);
- a few calibrated intervals are extremely wide (90–108 m: Thane, Dhule, Nandurbar).

Full numbers: `docs/history/TFT_VERSION_HISTORY.md` (§0 and the v4.1 section).

**v4.2 (built and run 2026-10-30, Colab T4, 31 min).** Post-processing only. Notebook `03_models/tft/notebooks/09_tft_colab_production_v4_2.ipynb`, pipeline `tft_v4.py` 4.6.
- **Calibration fixes.** ACI is fed one outcome per issue date and band; v4.1 fed the TFT about 10 correlated outcomes per issue date. A symmetric α range is also tried.
- **Reported interval.** Raw or calibrated is chosen per model and horizon on the development folds.
- **Pune guard.** A per-district stack guard. Pune's stack weights were already almost global; the problem is LightGBM in Pune (MASE 1.30 at t+30, vs 0.87 for seasonal-naive and the TFT).
- **Selection.** Three rules fixed in advance (G, C, D) on development folds 1–4 only. Folds 5–6 are trained only after the choices are made. The v4.1 post-processing is re-run on the same networks as the reference.
- **Checked locally.** With default settings the new code reproduces v4.1's output exactly.
- **Results** (`04_results/tft_v4_2/`):
  - **Reproduced v4.1 bit-for-bit.** The re-trained networks and LightGBM match v4.1 on all 220,215 forecast rows, so accuracy is unchanged.
  - **Rule G (Pune guard): rejected.** It fixed Pune on development folds (t+30 1.27 → 0.87), but made 22 other district-horizons worse.
  - **Rule C: static conformal (γ = 0).** One ACI outcome per issue date restored 4–9 points of the TFT's calibrated coverage, but no variant beat the raw intervals.
  - **Rule D: raw intervals** for the TFT, and for the Ensemble at t+7 / t+14; calibrated for the Ensemble at t+30.
  - **TFT intervals fixed.** Locked coverage 74/70/66 → 80/78/77%, interval score 11–19% better.
  - **Ensemble intervals about the same on average:** locked +1.5–3.3%, all folds +1.7 / −8.7 / −3.3%. But only 52–57% of districts are within 65–90% coverage on the locked test (v4.1: 76–100%).
  - **Pune is still open:** 1.000 / 1.050 / 1.128.
- **Next:**
  - freeze the model for the write-up;
  - optional v4.3 (post-processing only): a LightGBM-only guard for Pune, per-district interval evenness, a width cap;
  - then dashboards, rainfall forecasts, and a monsoon-2026 fold when CGWB publishes.
- Design, rules and full results: `docs/history/TFT_VERSION_HISTORY.md` (v4.2 section), `docs/REPORT.md` §9.8, §10.12, §12.4.

**v4.3 (built 2026-10-31, Colab run pending).** Notebook `03_models/tft/notebooks/09_tft_colab_production_v4_3.ipynb`, pipeline 4.7.
- **Post-processing:**
  - a Pune LightGBM-only guard (rule G2);
  - an interval width cap of 3 ×;
  - the evenness-aware interval rule D2.
- **Operational forecast** from the latest reading (`forecast_latest`) and `dashboard_export/` for the dashboards.
- **Checked locally:** v4.2 is reproduced exactly with the switches off, and the latest-forecast smoke test passes. 23 of 24 districts are issued; Nanded is skipped because only 40% of its last 60 days have readings.
- **Run on Colab:** about 40 min. Unzip to `04_results/tft_v4_3/`, then run `dev\scripts\build_data.ps1`.

**Dashboards (built 2026-10-31), in `dev/`.**
- Farmer and government views in English, Marathi and Hindi.
- A site-guide chatbot: offline now, Groq once a key is set.
- A Gnani speech plan and the plumbing for it.
- Real data only; the build never writes outside `dev/`.
- Run: `dev\scripts\setup.ps1` once, then `dev\scripts\run_local.ps1`.
- Details: `dev/README.md`, `docs/REPORT.md` §13.

Superseded:
- v4.0: locked 0.865 / 0.788 / 0.664;
- production v3: MASE 0.958 / 0.909 / 0.834; its evaluation had a window-end-date bug.

**v4 pipeline files:** `02_preprocessing/preprocess_groundwater.py` (rebuilt target),
`audit_groundwater_target.py`, `03_models/tft/tft_v4.py`, `run_v4.py`, `tune_snaive.py`,
`09_tft_colab_production_v4.ipynb`, `09_tft_colab_production_v4_1.ipynb` (both in `notebooks/archive/`), `09_tft_colab_production_v4_2.ipynb` (current), `RESULTS_GUIDE.md`,
dataset `master_dataset_v4target.csv`.
