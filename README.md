# Jalvani — Maharashtra Groundwater Level Prediction

**Scope:** Prediction of groundwater levels across Maharashtra districts using multi-source environmental time-series data, at t+7 / t+14 / t+30 day horizons with q10/q50/q90 quantile uncertainty.
**Time range:** 2023-01-01 onward — groundwater/rainfall through 2026-06-04, ET/soil moisture through present (ERA5-Land, ~5-day latency). See `docs/PROJECT_STATUS.md` §1 for per-source ranges.
**Type:** Research project — Temporal Fusion Transformer for hydrological forecasting, evaluated against a persistence baseline with statistical significance testing.

> ## Current status (2026-10-31): v4.2 done; v4.3 and the dashboards built
> The Temporal Fusion Transformer (§4.2) is implemented, trained and evaluated, so this is no longer a
> proposal. **For current results, go straight to `docs/history/TFT_VERSION_HISTORY.md`: §0 has the scores, and the
> v4.1 and v4.2 sections are at the end. The full write-up is `docs/REPORT.md`.**
>
> **What v4.1 is:** the rebuilt groundwater target, corrected evaluation and seasonal-naive baseline, plus an
> ensemble of TFT + LightGBM + seasonal-naive + no-change.
>
> **Headline (locked test, folds 5–6, Oct 2025 → Jun 2026, Ensemble), at t+7 / t+14 / t+30, < 1 = better:**
> - error ÷ "nothing changes" error: **0.839 / 0.763 / 0.616**;
> - error ÷ seasonal-naive error: **0.938 / 0.922 / 0.877**;
> - all six comparisons significant (p = 0.003); 80% intervals cover 79–82%;
> - the TFT alone: 80% intervals cover 77–80% (v4.2; v4.1 66–74%).
>
> **Over all 6 folds:** 0.903 / 0.830 / 0.712.
>
> **The TFT alone and LightGBM:**
> - On the locked test the TFT alone beats seasonal-naive by 2–4%, significant only at t+7.
> - Over all folds the TFT alone ties seasonal-naive.
> - Most of the v4.1 gain comes from the improved LightGBM, which is about as accurate as the Ensemble but has
>   intervals that are too narrow (61–75% coverage).
>
> **v4.2 (Colab run 2026-10-30, 31 min)** changed post-processing only. Every choice was made by rules fixed in advance, on development folds only.
> - It reproduced v4.1's forecasts bit-for-bit, so the accuracy above is unchanged.
> - **It fixed the TFT's intervals.** The network's own quantiles beat every calibration variant, so they are now reported.
> - **The Ensemble's intervals are about the same on average,** but less even across districts.
> - **The Pune stack guard was rejected:** it fixed Pune but hurt 22 other district-horizons. Pune is still open.
> - Details: `docs/REPORT.md` §9.8, §10.12, §12.4.
>
> **v4.3 (built, Colab run pending)** changes post-processing and adds the forecast the dashboards need:
> - a Pune fix and evenly honest intervals;
> - a forecast issued from the latest reading.
>
> **Dashboards (`dev/`)** are built and run locally: farmer and government views in English, Marathi and Hindi, with a site-guide chatbot.
> - Start with `dev\scripts\setup.ps1`, then `dev\scripts\run_local.ps1`.
> - See `dev/README.md`.
>
> **Run it:** `03_models/tft/notebooks/09_tft_colab_production_v4_3.ipynb` with `master_dataset_v4target.csv` (about 40 min on a T4).
> - The final results are in `04_results/tft_v4_2/`, and `docs/RESULTS_GUIDE.md` explains every file.
> - v4.1's run is in `04_results/tft_v4_1/`.
>
> **Superseded:**
> - v4.0: locked 0.865 / 0.788 / 0.664;
> - v3: MASE 0.958 / 0.909 / 0.834;
> - `04_results/tft_prod_v1/` is the v1 run.
>
> **Scores of every version (v1 → v4.2) in one place: `docs/history/TFT_VERSION_HISTORY.md` §0.**
>
> §4.2 below describes the **actual implemented architecture** (delta-target framing, data tiers,
> shrinkage toward persistence, vectorised variable-selection networks). §4.3 (spatial GAT layer) and
> §4.4 (physics-informed loss) remain **future work, not yet built** — see the note at the top of each
> for why they're deliberately on hold.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [How Each Dataset Drives Prediction](#2-how-each-dataset-drives-prediction)
3. [The Full Research Methodology](#3-the-full-research-methodology)
4. [Model Architectures](#4-model-architectures)
5. [Training Strategy](#5-training-strategy)
6. [Research Narrative](#6-research-narrative)
7. [Project Structure](#7-project-structure)
8. [Pipeline Order](#8-pipeline-order)

---

## 1. Problem Statement

Groundwater is not a static tank — it is a dynamic balance:

```
GWL change = Recharge − Discharge − Losses
```

Every dataset collected maps to one of these three forces. The model's job is to learn this equation from data, including the time lags and cross-feature interactions that make it non-trivial. Classical models treat inputs independently and ignore spatial aquifer connectivity between districts. This project addresses both gaps.

---

## 2. How Each Dataset Drives Prediction

### 2.1 Rainfall → Primary Recharge Signal

**Causal chain:**
> Rain falls → some evaporates → some runs off → the rest infiltrates soil → percolates through unsaturated zone → reaches the water table

**Why it matters:**
- `rainfall_mm` on day D does not recharge groundwater on day D. The percolation lag is 3–21 days depending on soil depth and aquifer type. This is why `rainfall_7d` and `rainfall_30d` are more powerful features than the raw daily value.
- A single 100mm rain event is very different from 100mm spread over 30 days — rolling windows capture this distinction.
- **Monsoon arithmetic:** Maharashtra's primary recharge window is June–September. If June–August rainfall is below average, GWL in October–December will be depressed regardless of September rainfall. The `rainfall_30d` feature carries this seasonal memory.
- **District-specific infiltration:** Infiltration rate varies significantly across Maharashtra — sandy soils in eastern districts (Akola, Amravati, Buldhana) behave very differently from the black cotton (Vertisol) soils dominant in Vidarbha and Marathwada. This dataset does not include explicit soil type data. The district embedding implicitly absorbs these fixed district-level soil characteristics from the GWL response patterns seen in training data, but it is not a substitute for actual soil texture features. See note below on adding ISRIC SoilGrids data to make this explicit.

**What the model learns:** A function like — "if 30-day cumulative rainfall in Nashik exceeds 150mm AND soil moisture is already above 0.3, expect GWL to rise ~0.8m over the next 14 days."

---

### 2.2 Evapotranspiration (ET) → The Loss Counter

**Causal chain:**
> Plants transpire + soil surface evaporates → water leaves the soil column → reduces soil moisture → two consequences: (a) less water available to percolate to groundwater, and (b) in irrigated agriculture, farmers pump groundwater to compensate → direct GWL drawdown

**Why it matters:**
- ET is the primary competitor to rainfall for soil moisture. The balance `Rainfall − ET` is the net water surplus available for recharge.
- High ET months (April–May, Maharashtra summer) mean even moderate rainfall barely compensates, let alone recharges aquifers.
- In irrigated districts (Nashik, Pune, Solapur, Sangli — sugarcane and grapes), high ET + low rainfall = guaranteed groundwater pumping season and predictable GWL drop.
- ET is the model's proxy for "how much energy the atmosphere is pulling from the soil right now."

**Model learns:** "District X has ET 6mm/day and rainfall 2mm/day → net deficit 4mm/day → soil moisture declining → irrigation demand rising → expect GWL to drop 0.15m/week."

---

### 2.3 Soil Moisture → The Gatekeeper Between Rain and Groundwater

**This is the most physically important feature and the most underappreciated.**

**Causal chain:**
> Soil moisture is the state of the unsaturated zone between the surface and the water table. It determines how much of any incoming rain actually reaches the aquifer versus being absorbed by the soil column first.

**Two critical scenarios:**

**Scenario A — Dry soil (post-summer, pre-monsoon):**
> Heavy rain arrives but soil moisture deficit is high → the first 50–80mm of rain fills the soil column → no recharge until soil is saturated → GWL does not respond immediately

**Scenario B — Saturated soil (late monsoon, September):**
> Even moderate rain (20mm) → soil is already at field capacity → all excess water goes directly to the aquifer → GWL rises sharply

**Why it matters:**
- Soil moisture is the **gating variable** that modulates how rainfall translates to recharge.
- It gives the model the current "state" of the hydrological system between rain events.
- Post-monsoon soil moisture staying elevated into November = slower depletion of GWL through the dry season.

**Model learns:** "Rainfall = 40mm AND soil_moisture = 0.18 → recharge ~5mm. Rainfall = 40mm AND soil_moisture = 0.45 → recharge ~28mm." This non-linear gating interaction is exactly what attention mechanisms and gated recurrent units are built to learn.

---

### 2.4 Temperature → Seasonal Stress Indicator + Pumping Proxy

**Causal chain:**
> High temperature → higher ET → more soil moisture loss → higher crop water stress → more irrigation pumping → faster GWL drawdown

**Why it matters:**
- Temperature encodes the seasonal energy state of the system. March–June temperature rise signals the irrigation demand surge before monsoon onset.
- In Marathwada (Latur, Osmanabad, Beed — semi-arid), summer temperatures exceed 42°C. ET spikes. These are also districts with the most severe groundwater stress. The model learns this correlation from district identity + temperature together.
- `temp_max` captures extreme stress events (heat waves) that cause farmers to irrigate more aggressively.
- Temperature + NDVI interaction: High temp + high NDVI = crops transpiring heavily = guaranteed high groundwater pumping.

---

### 2.5 Humidity → Atmospheric Moisture Demand

**Causal chain:**
> Low humidity → higher vapour pressure deficit (VPD) → ET increases → more water pulled from soil and plants → groundwater demand rises

**Why it matters:**
- Humidity is the atmospheric demand signal that modulates ET effectiveness.
- Pre-monsoon humidity surge (late May, early June) is a reliable signal that monsoon onset is imminent → model can anticipate upcoming recharge before the first rain arrives.
- In coastal districts (Ratnagiri, Sindhudurg, Raigad), humidity stays high year-round, dampening ET — these districts have fundamentally different GWL dynamics from semi-arid interior districts. The model learns this.
- Temperature + humidity together approximate VPD, the true physical driver of ET. The model implicitly learns VPD from these two features.

---

### 2.6 NDVI → Vegetation Cover & Agricultural Irrigation Demand

**This is where it gets interesting — NDVI encodes human behaviour (irrigation) indirectly.**

**Causal chain (two directions):**

**Direction A — High NDVI causes GWL to drop:**
> Lush green crops (high NDVI, Jan–April in Nashik's vineyards) → high transpiration → farmers irrigating from borewells → groundwater pumping → GWL drops. High NDVI in non-monsoon months is almost always irrigated agriculture.

**Direction B — GWL determines future NDVI:**
> If GWL drops too low → farmers cannot pump → crops stress → NDVI drops next month. This creates a feedback loop the model must learn.

**Why it matters:**
- NDVI in month M is a leading indicator of irrigation demand and therefore GWL drop in months M through M+2.
- Seasonal NDVI pattern encodes the crop calendar: rabi crop (Oct–March), kharif crop (June–Sept), summer vegetables.
- **District intelligence:** Nashik has high NDVI year-round (grapes, onions). Osmanabad has low NDVI except kharif. Gadchiroli is forested — NDVI high but low irrigation demand. District embeddings + NDVI together teach the model these distinctions.
- NDVI drop month-on-month signals drought stress → GWL stabilises because pumping reduces as crops fail.

---

### 2.7 NDWI → Surface Water Bodies & Lateral Recharge

**Causal chain:**
> High NDWI = lakes, rivers, reservoirs, flooded paddy fields are full → lateral seepage recharges adjacent aquifers → GWL rises near water bodies

**Why it matters:**
- Post-monsoon NDWI high = surface water bodies full = sustained lateral recharge for the next 2–3 months even without rain.
- Reservoir storage is the "savings account" of the hydrological system. High NDWI near a dam = slow-release recharge downstream for weeks.
- In Marathwada, NDWI detects whether major reservoirs (Jayakwadi, Ujani) are full → model can project multi-month GWL stability downstream.
- NDWI going to zero mid-season = water body dried up = recharge source gone = accelerated GWL decline begins.

---

### 2.8 NDMI → Vegetation Moisture Stress (Irrigation Trigger)

**Causal chain:**
> NDMI measures how much water is in the crop canopy. When NDMI drops, crops are physiologically stressed → farmers respond by irrigating more aggressively within days.

**Why it matters:**
- NDMI is a more sensitive irrigation demand signal than NDVI. NDVI only drops when crops visibly fail. NDMI drops as soon as stress begins, before any visible damage — it is an early warning signal.
- NDMI decline in March–April (pre-monsoon) = rabi crops stressed = peak pumping period = sharpest GWL decline of the year.
- NDMI × rainfall interaction: Low NDMI + no rain = farmers will definitely pump. This is a reliable drawdown predictor.

---

### 2.9 Soil Texture (Sand / Clay / Silt %) → Infiltration Rate per District

**Causal chain:**
> Soil texture determines how fast water moves through the unsaturated zone. Sandy soils drain quickly → fast but small recharge pulses. Clay-heavy soils (black cotton Vertisols of Marathwada) hold water longer → slower percolation but sustained recharge.

**Why it matters:**
- The single largest fixed factor controlling how much of any rainfall actually reaches the aquifer. Without it, the model cannot distinguish between equally rainy districts that have very different recharge rates.
- Explicitly resolves what the district embedding was previously forced to approximate: soil-driven infiltration capacity.
- Static feature — collected once per district, reused across all model training.

**Data collected:** Sand %, clay %, silt % at depths 0 cm, 10 cm, 30 cm for all 36 districts.

**Primary source:** OpenLandMap SoilGrids assets in Google Earth Engine (`soil_type_gee.py`) — full raster coverage over Maharashtra, spatially aggregated to district mean at 250 m resolution.

**Fallback source:** ISRIC SoilGrids REST API (`soil_type_api.py`) — point query at district centroids, no GEE required.

---

### 2.10 Sentinel-1 VV/VH SAR Backscatter → All-Weather Soil Moisture

**Why this matters:**
> Sentinel-2 (NDVI/NDWI/NDMI) goes blind during monsoon due to cloud cover. June–September is the most critical recharge period. Without SAR, there is no satellite signal during the most important months of the year.

**Causal chain:**
> SAR backscatter (VV, VH) correlates with soil surface roughness and dielectric constant. Dielectric constant of soil increases sharply with moisture content. High VH/VV ratio → high soil surface moisture → recent rainfall → recharge underway.

**Why it matters:**
- Fills the optical satellite data gap during monsoon — the exact period when knowing recharge state is most critical.
- VH backscatter tracks soil moisture independent of cloud cover.
- VH/VV ratio in July–August tells the model "soil is saturated in Pune district this week" → expect GWL rise in 7–14 days.
- In drier months, SAR serves as a second, independent soil moisture estimate alongside the WRIS soil moisture data.

---

## 2.11 Data Gaps & Recommended Additions

The following features are referenced in the methodology but are **not currently collected**. Adding them would strengthen the model significantly.

| Feature | Why it matters | Source |
|---------|---------------|--------|
| **Hydraulic conductivity** | Directly governs how fast water moves through the unsaturated zone to the water table | ISRIC SoilGrids (`hcc` layer) via GEE |
| **Aquifer type per station** (alluvial vs. hard rock basalt) | Hard rock aquifers (most of Deccan Plateau) behave very differently from alluvial aquifers — much lower storage, faster response to pumping | CGWB district aquifer maps (available as PDFs; would need manual digitisation per district) |
| **Crop area by type per district** (irrigated vs. rainfed, crop species) | NDVI encodes vegetation greenness but cannot distinguish irrigated wheat from a forest. Explicit crop-type area gives the model a direct pumping demand estimate. | ICRISAT Village Dynamics in South Asia (VDSA) or Kharif/Rabi crop area reports from Maharashtra Agriculture Dept |
| **Reservoir storage levels** (Jayakwadi, Ujani, Koyna, etc.) | NDWI detects surface water presence but not volume. Reservoir storage directly drives lateral recharge downstream. | India WRIS reservoir data (same API, different endpoint) |

> **Note:** Soil texture (sand %, clay %, silt %) has been added as dataset #9 via `soil_type_gee.py` and is no longer a gap.

---

## 3. The Full Research Methodology

The reasoning chain the model is trained to approximate, derived from hydrological first principles:

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — ATMOSPHERIC STATE                                         │
│                                                                     │
│    Temperature + Humidity                                           │
│         │                                                           │
│         ▼                                                           │
│    Implicit VPD (Vapour Pressure Deficit)                           │
│    High VPD → atmosphere is "thirsty" → ET will be high            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — LAND SURFACE RESPONSE                                     │
│                                                                     │
│    ET + NDVI + NDMI + Soil Moisture                                 │
│         │                                                           │
│         ▼                                                           │
│    Is vegetation healthy? (NDVI high, NDMI high, ET ≈ rainfall)     │
│      YES → System in equilibrium. Low pumping pressure.             │
│      NO  → NDMI falling, ET > rainfall                              │
│             Crops are stressed. Farmers will pump groundwater.      │
│             Expect GWL to fall.                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3 — RAINFALL & RECHARGE ESTIMATION                            │
│                                                                     │
│    rainfall_mm + rainfall_7d + rainfall_30d                         │
│    Sentinel-1 VH/VV ratio (soil surface wetness this week)          │
│         │                                                           │
│         ▼                                                           │
│    Actual recharge = f(rainfall, soil_moisture, NDVI, season)       │
│    Saturated soil + heavy rain   → HIGH recharge                    │
│    Dry soil + moderate rain      → soil fills first, LOW recharge   │
│    High NDVI + rain              → vegetation absorbs, LOW recharge │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4 — SURFACE WATER & LATERAL RECHARGE                         │
│                                                                     │
│    NDWI (lakes, reservoirs, flooded fields)                         │
│         │                                                           │
│         ▼                                                           │
│    NDWI high near major water bodies                                │
│      → sustained multi-week lateral recharge                        │
│    NDWI zero                                                        │
│      → recharge is rain-only, no buffer                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5 — TEMPORAL MEMORY                                           │
│                                                                     │
│    gwl_lag_7 + gwl_lag_30 + season + month                          │
│         │                                                           │
│         ▼                                                           │
│    Current GWL is a function of recent trajectory.                  │
│    A GWL that has been falling for 30 days will not                 │
│    reverse today unless recharge is exceptional.                    │
│    Season tells the model what is normal for this time of year.     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6 — DISTRICT IDENTITY (Hydrogeological Character)             │
│                                                                     │
│    District embedding (learned from training data)                  │
│         │                                                           │
│         ▼                                                           │
│    Each district has a unique character:                            │
│      Aquifer type (alluvial vs. hard rock basalt)                   │
│      Average depth to water table                                   │
│      Dominant crop and irrigation pattern                           │
│      Annual GWL cycle amplitude                                     │
│    The model learns this implicitly from years of data.             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────┐
            │  OUTPUT: Predicted GWL          │
            │  Next 7, 14, 30 days            │
            │  With confidence interval       │
            └─────────────────────────────────┘
```

---

## 4. Model Architectures

### 4.1 Baseline Models

Train these first. If your advanced model does not clearly beat XGBoost, the architecture has a problem.

#### SARIMA (per district)
- Pure statistical baseline. Captures seasonality and autocorrelation only.
- Train one model per district. No environmental features — time only.
- Purpose: establishes the floor. Shows how much ML adds over naive time-series methods.

#### XGBoost (all districts, single model)
- Features: all environmental variables + GWL lags at 1, 3, 7, 14, 30 days + month + season + district one-hot encoding.
- Train on rolling windows of 30 days, predict next day GWL.
- **Run SHAP analysis after training.** SHAP values will reveal which variables actually matter in your data — this shapes all subsequent architecture decisions.
- Often beats naive LSTM. If XGBoost beats your LSTM, the LSTM architecture needs fixing.

#### Vanilla LSTM (all districts, district embedding)
- Input: sequence of past 30 days of all features, with district ID as a learned embedding (dim=8).
- Single-step output: predict GWL tomorrow.
- Deep learning baseline before adding attention and spatial components.

---

### 4.2 Primary Architecture: Temporal Fusion Transformer (TFT) — IMPLEMENTED

Implementation: `03_models/tft/tft_model_v3.py` (from-scratch PyTorch). Training/evaluation notebook:
`03_models/tft/notebooks/archive/09_tft_colab_production_v3.ipynb`. Full results and design rationale:
`docs/history/TFT_VERSION_HISTORY.md` and `docs/history/V3_PLAN.md`.

**Why TFT fits this problem:**
- Explicitly handles **static covariates** (district identity, soil texture, aquifer category), **known future inputs** (month, season, horizon index — known in advance), and **observed past inputs** (rainfall, NDVI, GWL, etc.)
- **Variable Selection Networks** learn which features matter at each time step — a district's importance weights are printed as a bar chart in the notebook (`vsn_importance.png`) for both past and future feature groups.
- **Interpretable attention weights** show which past time steps the model focuses on.
- Predicts 7, 14 and 30 days simultaneously in a single forward pass with calibrated q10/q50/q90 uncertainty.

**Two decisions turned out to matter more than the architecture itself, and are worth stating up
front** (full evidence in `TFT_VERSION_HISTORY.md`):

1. **Predict the change, not the level.** Groundwater is a near-random-walk — today's level tells
   you almost everything about next week's. A model asked to predict the absolute level spends its
   capacity re-deriving the easy part (the level itself) and loses to a naive "assume nothing
   changes" forecast. The model instead predicts `Δ = gwl(t+h) − gwl(anchor)`, where `anchor` is the
   last observed encoder day, and every GWL input feature is made anchor-relative. **An output of
   exactly 0 reproduces persistence** — so persistence is the model's floor, not its competitor.
2. **Report MASE (model error ÷ persistence error) in every table, and treat MASE ≥ 1 as
   disqualifying, no matter how good RMSE/MAPE look.** An earlier full GPU run looked reasonable on
   RMSE/MAPE alone and was losing to persistence in 15 of 18 fold-horizon combinations — no metric
   without a baseline would have revealed this. See `TFT_VERSION_HISTORY.md` §1–§2 for the full story.

**Architecture, as actually built:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT LAYER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Static (fixed per district):
    district_id, aquifer_category_id → learned embeddings
    sand/clay/silt % at 0cm, 10cm, 30cm (soil texture)
    annual_recharge_bcm, annual_extraction_bcm, extractable_resource_bcm,
    stage_of_extraction_pct (CGWB district statics)

  Time-varying KNOWN (future values available at inference time):
    month_sin/cos, doy_sin/cos (cyclic calendar encoding)
    is_monsoon, season_id
    horizon_frac — explicit "how far into the 30-day decoder am I" index

  Time-varying OBSERVED (anchor-relative — see decision 1 above):
    gwl_mean, gwl_min, gwl_max, gwl_lag_{7,14,30}, gwl_roll_mean_{30,60,90}
      → (value − anchor) / delta_sd[district][horizon]
    gwl_7d_change, gwl_30d_change, gwl_roll_std_{30,60,90} → value / delta_sd
    gwl_observed, et_mm_observed, soil_moisture_observed (0/1 real-reading flags, untouched)
    rainfall_mm/_7d/_30d/_60d/_90d (log1p), API (exponential antecedent precip, 3 decay rates),
    cumulative rainfall departure, days-since-rain
    et_mm, pet_mm, soil_moisture (4 ERA5-Land depths), water_stress, net_water_balance
    temp_max/min/mean, humidity_mean, vpd_approx
    ndvi_mean, ndwi_mean, ndmi_mean, vv_db, vh_db, vh_vv_ratio (monthly, causal month-end fill)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIABLE SELECTION NETWORKS (VSN) — vectorised
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Separate VSNs for: static inputs, past time-varying, future time-varying.
  Each VSN learns soft gating weights over all features at every time step.
  Implemented as batched matrix ops (BatchedGRN/BatchedFeatureProjection) rather
  than a per-feature Python loop — same math, ~2-3x faster, which is what makes
  a 3-seed x 6-fold ensemble (18 trainings) finish in ~6 minutes on a T4.
  A d_select compression step (64->8 by default) keeps the selector's own
  parameter count from dwarfing the rest of the network — see the note below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEQUENCE ENCODER / DECODER — LSTM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Encoder input: past 60 days of VSN-selected features
  Decoder input: next 30 days of known-future features
  d_model = 32 by default (down from an earlier 192 — see capacity note)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERPRETABLE MULTI-HEAD SELF-ATTENTION (4 heads)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Causal mask: decoder positions attend to all encoder positions
  and to prior decoder positions only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATED RESIDUAL NETWORKS (GRN) — throughout, not just as a final layer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ELU activation + GLU gate + LayerNorm + residual skip, per Lim et al. (2021).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MONOTONE QUANTILE OUTPUT HEAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  q50 = raw median output
  q10 = q50 - softplus(a)      q90 = q50 + softplus(b)
  Quantile crossing (q10 > q90) is impossible by construction.
  Loss: masked pinball loss, zero gradient on unobserved target days.

  Output shape: [batch, forecast_horizon=30, 3_quantiles], reconstructed
  as  level = anchor + output * delta_sd[district][horizon]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST-PROCESSING (fitted on a held-out calibration block, never on test)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Shrinkage:  level_hat = anchor + lambda(district, horizon) * delta_hat
              lambda in [0,1], fitted per district/horizon-band; puts a floor
              under any district the model hasn't learned to forecast well.
  Conformal:  q10/q90 widened by a margin fit on the calibration block,
              so reported interval coverage is a genuine out-of-sample number.
```

**A capacity lesson worth stating explicitly:** an earlier configuration reported "205,173
parameters" but, once rebuilt and counted directly, actually instantiated **1,231,159** — the
variable-selection network's first layer alone (`Linear(n_features × d_model, d_model)`) was larger
than the rest of the network combined, on a problem with only ~300 genuinely independent training
windows per fold. The current model genuinely builds **~222,000 parameters** via the `d_select`
compression step above, and the notebook prints a params-per-independent-window ratio every fold so
this can't silently drift again.

**Training setup (as actually run):**

| Parameter | Value |
|-----------|-------|
| Encoder / decoder length | 60 days / 30 days |
| Evaluation horizons | t+7, t+14, t+30 (both a daily target and a 7-day-mean target are reported) |
| Split | **6-fold walk-forward** (rolling-origin), each fold: train → early-stopping block → separate calibration block → test — never randomly split |
| Folds | monsoon 2024, post-monsoon 2024, dry 2025, monsoon 2025, post-monsoon 2025, dry 2026 — one full seasonal cycle per test window |
| Districts | 24–26 of 36, gated by a `data_tier` computed from station count + record completeness (see `docs/PROJECT_STATUS.md` §4.5); 1 excluded from training entirely, 4 reported separately as low-station |
| Ensemble | 3 seeds, averaged in prediction space |
| Optimizer | AdamW, lr=1e-3 with 5-epoch warmup, `ReduceLROnPlateau` |
| Batch size | 128 |
| Early stopping | Patience 10 epochs on the early-stopping validation block only |
| Baseline ladder | persistence, seasonal climatology-of-change, ridge, LightGBM — all on identical windows/targets |
| Significance testing | Diebold–Mariano vs. persistence (HAC-corrected) + bootstrap CI on median MASE |

> **Split is always by time, walk-forward, never random** — random splits leak because consecutive
> days are highly correlated, and a single train/val/test split (the original plan above) hides
> seasonal generalisation failures that walk-forward evaluation catches.

---

### 4.3 Advanced Architecture: Spatial-Temporal Graph Attention + TFT — FUTURE WORK, HELD

> **Status: not implemented, and deliberately held rather than started.** An independent 2026 study
> over 2,951 German groundwater wells found globally-trained models had no systematic advantage over
> per-well models, and that dynamic similarity between wells mattered far more than training-set
> volume than raw station count. With several Maharashtra districts still thin (2–4 stations, tier C
> — see `docs/PROJECT_STATUS.md` §4.5), a spatial graph risks adding capacity the data can't yet support.
> Revisit once tier-C districts have more stations, or start with the cheaper alternative noted below.
> Full citation and reasoning: `docs/history/TFT_VERSION_HISTORY.md` §15, `docs/history/V3_PLAN.md` §1 (M13).

This remains the most novel planned contribution and, if the data catches up, the strongest research result.

**The insight:** Districts are not independent. Nashik's GWL affects Ahmednagar's because they share the Godavari basin. A model that treats each district independently misses these spatial dependencies — and they matter, especially for downstream recharge propagation.

**Graph construction:**

```
Nodes:   36 districts of Maharashtra

Edges (two types):
  1. Spatial adjacency — districts that share a border (static)
  2. River basin membership — districts in the same basin share aquifer dynamics

  Godavari basin:  Nashik, Ahmednagar, Aurangabad, Nanded, Osmanabad, Latur...
  Krishna basin:   Pune, Satara, Sangli, Kolhapur, Solapur...
  Tapi basin:      Dhule, Jalgaon, Nandurbar...
  Konkan coast:    Ratnagiri, Sindhudurg, Raigad, Palghar, Thane, Mumbai...

Edge weights: inverse distance between district centroids
```

**Full architecture:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NODE ENCODING — TFT Encoder (per district, independently)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Each of the 36 districts runs its own TFT encoder.
  Output: context vector [batch, 36 districts, hidden_dim=128]
  This encodes the full temporal history per district.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPATIAL PROPAGATION — Graph Attention Network (GAT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  2 GAT layers, 4 attention heads each.
  Aggregates information from neighbouring districts.

  Example: "Nashik received 200mm of rain last week.
            Propagate this signal to Ahmednagar's hidden state
            because they are downstream in the Godavari basin."

  The learned attention weights on edges ARE a scientific result:
  they reveal which district pairs have the strongest hydrological
  coupling — this is aquifer connectivity learned from data.

  Output: spatially-enriched node states [batch, 36, hidden_dim=128]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECODER — TFT Decoder (per district, using spatially-enriched state)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Multi-step prediction over 30-day horizon.
  Quantile output: q10, q50, q90 per day per district.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOSS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Sum of quantile losses across all 36 districts simultaneously.
  The model optimises for all districts in one training loop.
```

---

### 4.4 Physics-Informed Loss (Optional Variant) — FUTURE WORK, NOT IMPLEMENTED

Add a regularisation term that encodes the water balance equation. This prevents physically impossible predictions and is itself a publishable contribution.

```python
# Physical constraint:
# GWL change should be consistent with net recharge
# (rainfall - ET) integrated over recent days

gwl_change_predicted = pred_gwl_t - pred_gwl_t_minus_1
net_recharge_proxy   = rainfall_7d - et_7d   # approximate water balance

# Penalise predictions where GWL rises with no recharge,
# or falls during heavy sustained recharge
physics_residual = gwl_change_predicted - alpha * net_recharge_proxy
physics_loss     = torch.mean(physics_residual ** 2)

total_loss = quantile_loss + lambda_phys * physics_loss
```

The model learns `alpha` implicitly. The physics term acts as a constraint that keeps predictions in the physically plausible space while still allowing the neural network to capture patterns that pure physics equations miss.

---

## 5. Training Strategy

### Model Comparison Table

Train in this order. Each phase adds one capability and should show a measurable improvement.

| Phase | Model | New Capability Added | Key Metric | Status |
|-------|-------|---------------------|------------|--------|
| 1 | SARIMA (per district) | Temporal seasonality only | RMSE, MAPE | ✅ done |
| 2 | XGBoost (all districts) | Non-linear feature interactions | RMSE + SHAP | ✅ done |
| 3 | Vanilla LSTM | Deep temporal memory | RMSE, MAE | ✅ done |
| 4 | TFT (single-step) | Feature selection + attention | Pinball Loss | ✅ done (superseded by 5) |
| 5 | TFT (multi-step, 7/14/30d, delta-target, walk-forward) | Multi-horizon calibrated forecasts vs. persistence | MASE, Pinball Loss | ✅ **done — current results** |
| 6 | GAT + TFT (spatial-temporal) | Inter-district recharge propagation | RMSE all districts | ⏸️ held, see §4.3 |
| 7 | TFT + physics loss | Physical plausibility constraints | RMSE + violation rate | ⏸️ not started |

The baseline ladder in phase 5's actual implementation additionally includes ridge regression and
LightGBM, both fitted on the identical windows/target as the TFT — see `TFT_VERSION_HISTORY.md` §9.

### Curriculum Learning

Train the TFT in phases to prevent the model from seeing long-horizon error before learning basic patterns:

- **Phase 1** (epochs 1–30): Single-district, single-step prediction. Model learns basic temporal patterns per district.
- **Phase 2** (epochs 31–60): Extend to 7-day horizon. Add all environmental features.
- **Phase 3** (epochs 61–100): Full 30-day multi-district prediction with spatial graph.

### Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| RMSE | Overall prediction accuracy in metres |
| MAE | Average absolute error, less sensitive to outliers than RMSE |
| MAPE | Percentage error, comparable across districts with different GWL scales |
| Pinball Loss | Calibration of the q10/q50/q90 quantile forecasts |
| Coverage | Fraction of true values falling within the [q10, q90] interval (target: 80%) |

Report all metrics **per district** and as an aggregate. Districts with SPARSE groundwater data coverage will have higher error — report this honestly and separately.

### Interpretability Analysis (Required for Publication)

After training the TFT:

1. **Variable importance per season:** Extract VSN weights averaged over summer, monsoon, post-monsoon, winter. Expected finding: soil_moisture and rainfall dominate in monsoon; NDVI and temperature dominate in summer.

2. **Temporal attention heatmap:** Average attention weights over the 60-day lookback window. Expected finding: peaks at 7–10 days (soil percolation lag) and 21–30 days (deeper aquifer recharge lag).

3. **District attention weights (GAT):** Which district pairs have the strongest learned edges. Expected finding: basin-aligned clusters (Godavari, Krishna, Tapi).

These three analyses are the scientific contribution of the paper — they show that the model has not just learned correlation but has recovered known hydrological principles from data.

---

## 6. Research Narrative

> Groundwater level in Maharashtra is determined by the interplay of atmospheric recharge (rainfall), land surface mediation (soil moisture, evapotranspiration, vegetation state) and human extraction pressure (inferred indirectly from NDVI crop cycles and temperature-driven irrigation demand). Classical models treat these inputs independently and ignore spatial aquifer connectivity between districts. We propose a Spatial-Temporal Graph Attention Transformer that models all 36 Maharashtra districts as an interconnected hydrological graph, uses variable selection networks to dynamically weight features by season and district, and outputs calibrated probabilistic forecasts at 7, 14 and 30-day horizons.
>
> Our interpretability analysis reveals that the model has learned physically consistent behaviour: it attends to monsoon-lagged rainfall (14–21 days) for recharge prediction, weights soil moisture as the primary gating variable between rainfall and recharge, and identifies NDMI decline as a leading indicator of irrigation demand and impending GWL drawdown. The learned graph attention weights reveal basin-aligned inter-district coupling consistent with known Godavari and Krishna basin hydrogeology. These results constitute evidence that the model has learned real hydrology from data, rather than spurious statistical correlation.

---

## 7. Project Structure

The project is split by **what a thing is**: code for each pipeline stage, all data, all results, all documents,
and an archive that holds every backup and duplicate (nothing was deleted). The full old-path → new-path table
is in `docs/STRUCTURE.md`.

```
Major Project/
├── README.md                         this file
├── docs/                             all documents
│   ├── REPORT.md                     the project report (start here for the full story)
│   ├── PROJECT_STATUS.md             data-collection and preprocessing log
│   ├── RESULTS_GUIDE.md              how to read a results folder
│   ├── STRUCTURE.md                  every folder explained + old -> new path table
│   └── history/                      TFT_VERSION_HISTORY.md (v1 -> v4.1, scores of every version),
│                                     V3_PLAN.md, TFT_IMPROVEMENT_PLAN.md
├── data/                             ALL data, one live copy each
│   ├── raw/                          what the collectors downloaded (+ cgwb_pdfs/)
│   ├── clean/                        what preprocessing produced (master_dataset.csv = latest build)
│   ├── master/                       frozen training sets: master_dataset_v4target.csv (CURRENT),
│   │                                 master_dataset_v3target.csv (+ its district_data_quality)
│   └── audit/                        groundwater-target audit plots and tables
├── 01_data_collection/               collectors (run in this folder or anywhere)
│   ├── *.py                          active: groundwater_data, chirps_rainfall, era5_land, temperature_openmeteo,
│   │                                 sentinel_indices, soil_type_gee / soil_type_api, cgwb_extraction,
│   │                                 consolidate_cgwb, cwc_reservoir (not run), run_*_incremental, chirps_* tail tools
│   ├── legacy/                       superseded WRIS collectors (CWC rainfall, NRSC ET / soil moisture, ...)
│   ├── progress/                     resume files for each collector
│   └── logs/                         fetch and validation logs
├── 02_preprocessing/                 preprocess_groundwater, audit_groundwater_target, preprocess_env_features,
│   │                                 merge_master, extract_station_geojson
│   └── legacy/                       preprocess_rainfall (CWC), merge_cgwb_into_master (now inside merge_master)
├── 03_models/
│   ├── baselines_phase1/             Phase 1 ARIMA / XGBoost / LSTM: notebooks/ + scripts/
│   └── tft/                          tft_model_v3.py (network), tft_v4.py (pipeline), run_v4.py, tune_snaive.py
│       ├── notebooks/                09_tft_colab_production_v4_3.ipynb  <- CURRENT notebook
│       │   └── archive/              production v1, v2, v3, v4, v4.1, v4.2 notebooks + early prototypes
│       └── legacy/                   tft_model.py (v1/v2 network)
├── 04_results/                       ALL results, one folder per run
│   ├── phase1_baselines/             Phase 1 outputs and plots
│   ├── tft_prod_v1/                  first full TFT run (worse than "no change"; kept for comparison)
│   ├── tft_v4_0_dev/                 v4.0 development runs, ablations, seasonal-naive tuning, caches
│   ├── tft_v4_1/                     v4.1 run (same forecasts; its own guide: RESULTS_GUIDE_as_shipped.md)
│   ├── tft_v4_2/                     v4.2 results  <- FINAL (read with docs/RESULTS_GUIDE.md)
│   ├── tft_runs/                     where new local runs of run_v4.py are written
│   └── zips/                         the original results zip(s)
├── dev/                              farmer + government dashboards (React + FastAPI, en/mr/hi, site guide): see dev/README.md
└── archive/                          nothing here is used by the pipeline
    ├── MANIFEST.csv                  every archived file: md5, original path, new path, reason
    ├── backups/                      2026-04_pre_extend, 2026-10_pre_extend, 2026-10_v3_target
    ├── duplicates/                   byte-identical extra copies
    └── notebook_checkpoints/         Jupyter autosaves
```

---

## 8. Pipeline Order

Scripts find their files relative to their own location, so they can be started from any folder.

```bash
# ── Step 1: collect (01_data_collection/) → writes data/raw/ ─────────────────────
python 01_data_collection/groundwater_data.py
python 01_data_collection/chirps_rainfall.py        # needs: earthengine authenticate
#   then copy data/raw/rainfall_chirps_maharashtra_23_25.csv -> data/clean/rainfall_clean.csv
python 01_data_collection/era5_land.py              # ET, PET, 4 soil-moisture layers (needs Earth Engine)
python 01_data_collection/temperature_openmeteo.py
python 01_data_collection/sentinel_indices.py       # needs Earth Engine
python 01_data_collection/soil_type_gee.py          # or soil_type_api.py without Earth Engine
python 01_data_collection/cgwb_extraction.py
# extending an existing collection: run_gw_incremental.py, run_chirps_incremental.py,
# run_sentinel_incremental.py, run_temp_incremental.py, run_om_precip.py, chirps_extend_assemble.py

# ── Step 2: preprocess (02_preprocessing/) → reads data/raw/, writes data/clean/ ──
python 02_preprocessing/preprocess_groundwater.py
python 02_preprocessing/audit_groundwater_target.py # optional: target audit -> data/audit/
python 02_preprocessing/preprocess_env_features.py
python 02_preprocessing/merge_master.py             # last: writes data/clean/master_dataset.csv
#   to freeze a new training set, copy it to data/master/ under a versioned name

# ── Step 3: train + evaluate the TFT (Colab, T4 GPU) ─────────────────────────────
# Open 03_models/tft/notebooks/09_tft_colab_production_v4_3.ipynb in Colab and upload
# 03_models/tft/tft_model_v3.py, 03_models/tft/tft_v4.py, data/master/master_dataset_v4target.csv
# and docs/RESULTS_GUIDE.md. Unzip the downloaded results into 04_results/<run name>/.
# Locally: python 03_models/tft/run_v4.py --help   (writes to 04_results/tft_runs/)
```

All collection scripts are **resumable** — they track completed districts in `01_data_collection/progress/*.txt` and skip already-fetched districts on re-run. If a script is interrupted, simply re-run it and it continues from where it stopped.
