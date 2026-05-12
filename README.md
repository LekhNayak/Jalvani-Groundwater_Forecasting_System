# Jalvani — Maharashtra Groundwater Level Prediction

**Scope:** Spatiotemporal prediction of groundwater levels across all 36 districts of Maharashtra using multi-source environmental time-series data.
**Time range:** 2023-01-01 to 2025-04-22
**Type:** Research project — novel spatial-temporal deep learning architecture for hydrological forecasting

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

### 4.2 Primary Research Architecture: Temporal Fusion Transformer (TFT)

The most appropriate architecture for this problem. Designed by Google specifically for multi-horizon time-series forecasting with mixed input types. It is publication-grade and interpretable.

**Why TFT fits this problem:**
- Explicitly handles **static covariates** (district identity), **known future inputs** (month, season — known in advance), and **observed past inputs** (rainfall, NDVI, GWL, etc.)
- **Variable Selection Networks** learn which features matter at each time step — it will automatically learn that NDVI matters more in summer, VH backscatter matters more during monsoon, and GWL lag matters year-round.
- **Interpretable attention weights** show which past time steps the model focuses on — this is itself a publishable scientific result.
- Predicts next 7, 14, 30 days simultaneously in a single forward pass with calibrated uncertainty.

**Full architecture:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT LAYER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Static (fixed per district):
    district_id → learned embedding (dim=16)
    sand_0cm, clay_0cm, silt_0cm       (soil texture at surface)
    sand_10cm, clay_10cm, silt_10cm    (soil texture at root zone)
    sand_30cm, clay_30cm, silt_30cm    (soil texture at deeper zone)

  Time-varying KNOWN (future values available at inference time):
    month, season (one-hot)
    gwl_lag_7, gwl_lag_30

  Time-varying OBSERVED (only known up to present):
    gwl_mean              ← target (observed in past, predicted in future)
    rainfall_mm, rainfall_7d, rainfall_30d
    et_mm, soil_moisture
    temp_max, temp_min, temp_mean, humidity_mean
    ndvi_mean, ndwi_mean, ndmi_mean   (monthly, interpolated to daily)
    vv_db, vh_db, vh_vv_ratio         (monthly, interpolated to daily)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIABLE SELECTION NETWORKS (VSN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Separate VSNs for: static inputs, past time-varying, future time-varying.
  Each VSN learns soft gating weights over all features at every time step.
  Output: "in this month, rainfall_30d and soil_moisture are most
           important; NDVI is secondary; humidity is low priority."
  These weights are interpretable and publishable as a seasonal
  feature importance table.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEQUENCE ENCODER — LSTM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Input sequence: past 60 days of VSN-selected features
  Hidden dim: 128
  Output: encoder hidden state + full sequence of hidden states

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-HEAD SELF-ATTENTION (4 heads)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Applied over the encoder sequence.
  Learns which past days matter most for predicting future GWL.
  Attention weights per time step = interpretable result.
  Expected finding: model will attend to rainfall 14–21 days ago
  (recharge lag) and to NDMI drops 7–14 days ago (irrigation signal).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATED RESIDUAL NETWORKS (GRN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Non-linear interaction layer.
  Learns cross-feature interactions (e.g. rainfall × soil_moisture).
  ELU activation + Layer Norm + Dropout (0.1).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTILE REGRESSION OUTPUT HEAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Predicts three quantiles: q10, q50, q90
    q50 = point forecast
    [q10, q90] = 80% prediction interval
  Loss: Pinball (quantile) loss summed over all three quantiles.
  
  This is critical for water resource management:
  officials need worst-case scenarios (q10), not just averages.
  Reporting confidence intervals is also a research requirement.

  Output shape: [batch, forecast_horizon=30, 3_quantiles]
```

**Training setup:**

| Parameter | Value |
|-----------|-------|
| Lookback window | 60 days |
| Forecast horizon | 30 days |
| Train / Val / Test split | 2023–2024 train · Jan–Sep 2025 val · Oct 2025+ test |
| Optimizer | Adam, lr=1e-3 → decay to 1e-5 |
| Batch size | 64 |
| Early stopping | Patience = 20 epochs on validation quantile loss |

> **Important:** Split by time only. Never split randomly — random splits cause data leakage because consecutive days are highly correlated.

---

### 4.3 Advanced Architecture: Spatial-Temporal Graph Attention + TFT

This is the most novel contribution and the strongest research result.

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

### 4.4 Physics-Informed Loss (Optional Variant)

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

| Phase | Model | New Capability Added | Key Metric |
|-------|-------|---------------------|------------|
| 1 | SARIMA (per district) | Temporal seasonality only | RMSE, MAPE |
| 2 | XGBoost (all districts) | Non-linear feature interactions | RMSE + SHAP |
| 3 | Vanilla LSTM | Deep temporal memory | RMSE, MAE |
| 4 | TFT (single-step) | Feature selection + attention | Pinball Loss |
| 5 | TFT (multi-step, 30d) | Multi-horizon calibrated forecasts | Pinball Loss |
| 6 | GAT + TFT (spatial-temporal) | Inter-district recharge propagation | RMSE all districts |
| 7 | TFT + physics loss | Physical plausibility constraints | RMSE + violation rate |

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

```
MajorProject-Jalvani/
│
├── Data Collection/
│   ├── groundwater_data.py          # CGWB groundwater from India WRIS
│   ├── CWC_Rainfall.py              # CWC rainfall from India WRIS
│   ├── ET_data.py                   # NRSC ET from India WRIS
│   ├── soil_moisture_data.py        # NRSC soil moisture from India WRIS
│   ├── temperature_openmeteo.py     # ERA5 temp + humidity from Open-Meteo (primary)
│   ├── temperature_data.py          # CWC temperature from India WRIS (superseded by Open-Meteo)
│   ├── humidity_data.py             # CWC humidity from India WRIS (superseded by Open-Meteo)
│   ├── sentinel_indices.py          # Sentinel-1 and Sentinel-2 from GEE
│   ├── soil_type_gee.py             # Soil texture (sand/clay/silt) via OpenLandMap in GEE (primary)
│   ├── soil_type_api.py             # Soil texture via ISRIC SoilGrids REST API (fallback, no GEE needed)
│   └── txt files/                   # Per-dataset progress tracking files
│
├── Data Preprocessing/
│   ├── preprocess_groundwater.py    # Clean + aggregate + gap-fill groundwater
│   ├── preprocess_rainfall.py       # Clean + aggregate + rolling features rainfall
│   ├── preprocess_env_features.py   # Clean + merge ET, soil moisture, temp, humidity
│   ├── merge_master.py              # Final merge + feature engineering → master CSV
│   ├── extract_station_geojson.py   # Export station locations to GeoJSON
│   └── clean datasets/
│       ├── groundwater_clean.csv
│       ├── rainfall_clean.csv
│       ├── env_features_clean.csv
│       ├── master_dataset.csv       # Final training dataset
│       └── groundwater_stations.geojson
│
├── PROJECT_STATUS.md                # Data collection and preprocessing log
└── README.md                        # This file
```

---

## 8. Pipeline Order

Run in this exact order to reproduce all datasets from scratch:

```bash
# ── Step 1: Data Collection (run from Data Collection/) ────────────────
python groundwater_data.py
python CWC_Rainfall.py
python ET_data.py
python soil_moisture_data.py
python temperature_openmeteo.py
python sentinel_indices.py          # requires: earthengine authenticate
python soil_type_gee.py             # requires: earthengine authenticate + registered GCP project
# fallback if GEE unavailable:
# python soil_type_api.py           # uses ISRIC REST API, no GEE needed

# ── Step 2: Data Preprocessing (run from Data Preprocessing/) ──────────
python extract_station_geojson.py   # optional — exports station map
python preprocess_groundwater.py
python preprocess_rainfall.py
python preprocess_env_features.py
python merge_master.py              # must run last — depends on all above
```

All collection scripts are **resumable** — they track completed districts in `txt files/*.txt` and skip already-fetched districts on re-run. If a script is interrupted, simply re-run it and it continues from where it stopped.
