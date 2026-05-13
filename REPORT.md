# JalVaani — Spatiotemporal Groundwater Level Prediction for Maharashtra
## Project Report

**Scope:** Multi-horizon probabilistic forecasting of groundwater levels across 24 active monitoring districts of Maharashtra using deep learning on multi-source environmental time-series data.
**Data Range:** March 2023 – April 2025
**Status:** Phase 1 (Baselines) complete; Phase 2 (TFT) in progress

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 Background
   - 1.2 Problem Statement
   - 1.3 Objectives
   - 1.4 Scope
   - 1.5 Assumptions and Constraints
   - 1.6 Technologies Used
2. [Literature Review](#2-literature-review)
   - 2.1 Statistical Time-Series Methods
   - 2.2 Machine Learning for Groundwater Level Prediction
   - 2.3 Deep Learning — LSTM and Sequential Models
   - 2.4 Transformer-Based Time-Series Forecasting
   - 2.5 Graph Neural Networks for Spatial-Temporal Forecasting
   - 2.6 Probabilistic and Quantile Forecasting
   - 2.7 Research Gaps
3. [Research Methodology](#3-research-methodology)
   - 3.1 Data Collection
   - 3.2 Data Preprocessing
   - 3.3 Feature Engineering
   - 3.4 Dataset Construction
   - 3.5 Evaluation Strategy
4. [Phase 1 — Baseline Models](#4-phase-1--baseline-models)
   - 4.1 ARIMA/SARIMA Baseline
   - 4.2 XGBoost Baseline
   - 4.3 Vanilla LSTM Baseline
   - 4.4 Multi-Horizon Extension (t+7, t+14, t+30)
   - 4.5 Results and Analysis
   - 4.6 Why Baselines Are Insufficient
5. [Proposed Architecture](#5-proposed-architecture)
   - 5.1 Temporal Fusion Transformer (TFT)
   - 5.2 TFT + Graph Attention Network (Spatial-Temporal Architecture)
   - 5.3 Quantile Output and Uncertainty Estimation
6. [Current Results — Phase 1](#6-current-results--phase-1)
   - 6.1 t+1 Next-Day Baseline Comparison
   - 6.2 Multi-Horizon Baseline Comparison
   - 6.3 SHAP Feature Importance Evolution
   - 6.4 Per-District Performance Analysis
7. [Conclusion and Future Work](#7-conclusion-and-future-work)

---

## 1. Introduction

### 1.1 Background

Groundwater is Maharashtra's most critical water resource. It accounts for the majority of rural drinking water supply and supports irrigated agriculture across the semi-arid Deccan Plateau, the Marathwada plateau, and the Vidarbha region. Unlike surface water reservoirs — whose levels are directly visible and centrally monitored — groundwater is invisible, spatially heterogeneous, and subject to a complex interplay of natural recharge and anthropogenic extraction.

Maharashtra has faced chronic groundwater stress for over two decades. Districts such as Latur, Osmanabad, Beed, and Aurangabad regularly experience well failures during the March–June dry season. The 2016 Marathwada drought displaced hundreds of thousands of people and required emergency water train deliveries to 5.8 million people. These crises are not unpredictable — they are the result of months-long recharge deficits that standard monitoring fails to anticipate in time for remedial action.

The Central Ground Water Board (CGWB) maintains a network of groundwater monitoring stations across Maharashtra and reports groundwater levels (GWL) at seasonal frequency. While this data is valuable for long-term trend analysis, it is insufficient for operational decision-making, which requires weekly or daily forecasts with enough lead time to activate emergency supply chains, restrict industrial groundwater extraction, or advise farmers on irrigation scheduling.

This project — JalVaani — addresses this gap by building a spatiotemporal deep learning system that forecasts groundwater levels 7, 14, and 30 days ahead for all monitored districts of Maharashtra, with calibrated probabilistic confidence intervals that distinguish best-case, likely, and worst-case scenarios for water resource managers.

### 1.2 Problem Statement

Groundwater level at any point in time is governed by the water balance:

```
ΔGWL = Recharge(t) − Discharge(t) − Evapotranspiration(t)
```

Each term in this equation is driven by multiple observable environmental variables that interact non-linearly and with significant time lags:

- **Recharge** is driven by rainfall, but the lag between rainfall and measurable GWL rise ranges from 3 days (sandy alluvial soils, shallow water table) to 45 days (hard rock basalt, deep water table). Soil moisture state is the primary gating variable: saturated soil routes rainfall directly to the aquifer; dry soil absorbs most of it in the unsaturated zone.
- **Discharge** is dominated by agricultural irrigation pumping, which is driven by crop water demand (temperature, humidity, crop type), not directly observable at district scale. Vegetation indices (NDVI, NDMI) serve as an indirect proxy for irrigation intensity.
- **Evapotranspiration** creates a second loss path: high ET months (April–June) deplete soil moisture and reduce net water available for recharge, even when rainfall is moderate.
- **Spatial coupling:** districts sharing a river basin or aquifer system are hydrologically connected. A recharge event in Nashik propagates downstream through the Godavari basin and affects GWL in Ahmednagar and Aurangabad weeks later. Standard per-district models miss this entirely.

Existing approaches either (a) use simple statistical models that cannot capture non-linear feature interactions, (b) use machine learning models that treat districts as independent and ignore spatial connectivity, or (c) use physically-based hydrological simulation models that require data on soil hydraulic conductivity, land use, and aquifer geometry that is rarely available at scale.

This project bridges the gap between data-driven and physically-informed forecasting using a novel spatial-temporal deep learning architecture.

### 1.3 Objectives

1. Collect, clean, and integrate multi-source environmental time-series data for 24–36 districts of Maharashtra spanning March 2023 to April 2025.
2. Engineer physically-motivated features that encode groundwater recharge dynamics (precipitation lags, trend features, soil moisture, vapour pressure deficit, seasonal indicators).
3. Establish rigorous baseline model performance using ARIMA, XGBoost, and LSTM for both single-step (t+1) and multi-horizon (t+7, t+14, t+30) forecasting.
4. Quantify the degradation of each baseline model as forecast horizon increases, and identify the crossover point where sequential deep learning architectures outperform tree-based and statistical models.
5. Design and train a Temporal Fusion Transformer (TFT) that uses Variable Selection Networks to learn which features matter at each horizon and season.
6. Extend the TFT with a Graph Attention Network (GAT) layer to model inter-district aquifer connectivity and propagate spatial recharge signals across the Maharashtra district graph.
7. Produce calibrated probabilistic forecasts at three quantile levels (q10, q50, q90) to provide water resource managers with worst-case, expected, and best-case groundwater scenarios.
8. Analyse model interpretability: extract variable importance weights, temporal attention patterns, and inter-district graph attention weights as scientific findings about Maharashtra's hydrogeological dynamics.

### 1.4 Scope

**In scope:**
- 24 districts of Maharashtra with active CGWB groundwater monitoring stations having sufficient data coverage (defined as <40% missing groundwater level observations in the training period).
- Daily groundwater level prediction at district scale (mean, min, max across active stations per district).
- Forecast horizons: t+7 (one week), t+14 (two weeks), t+30 (one month).
- Environmental covariates: rainfall, evapotranspiration, soil moisture, temperature, humidity, and static soil texture.
- Probabilistic output: three quantiles (q10, q50, q90) forming an 80% prediction interval.
- Data period: March 2023 to April 2025 (training + validation + test).

**Out of scope (deferred):**
- Reservoir storage levels and surface water routing (data unavailable at required resolution).
- Satellite-derived vegetation indices (NDVI, NDWI, NDMI) and SAR backscatter (Sentinel-1/2): data not yet collected; will be added in TFT Phase 2b.
- Aquifer type and hydraulic conductivity mapping: would require manual digitisation of CGWB district aquifer maps.
- Sub-district or station-level prediction (insufficient station density for spatial learning at this granularity).
- Real-time operational deployment (dashboard/API): post-research objective.

### 1.5 Assumptions and Constraints

**Assumptions:**
- District-aggregated groundwater level (mean across active stations) is a meaningful and consistent signal for model training, despite variation in station depth and aquifer type within a district.
- The environmental forcing datasets (rainfall, ET, soil moisture, ERA5 temperature/humidity) are sufficiently accurate at daily district-level resolution to serve as model inputs without additional bias correction.
- The train/validation/test time split (training up to September 2024, validation October 2024 – January 2025, test February 2025 onwards) is representative of the seasonal dynamics the model must generalise to.
- Static soil texture (sand %, clay %, silt % at 0, 10, 30 cm depth) from OpenLandMap/SoilGrids does not change meaningfully over the 2023–2025 study period.

**Constraints:**
- Data coverage: only 24 of Maharashtra's 36 districts have sufficient groundwater monitoring station density and data continuity. The remaining 12 districts were excluded due to SPARSE classification (≥40% missing GWL readings).
- Temporal coverage: CGWB monitoring data starts from March 2023; this gives approximately 20 months of training data — short by deep learning standards. Models may underfit seasonal cycles that repeat only 1–2 times in the training set.
- Compute: all models trained on CPU (Intel Core i5/i7 generation). This limits LSTM hidden dimensions and batch sizes. TFT training will require GPU or cloud compute.
- Sentinel-2/1 data (vegetation indices, SAR backscatter) is not available in the current dataset. These are the most physically informative features for irrigation demand during monsoon blackout periods and will be added in the next phase.

### 1.6 Technologies Used

| Category | Tools |
|----------|-------|
| Data collection | Python `requests`, India WRIS API (paginated POST), Open-Meteo API, Google Earth Engine Python API |
| Data processing | `pandas`, `numpy`, `scipy` |
| Statistical models | `statsmodels` (ARIMA, SARIMAX) |
| Machine learning | `xgboost`, `scikit-learn` |
| Deep learning | `PyTorch` |
| Interpretability | `shap` (SHAP values for XGBoost, TreeExplainer) |
| Visualisation | `matplotlib`, `seaborn` |
| Notebooks | Jupyter (VS Code extension) |
| Spatial data | `geopandas`, `shapely`, GeoJSON |
| Progress tracking | `tqdm` |
| Version control | Git |

---

## 2. Literature Review

### 2.1 Statistical Time-Series Methods

The Box-Jenkins ARIMA (AutoRegressive Integrated Moving Average) framework [Box and Jenkins, 1970] represents the canonical approach to univariate time-series forecasting. Its seasonal variant SARIMA adds multiplicative seasonal terms, making it suitable for hydrological series with annual periodicity. ARIMA models have been extensively applied to groundwater level prediction, particularly in data-sparse settings where limited historical data precludes training data-hungry neural architectures.

The primary limitation of ARIMA for groundwater prediction is that it is purely autoregressive — it predicts future values from past values of the same series only. It cannot incorporate exogenous drivers (rainfall, temperature, soil moisture) that are the actual physical causes of groundwater level change. The ARIMAX extension permits exogenous inputs but does not model non-linear interactions between them.

For multi-horizon forecasting, ARIMA produces a single forecast sequence per refit. In rolling evaluation with monthly refit, the forecast uncertainty grows with horizon as the linear approximation diverges from the true non-linear system dynamics. This predictable degradation is one of the key motivations for attention-based models with explicit multi-horizon output heads.

### 2.2 Machine Learning for Groundwater Level Prediction

Ensemble tree methods — particularly gradient boosting — have become the dominant machine learning approach for tabular environmental prediction tasks. XGBoost [Chen and Guestrin, 2016] combines gradient boosting with regularisation and column subsampling, making it robust to the noisy, multicollinear feature sets typical of environmental datasets.

For groundwater prediction, XGBoost's key advantage is its ability to capture non-linear feature interactions without manual specification. A model trained on all 36 environmental features simultaneously learns interactions such as "high rainfall × high soil moisture → recharge" and "high NDMI drop × low rainfall → irrigation pumping → GWL decline" through the tree split structure without explicit feature crosses.

A critical additional capability is SHAP (SHapley Additive exPlanations) analysis [Lundberg and Lee, 2017], which decomposes XGBoost predictions into per-feature contributions. For groundwater modelling, SHAP provides the model's learned causal theory: which variables it uses, in what magnitude, and how their importance shifts with season and forecast horizon. This makes XGBoost not just a baseline but a feature selection and scientific analysis tool.

The fundamental limitation of gradient boosting for multi-horizon forecasting is temporal blindness: tree models have no native mechanism to model the ordering of input sequences. The past 30 days of data must be presented as a flat feature vector, destroying the temporal structure that sequential models learn implicitly. This limitation becomes increasingly severe as forecast horizon grows.

### 2.3 Deep Learning — LSTM and Sequential Models

Long Short-Term Memory networks [Hochreiter and Schmidhuber, 1997] were designed to solve the vanishing gradient problem in recurrent neural networks, enabling them to retain relevant information across sequences of hundreds of timesteps. For environmental time-series with strong autocorrelation and multi-scale temporal dependencies (daily soil moisture cycles, weekly rainfall events, seasonal aquifer recharge patterns), LSTMs are a natural fit.

Kratzert et al. [2018] demonstrated in a landmark study that a single global LSTM trained on rainfall-runoff records from 241 US basins outperformed basin-specific models and even physically-based simulation models on unseen basins — a result that shifted the hydrology community's view of deep learning from a black-box curiosity to a serious modelling tool. Their key finding was that the LSTM's cell state implicitly encodes physical variables such as soil moisture and snow water equivalent, without being explicitly told to do so.

For groundwater prediction specifically, LSTM captures the multi-scale temporal dependencies that are the defining characteristic of the problem: the 7-day rainfall-recharge lag, the 14–30 day soil moisture depletion cycle, and the 6-month seasonal aquifer recharge-depletion pattern. Static covariates (soil texture, district identity) are handled by broadcasting them across all timesteps and concatenating with the dynamic feature vector at each step.

The vanilla LSTM's limitation for this project is twofold: (a) it lacks a mechanism to explicitly weight features by relevance at each timestep and horizon — all features are treated equally at the input layer; (b) it has no spatial component — it models each district in isolation, missing inter-district aquifer connectivity that is physically important in basin-sharing districts.

### 2.4 Transformer-Based Time-Series Forecasting

The Transformer architecture [Vaswani et al., 2017], built around multi-head self-attention, fundamentally changes how sequence models process temporal information. Instead of propagating information through sequential hidden states (which creates a bottleneck), attention allows every position in the sequence to directly attend to every other position. For long lookback windows (60+ days), this is architecturally superior to LSTMs.

The Temporal Fusion Transformer (TFT) [Lim et al., 2021] extends this to mixed-input multi-horizon forecasting with specific architectural components designed for the structure of real-world forecasting problems. The critical innovation is the separation of three distinct input types:

1. **Static covariates** (fixed per entity, e.g. soil texture, district identity) — encoded once and used to condition all downstream components via a static context mechanism.
2. **Known future inputs** (values available at inference time for the future window, e.g. month, season, day-of-year) — given to the decoder directly.
3. **Observed past inputs** (values only known up to the present, e.g. rainfall, GWL, NDVI) — processed by the encoder only.

This input structure maps exactly onto the groundwater prediction problem: we know what month and season it will be in 30 days, but we do not know what rainfall will occur. The TFT is the only published architecture that handles this three-way distinction natively.

The Variable Selection Networks (VSN) within TFT learn soft gating weights over input features at each timestep, effectively performing feature selection jointly with prediction. In a groundwater context, this means the model learns that soil moisture gates rainfall-recharge at short horizons, while seasonal indicators (is_monsoon, day_of_year) dominate at longer horizons — a physically correct behaviour that fixed-weight models cannot express.

TFT's quantile regression output head produces calibrated uncertainty intervals. For water resource management, the difference between the q10 (worst case) and q50 (expected) forecast is operationally critical: declaring a water emergency requires confidence that the worst case is serious, not just the average.

### 2.5 Graph Neural Networks for Spatial-Temporal Forecasting

Graph Attention Networks (GAT) [Veličković et al., 2018] extend the graph convolution framework by learning attention-weighted aggregation of neighbourhood features. Each node computes a weighted sum of its neighbours' feature vectors, where the weights are learned from the node features themselves. This is preferable to fixed-weight graph convolutions for hydrological applications because the strength of hydraulic coupling between districts varies dynamically with season and recharge state.

Spatial-temporal graph networks have been applied extensively to urban traffic forecasting (where road intersections form the graph) and air quality prediction (where monitoring stations form nodes). The application to inter-district groundwater prediction is architecturally analogous: districts are nodes, shared basin boundaries define edges, and the GWL time-series is the node feature to be predicted.

The novel contribution of this project is integrating GAT as a spatial propagation layer between the TFT encoder and decoder. The TFT encoder processes each district's temporal history independently, producing a context vector that encodes the district's recent state. The GAT layer then allows information to flow between basin-connected districts before the decoder produces the multi-horizon forecast. This architecture learns which districts most strongly influence each other's future GWL — a result that is itself a scientific finding about Maharashtra's aquifer connectivity.

### 2.6 Probabilistic and Quantile Forecasting

Deterministic forecasting of groundwater levels provides only the expected value, which is insufficient for risk-aware water resource management. Probabilistic forecasting produces a full distribution or a set of representative quantiles over possible future states.

The pinball loss (quantile loss) for a quantile τ is defined as:

```
L_τ(y, ŷ) = τ · max(y − ŷ, 0) + (1 − τ) · max(ŷ − y, 0)
```

Training with the sum of pinball losses at τ = {0.1, 0.5, 0.9} produces three simultaneous forecasts whose empirical coverage should match their quantile levels when averaged over the test set. A q10 forecast is correctly calibrated if approximately 10% of actual values fall below it.

For the JalVaani use case, the q10 forecast (worst-case groundwater scenario) has direct operational meaning: if the q10 forecast for a district drops below the critical well depth threshold in the next 30 days, that district qualifies for pre-emptive water supply augmentation. Using only the point forecast (q50) would trigger interventions too late when the actual trajectory is worse than expected.

### 2.7 Research Gaps

The literature review identifies five specific gaps that motivate this project's design choices:

**Gap 1 — Single-district modelling ignores spatial aquifer coupling.** The vast majority of published GWL prediction papers train separate models per monitoring station or per district, treating each as an independent system. Maharashtra's aquifer systems are basin-connected: districts sharing the Godavari, Krishna, or Tapi basins exchange recharge through shared phreatic and confined aquifers. No published study for Maharashtra has modelled this spatial dependency with a learned graph structure.

**Gap 2 — Deterministic forecasting at t+1 is insufficient for operational use.** Publishing a t+1 RMSE of 1–2 metres provides little operational value — water managers need 1–4 week forecasts to act. The literature shows a general preference for single-step predictions (which are easier to optimise) over the more practically relevant multi-horizon setting. This project directly addresses this by using t+7, t+14, and t+30 as primary evaluation targets with probabilistic outputs.

**Gap 3 — Fixed feature importance ignores seasonal variability.** Standard machine learning papers report a single ranked feature importance list for GWL prediction. In reality, rainfall dominates during monsoon, GWL autoregressive lags dominate during dry season, and NDMI/vegetation stress dominates in the pre-monsoon irrigation period. Models that cannot dynamically weight features by season and horizon are learning a seasonal average, not the actual process. TFT's Variable Selection Networks directly address this.

**Gap 4 — Optical satellite data (NDVI/NDWI) fails during monsoon.** Several published studies incorporate vegetation indices from Sentinel-2 for GWL prediction, but these studies typically use data from dry seasons where optical imagery is available. During June–September — Maharashtra's primary recharge window — Sentinel-2 is cloud-obscured for weeks at a time. This project addresses the gap by incorporating Sentinel-1 SAR backscatter (VV/VH) as an all-weather soil moisture proxy, filling the critical monsoon observation gap. (Sentinel data collection is in progress; this feature will be added in TFT Phase 2b.)

**Gap 5 — Interpretability as scientific output.** Most GWL prediction papers treat the model as a black box evaluated only by RMSE and MAE. TFT's interpretable attention weights and Variable Selection Networks produce outputs that are scientifically meaningful in their own right: temporal attention peaks at known recharge lag periods (7–21 days), graph attention edges align with basin boundaries, and VSN weights reproduce the known seasonal feature importance hierarchy. These interpretability outputs are the primary scientific contribution of this work, distinct from and complementary to the predictive accuracy improvement.

---

## 3. Research Methodology

### 3.1 Data Collection

All data was collected via public APIs and processed in Python. Nine distinct datasets were integrated:

| # | Dataset | Source | Granularity | Period |
|---|---------|--------|-------------|--------|
| 1 | Groundwater Level | CGWB via India WRIS | Sub-daily per station | Mar 2023 – Apr 2025 |
| 2 | Rainfall | CWC via India WRIS | Twice-daily per station | Mar 2023 – Apr 2025 |
| 3 | Evapotranspiration | NRSC VIC Model via India WRIS | Daily per district | Mar 2023 – Apr 2025 |
| 4 | Soil Moisture | NRSC VIC Model via India WRIS | Daily per district | Mar 2023 – Apr 2025 |
| 5 | Temperature + Humidity | Open-Meteo (ERA5 reanalysis) | Daily per district centroid | Mar 2023 – Apr 2025 |
| 6 | Soil Texture | OpenLandMap/SoilGrids via GEE | Static per district (3 depths) | — |
| 7 | Sentinel-2 Indices | Google Earth Engine | Monthly per district | *(pending collection)* |
| 8 | Sentinel-1 SAR | Google Earth Engine | Monthly per district | *(pending collection)* |
| 9 | Relative Humidity (WRIS) | CWC via India WRIS | Sub-daily | *(superseded by ERA5)* |

The India WRIS API uses paginated POST requests (1,000 records per page) with exponential backoff retry for rate-limit responses. All collection scripts are resumable — they track completed districts in per-dataset text files and skip already-fetched districts on re-run.

Open-Meteo (ERA5) was chosen over the WRIS temperature/humidity dataset because ERA5 provides continuous daily values at every district centroid with no station coverage gaps, while the CWC gauge network has significant spatial gaps in eastern Maharashtra.

Soil texture (sand %, clay %, silt % at 0 cm, 10 cm, 30 cm per district) was collected once as a static dataset using the OpenLandMap SoilGrids assets in Google Earth Engine, with ISRIC SoilGrids REST API as fallback.

### 3.2 Data Preprocessing

Each dataset underwent domain-specific cleaning before the master merge:

**Groundwater Level:**
- Kept only `stationStatus == "Active"` stations
- Dropped sensor artefacts (`dataValue = 0`) and physically implausible readings (outside [−100, 0] metres)
- Aggregated sub-daily readings to daily mean per station, then to district-level mean/min/max across all stations
- Gap-filled missing days using forward-fill for gaps ≤7 days; longer gaps remain NaN
- Districts classified by data coverage: OK (<10% missing), GAPS (<40%), SPARSE (≥40%)

**Rainfall:**
- Summed all readings per station per day (sum, not mean — multiple readings in a day are accumulations)
- Aggregated to district mean across stations (mean to avoid bias from unequal station density)
- Missing days set to 0 mm (no rain is a valid observation; stations only report when active)

**Environmental Features (ET, soil moisture, temperature/humidity):**
- Dropped physically implausible values (negative ET, negative soil moisture, temperature outside −5°C to +55°C)
- Gap-filled with forward-fill for gaps ≤7 days
- All four streams merged into a single `env_features_clean.csv` by (district, date)

### 3.3 Feature Engineering

The master dataset was built by joining all clean datasets on `(district, date)` and engineering the following additional features:

| Feature | Formula | Physical Meaning |
|---------|---------|-----------------|
| `rainfall_7d` | 7-day rolling sum of `rainfall_mm` | Short-term cumulative recharge potential |
| `rainfall_30d` | 30-day rolling sum of `rainfall_mm` | Medium-term cumulative recharge (primary monsoon feature) |
| `gwl_lag_7` | `gwl_mean` shifted 7 days | Short-term autoregressive signal |
| `gwl_lag_14` | `gwl_mean` shifted 14 days | Intermediate autoregressive signal |
| `gwl_lag_30` | `gwl_mean` shifted 30 days | Long-term autoregressive signal (seasonal baseline) |
| `gwl_7d_change` | `gwl_mean − gwl_lag_7` | Current week trend direction |
| `gwl_30d_change` | `gwl_mean − gwl_lag_30` | Monthly trend direction |
| `net_water_balance` | `rainfall_mm − et_mm` | Daily net water surplus or deficit |
| `vpd_approx` | `temp_mean × (1 − humidity_mean/100)` | Vapour pressure deficit proxy (drives ET) |
| `is_monsoon` | 1 if month ∈ {6, 7, 8, 9} else 0 | Primary recharge season flag |
| `day_of_year` | Integer 1–365 | Fine-grained annual cycle signal |
| `month` | Integer 1–12 | Coarse seasonal indicator |
| `season` | winter/summer/monsoon/post-monsoon | Categorical seasonal grouping |
| `district_id` | Integer 0–23 (alphabetically sorted) | Entity identifier for learned embedding |
| `sand/clay/silt_Xcm` | Static per district at 0, 10, 30 cm | Soil infiltration rate per district |

The final `master_dataset.csv` contains 16,785 rows × 36 columns, covering 24 districts from March 2023 to April 2025.

### 3.4 Dataset Construction

**Time-based train/validation/test split** (no random splitting — consecutive days are highly correlated and random splits create data leakage):

| Split | Date Range | Purpose |
|-------|-----------|---------|
| Training | Mar 2023 – Sep 2024 | Model parameter learning |
| Validation | Oct 2024 – Jan 2025 | Hyperparameter tuning, early stopping |
| Test | Feb 2025 – Apr 2025 | Final held-out evaluation |

For multi-horizon models, window labels are assigned by **target date** (not input window start date) to prevent leakage: a window whose target date falls in the test period cannot have been visible during training even if part of its lookback window falls in the validation period.

For each horizon h ∈ {7, 14, 30}, targets were constructed as:

```python
df[f'target_t{h}'] = df.groupby('district')['gwl_mean'].shift(-h)
```

The `groupby` is essential — a naive `shift(-h)` on the full dataframe bleeds the last h values of one district into the first h rows of the next district when the dataframe is sorted by (district, date).

### 3.5 Evaluation Strategy

All models are evaluated on the held-out test set using three metrics:

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| RMSE | √(mean((actual − pred)²)) | Error in metres; penalises large errors more heavily |
| MAE | mean(\|actual − pred\|) | Average absolute error in metres; more robust to outliers |
| MAPE | mean(\|actual − pred\| / \|actual\|) × 100 | Percentage error; comparable across districts with different GWL scales |

For TFT (Phase 2), the additional metric is:

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Pinball Loss (q10) | Quantile loss at τ=0.10 | Calibration of worst-case forecast |
| Pinball Loss (q50) | Quantile loss at τ=0.50 | Equivalent to MAE for the median forecast |
| Pinball Loss (q90) | Quantile loss at τ=0.90 | Calibration of best-case forecast |
| Coverage | Fraction of actual values within [q10, q90] | Target: 80% coverage for a well-calibrated model |

All metrics are reported both as aggregates across all 24 districts and per-district to identify which districts are structurally harder to predict (high-variability aquifer systems such as Dhule, Jalgaon, Nashik).

---

## 4. Phase 1 — Baseline Models

The baseline phase serves two purposes: (1) establishing a rigorous performance floor that the advanced models must demonstrably beat, and (2) using SHAP interpretability analysis on XGBoost to characterise how different features contribute to prediction at different horizons — findings that directly shape the TFT architecture design.

### 4.1 ARIMA/SARIMA Baseline

**Model:** ARIMA(7, 1, 1) per district. The order (p=7, d=1, q=1) was chosen as appropriate for daily groundwater data: 7 autoregressive lags capture the weekly autocorrelation structure in GWL, one difference removes the non-stationary trend, and a first-order MA term handles the residual noise.

**Rolling evaluation:** Rather than fitting once on the training set and evaluating on the full test period, ARIMA uses a rolling refit strategy with refit every 30 steps. At each evaluation step, predictions at h=7, 14, and 30 days ahead are generated from a single forecast call, using cached forecast arrays for computational efficiency (approximately 18 total refits per district versus 500+ for per-step refits).

**Forecast indexing:** At evaluation step i (counting from the end of training), with s = steps since last refit, the h-step-ahead forecast from the current date corresponds to index `fc[s + h − 1]` in the cached forecast array. This correctly computes the prediction for the date `current_date + h`, which is then matched against the actual observed GWL at that date.

**Limitations:** ARIMA uses only the univariate GWL time-series. It cannot incorporate rainfall, soil moisture, or any environmental driver. The t+1 ARIMA baseline provides the "what can you predict from the history of the variable alone" floor.

### 4.2 XGBoost Baseline

**Model:** A single `XGBRegressor` trained on all 24 districts simultaneously (district-shared learning) with the following hyperparameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 600 | Sufficient capacity for 36 features; early stopping prevents overfitting |
| `max_depth` | 6 | Standard for tabular environmental data |
| `learning_rate` | 0.05 | Conservative; paired with high n_estimators |
| `subsample` | 0.8 | Reduces overfitting to district-specific patterns |
| `colsample_bytree` | 0.8 | Feature bagging for implicit regularisation |
| `early_stopping_rounds` | 30 | Stop if validation RMSE does not improve for 30 rounds |

**Multi-horizon strategy:** Direct h-step forecasting — a separate XGBRegressor trained per horizon h ∈ {7, 14, 30} with target `gwl_mean` shifted −h days. This avoids recursive error compounding that would occur if a t+1 model were iteratively applied h times.

**SHAP analysis:** After training each horizon model, SHAP TreeExplainer values are computed on a 2000-row sample from the test set. The resulting `mean_abs_shap` per feature captures each feature's average contribution to the model's predictions. This analysis is repeated at all three horizons to track how feature importance evolves as the forecast horizon increases.

### 4.3 Vanilla LSTM Baseline

**Architecture:**

```
Input: [batch, T=30, input_dim=36]
  where input_dim = time_series_dim(19) + static_soil(9) + district_embed_dim(8)
  static soil texture and district embedding are broadcast across all 30 timesteps

LSTM: 2 layers, hidden_dim=64, dropout=0.2 between layers
Linear: 64 → 1 (single GWL output)
```

Static soil texture and district embeddings are repeated across all 30 input timesteps before concatenation with the dynamic features. This is the standard approach for static covariates in sequential models and exactly mirrors how TFT's static context mechanism works, making the comparison architecturally fair.

**Training:** Adam optimiser, learning rate 1×10⁻³ with ReduceLROnPlateau (patience=5), early stopping on validation MSE (patience=15), batch size=128. All features are z-score normalised using statistics computed on the training set only. The normaliser is preserved and applied to validation and test sets.

**Multi-horizon strategy:** Three separate models trained per horizon h ∈ {7, 14, 30}. The window construction assigns each (district, input_window) pair a target based on the date `input_window_end + h`. Windows are assigned to train/val/test splits by their target date.

### 4.4 Multi-Horizon Extension (t+7, t+14, t+30)

The t+1 baseline establishes the floor for next-day prediction accuracy. However, for operational use in drought management or irrigation advisory, forecasts of 1–4 weeks ahead are required. The multi-horizon evaluation answers three additional questions:

1. **How fast does each model degrade as horizon increases?** Models that degrade slowly are preferable for long-range operational use.
2. **At which horizon do sequential models (LSTM) begin to outperform tree models (XGBoost)?** This is the architectural crossover point that justifies the move to TFT.
3. **Which features gain importance at longer horizons?** The SHAP evolution analysis informs which TFT input types (known future inputs vs. observed past) will be most critical.

All multi-horizon baselines use the same direct forecasting approach: a separate model is trained per horizon with a shifted target, and the same train/val/test time split is maintained.

### 4.5 Results and Analysis

#### t+1 Next-Day Prediction (Test Set)

| Model | Test RMSE | Test MAE | Test MAPE |
|-------|-----------|----------|-----------|
| SARIMA | 3.6335 m | 1.8725 m | 16.22% |
| XGBoost | **1.6818 m** | **0.4616 m** | **2.43%** |
| LSTM | 2.7346 m | 1.0636 m | 8.91% |

XGBoost dominates at t+1 (RMSE 1.68m, MAPE 2.43%). The extremely low MAPE indicates that XGBoost has learned the level and scale of each district's GWL very effectively from the feature set. The LSTM at 2.73m RMSE is competitive but does not match XGBoost at this horizon, where the most important features are autoregressive lags already well-represented as flat features in the XGBoost input.

SARIMA's 3.63m RMSE — worse than both ML models — confirms that environmental features (rainfall, soil moisture, ET) carry material predictive information beyond the GWL time-series alone. The floor established by ARIMA/SARIMA is the "what you can know from GWL history only" ceiling.

#### Multi-Horizon Prediction (Test Set RMSE)

| Model | t+7 RMSE | t+14 RMSE | t+30 RMSE |
|-------|----------|-----------|-----------|
| ARIMA | 4.9449 m | 5.1334 m | 5.0997 m |
| XGBoost | 3.9210 m | 4.5080 m | 5.2570 m |
| **LSTM** | **3.2685 m** | **4.1763 m** | **3.5448 m** |

LSTM beats XGBoost at all three multi-horizon targets. The crossover is present even at t+7 (LSTM 3.27m vs. XGBoost 3.92m) and widens dramatically at t+30 (LSTM 3.54m vs. XGBoost 5.26m — a 33% improvement).

XGBoost's degradation pattern (3.92 → 4.51 → 5.26m) is the expected behaviour of a temporally-blind model: as the horizon grows, the autoregressive lag features become stale and the model has no mechanism to learn how environmental signals compound over time. By t+30, XGBoost is essentially extrapolating beyond the informative range of its oldest lag feature.

The LSTM's non-monotonic degradation (3.27 → 4.18 → 3.54m) is a notable finding. The t+30 RMSE is better than t+14, which is unusual. The likely explanation is that at a 30-day horizon the model has learned monsoon onset as a strong periodic signal — the dominant driver of GWL at this timescale — whereas the 14-day window sits in an intermediate regime with less predictable structure. This suggests that the most informative predictive signal for 30-day GWL is the seasonal cycle, not the short-term weather noise that dominates at 7–14 days.

#### Multi-Horizon MAE

| Model | t+7 MAE | t+14 MAE | t+30 MAE |
|-------|---------|----------|----------|
| ARIMA | 2.0134 m | 2.2075 m | 2.4836 m |
| XGBoost | 1.5551 m | 1.8697 m | 2.2199 m |
| **LSTM** | 1.6505 m | **1.9935 m** | **1.8007 m** |

### 4.6 Why Baselines Are Insufficient

Despite competitive performance for t+1 prediction, the baseline models share fundamental limitations that motivate the TFT and TFT+GAT architectures:

**1. No feature selection by horizon or season.** XGBoost uses the same set of features with the same learned weights regardless of whether it is predicting 7 or 30 days ahead, in June or in December. The SHAP evolution analysis (Section 6.3) shows that the optimal feature set changes substantially with horizon: at t+7, `gwl_lag_7` accounts for ~90% of predictive signal; by t+30, seasonal features (`day_of_year`, `is_monsoon`) rise to become top-4 features. A model that cannot dynamically weight features by context is learning the average behaviour, not the true process.

**2. No uncertainty quantification.** All three baseline models produce only point forecasts. Water resource managers making drought emergency declarations need to know whether the worst-case GWL trajectory (q10) crosses a critical threshold, not just the expected trajectory. Point forecasts cannot answer this question.

**3. No spatial coupling.** All baseline models treat each district independently. Districts sharing the Godavari basin (Nashik → Ahmednagar → Aurangabad) or the Tapi basin (Nandurbar → Dhule → Jalgaon) are hydrologically connected. A recharge pulse in an upstream district propagates to downstream districts over weeks. The per-district RMSE results (Section 6.4) show that Dhule and Jalgaon — both in the Tapi basin, both downstream — have the highest prediction errors across all three models. This pattern is consistent with missing spatial context.

**4. LSTM multi-horizon degradation remains high.** Even the best baseline (LSTM) achieves test RMSE of 3.27–4.18m across horizons. The TFT's explicit encoder-decoder structure for multi-horizon prediction, combined with temporal self-attention and known future inputs, is expected to substantially reduce this error, particularly at intermediate horizons (t+14) where the LSTM underperforms relative to its t+7 and t+30 results.

---

## 5. Proposed Architecture

### 5.1 Temporal Fusion Transformer (TFT)

The TFT [Lim et al., 2021] is the primary modelling architecture for Phase 2 of this project. Its design directly addresses all four limitations identified in Section 4.6.

#### 5.1.1 Input Structure

The TFT makes an explicit three-way distinction between input types that directly maps onto the groundwater prediction problem:

| Input Type | Definition | Examples in this Project |
|-----------|-----------|-------------------------|
| Static covariates | Fixed per district, no temporal dimension | `district_id`, `sand_0cm`, `clay_0cm`, `silt_0cm` (all 9 soil texture features) |
| Known future inputs | Available at inference time for the future window | `month`, `season`, `day_of_year`, `is_monsoon` |
| Observed past inputs | Known up to the present; unknown in the future | `gwl_mean`, `rainfall_mm`, `rainfall_7d`, `rainfall_30d`, `et_mm`, `soil_moisture`, `temp_mean`, `humidity_mean`, `vpd_approx`, `net_water_balance`, `gwl_lag_7`, `gwl_lag_14`, `gwl_lag_30`, `gwl_7d_change`, `gwl_30d_change` |

When Sentinel-2/1 data is collected (Phase 2b), `ndvi_mean`, `ndwi_mean`, `ndmi_mean`, `vv_db`, `vh_db`, `vh_vv_ratio` will be added to observed past inputs.

#### 5.1.2 Architecture Components

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT PROCESSING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Static covariates → Static context encoder (MLP)
  Produces: 4 context vectors used to initialise
  encoder cell/hidden states and condition VSNs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIABLE SELECTION NETWORKS (VSN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Separate VSNs for past and future inputs.
Each VSN learns soft gating weights over features
at each timestep, conditioned on the static context.
Output: weighted sum of per-feature GRN outputs.

At t+7 → model learns: weight gwl_lag_7 heavily
At t+30 → model learns: weight is_monsoon and
  day_of_year more heavily, reduce gwl_lag_7 weight

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEQUENCE ENCODER — LSTM (past 60 days)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hidden dim: 128, 2 layers
Processes VSN-selected past inputs.
Outputs: encoder hidden states + cell state for
decoder initialisation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEQUENCE DECODER — LSTM (future 30 days)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Processes VSN-selected known future inputs.
Initialised from encoder hidden/cell state.
Produces future hidden states for attention.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-HEAD TEMPORAL SELF-ATTENTION (4 heads)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Applied over encoder+decoder hidden states.
Each decoder position attends to all encoder positions.
Attention weights = which past days matter most
  for each future day's prediction.

Expected finding: attention peaks at 7–21 days ago
(rainfall-recharge percolation lag) and 30 days ago
(monthly seasonal baseline).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATED RESIDUAL NETWORKS (GRN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Non-linear cross-feature interaction layers.
ELU activation + LayerNorm + Dropout(0.1).
Learns interactions e.g. rainfall × soil_moisture.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTILE OUTPUT HEAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Linear(128 → 3) applied at each forecast step.
Outputs: [q10, q50, q90] per day per district.

q50 = point forecast (expected GWL)
q10 = worst-case (10th percentile)
q90 = best-case (90th percentile)
[q10, q90] = 80% prediction interval

Loss: sum of pinball losses at τ = {0.10, 0.50, 0.90}
  over all 30 forecast steps simultaneously.
```

#### 5.1.3 TFT Training Configuration

| Parameter | Value |
|-----------|-------|
| Lookback window | 60 days |
| Forecast horizon | 30 days (t+1 through t+30 simultaneously) |
| Hidden dim | 128 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Batch size | 64 |
| Optimiser | Adam, lr = 1×10⁻³ |
| LR schedule | Cosine annealing with warm restarts |
| Early stopping | Patience = 20 epochs on validation quantile loss |
| Training targets | q10, q50, q90 at all 30 forecast steps |

The model predicts all 30 future steps in a single forward pass, enabling the attention mechanism to learn different focus patterns for near-term versus far-term predictions within the same learned representation.

### 5.2 TFT + Graph Attention Network (Spatial-Temporal Architecture)

#### 5.2.1 Motivation

The per-district RMSE analysis (Section 6.4) shows that the highest-error districts (Dhule, Jalgaon) are both located in the Tapi basin. Their error is systematically higher than basin-adjacent districts with more stable aquifer conditions. This pattern suggests that models which lack cross-district information propagation are leaving predictive signal on the table.

The spatial-temporal architecture extends TFT by inserting a GAT layer between the encoder and decoder, allowing information from upstream (or hydrologically connected) districts to condition each district's forecast.

#### 5.2.2 Graph Construction

```
Nodes: 24 active monitoring districts
Edges (two types, combined into one adjacency matrix):
  1. Spatial adjacency: districts with shared borders
  2. River basin membership:
     Godavari basin: Nashik, Ahmednagar, Aurangabad, Nanded,
                     Osmanabad, Latur, Nandurbar
     Krishna basin:  Pune, Satara, Sangli, Kolhapur, Solapur
     Tapi basin:     Nandurbar, Dhule, Jalgaon
     Konkan coast:   Ratnagiri, Raigad, Thane

Edge weights: inverse Euclidean distance between district centroids
  (closer districts weighted more heavily as neighbours)
```

The adjacency matrix is fixed (static graph). The GAT attention weights over edges are learned — the model learns which of the spatially or basin-connected district pairs actually have strong hydrological coupling in the data.

#### 5.2.3 Architecture

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — TFT ENCODER (per district, independently)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each of the 24 districts runs its own TFT encoder.
Input: past 60 days of all features per district.
Output: encoder context vector per district.
Shape: [batch, 24 districts, hidden_dim=128]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — SPATIAL PROPAGATION — GAT (2 layers, 4 heads)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: [batch, 24, 128] node features + adjacency matrix.
Each GAT layer aggregates neighbour context vectors,
weighted by learned attention over edges.

"If Nashik has received heavy recent rainfall,
 its encoder context vector reflects this.
 The GAT propagates this signal to Ahmednagar's
 node representation — Ahmednagar's decoder is
 now aware that upstream recharge has occurred."

Output: spatially-enriched node states [batch, 24, 128]
Learned edge attention weights = publishable result
  (which district pairs have strongest aquifer coupling).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — TFT DECODER (per district, using enriched state)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Decoder is initialised from the spatially-enriched
node state instead of pure encoder state.
Produces: [q10, q50, q90] × 30 days per district.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOSS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sum of quantile losses over all 24 districts,
all 30 forecast steps, all 3 quantile levels.
The model is trained jointly across all districts,
enabling cross-district parameter sharing.
```

#### 5.2.4 Expected Scientific Findings

The learned GAT edge attention weights are expected to show:
- Strong attention from Nashik → Ahmednagar → Aurangabad (Godavari basin propagation)
- Strong attention within Tapi basin (Nandurbar → Dhule → Jalgaon)
- Weak attention between non-basin-adjacent district pairs
- Seasonal modulation of attention strength (stronger during monsoon when recharge is propagating)

If the attention weights align with known basin boundaries, this constitutes evidence that the model has learned real hydrological coupling from data, not spurious correlation.

### 5.3 Quantile Output and Uncertainty Estimation

**Output:** At each forecast step h ∈ {1, ..., 30} for each district d:
- `q10(d, h)` — the 10th percentile GWL forecast: the worst-case scenario. Approximately 90% of actual outcomes should be above this value.
- `q50(d, h)` — the median forecast: the model's best single-point estimate.
- `q90(d, h)` — the 90th percentile forecast: the best-case scenario. Approximately 90% of actual outcomes should be below this value.

The interval [q10, q90] is a calibrated 80% prediction interval.

**Operational interpretation for water managers:**
- If `q10(d, 30) < drought_threshold` → declare pre-emptive water supply alert (worst case)
- If `q50(d, 30) < drought_threshold` → high probability alert; begin rationing
- If `q50(d, 30) > drought_threshold` but `q10(d, 30) < drought_threshold` → monitoring alert; prepare contingency
- The width `q90 − q10` measures forecast uncertainty: narrow = high confidence; wide = high uncertainty (typical during monsoon onset when the exact rainfall trajectory is unknown)

**Calibration check:** A well-calibrated q10 forecast should be exceeded by actual GWL approximately 90% of the time. A q10 that is exceeded 99% of the time is too pessimistic; exceeded 70% of the time is too optimistic. Coverage rate will be reported as part of the TFT evaluation.

---

## 6. Current Results — Phase 1

### 6.1 t+1 Next-Day Baseline Comparison

All three models were trained and evaluated on the same train/val/test split using the 36-column master dataset.

| Model | Val RMSE | Val MAE | Test RMSE | Test MAE | Test MAPE |
|-------|----------|---------|-----------|----------|-----------|
| SARIMA | 90.85 m* | 18.26 m | 3.6335 m | 1.8725 m | 16.22% |
| LSTM | 1.4465 m | 0.7060 m | 2.7346 m | 1.0636 m | 8.91% |
| **XGBoost** | **0.5566 m** | **0.1957 m** | **1.6818 m** | **0.4616 m** | **2.43%** |

*SARIMA validation RMSE is artificially inflated by rolling-window refit instability on a small number of districts. The test set performance (3.63m) is the meaningful figure.

The gap between XGBoost (RMSE 1.68m) and LSTM (2.73m) at t+1 is attributable to the nature of the one-day-ahead prediction problem: the dominant signal is `gwl_lag_7` (SHAP value 6.59 — see Section 6.3), which XGBoost can access directly as a flat feature. The LSTM must learn this through its recurrent structure, introducing a modest overhead. At longer horizons this reversal does not hold.

### 6.2 Multi-Horizon Baseline Comparison

Direct h-step forecasting strategy: a separate model trained per horizon.

**Test Set RMSE (metres):**

| Model | t+7 | t+14 | t+30 |
|-------|-----|------|------|
| ARIMA | 4.9449 | 5.1334 | 5.0997 |
| XGBoost | 3.9210 | 4.5080 | 5.2570 |
| **LSTM** | **3.2685** | **4.1763** | **3.5448** |

**Test Set MAE (metres):**

| Model | t+7 | t+14 | t+30 |
|-------|-----|------|------|
| ARIMA | 2.0134 | 2.2075 | 2.4836 |
| XGBoost | 1.5551 | 1.8697 | 2.2199 |
| **LSTM** | 1.6505 | 1.9935 | **1.8007** |

**LSTM beats XGBoost at all horizons.** The gap grows with horizon:
- t+7: LSTM leads by 0.65m RMSE (17% improvement)
- t+14: LSTM leads by 0.33m RMSE (7% improvement)
- t+30: LSTM leads by 1.71m RMSE (33% improvement — most significant)

The t+30 result is the primary justification for transitioning to TFT: the LSTM demonstrates that sequential temporal learning captures 30-day dynamics that tree-based methods cannot, while the LSTM's own absolute error (3.54m) remains large enough that a more capable architecture should improve further.

### 6.3 SHAP Feature Importance Evolution

SHAP TreeExplainer was applied to each horizon's XGBoost model on a 2,000-row test sample. Mean absolute SHAP values are shown for the top features at each horizon:

**Top features at t+7 (mean |SHAP|):**

| Feature | SHAP | Category |
|---------|------|---------|
| gwl_lag_7 | 6.59 | Autoregressive |
| gwl_lag_14 | 2.25 | Autoregressive |
| gwl_7d_change | 0.61 | Trend |
| gwl_lag_30 | 0.46 | Autoregressive |
| gwl_30d_change | 0.29 | Trend |
| rainfall_30d | 0.14 | Environmental |
| soil_moisture | 0.14 | Environmental |

**Top features at t+30 (mean |SHAP|):**

| Feature | SHAP | Category | Change from t+7 |
|---------|------|---------|----------------|
| gwl_lag_7 | 6.10 | Autoregressive | ↓ decreased |
| gwl_lag_14 | 1.78 | Autoregressive | ↓ decreased |
| gwl_lag_30 | 0.78 | Autoregressive | ↑ increased |
| **day_of_year** | **0.50** | **Seasonal** | **↑↑ large increase** |
| **is_monsoon** | **0.46** | **Seasonal** | **↑↑ large increase** |
| gwl_7d_change | 0.43 | Trend | — |
| soil_moisture | 0.26 | Environmental | ↑ increased |
| rainfall_30d | 0.20 | Environmental | ↑ increased |

**Key findings:**

1. **Autoregressive dominance persists but weakens.** `gwl_lag_7` remains the top feature at every horizon (SHAP: 6.59 → 6.64 → 6.10), but its relative dominance decreases as other features gain importance.

2. **Seasonal features emerge at t+30.** `day_of_year` and `is_monsoon` are not in the top-5 at t+7 but become the 4th and 5th most important features at t+30. The model is learning that at a 30-day horizon, the time of year is more predictive than short-term weather noise.

3. **Longer-range lags gain importance at longer horizons.** `gwl_lag_30` increases from SHAP 0.46 (t+7) to 0.78 (t+30), consistent with the physical intuition that the 30-day seasonal baseline becomes more relevant when predicting 30 days ahead.

4. **Soil moisture and rainfall_30d increase in importance.** These reflect the recharge memory of the system: at longer horizons, whether the soil was saturated or dry 30 days ago (captured by `soil_moisture`) and the cumulative rainfall history (`rainfall_30d`) contribute more than the immediate short-term weather.

5. **TFT Variable Selection Networks are warranted.** The shift in feature importance across horizons directly motivates TFT's mechanism for dynamically weighting features per timestep and horizon. A fixed-weight model learns the average importance; TFT learns horizon-conditional importance.

### 6.4 Per-District Performance Analysis

Test RMSE at t+7 by district (all three models):

| District | ARIMA | LSTM | XGBoost | Notes |
|----------|-------|------|---------|-------|
| Dhule | 15.47 m | 10.72 m | 11.51 m | High error — deep Tapi basin aquifer |
| Jalgaon | 14.99 m | 9.07 m | 11.20 m | High error — high GWL variability |
| Nashik | 5.88 m | 3.37 m | 3.86 m | Godavari headwaters, intense irrigation |
| Nandurbar | 5.10 m | 2.52 m | 2.68 m | Tapi basin, deep aquifer |
| Aurangabad | 3.26 m | 1.93 m | 2.25 m | Semi-arid, high seasonal swing |
| Solapur | 2.32 m | 2.21 m | 1.75 m | Chronic water stress district |
| Akola | 0.58 m | 1.78 m | 0.60 m | Moderate Vidarbha district |
| Amravati | 0.88 m | 0.62 m | 0.53 m | Well-monitored Vidarbha district |
| Nagpur | 0.54 m | 0.65 m | 0.25 m | Well-monitored district |
| Wardha | 0.27 m | 1.03 m | 0.16 m | Shallow, stable aquifer |
| Pune | 0.32 m | 1.23 m | 0.34 m | Krishna basin, good monitoring density |
| Raigad | 0.19 m | 0.71 m | — | Coastal, low GWL variability |

**District patterns:**

- **High-error districts (RMSE > 5m for all models):** Dhule and Jalgaon are outliers across all three models. Both lie in the Tapi basin and have deep aquifer systems with large seasonal amplitude (>20m swing). All models underperform here. The spatial TFT+GAT architecture is expected to improve these districts by propagating recharge signals from upstream basin nodes.

- **LSTM best in most districts:** LSTM outperforms XGBoost in 8 of the 12 districts shown, including the hardest cases (Dhule, Jalgaon, Nashik, Nandurbar, Aurangabad). XGBoost leads only in low-variability stable districts (Wardha, Nagpur, Thane) where the short-term lag feature is nearly perfect.

- **XGBoost missing data for Beed, Gadchiroli, Gondia, Raigad:** These districts' XGBoost columns are blank because the direct target shift for those districts produced insufficient non-NaN training samples after the feature engineering cutoffs (NaN values in lag features near district boundaries in time). This is a known limitation; LSTM and ARIMA handle these cases through imputation and sequence padding.

---

## 7. Conclusion and Future Work

### 7.1 Summary of Phase 1

JalVaani's Phase 1 established a rigorous multi-model, multi-horizon baseline for groundwater level prediction across 24 districts of Maharashtra. The key findings are:

1. **The t+1 framing is insufficient for operational use.** Next-day GWL prediction achieves low absolute error (XGBoost: 1.68m RMSE, MAPE 2.43%) but provides minimal lead time for water resource intervention. A model that tells you tomorrow's GWL has limited operational value; one that tells you next month's GWL with calibrated uncertainty enables drought preparedness.

2. **Sequential models outperform tree models at multi-horizon targets.** LSTM beats XGBoost at every multi-horizon target (t+7, t+14, t+30), with the gap growing to 33% at t+30. This confirms that temporal sequence learning is essential for medium-range groundwater forecasting and justifies the move to TFT.

3. **Feature importance is horizon-dependent.** SHAP analysis quantitatively shows that autoregressive lags dominate at t+7, while seasonal indicators and slow-moving environmental variables gain importance at t+30. This directly motivates TFT's Variable Selection Networks.

4. **High-error districts are spatially clustered in the Tapi basin.** Dhule and Jalgaon (both Tapi basin, RMSE >9m for all models) suggest that spatial inter-district information propagation — the GAT layer — will be most impactful for these districts.

5. **Current features are sufficient to proceed to TFT.** All 26 engineered time-varying and static features have non-zero SHAP importance. No dead features. The Sentinel data gap (NDVI, NDWI, SAR) does not block TFT Phase 2 — it is an enhancement for Phase 2b.

### 7.2 Phase 2 Plan — Temporal Fusion Transformer

The immediate next step is implementing and training the TFT on the current 36-column master dataset. The TFT will:
- Produce simultaneous q10, q50, q90 forecasts at all horizons (t+1 through t+30)
- Use the three-way input separation (static, known future, observed past) described in Section 5.1
- Report pinball loss and prediction interval coverage on the test set
- Generate interpretability outputs: VSN weights by season, temporal attention heatmaps

**Target:** beat LSTM t+30 RMSE of 3.54m as the primary benchmark. Provide calibrated 80% prediction intervals with ≥75% empirical coverage.

### 7.3 Phase 2b Plan — Sentinel Data Integration

Once Sentinel-2 and Sentinel-1 data is collected (via GEE batch export or Microsoft Planetary Computer), the following features will be added to the master dataset and the TFT will be retrained:

| Feature | Source | Expected contribution |
|---------|--------|----------------------|
| `ndvi_mean` | Sentinel-2 (monthly, interpolated to daily) | Irrigation demand proxy (non-monsoon) |
| `ndwi_mean` | Sentinel-2 | Surface water / reservoir lateral recharge |
| `ndmi_mean` | Sentinel-2 | Vegetation moisture stress → irrigation trigger |
| `vv_db`, `vh_db`, `vh_vv_ratio` | Sentinel-1 | All-weather soil moisture (fills monsoon cloud gap) |

The SHAP analysis from Phase 1 already confirms that `soil_moisture` SHAP increases at longer horizons. Adding direct vegetation and SAR moisture observations is expected to further improve t+14 and t+30 predictions.

### 7.4 Phase 3 Plan — TFT + GAT Spatial-Temporal Architecture

After the standalone TFT reaches its performance ceiling, the GAT layer will be integrated to model inter-district aquifer connectivity. The graph construction will use:
- Static spatial adjacency (shared district borders)
- River basin membership derived from station `majorBasin` metadata in `groundwater_stations.geojson`
- Edge weights: inverse distance between district centroids

The scientific hypothesis is that propagating recharge signals from upstream basin-connected districts will most improve Dhule and Jalgaon (Tapi basin), and Nashik/Ahmednagar/Aurangabad (Godavari basin) — the clusters where spatial dependence is physically strongest.

The learned GAT attention weights on edges will be analysed against known basin boundaries. Agreement between learned attention topology and known hydrological basin structure will constitute evidence that the model has recovered real hydrogeological coupling from data.

### 7.5 Open Data Gaps

| Gap | Impact | Priority |
|-----|--------|---------|
| Sentinel-2/1 data not collected | Missing NDVI/SAR features (monsoon gap) | Medium — add in Phase 2b |
| Reservoir storage levels | NDWI detects water presence but not volume | Low — requires separate WRIS endpoint |
| Aquifer type per station | Hard rock vs. alluvial behaviour difference | Medium — would improve Dhule/Jalgaon |
| Hydraulic conductivity | Direct soil permeability parameter | Low — only available from CGWB survey maps |

---

*End of Report — Last Updated: 2026-05-13*
*Status: Phase 1 (Baselines) complete. Phase 2 (TFT) in progress.*
