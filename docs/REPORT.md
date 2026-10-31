# JalVaani — Spatiotemporal Groundwater Level Prediction for Maharashtra
## Project Report

**Scope:** multi-horizon probabilistic forecasting of district groundwater levels across Maharashtra.
- **Forecasts:** 7, 14 and 30 days ahead, with q10 / q50 / q90 uncertainty.
- **Inputs:** multi-source environmental time series.
- **Models:** a Temporal Fusion Transformer (TFT) and a TFT-led ensemble.

**Data range:**
- **Groundwater target:** 2023-01-01 to 2026-06-04. That is the last reading published by CGWB.
- **Covariates:** collected to 2026-10-01.

**Districts:** 36 in the state; 24 have usable groundwater data and are modelled; 21 form the headline set.

**Status (2026-11-01):**
- Phase 1 (baselines) is complete.
- Phase 2 (TFT) is complete through production version **v4.3** (Colab run of 31 Oct – 1 Nov 2026). This is the final model, chosen by the rules fixed in advance.
- v4.2 and v4.3 changed only post-processing. The networks and LightGBM re-trained bit-for-bit identically to v4.1 both times.
  - v4.2 fixed the TFT's intervals.
  - v4.3 made the Ensemble's 7-day intervals even across districts, removed most extreme intervals, and added a Pune-only fix (with mixed results, §9.9) and the **operational forecast** (§9.8–§9.9, §10.12–§10.13).
- The **farmer and government dashboards** in `dev/` now show that forecast. They are in English, Marathi, Hindi and Gujarati, with a site-guide chatbot and light / dark themes (§13).
- The modelling is frozen for the write-up. The open items are in §12.4.

> **Headline result (final Ensemble = v4.3; locked test = Oct 2025 – Jun 2026, never used for any choice since v4.0):**
>
> | Comparison | t+7 / t+14 / t+30 |
> |---|---|
> | Error ÷ error of the "no change" forecast (MASE) | **0.841 / 0.766 / 0.620** (16–38% less error) |
> | Error ÷ error of the seasonal forecast | **0.940 / 0.926 / 0.882** (6–12% less error) |
> | 80% intervals contain the true level | 78–80% of the time |
> | TFT alone: 80% intervals contain the true level | 77–80% of the time (v4.1: 66–74%) |
>
> All six comparisons are statistically significant (p = 0.003, Holm-adjusted).
>
> For reference, the first full TFT run (production v1, August 2026) was **worse than "no change"**, with a MASE of 2.12 at t+7.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Project Timeline — What Was Done and When](#3-project-timeline--what-was-done-and-when)
4. [Data — What We Have and What We Do Not Have](#4-data--what-we-have-and-what-we-do-not-have)
5. [Data Preprocessing Pipeline and How It Changed](#5-data-preprocessing-pipeline-and-how-it-changed)
6. [Evaluation Methodology and How It Changed](#6-evaluation-methodology-and-how-it-changed)
7. [Phase 1 — Baseline Models (historical)](#7-phase-1--baseline-models-historical)
8. [Phase 2 — The Temporal Fusion Transformer as Built](#8-phase-2--the-temporal-fusion-transformer-as-built)
9. [Model Pipeline Versions v1 → v4.3 — What Changed in Each](#9-model-pipeline-versions-v1--v43--what-changed-in-each)
10. [Results](#10-results)
11. [Discussion](#11-discussion)
12. [Conclusion and Future Work](#12-conclusion-and-future-work)
13. [Dashboards and Decision Support](#13-dashboards-and-decision-support)
- [Appendix A — Every Bug Found and Fixed](#appendix-a--every-bug-found-and-fixed)
- [Appendix B — Repository Map](#appendix-b--repository-map)
- [Appendix C — Glossary](#appendix-c--glossary)
- [Appendix D — References](#appendix-d--references)

---

## 1. Introduction

### 1.1 Background

Groundwater is Maharashtra's most important water resource.
- It supplies most rural drinking water.
- It supports irrigated agriculture across the semi-arid Deccan Plateau, Marathwada and Vidarbha.

It is also hard to manage. Reservoir levels can be seen and are monitored centrally. Groundwater cannot be seen, it varies from place to place, and it depends on a complex interplay of natural recharge and human extraction.

The state has faced chronic groundwater stress for more than two decades.
- Districts such as Latur, Osmanabad, Beed and Aurangabad regularly see wells fail in the March–June dry season.
- During the 2016 Marathwada drought, water had to be brought in by train.

These crises are not sudden. They build from recharge deficits over several months, which routine monitoring does not anticipate in time.

The Central Ground Water Board (CGWB) runs a groundwater monitoring network in Maharashtra and reports groundwater levels (GWL).
- That record is valuable for long-term trends.
- Operational decisions need more. Supplying emergency water, restricting extraction or advising farmers on irrigation all require forecasts days to weeks ahead, with honest uncertainty.

**JalVaani** addresses this gap. It forecasts district groundwater levels 7, 14 and 30 days ahead, with 80% prediction intervals that give a low case (q10), a likely case (q50) and a high case (q90).

### 1.2 Problem Statement

Groundwater level follows a water balance:

```
ΔGWL = Recharge(t) − Discharge(t) − Evapotranspiration(t)
```

Each term depends on observable environmental variables. They interact non-linearly and with time lags:

- **Recharge** is driven by rainfall.
  - The lag between rain and a measurable rise ranges from days (shallow, permeable aquifers) to weeks (deep hard-rock basalt).
  - Soil moisture gates how much rain reaches the aquifer.
- **Discharge** is dominated by irrigation pumping.
  - Pumping is driven by crop water demand and is not observed at district scale.
  - Vegetation indices are an indirect proxy for it.
- **Evapotranspiration** removes water before it can recharge, especially in April–June.

The project also met a problem that a classical problem statement does not mention. **Groundwater is close to a random walk:** today's level already says most of what next week's level will be.
- The median real change in this data is **0.29 m over 7 days** and **0.90 m over 30 days**.
- So a model is useful only if it beats the trivial forecast "the level will not change" (persistence).
- It should also beat the seasonal forecast: today's level plus the district's usual change for that time of year.

This became the central evaluation principle of the project. Section 6 covers it, and Section 9.1 shows how the project discovered it.

### 1.3 Objectives and Their Status

| # | Objective (as set in May 2026) | Status (October 2026) |
|---|---|---|
| 1 | Collect, clean and integrate multi-source environmental time series for Maharashtra districts | **Achieved.** All 36 districts collected; 24 have usable groundwater. Target 2023-01-01 to 2026-06-04; covariates to 2026-10-01 (§4) |
| 2 | Engineer physically motivated features (rainfall lags, trends, soil moisture, vapour pressure deficit, season) | **Achieved.** 47 past inputs, 8 known-future inputs, 13 static inputs plus 2 categorical (§5.8) |
| 3 | Establish baselines (ARIMA, XGBoost, LSTM) at t+1 and t+7/14/30 | **Achieved in Phase 1, on the old dataset** (§7). The final evaluation uses stronger baselines on identical forecast windows: persistence, seasonal-naive, ridge and LightGBM (§6.4) |
| 4 | Measure how each baseline degrades with horizon | Done in Phase 1. The interpretation was later revised (§7.6) |
| 5 | Design and train a TFT with variable selection networks | **Achieved** (§8, §9) |
| 6 | Extend the TFT with a Graph Attention Network (GAT) for inter-district coupling | **Not built; deliberately on hold.** The evidence points to data completeness, not missing spatial links, as the limit (§8.11) |
| 7 | Produce calibrated q10/q50/q90 forecasts | **Achieved for the Ensemble** (80–81% coverage on the locked test). The TFT alone under-covers after the calibration step, which is a known issue with a planned fix (§10.9) |
| 8 | Interpretability: variable importance, attention, graph attention | **Partly.** Variable-selection importance is reported (§10.10). Attention maps are not analysed, and graph attention does not exist because there is no GAT |

### 1.4 Scope

**In scope (as delivered):**
- The 24 Maharashtra districts with usable CGWB telemetric groundwater data.
  - 21 headline districts.
  - 3 single-well districts reported separately: Nanded, Raigad, Thane.
- Daily district-level groundwater level, built from cleaned telemetric wells with a factor model (§5.5).
- Forecast horizons t+7, t+14 and t+30, all predicted at once (the model outputs all 30 days).
- Probabilistic output: q10, q50, q90, forming an 80% prediction interval.
- Inputs:
  - groundwater history;
  - rainfall (CHIRPS);
  - evapotranspiration, potential ET and four soil-moisture layers (ERA5-Land);
  - temperature and humidity;
  - Sentinel-2 vegetation and water indices;
  - Sentinel-1 radar backscatter;
  - soil texture;
  - CGWB district groundwater-resource statistics.
- Walk-forward evaluation over six seasonal test periods, from July 2024 to June 2026.

**Out of scope or not achieved:**
- Reservoir storage and canal releases: the collector was written but needs an API key, so it was never run.
- Rainfall *forecasts* as an input: the model does not know future rain.
- Station-level (well-level) forecasts.
- The spatial GAT layer and a physics-informed loss (§8.11).
- Operational deployment: dashboard and API.

### 1.5 Assumptions and Constraints

**Assumptions:**
1. A district groundwater series can be built from its telemetric wells. Each well needs its own offset and amplitude so that wells joining or leaving the record do not move the district value (§5.5).
   - The May 2026 version assumed a plain mean across stations was enough. **This turned out to be false** and was the largest single source of error in the project (§5.3).
2. The reanalysis and satellite covariates (CHIRPS, ERA5-Land, Open-Meteo, Sentinel) are accurate enough at daily district scale without bias correction.
   - The exception is the last 31 days of rainfall, which are provisional (§4.1).
3. Soil texture and the CGWB annual statistics are treated as static over 2023–2026.
4. Covariates are used only as they would have been known at forecast time.
   - Monthly satellite composites are stamped at month-end.
   - Rainfall departures use training-period means only.

**Constraints:**
- **Short record.** Groundwater starts in 2023, so there are only three monsoons (2023, 2024, 2025). The locked test period (Oct 2025 – Jun 2026) contains no monsoon.
- **Sparse monitoring.** 12 of 36 districts have no usable groundwater data. Several modelled districts rely on one to four wells (§4.3).
- **Few independent samples.** Consecutive forecast windows overlap almost entirely, so the number of truly independent training windows is in the hundreds. This limits model size (§9.2).
- **Compute.** Phase 1 was trained on a laptop CPU. All TFT runs use Google Colab with a free T4 GPU.

### 1.6 Technologies Used

| Category | Tools |
|---|---|
| Data collection | Python `requests` (India-WRIS paginated POST API), Open-Meteo API, Google Earth Engine Python API (CHIRPS, ERA5-Land, Sentinel-1/2, OpenLandMap) |
| Data processing | `pandas`, `numpy`, `scipy` |
| Statistical models | `statsmodels` (ARIMA / SARIMA) |
| Machine learning | `xgboost` (Phase 1), `lightgbm` (ensemble component), `scikit-learn` (ridge, scaling) |
| Deep learning | `PyTorch`, with a TFT written from scratch (`tft_model_v3.py`) |
| Evaluation | district-cluster bootstrap, Diebold–Mariano test, Holm correction, conformal prediction (own code in `tft_v4.py`) |
| Interpretability | `shap` (Phase 1 XGBoost); TFT variable-selection weights |
| Visualisation | `matplotlib`, `seaborn` |
| Compute | Jupyter (VS Code) for local work; Google Colab T4 GPU for TFT training |
| Spatial data | `geopandas`, `shapely`, GeoJSON, FAO GAUL district boundaries |
| Version control | Git / GitHub |

---

## 2. Literature Review

### 2.1 Statistical Time-Series Methods

The Box–Jenkins ARIMA framework [Box and Jenkins, 1970] is the classical approach to univariate forecasting. Its seasonal variant, SARIMA, adds seasonal terms suited to hydrological series with an annual cycle. ARIMA is widely used for groundwater level prediction in data-sparse settings.

Its limitation is that it is purely autoregressive: it predicts a series from its own past.
- It cannot use the physical drivers (rainfall, temperature, soil moisture).
- ARIMAX allows exogenous inputs, but only linearly.

### 2.2 Machine Learning for Groundwater Level Prediction

Gradient-boosted trees are the dominant machine-learning method for tabular environmental prediction.
- **XGBoost** [Chen and Guestrin, 2016] and **LightGBM** [Ke et al., 2017] learn non-linear interactions (for example, rainfall × soil moisture → recharge) without hand-made feature crosses.
- **SHAP values** [Lundberg and Lee, 2017] decompose a tree model's predictions into per-feature contributions.

Trees have no built-in sense of time order. Past values must be supplied as flat features (lags, rolling sums, calendar terms).

A finding of this project (§10.8) is that once LightGBM is given the right seasonal features, it becomes one of the strongest single forecasters.

### 2.3 Deep Learning — LSTM and Sequential Models

LSTMs [Hochreiter and Schmidhuber, 1997] keep relevant information over long sequences.
- Kratzert et al. [2018] showed that a single LSTM trained on many basins could beat basin-specific and physically based rainfall–runoff models.
- For groundwater, LSTMs can represent the rainfall–recharge lag, soil-moisture depletion and the seasonal recharge cycle.

Their limits for this problem:
- They have no explicit mechanism for weighting features by relevance.
- They have no spatial component.

### 2.4 Transformer-Based Time-Series Forecasting

The Transformer [Vaswani et al., 2017] uses self-attention, so every time step can attend to every other directly. The **Temporal Fusion Transformer** [Lim et al., 2021] adapts this to multi-horizon forecasting with mixed inputs. It separates three kinds of input:

1. **Static covariates** (per district: soil texture, district identity, aquifer type). They condition the whole network.
2. **Known-future inputs** (calendar terms: month, day of year, season). These are fed to the decoder.
3. **Observed-past inputs** (rainfall, groundwater, vegetation). These are known only up to the forecast date.

This matches the groundwater problem: we know which month it will be in 30 days, but not how much rain will fall. The TFT is one of the few published architectures that handles this three-way split natively.

Its **variable selection networks (VSNs)** learn soft weights over inputs at each time step. Its **quantile output head** produces q10 / q50 / q90.

A lesson of this project is that a quantile head does not guarantee honest intervals: coverage has to be measured out of sample (§2.7, §10.9).

### 2.5 Graph Neural Networks for Spatial-Temporal Forecasting

Graph Attention Networks [Veličković et al., 2018] learn attention-weighted aggregation over a node's neighbours. Spatial-temporal graph networks are well established for traffic and air-quality forecasting.

For groundwater, the original plan was:
- districts would be nodes, with shared basins and borders as edges;
- a GAT layer between the TFT encoder and decoder would pass recharge signals between districts.

A 2026 study of 2,951 German wells [HESS 30, 2373, 2026] found:
- globally trained models had no systematic advantage over per-well models;
- similar dynamics between wells mattered far more than data volume.

With the project's own evidence that per-district accuracy is limited by data completeness (§10.7), this led to the GAT being put on hold (§8.11).

### 2.6 Probabilistic and Quantile Forecasting

The pinball (quantile) loss for quantile τ is:

```
L_τ(y, ŷ) = τ · max(y − ŷ, 0) + (1 − τ) · max(ŷ − y, 0)
```

Training with τ = 0.1, 0.5 and 0.9 gives three forecasts that form an 80% interval.

For water management the low quantile matters most. If the q10 level crosses a critical depth within 30 days, the district qualifies for early action.

The interval must be **calibrated**: about 10% of outcomes should fall below q10 and about 10% above q90. The **interval (Winkler) score** [Gneiting and Raftery, 2007] combines width and misses into one number. It rewards intervals that are both narrow and honest.

### 2.7 Forecast Evaluation: Baselines, Skill Scores and Significance (added 2026-10)

- **Baselines.** For near-random-walk series, the honest benchmark is the naive forecast (persistence).
  - **MASE** [Hyndman and Koehler, 2006] is the model's error divided by the naive forecast's error. Below 1 means the model beats doing nothing.
  - For seasonal series the stronger benchmark is the **seasonal-naive** forecast. In this project it is the anchor level plus the district's typical change for that day of year, using past data only.
- **Walk-forward (rolling-origin) evaluation.** Each test period is forecast by models trained only on data before it. Random splits of time series leak information.
- **Significance.**
  - The **Diebold–Mariano test** [Diebold and Mariano, 1995] compares two forecasts' errors, with a correction for autocorrelation.
  - A **cluster bootstrap** that resamples whole districts gives confidence intervals that respect the correlation between a district's forecasts.
  - The **Holm correction** [Holm, 1979] controls the error rate across the many comparisons.

### 2.8 Conformal Prediction and Adaptive Calibration (added 2026-10)

- **Conformal prediction** turns any model's intervals into intervals with a coverage guarantee. It uses errors measured on held-out data that the model was not fitted on.
- **Conformalised quantile regression (CQR)** [Romano et al., 2019] calibrates each side of a quantile interval.
- **Adaptive conformal inference (ACI)** [Gibbs and Candès, 2021] keeps adjusting the miss rate online to cope with distribution shift.

Fitting the calibration on the same data it is applied to makes the coverage claim meaningless. Section 9.3 explains how this project found and fixed that mistake.

### 2.9 Ensembles and Stacking (added 2026-10)

**Stacked generalisation** [Wolpert, 1992] combines several forecasters, with weights learned on held-out errors. A stack can keep the strengths of each component; in this project those are the TFT's intervals and LightGBM's accuracy. It can also fall back to simple baselines where the complex models are unreliable.

### 2.10 Research Gaps and How This Project Addressed Them

| Gap identified (May 2026) | What the project did |
|---|---|
| **1. Per-district models ignore spatial aquifer coupling** | Not addressed by a GAT, which is on hold (§8.11). The evidence points to data completeness as the binding constraint |
| **2. Single-step forecasts are not useful operationally** | Addressed: t+7 / t+14 / t+30 forecasts with 80% intervals |
| **3. Fixed feature importance ignores seasonal variability** | Addressed: the TFT's variable selection networks, and calendar and seasonal-naive features given to both the TFT and LightGBM |
| **4. Optical satellite data fails in the monsoon** | Addressed: Sentinel-1 radar backscatter (VV, VH, VH/VV) was collected alongside Sentinel-2 and made causal (month-end stamping) |
| **5. Interpretability as scientific output** | Partly addressed: variable-selection importance is reported (§10.10) |
| **6. (new) Published groundwater models are rarely tested against naive and seasonal baselines, or with honest out-of-sample intervals** | Addressed throughout: every result is reported against persistence and seasonal-naive, with significance tests and out-of-sample coverage |

---

## 3. Project Timeline — What Was Done and When

| Date (2026) | Phase | What was done | Outcome |
|---|---|---|---|
| **Up to May** | Data collection v1 | India-WRIS groundwater, CWC rainfall, NRSC ET and soil moisture; Open-Meteo temperature and humidity; soil texture via Earth Engine | Master dataset: 24 districts, Mar 2023 – Apr 2025, 16,785 rows × 36 columns |
| **May** | **Phase 1 baselines** | ARIMA/SARIMA, XGBoost and LSTM at t+1 and t+7/14/30; SHAP analysis | LSTM best at multiple horizons (RMSE 3.3–4.2 m). Later found to be dominated by data artefacts (§7.6) |
| **Before July** | TFT prototype | First TFT notebooks (`09_tft_groundwater.ipynb`, `09_tft_colab.ipynb`); Sentinel-1/2 collected (Phase 2b) | Validation-only scores, no test set and no baseline. Superseded; not reproducible |
| **4 Jul** | **Data audit and master rebuild** | Documentation found to be out of date; rainfall switched to CHIRPS; a calendar-gap windowing bug found; full daily calendar; date-true lags; district gate; GAUL name bugs fixed; walk-forward evaluation designed | 24 districts × 843 days = 20,232 rows × 52 columns. Every window is genuinely continuous |
| **Jul–Aug** | Data extension | Groundwater and covariates extended to the last published CGWB reading (2026-06-04) | Enables six walk-forward test seasons, Jul 2024 – Jun 2026 |
| **14 Aug** | **Production v1** | First full six-fold TFT run on a Colab T4 (11.6M parameters, absolute-level target), about 3 h | Looked plausible (MAPE 18–23%) but **lost to persistence in 15 of 18 fold-horizon cases** |
| **15 Aug** | **Production v2** | Groundwater panel filter, per-well anomaly aggregation and drift filter; change-from-anchor target; model shrunk; MASE introduced | Median MASE 1.004 / 0.940 / 0.827. Tied persistence at t+7; 7 districts still lost everywhere |
| **15 Aug** | **Production v3** | Full audit: parameter count wrong by 6×; Sentinel future leak; ET and soil moisture 100% fabricated after 2025-06; self-referential calibration. Fixed with ERA5-Land, causal Sentinel, data tiers, per-horizon scaling, shrinkage, monotone quantiles, baseline ladder, vectorised network, significance tests | MASE 0.958 / 0.909 / 0.834, significant. Run time 3 h → 6.5 min |
| **2 Oct** | Data extension 2 | All covariates extended to 2026-10-01; groundwater re-checked (nothing after 2026-06-04); master rebuilt to end at the groundwater end | 26 districts × 1,251 days = 32,526 rows × 63 columns |
| **3 Oct** | **Production v4** | Groundwater target rebuilt (placeholders, stuck loggers, two-sided Hampel filter, step and regime detection, factor-model aggregation, no fills); evaluation fixed (issue-date windows); seasonal-naive baseline; residual-on-seasonal target; pooled adaptive calibration; stacked ensemble; locked folds 5–6 | Locked Ensemble 0.865 / 0.788 / 0.664 vs persistence, 0.967 / 0.952 / 0.945 vs seasonal |
| **26 Oct** | v4.1 built | Shrinkage toward seasonal-naive; stronger LightGBM; refit pass; stride-1 sampling; sweeps; 10 seeds; simple per-district outputs; results guide | Pipeline 4.5, notebook `09_tft_colab_production_v4_1.ipynb` |
| **29 Oct** | **Production v4.1 run** | Colab T4, 54 min, pre-registered selection on dev folds 1–4 | **Locked Ensemble 0.839 / 0.763 / 0.616 vs persistence, 0.938 / 0.922 / 0.877 vs seasonal (still the final forecasts).** Interval-calibration issue found for the TFT alone |
| **30 Oct** | v4.2 built | Post-processing only: ACI fed one outcome per issue date and band, symmetric α range, raw-or-calibrated interval per model and horizon, per-district stack guard (Pune); three rules fixed in advance on development folds | Pipeline 4.6, notebook `09_tft_colab_production_v4_2.ipynb`; defaults reproduce v4.1 exactly |
| **30 Oct** | **Production v4.2 run** | Colab T4, 31 min, 10 seeds; three rules fixed in advance, applied on development folds 1–4 | Forecasts bit-for-bit identical to v4.1. **TFT intervals fixed:** locked coverage 66–74% → 77–80%, interval score 11–19% better. Ensemble intervals about the same. Pune guard rejected by its rule |
| **31 Oct** | v4.3 built; dashboards built | Post-processing (Pune LightGBM-only guard, width cap, evenness-aware interval rule), operational forecast + dashboard export; farmer and government dashboards in `dev/` (en / mr / hi, site guide) | Pipeline 4.7, notebook `09_tft_colab_production_v4_3.ipynb` |
| **1 Nov** | **Production v4.3 run** | Colab T4, 39 min, 10 seeds; rules G2 and D2 on development folds | Guard adopted (Pune and Raigad only); 7-day Ensemble intervals even in all districts; intervals > 20 m 332 → 105; latest forecast issued from 4 Jun 2026 for 23 districts; locked headline 0.841 / 0.766 / 0.620 (Pune worse on the locked folds) |

> **Naming note.** The repository has two unrelated "v1/v2" lines.
> - The **prototype** TFT runs of May–June are described in `docs/PROJECT_STATUS.md` §9.
> - The **production** line v1 → v4.1 is the one this report uses from §9 on. It is documented in `docs/history/TFT_VERSION_HISTORY.md`.
>
> The `04_results/tft_prod_v1/` folder holds the production **v1** run.

---

## 4. Data — What We Have and What We Do Not Have

### 4.1 Data Inventory (current, as of 2026-10-29)

| # | Dataset | Source | Granularity | Extent | Notes |
|---|---|---|---|---|---|
| 1 | **Groundwater level** (target) | CGWB via India-WRIS | sub-daily per telemetric station | **2023-01-01 → 2026-06-04** | Last published reading; WRIS returns nothing later for any district or agency |
| 2 | **Rainfall** | CHIRPS v2.0 via Earth Engine | daily, gridded, averaged per district | final to 2026-08-31; **provisional** to 2026-10-01 | 2026-09-01 to 09-25: CHIRPS preliminary rasters, scaled per district to match July–August. 09-26 to 10-01: Open-Meteo rainfall, also scaled. All 31 days flagged `rainfall_observed=0` |
| 3 | **ET, potential ET, soil moisture at 4 depths** (0–7, 7–28, 28–100, 100–289 cm) | ERA5-Land via Earth Engine | daily per district | real to 2026-09-24; filled with flags to 10-01 | Replaced the NRSC VIC feed, which stopped on 2025-06-30 |
| 4 | **Temperature (max/min/mean), relative humidity** | Open-Meteo (ERA5 reanalysis) | daily per district centroid | to 2026-10-01 | |
| 5 | **Sentinel-2 indices** (NDVI, NDWI, NDMI) | Copernicus S2 via Earth Engine | monthly per district, stamped at month-end | to Sep 2026 | Sparse in Jul–Aug because of monsoon cloud; held forward |
| 6 | **Sentinel-1 radar backscatter** (VV, VH, VH/VV) | Copernicus S1 via Earth Engine | monthly per district, stamped at month-end | to Sep 2026 | Works through cloud |
| 7 | **Soil texture** (sand, clay, silt % at 0/10/30 cm) | OpenLandMap / SoilGrids via Earth Engine (ISRIC REST fallback) | static per district | – | |
| 8 | **CGWB groundwater-resource assessment** (annual recharge, extraction, extractable resource, stage of extraction) | CGWB | yearly per district, used as static | held forward to 2026 | |
| 9 | **Station metadata** (aquifer type, well depth, basin) | India-WRIS | static per station | – | Used for aquifer category and station counts |

**Training data actually used:** `master_dataset_v4target.csv`.
- 25 districts × 1,251 days = **31,275 rows × 64 columns**, from 2023-01-01 to 2026-06-04.
- The master ends where the groundwater target ends. Covariates after 2026-06-04 are collected but unused until new groundwater readings appear.

### 4.2 How the Data Sources Changed During the Project

| Data | Original source (May 2026) | Current source | Why it changed |
|---|---|---|---|
| Rainfall | CWC station gauges (India-WRIS), mean across stations | **CHIRPS satellite-gridded rainfall** | Station gaps; CHIRPS covers all 36 districts evenly (July 2026) |
| ET and soil moisture | NRSC VIC model (India-WRIS) | **ERA5-Land** | The NRSC feed stopped on 2025-06-30. Every value after that was a day-of-year average, so evaluation folds 4–6 had 25%, 0% and 0% real data. ERA5-Land also adds potential ET and four soil depths (August 2026) |
| Humidity | CWC stations (India-WRIS) | Open-Meteo (ERA5) | Continuous, with no station gaps |
| Groundwater wells | all CGWB stations, including ~1,254 manual-survey wells | **telemetric wells only** (~125 → ~109 after the panel filter) | The manual wells have exactly 2 readings each (Jan and May 2023). Mixing them in created false jumps of up to ~94 m (§5.3) |
| Sentinel-1/2 | not collected | collected, monthly, causal | Added in Phase 2b; the future leak was fixed in v3 (§5.4) |

### 4.3 District Coverage

| Group | Districts | Reason |
|---|---|---|
| **No usable groundwater data (10)** | Ahmednagar, Buldhana, Palghar, Sindhudurg (no CGWB data returned); Satara (one mixed-sign station); Mumbai City, Mumbai Suburban, Hingoli, Osmanabad, Sangli (fewer than 100 readings) | Nothing to collect; not a processing error |
| **Collected but unusable (2)** | **Beed**: its only well is a stuck or dry sensor, so after the v4 cleaning ≥ 60% of days are missing. **Latur**: tier D, 44% observed, one station | Listed as "no usable data" in all results |
| **Modelled (24)** | see the tier table below | |
| **Headline set (21)** | the 24 modelled minus the single-well Nanded, Raigad and Thane | Single-well districts are reported but kept out of headline averages |

**Data tiers.** These are assigned from station count and record completeness only, never from model results, so they can explain per-district accuracy independently:

```
D  overall observed < 55%                              -> excluded from training
A  >= 6 stations AND >= 65% of 2023 observed           -> dense, complete
C  <= 4 stations AND <  55% of 2023 observed           -> thin panel, gappy 2023
B  everything else
```

| Tier | Districts (v4.1 run) |
|---|---|
| A (9) | Amravati, Chandrapur, Gondia, Nagpur, Ratnagiri, Solapur, Wardha, Washim, Yavatmal |
| B (7) | Akola, Jalgaon, Jalna, Nandurbar, Nashik, Parbhani, Raigad |
| C (8) | Aurangabad, Bhandara, Dhule, Gadchiroli, Kolhapur, Nanded, Pune, Thane |
| D (1) | Latur (excluded) |

Across districts, predictability tracks data completeness, not how much the water table moves:
- 2023 coverage: Spearman −0.54 with error (v3 ridge probe);
- station count: −0.51;
- typical movement: −0.10.

### 4.4 What We Do Not Have

| Missing | Effect | Could it be obtained? |
|---|---|---|
| **Rainfall forecasts for the next 30 days** (no known-future weather input) | **The largest remaining source of error**, especially at t+30 in the monsoon | Yes: IMD or ECMWF extended-range forecasts could be added as known-future inputs (§12.2) |
| Groundwater after 2026-06-04 | No newer test data; the locked test has no monsoon | Only when CGWB publishes more |
| Groundwater for 12 of 36 districts (§4.3) | These districts cannot be forecast | Needs more stations (CGWB, or the state GSDA network, which returned nothing via WRIS) |
| More wells in tier-C districts (≤ 4 wells) | These districts are the weakest; one well is not a district signal | Same as above |
| Reservoir storage and canal releases | A missing recharge and discharge driver, especially in Marathwada | `01_data_collection/cwc_reservoir.py` is written but needs a data.gov.in API key; never run |
| Daily pumping or extraction | Only yearly CGWB statistics exist | Not publicly available |
| Aquifer hydraulic properties (conductivity, specific yield) | Hard-rock and alluvial behaviour differ | Only from CGWB survey maps (manual digitisation) |
| A longer record | Only three monsoons; seasonal effects are estimated from 2–3 examples | Time |
| Final CHIRPS rainfall for Sep 2026 | The last 31 rainfall days are provisional | Re-run `run_chirps_incremental.py` and `chirps_extend_assemble.py` when published. They lie after the target end, so current results are unaffected |

---

## 5. Data Preprocessing Pipeline and How It Changed

The pipeline runs in order:
1. collectors in `01_data_collection/`;
2. `preprocess_groundwater.py`, `preprocess_env_features.py` and `merge_master.py` in `02_preprocessing/`;
3. the model pipeline in `03_models/tft/`.

The groundwater step was rewritten three times. Each rewrite fixed a problem that had been inflating forecast errors.

### 5.1 Original Pipeline (May 2026)

- **Groundwater:**
  - active stations only, no zero readings, values within [−100, 0] m;
  - daily mean per station, then mean, minimum and maximum per district;
  - gaps of up to 7 days filled forwards and backwards.
- **Rainfall:** sub-daily readings summed per station, averaged per district; missing days set to 0.
- **ET, soil moisture, temperature, humidity:** implausible values dropped; gaps of up to 7 days filled forward.
- **Master:** inner join on (district, date) with lag and season features. This dropped every day with no groundwater reading.

### 5.2 July 2026 — Master Rebuild (calendar and windowing fix)

- **The windowing bug.**
  - The old master dropped days with missing groundwater, leaving holes in each district's calendar.
  - Lags such as `gwl_lag_7 = shift(7)` were therefore 7 *rows*, not 7 days.
  - The TFT's 150-day windows were built over row positions, so many spanned months with invisible gaps. Only 75% of Raigad's windows and 77% of Nanded's were truly continuous.
  - The gappiest districts got the most corrupted windows and the fewest of them. This matched the "good district / bad district" split seen in early results.
- **The fix.** A full daily calendar per district, with no rows dropped.
  - A `gwl_observed` mask marks real readings, so the loss ignores filled days.
  - All lags are date-true; this was asserted and spot-checked.
- **District gate.** Districts with ≥ 60% of groundwater missing are excluded and logged in `excluded_districts.csv`.
- **Silent name bugs fixed.**
  - The soil-texture collector had no district-name mapping, so Beed, Gondia, Raigad and Gadchiroli got no soil data.
  - The Sentinel and CHIRPS collectors misspelled Gadchiroli.
  - Soil texture now covers 35 of 36 districts; CHIRPS covers 36 of 36.
- **Rainfall** switched to CHIRPS.
- **Result:** 24 districts × 843 days = 20,232 rows × 52 columns, with 100% calendar and window continuity.

### 5.3 August 2026 — Groundwater Fix 1 (station composition; production v2)

**The problem.** The raw CGWB feed mixes two measurement programmes:

| Programme | Count | Coverage |
|---|---|---|
| Telemetric loggers | ~125 | 20–90% of days, continuous |
| Manual-survey wells | ~1,254 | exactly 2 readings each (Jan and May 2023) |

When the manual wells dropped out after May 2023, the "district level" jumped to whatever the few deep boreholes read. Aurangabad is the clearest example:

```
2023-01    -4.94 m   (40 manual dug wells)
2023-06   -98.88 m   ( 1 deep telemetric borehole)   <- a ~94 m "change" that never happened
2024-09    -1.28 m   ( 3 stations)
```

**The fix:**
1. **Panel filter.** Keep only stations reporting on at least 10% of days: 1,379 → ~109 stations.
2. **Per-station anomaly aggregation.** Each station is expressed as its deviation from its own mean before districts are aggregated.
3. **Drift filter.** Drop newly commissioned boreholes still filling up: trend R² ≥ 0.6 *and* range > 20 m.
   - Example: CGWBNG0515 drifted 102 m.
   - **Range alone is the wrong test.** Dhule and Nandurbar genuinely swing 30–50 m with the seasons, with trend R² below 0.1.

**Effect of the data fix alone, before any model change:**

| Metric | Before | After |
|---|---|---|
| Worst district standard deviation | 30.47 m (Aurangabad) | 10.24 m (Dhule) |
| Worst one-day jump | 29.10 m | 14.75 m |
| Overall groundwater missing | 24.2% | 12.3% |
| Persistence RMSE at t+7 / t+14 / t+30 | 2.60 / 3.04 / 3.64 m | **1.07 / 1.49 / 2.33 m** |

About 60% of what had looked like "prediction error" was noise from station composition.

### 5.4 August 2026 — Covariate Fixes and Data Tiers (production v3)

- **ET and soil moisture were fabricated after 2025-06-30.** The NRSC feed had stopped; the rest was day-of-year averages fed in as if they were real.
  - Re-sourced to **ERA5-Land** (`era5_land.py`).
  - This also added potential ET and four soil depths. The deep layers (28–100 and 100–289 cm) sit on the actual recharge path.
- **Sentinel future leak.**
  - Monthly composites were stamped at the 1st of the month and interpolated to daily. A value on 15 January was half January and half *February*: up to 15 days of the future.
  - They are now stamped at month-end and held constant until the next composite, which is strictly causal.
- **Data tiers** (§4.3) written by `merge_master.py`, together with a `district_data_quality.csv` scorecard.

### 5.5 October 2026 — Groundwater Target Rebuild (production v4)

**The audit.** Even after §5.3, the district series had **376 day-to-day jumps larger than 2 m**.
- 81–84% of them fell on days when the set of reporting wells changed, because the district value was still a plain median of whichever wells reported.
- The audit (`audit_groundwater_target.py`) also found these faults:

| Fault | Example |
|---|---|
| A telemetry placeholder value of exactly **+1.000** (2,709 readings) | Akola CGWBNG0501 sent it for 24 days. It flipped stations under the old sign rule, so 12 good telemetric wells were dropped (Nashik kept 2 of 5) |
| Mixed-sign stations | Their positive readings are fault periods, not a different sign convention. Bhandara read +20 to +50 m against a real level of about −3.4 m |
| **484 stuck-logger runs** (identical repeated readings) | |
| Dry-well plateaus | Dhule CGWNAG2183 pinned at −54.55 m, the sensor depth |
| Spike filter window too short | 7 *readings* (~1.75 days), so multi-day faults passed |
| Jump cap compared against the wrong value | Compared against the previous *uncapped* value, so it removed only the first day of a multi-day fault |
| Gap filling used future readings | Backward fill copied later readings into earlier days |

**The new groundwater pipeline (`preprocess_groundwater.py`, `AGG_METHOD = "factor_med"`):**
1. **Raw readings.** Telemetric panel only; |value| ≤ 100 m; exact 0 and +1.000 dropped.
2. **Sign.** Per-station majority sign. Minority-sign readings are dropped, together with majority readings within ±2 days of them.
3. **Stuck runs.** 12 or more identical consecutive readings are dropped.
4. **Per-station filters on daily means:**
   - a two-sided Hampel filter over ±7 days, where a spike must differ from both the week before and the week after, so a real step change is kept;
   - a plateau / censoring check;
   - removal of dead sensors and of stations with fewer than 180 good days.
5. **Step detection** against the last *accepted* value. A large jump is:
   - kept if the district's other wells moved the same way;
   - treated as a **regime break** (the station is split into a new segment with its own offset) if the new level persists after an outage;
   - rejected otherwise.
6. **Drift filter** per segment.
7. **District aggregation with a rank-1 factor model.** Each segment is modelled as `x_p(t) = μ_p + β_p · D(t)`, with its own offset and an amplitude shrunk toward 1. The district signal D(t) is the median of `(x_p − μ_p) / β_p` over the segments reporting that day. A well joining or leaving no longer moves the series.
8. **Observed flag.** A day counts as observed only if at least `ceil(0.34 × active)` segments really reported. No forward or backward fill.

**Result:**
- The aggregation audit chose the factor model: jumps larger than 2 m fell from 329 to 193, and it had the best leave-one-station-out correlation (0.458 at 30 days).
- A QC log (`station_qc_log.csv`) records every removal and the reason for it.
- On the v4 development folds the cleaner target lowered t+7 MAE from 0.57 to 0.51 m and t+30 MAE from 1.31 to 1.18 m (§10.2).
- Beed dropped out at this stage: its only well was a stuck sensor.

### 5.6 October 2026 — Data Extension to 2026-10-01

- Incremental collectors were added:
  - `run_gw_incremental.py`, `run_chirps_incremental.py`, `run_sentinel_incremental.py`, `run_temp_incremental.py`, `run_om_precip.py`;
  - `chirps_prelim_fill.py` and `chirps_extend_assemble.py` for the provisional rainfall tail.
- Pre-extension files are backed up in `archive/backups/2026-10_pre_extend/`.
- **Decision:** the master calendar ends at the last groundwater reading (2026-06-04), so the training data contains no provisional rainfall.
- **A trap that was found and fixed:**
  - The old rainfall file had zero-filled days from 2026-06-01, because CHIRPS had not yet been published when it was collected and the collector filled missing days with 0.
  - Fixed by re-fetching.
  - Any "reindex and fill with 0" step on a source that is published with a delay has the same risk.

### 5.7 Master Dataset — How It Grew

| Version | Districts | Days | Rows × columns | Key change |
|---|---|---|---|---|
| May 2026 | 24 | Mar 2023 – Apr 2025 | 16,785 × 36 | days without groundwater dropped (gappy calendar) |
| July 2026 | 24 | 843 (to 2025-04-22) | 20,232 × 52 | full calendar, date-true lags, district gate, CHIRPS |
| August 2026 (v3) | 26 | to 2026-06-04 | 55 columns | ERA5-Land, causal Sentinel, `data_tier` |
| October 2026 | 26 | 1,251 | 32,526 × 63 | covariates to 2026-10-01; master trimmed to the groundwater end |
| **October 2026 (v4 target)** | **25** | **1,251** | **31,275 × 64** | rebuilt groundwater target, `n_active`, `gwl_frac_real`; Beed dropped |

### 5.8 Features Used by the Model (v4 / v4.1)

**Observed-past inputs (47), by group:**

| Group | Features |
|---|---|
| Groundwater level | `gwl_mean/min/max`, `gwl_lag_7/14/30`, `gwl_roll_mean_30/60/90` (all made relative to the anchor), `gwl_7d_change`, `gwl_30d_change`, `gwl_roll_std_30/60/90`, `gwl_observed`, `gwl_frac_real`, `station_count`, `n_active` |
| Rainfall | log rainfall (1, 7 and 30 days), 60 and 90-day totals, antecedent precipitation index at three decay rates (`api_fast/mid/slow`), cumulative rainfall departure (`crd`, using the training-period mean), `days_since_rain` |
| Meteorology | `et_mm`, `net_water_balance`, `temp_max/min/mean`, `humidity_mean`, `vpd_approx` |
| ERA5-Land | `pet_mm`, `water_stress`, soil moisture `sm_l1`–`sm_l4` |
| Sentinel | `ndvi_mean`, `ndwi_mean`, `ndmi_mean`, `vv_db`, `vh_db`, `vh_vv_ratio` |

**Known-future inputs (8):**
- `month_sin/cos`, `doy_sin/cos`, `is_monsoon`, `season_id`;
- `horizon_frac` (position in the 30-day forecast);
- `sn_prior` (the seasonal-naive change for that step).

**Static inputs:**
- 9 soil-texture values (sand, clay and silt at three depths);
- 4 CGWB resource statistics;
- learned embeddings for `district_id` and `aquifer_category_id`.

**Dropped in v4:**
- `soil_moisture`, which duplicates `sm_l2`;
- the `*_observed` flags for ET and soil moisture, which are constant in the ERA5 master.

Ablation result (v4 development folds): removing the ERA5 and Sentinel inputs made the Ensemble slightly worse (mean ratio vs seasonal 0.978 vs 0.963), so these inputs carry a small amount of real information.

---

## 6. Evaluation Methodology and How It Changed

### 6.1 Phase 1: a Single Time Split

- Train Mar 2023 – Sep 2024; validate Oct 2024 – Jan 2025; test Feb–Apr 2025.
- This tests one dry season only, with no naive baseline and no significance tests.

### 6.2 Walk-Forward Evaluation (from July 2026)

Six test periods. Each is forecast by models trained only on data before it:

| Fold | Season | Test period | Role from v4 on |
|---|---|---|---|
| 1 | Monsoon 2024 | 2024-07-01 → 2024-09-30 | development (tuning) |
| 2 | Post-monsoon 2024 | 2024-10-01 → 2024-12-31 | development |
| 3 | Dry 2025 | 2025-01-01 → 2025-05-31 | development |
| 4 | Monsoon 2025 | 2025-06-01 → 2025-09-30 | development |
| **5** | **Post-monsoon 2025** | 2025-10-01 → 2025-12-31 | **locked test** |
| **6** | **Dry 2026** | 2026-01-01 → 2026-06-04 | **locked test** |

**Locked folds (v4 on).** All set-up choices are made on folds 1–4 with three seeds, using a selection rule fixed in advance. Folds 5–6 are reported only for the chosen set-up.
- They were seen once, in the v4.0 final run.
- v4.1 was selected on folds 1–4 only.
- The development-fold scores held up on the locked folds (§10.3).

### 6.3 Inside Each Fold

```
| ---------- training ---------- | early-stopping block | calibration block | ------ test ------ |
                                 \_____ last 3 months before the test: 60% / 40% _____/
```

- The network trains on the training block and stops early on the early-stopping block.
- The calibration block is never trained on. It is used to fit:
  - the shrinkage weights;
  - the ensemble weights;
  - the interval calibration.
- From v4 on, these are fitted on a **pool** made of the calibration block plus earlier folds' already-scored test forecasts. Rows from the same season get weight 1.0; others get 0.3.
- From v4.1, LightGBM is "fresh":
  - its calibration-block model is trained up to the end of the early-stopping block;
  - its test model is trained up to the day before the test period.

**Windows:**
- 60-day encoder and 30-day decoder.
- Training windows: stride 7 (v4), or stride 1 with one-seventh of windows drawn per epoch (v4.1).
- Validation windows: stride 2. Test windows: stride 1, so every issue date is scored.
- From v4 on, a window is keyed by its **issue (anchor) date**. It belongs to a split if any of its target dates falls inside the split, and `target_date = anchor + h` for every scored row.

### 6.4 Baselines (all on identical windows and targets)

| Baseline | Definition |
|---|---|
| **Persistence ("no change")** | `gwl(t+h) = gwl(t)` |
| **Seasonal-naive (SN)** | anchor level plus the district's typical h-day change for that day of year, from past data only. It uses a 30-day circular Gaussian kernel on day of year, shrunk toward 0 with a pseudo-count of 20 and damping 1.0. These settings were tuned on SN's own development-fold skill (`tune_snaive.py`). SN alone scores 0.948 / 0.892 / 0.807 vs persistence on the development folds |
| Ridge | linear model on the tabular features |
| **LightGBM** | gradient-boosted trees on tabular features. From v4.1 (`tab_features='v2'`) the features include day-of-year sin/cos, the SN change, its scale, the anchor level and the year-on-year change |
| Climatology-of-change | the v3 ladder only; replaced by SN |

### 6.5 Metrics

| Metric | Meaning | Good value |
|---|---|---|
| MAE, RMSE (m) | typical and large-error-weighted miss | smaller |
| **MASE** (rMAE vs persistence) | model MAE ÷ persistence MAE, pooled over the headline districts | **< 1** beats "no change" |
| **rMAE vs seasonal-naive** | model MAE ÷ SN MAE | **< 1** beats the seasonal forecast (harder) |
| MAPE (%) | error as % of depth, with depth floored at 0.5 m | smaller; never quoted alone, because it explodes near 0 m (r = 0.59 with "share of readings within 1 m of the surface" in v1) |
| **Coverage (%)** | share of actual levels inside [q10, q90] | about 80 (75–85) |
| Below q10 / above q90 (%) | misses on each side | about 10 each |
| Interval width (m) | q90 − q10 | narrower, at the right coverage |
| **Interval (Winkler) score (m)** | width + 10 × the size of any miss | smaller |
| Pinball loss, NSE | quantile loss; Nash–Sutcliffe efficiency | smaller; closer to 1 |
| Districts beating persistence | count of headline districts with MASE < 1 | 21/21 |
| Worst-5 minus best-5 gap | spread of per-district MASE | smaller = more even |

### 6.6 Significance

- **District-cluster bootstrap.** 2,000 resamples of whole districts, giving the pooled rMAE with a 95% confidence interval and a one-sided p-value. The floor is p ≈ 0.003 after the Holm adjustment.
- **Diebold–Mariano test**, with an autocorrelation (HAC) correction for overlapping windows.
- **Holm correction** across horizons and baselines.

### 6.7 Evaluation Mistakes That Were Found and Fixed

| Mistake | Version | Effect | Fix |
|---|---|---|---|
| No naive baseline in any table | Phase 1 – v1 | A model worse than "no change" looked fine for three generations | MASE in every table (v2); seasonal-naive added (v4) |
| Median over (district × fold) as the headline | v1 | Hid bad districts: median RMSE 1.8–2.2 m vs pooled 2.9–9.1 m | Pooled ratios plus per-district tables |
| Validation used as the test | prototype | Optimistic, dry-season-only scores | Walk-forward with a separate test period |
| Row-position windows over a gappy calendar | prototype | Corrupted windows and lags | Full calendar (July 2026) |
| Conformal calibration fitted on the test rows it calibrated | v2 | "80% coverage" was guaranteed by construction, so it was not a result | Held-out calibration block (v3) |
| Windows keyed by the decoder end date | v1–v3 | Some t+7 test targets fell in the validation block; the last 23 days of each season were never scored | Issue-date keying (v4) |
| Rainfall departure (`crd`) used the full-series mean | v3 | Small look-ahead | Training-period mean (v4) |

### 6.8 Accuracy Checks Added Over Time — When, Why, and What Each Revealed

Each version added measurements because the previous ones had missed a real problem.

| When | Check added | Why it was added | What it revealed |
|---|---|---|---|
| **Phase 1** (May) | RMSE, MAE, MAPE; per-district RMSE; SHAP per horizon | Standard regression metrics; which inputs matter | LSTM best at multiple horizons. Feature importance shifts with horizon. **No baseline, so real skill could not be shown** |
| **Prototype** | Pinball loss; coverage of q10–q90 | The TFT produces quantiles | Scored on the validation block (no separate test), dry season only, so optimistic |
| **v1** (14 Aug) | Six-fold walk-forward test; median RMSE and MAPE over district × fold | A real out-of-sample test across seasons | Looked plausible (MAPE 18–23%). A **persistence comparison added afterwards** showed it lost in 15 of 18 fold-horizon cases, and the median had hidden pooled RMSEs of 2.9–9.1 m. Raw coverage was only 52–53% |
| **v2** (15 Aug) | Persistence MAE and RMSE in every table; **MASE**; a written rule that MASE ≥ 1 means revisit the framing rather than tune; per-district MASE; raw vs calibrated coverage | Make "worse than doing nothing" impossible to miss | t+7 tied with persistence (1.004); 7 districts lost at every horizon |
| **v3** (15 Aug) | Skill %; baseline ladder (climatology, ridge, LightGBM on identical windows); Diebold–Mariano test with HAC correction; bootstrap 95% CI on the median MASE; districts beating persistence per data tier; coverage on a held-out calibration block; weekly-mean target; parameters-per-independent-window printed every fold; **covariate reality table** (share of truly observed ET and soil moisture per fold); **v2→v3 regression gate** (any district > 10% worse is flagged); an independent ridge probe | Is the gain significant, and better than simpler models? Are the inputs real? Is the model size honest? | The parameter count was wrong by 6×. ET and soil moisture were 0–25% real in folds 4–6. Coverage was really 63–70%, not 80%. 6 of the 7 failing districts were a data limit (ridge failed on the same ones) |
| **v4** (3 Oct) | **rMAE vs seasonal-naive**; separate tails (below q10 and above q90); interval width; NSE; the **worst-5 minus best-5 district gap**; count of districts below 1; **district-cluster bootstrap** (2,000 draws) plus Holm correction; development and locked folds; `ablations.csv` recording every run | Is the model better than the seasonal forecast? Are the misses one-sided? Is the gain even across districts? Is the selection honest? | **v3's TFT was worse than seasonal-naive** (1.03–1.07). Intervals missed mostly low. The district gap did not shrink |
| **v4.1** (26–29 Oct) | **Winkler interval score**; selection rule fixed in advance (a simpler set-up wins within 0.002); sweep tables; a check that development-fold scores hold on the locked folds; `v4_vs_v4_1.csv`; per-season tables; simple per-district files for both models; comparison with v1; variable importance; training curves and best epochs; stacking weights; then, after the run, **calibrated vs uncalibrated intervals compared on development folds** | One number for interval quality; protection against picking set-ups on noise; seeing what the Ensemble relies on | The gains held on the locked folds. LightGBM carries most of the accuracy. **The calibration step makes the TFT's own intervals worse** (§10.9) |

### 6.9 Integrity Checks Run Automatically in Every Fold

These are built into `tft_v4.py`. A run stops with an error if any of them fails.

| Check | Where | What it guarantees | Why it exists |
|---|---|---|---|
| No target beyond its cut-off in the training, early-stopping, calibration or refit windows (`_assert_no_leak`) | `train_fold`, `tab_fold` | No future information reaches any fitted model | Earlier versions leaked through interpolation and window keying (Appendix A) |
| Every model has an identical set of (district, target date, horizon) keys | `postprocess_fold` | Every model is scored on exactly the same targets | Different models silently dropping different hard days would bias the comparison |
| Every scored target lies inside its fold's test period, and `target_date − anchor_date = horizon` | `postprocess_fold` | Correct horizons; no target spilling into another fold | v1–v3 keyed windows by the decoder end date (bug 15) |
| LightGBM and ridge windows are identical to the network's | `tab_fold` | The tabular baselines are fair | The two stages are built and cached separately |
| Window gates: encoder ≥ 70% observed, anchor day observed; training and early-stopping targets ≥ 50% observed | `GWLDataset` | Forecasts start from real data | Gappy history and invented anchors made v1's district results unreliable |
| Cache keys include the data file's MD5 and `PIPELINE_VERSION` | `train_key`, `tab_key` | A result computed with older code or data is never reused | Found during v4 development (bug 21) |
| Selection only on development folds 1–4; locked folds 5–6 reported once | notebook | The headline is not tuned on the test | Standard protection against over-fitting the test set |

### 6.10 How a Change Is Judged — the Procedure Used Since v4

Every improvement in v4 and v4.1 went through these steps, in this order:

1. **Diagnose.** Find a specific weakness in the last results, for example "the TFT alone ties seasonal-naive" or "LightGBM cannot see the time of year".
2. **Implement it as a switch.** Add a configuration option whose default reproduces the previous version exactly. Examples: `target_mode`, `calib_mode`, `combine`, `lambda_center`, `tab_features`, `refit`, `epoch_frac`.
3. **Bump `PIPELINE_VERSION`**, so no cached result from older code can be reused.
4. **Smoke test locally**: 1 fold, 1 seed, 1–2 epochs. This checks that every new code path runs and every output file appears. The numbers mean nothing.
5. **Screen on development folds 1–4 with 3 seeds**, against a reference set-up (`ref`), with all candidates scored on identical windows.
6. **Apply the selection rule fixed in advance**: the lowest mean Ensemble error ÷ seasonal-naive error, where a simpler candidate wins if it is within 0.002.
7. **Sweep the cheap post-processing settings** on the chosen set-up (stacking strength, interval adaptivity), with the same rule. Interval settings must also keep each tail at 5–15%.
8. **Final run**: all six folds with 10 seeds.
9. **Report the locked folds 5–6 separately**, with significance tests against persistence and seasonal-naive, and check that the development-fold gains held.
10. **Compare with the previous version** on identical folds (`v4_vs_v4_1.csv`). Keep negative results; update the documentation.

---

## 7. Phase 1 — Baseline Models (historical)

> **Read with §7.6.**
> - These results were produced in May 2026 on the original dataset, which had station-composition jumps, row-based lags and a single dry-season test window.
> - They are kept as the project's starting point.
> - They are **not comparable** with §10.

### 7.1 ARIMA / SARIMA

ARIMA(7, 1, 1) per district:
- 7 autoregressive lags for weekly autocorrelation;
- one difference;
- one moving-average term.

Evaluation was rolling, with a refit every 30 steps. The h-step forecast for each date was taken from a cached forecast array. The model uses groundwater history only.

### 7.2 XGBoost

- One `XGBRegressor` across all 24 districts: 600 trees, depth 6, learning rate 0.05, subsample and column sample 0.8, early stopping after 30 rounds.
- Direct multi-horizon forecasting, with one model per horizon on the target shifted by −h.
- SHAP TreeExplainer on 2,000 test rows per horizon.

### 7.3 Vanilla LSTM

- Input: [batch, 30 days, 36 features]. These are 19 time-series features, 9 static soil features and an 8-dimensional district embedding, all broadcast over time.
- Two LSTM layers of 64 units with 0.2 dropout, then a linear output.
- Adam (learning rate 1e-3), ReduceLROnPlateau, early stopping after 15 epochs, batch 128, z-scores computed on training data. One model per horizon.

### 7.4 Results (test period Feb–Apr 2025)

**t+1:**

| Model | Test RMSE | Test MAE | Test MAPE |
|---|---|---|---|
| SARIMA | 3.63 m | 1.87 m | 16.22% |
| **XGBoost** | **1.68 m** | **0.46 m** | **2.43%** |
| LSTM | 2.73 m | 1.06 m | 8.91% |

**Multi-horizon RMSE (m):**

| Model | t+7 | t+14 | t+30 |
|---|---|---|---|
| ARIMA | 4.94 | 5.13 | 5.10 |
| XGBoost | 3.92 | 4.51 | 5.26 |
| **LSTM** | **3.27** | **4.18** | **3.54** |

**Multi-horizon MAE (m):**

| Model | t+7 | t+14 | t+30 |
|---|---|---|---|
| ARIMA | 2.01 | 2.21 | 2.48 |
| XGBoost | 1.56 | 1.87 | 2.22 |
| LSTM | 1.65 | 1.99 | 1.80 |

### 7.5 SHAP Feature Importance (XGBoost)

| Feature | Mean \|SHAP\| at t+7 | Mean \|SHAP\| at t+30 |
|---|---|---|
| gwl_lag_7 | 6.59 | 6.10 |
| gwl_lag_14 | 2.25 | 1.78 |
| gwl_lag_30 | 0.46 | 0.78 |
| day_of_year | – (not in the top 7) | **0.50** |
| is_monsoon | – | **0.46** |
| soil_moisture | 0.14 | 0.26 |
| rainfall_30d | 0.14 | 0.20 |

What the SHAP values show:
- Autoregressive lags dominate at every horizon.
- Seasonal terms rise to the top five at t+30.
- Slow variables (soil moisture, 30-day rainfall) gain importance with horizon.

This motivated horizon-dependent feature weighting, which the TFT's VSN provides. It also motivated the later seasonal-naive baseline.

### 7.6 Phase 1 in Hindsight — What Was Later Revised

1. **Most of the multi-metre error was in the data, not the models.**
   - The 2023 district series contained jumps caused by station composition, such as Aurangabad's −4.9 → −98.9 m (§5.3).
   - The lags were row-based on a gappy calendar (§5.2).
   - After the fixes, persistence RMSE at t+7 fell from 2.60 to 1.07 m with no model change at all.
2. **There was no "no change" benchmark.**
   - The t+1 XGBoost result (MAPE 2.43%) is mostly the model copying the last level.
   - Without persistence in the table, none of the Phase 1 models can be shown to have skill.
3. **The spatial explanation for Dhule and Jalgaon was not supported.**
   - Phase 1 attributed their high errors to missing basin coupling.
   - The July audit found Dhule 49% missing and Jalgaon a single well.
   - In v3, a ridge model failed on the same districts as the TFT, which points to a data limit rather than a missing spatial layer.
4. **One dry-season test window** cannot show monsoon skill. Walk-forward evaluation replaced it.
5. **What remains valid:**
   - Sequence models and seasonal information matter more at longer horizons.
   - Feature importance changes with horizon.

---

## 8. Phase 2 — The Temporal Fusion Transformer as Built

### 8.1 Implementation

- **Network:** `03_models/tft/tft_model_v3.py`, written from scratch in PyTorch.
- **Pipeline:**
  - `03_models/tft/tft_v4.py` (version 4.7 = v4.3; v4.2 ran on 4.6, v4.1 on 4.5): data, windows, training, baselines, post-processing, metrics, significance, exports;
  - `run_v4.py`: command-line interface.
- **Notebook:** `03_models/tft/notebooks/09_tft_colab_production_v4_3.ipynb` (current; earlier ones are in `notebooks/archive/`), run on a Colab T4. It also issues the operational forecast and the dashboard export.

### 8.2 One Run, Step by Step — What Happens, When, and Why

This is the exact order in which `tft_v4.run()` works. The function names refer to `03_models/tft/tft_v4.py`.

#### Stage 0 — load the master and build features once (`load_data`)

| Step | What happens | Why |
|---|---|---|
| 0.1 | Read `master_dataset_v4target.csv` and sort by district and date | – |
| 0.2 | Keep a **raw target** (`gwl_mean_raw`: the level on observed days, empty otherwise). Then linearly interpolate `gwl_mean/min/max` to give the network a continuous *input* history | The network needs gap-free inputs. It must never be trained or scored on an invented level, so the target stays raw. Interpolation cannot leak because every forecast's anchor day must be observed (step 5) |
| 0.3 | Recompute the lags (7/14/30 days) and the 7 and 30-day changes on the full calendar | Lags must be date-true (bug 1, Appendix A) |
| 0.4 | Calendar features: month and day-of-year as sin/cos; `is_monsoon`; `season_id` | Known in advance, so they can be decoder (future) inputs |
| 0.5 | Rainfall features: log of 1/7/30-day rainfall; 60 and 90-day totals; antecedent precipitation index `API_t = k·API_{t−1} + rain_t` with k = 0.90 / 0.95 / 0.98; days since rain (> 1 mm), capped at 120 | Recharge responds to accumulated wetness over different time scales, not to one day's rain |
| 0.6 | Rolling 30/60/90-day mean and SD of the level, **shifted by one day** | Describes the recent state using only the past |
| 0.7 | Assign each column a role: 47 observed-past, 8 known-future, 13 static numeric, 2 static categorical. Drop columns that are constant or duplicated | §5.8 |
| 0.8 | Drop tier-D districts (Latur), leaving 24 | Too little data to train on (§4.3) |

#### Stage 1 — for each fold in date order (fold 1 → fold 6)

Folds run in date order because each fold's post-processing uses the earlier folds' already-scored forecasts (step 11).

| Step | What happens | When (date cut-off) | Why |
|---|---|---|---|
| 1 | **Split dates** (`fold_dates`): the 3 months before the test period become the validation span. Its first 60% is the early-stopping block; the remaining 40% is the calibration block | see the table below | The network needs data to stop on, and post-processing needs data the network never saw |
| 2 | **Seasonal-naive tables** (`fit_snaive_table`): the typical h-day change for each day of the year, per district, for h = 1…30 | fitted on targets ≤ `fit_end` for the training, early-stopping and calibration windows; refitted ≤ `train_end` for the test windows | SN may use only the past. For training windows, the table is rebuilt **leaving that window's own year out**, so a training target never helps make its own baseline |
| 3 | **Normalise** (`normalize_for_fold`): missing covariates are filled with that district's training median; cumulative rainfall departure is computed against the training mean rainfall; a `StandardScaler` is fitted on training rows only | training rows ≤ `fit_end` | Nothing about the test period may influence the scaling |
| 4 | **Change scales** (`compute_delta_scales`): the SD of the h-step change, *minus the SN change*, per district and for each of the 30 horizons | training rows ≤ `fit_end` | Each horizon gets a comparable training signal (§8.7) |
| 5 | **Build windows** (`GWLDataset`) | – | – |
| | – one window per issue (anchor) day: 60 days of history and the next 30 days | | |
| | – the window belongs to a split if any of its 30 target dates falls inside the split | | |
| | – skipped if the encoder is < 70% observed or the anchor day is not observed; training and early-stopping windows also need ≥ 50% of their in-split target days observed | | Keeps windows with enough real data. The anchor must be real because every forecast is measured from it |
| | – groundwater inputs re-expressed as (value − anchor) ÷ scale | | The network never sees an absolute depth, so it predicts change |
| | – target = (actual − anchor − SN change) ÷ scale; mask = observed and inside the split | | Residual-on-SN target (§8.7); unobserved days get no gradient |
| | – strides: training 1, early-stopping and calibration 2, test 1 | | Every test issue day is scored |
| 6 | **Leak checks** (`_assert_no_leak`): no training, early-stopping or calibration target may lie beyond its cut-off | – | Stops the run if anything leaks (§6.9) |
| 7 | **Train the networks**, one per seed (10 seeds; §8.5) | train ≤ `fit_end`; early stopping on the early-stopping block | – |
| 8 | **Predict** the calibration and test windows. Average the 10 seeds in scaled units, then convert to metres: `Δ = output × scale + SN change` | – | Seed averaging cuts variance; the conversion undoes step 5 |
| 9 | **Tabular models** (`tab_fold`): ridge, and LightGBM with quantile loss for q10/q50/q90 (400 trees, learning rate 0.03, 15 leaves, at least 100 rows per leaf), one model per evaluated horizon, predicting the same residual target | the calibration-block model is trained ≤ `es_end`; the test model ≤ `train_end` | Each model is as fresh as possible without seeing the block it is evaluated on. The code asserts that its windows are identical to the network's |
| 10 | **Seasonal-naive baseline intervals**: SN median plus the 10th/90th percentile of training residuals, per district and step | training rows | So SN can be scored like the other models |
| 11 | **Post-processing** (`postprocess_fold`, §8.8): shrinkage, then stacking, then interval calibration, all fitted on the **pool** (this fold's calibration block plus all earlier folds' test forecasts) | – | The pool contains same-season errors (fold 4's monsoon learns from fold 1's monsoon) |
| 12 | **Scored rows**: one row per (district, issue date, horizon) for TFT (raw), TFT, TFT (calibrated), Ensemble, Ensemble (calibrated), Seasonal-naive, Ridge, LightGBM, plus a weekly-mean version (needs ≥ 4 observed days in the week). The code asserts identical keys for every model and `target_date = anchor + h` | – | Fair comparison: every model is scored on exactly the same targets (§6.9) |

This fold's test forecasts are then added to the pool for the next fold.

#### Stage 2 — after all folds

1. Metrics for the 21 headline districts: overall, per fold, per district, per season (§6.5).
2. The worst-5 minus best-5 district gap, and the count of districts beating persistence.
3. Significance tests for each model against persistence and seasonal-naive (§6.6).
4. The baseline ladder, coverage tables and `ablations.csv`, which records every run's settings and numbers.
5. Exports:
   - locked-fold subsets (`subset_folds`);
   - simple per-district files and plots for both models (`export_simple`);
   - variable importance and training curves (`export_shared`);
   - `RESULTS_TO_QUOTE.csv`.

#### The six folds' date cut-offs (from `fold_dates`)

| Fold | Training targets up to (`fit_end`) | Early-stopping block | Calibration block | Test period |
|---|---|---|---|---|
| 1 Monsoon 2024 | 2024-03-30 | 2024-03-31 → 05-24 | 2024-05-25 → 06-30 | 2024-07-01 → 09-30 |
| 2 Post-monsoon 2024 | 2024-06-30 | 2024-07-01 → 08-24 | 2024-08-25 → 09-30 | 2024-10-01 → 12-31 |
| 3 Dry 2025 | 2024-09-30 | 2024-10-01 → 11-24 | 2024-11-25 → 12-31 | 2025-01-01 → 05-31 |
| 4 Monsoon 2025 | 2025-02-28 | 2025-03-01 → 04-24 | 2025-04-25 → 05-31 | 2025-06-01 → 09-30 |
| 5 Post-monsoon 2025 | 2025-06-30 | 2025-07-01 → 08-24 | 2025-08-25 → 09-30 | 2025-10-01 → 12-31 |
| 6 Dry 2026 | 2025-09-30 | 2025-10-01 → 11-24 | 2025-11-25 → 12-31 | 2026-01-01 → 06-04 |

**Why training stops three months before each test period.** The early-stopping and calibration blocks must stay unseen by the network, or the stopping point and the intervals would be optimistic.
- The cost: the network never learns from the most recent three months.
- The v4.1 refit pass tried to recover them, and did not help (§9.5).
- LightGBM, which needs no early stopping, *is* refitted up to the day before the test period (step 9). This is part of why it became so strong.

### 8.3 Architecture

```
STATIC INPUTS (per district)
  district_id, aquifer_category_id -> learned embeddings
  soil texture (9), CGWB resource statistics (4)
        |
        v
  Static covariate encoder -> context vectors that condition the VSNs,
                              initialise the LSTM states and enrich the decoder

VARIABLE SELECTION NETWORKS (static / past / future)  -- vectorised
  each input -> projection -> compressed to d_select = 8 -> gated residual network
  softmax weights over the inputs at every time step (interpretable)

ENCODER  LSTM over the past 60 days (47 observed-past inputs, made relative to the anchor)
DECODER  LSTM over the next 30 days (8 known-future inputs)
  gated skip connections, static enrichment

INTERPRETABLE MULTI-HEAD ATTENTION (4 heads, causal mask)
  each decoder day attends to all encoder days and to earlier decoder days

POSITION-WISE GATED RESIDUAL NETWORK  (ELU + GLU gate + LayerNorm + residual)

MONOTONE QUANTILE HEAD (every decoder day)
  q50 = output;  q10 = q50 - softplus(a);  q90 = q50 + softplus(b)   -> the quantiles cannot cross
  loss: masked pinball loss at tau = 0.1 / 0.5 / 0.9 (no gradient on unobserved target days)

OUTPUT  [30 days x 3 quantiles], in units of the district's per-horizon change scale
  level = anchor + SN_change(h) + output(h) * scale(district, h)     (v4 residual-on-SN target)
```

### 8.4 What the Network Does With One Window — Forward Pass, Step by Step

This follows `TemporalFusionTransformer.forward` in `tft_model_v3.py`.

| Step | What happens | Why |
|---|---|---|
| 1 | **Static inputs are embedded.** `district_id` and `aquifer_category_id` become learned 8-dimensional embeddings, merged into one "variable". Each of the 13 numeric statics gets its own linear projection to d_model = 32 | Each district gets a learned identity, and soil and extraction statistics can shift its behaviour |
| 2 | **Static variable selection, then four context vectors.** A VSN weights the static variables. Four separate GRNs then turn the result into contexts for: (a) static enrichment, (b) conditioning the time-varying VSNs, (c) the LSTM's initial hidden state, (d) its initial cell state | One district summary steers every later stage. This is how, for example, a hard-rock district learns different input weights from an alluvial one |
| 3 | **Time-varying variable selection.** Every past input (47) and future input (8) is projected, compressed to `d_select = 8` and passed through its own small GRN. A softmax over the variables, conditioned on context (b), gives one weight per variable per day. The weighted sum is the day's input vector | Learns which inputs matter, and when. These are the importance weights reported in §10.10. The compression keeps the selector from outgrowing the network (§8.9) |
| 4 | **LSTM encoder, then decoder.** The encoder reads the 60 past days, starting from the static states (c) and (d). The decoder continues from the encoder's final state over the 30 future days, using only known-future inputs | Short-range memory; the forecast starts from the latest state |
| 5 | **Gated skip connection** (GLU and LayerNorm) over the LSTM outputs | The network can bypass the LSTM where it does not help, which keeps training stable |
| 6 | **Static enrichment.** A GRN mixes context (a) into every time step | Lets the same day of history mean different things in different districts |
| 7 | **Interpretable multi-head attention** (4 heads, shared value projection) with a causal mask: each decoder day sees all 60 history days and only earlier forecast days. Followed by a gate and skip | Direct access to any past day, such as the start of the monsoon or a rain event three weeks ago, without passing through the LSTM |
| 8 | **Position-wise GRN**, then a gate and skip | Final non-linear mixing per day |
| 9 | **Monotone quantile head** on the 30 decoder days only: `q50 = m`, `q10 = m − softplus(a)`, `q90 = m + softplus(b)`. The bias starts at (−0.5, 0, −0.5), so the first interval is about half a typical movement wide | q10 ≤ q50 ≤ q90 is guaranteed. A sensible starting width avoids degenerate early training |
| 10 | **Loss: masked pinball loss** at τ = 0.1 / 0.5 / 0.9, averaged only over observed target days inside the split (`MaskedQuantileLoss`) | Teaches all three quantiles at once. Days without a real reading give no gradient |

### 8.5 How the Network Is Trained — Step by Step

This follows `train_one` in `tft_v4.py`. It is repeated for each of the 10 seeds in each fold.

| Step | What happens | Why |
|---|---|---|
| 1 | Set the seed; build a fresh network (about 222k parameters) | Each seed is an independent model, so averaging them reduces variance |
| 2 | **District-balanced sampler.** Each training window is drawn with weight 1 ÷ (number of windows in its district), capped so that no district's windows are drawn more than 1.5× as often as the largest district's | Large districts would otherwise dominate the gradient. The cap stops tiny districts being over-fitted |
| 3 | **Windows per epoch:** one-seventh of all stride-1 windows, drawn at random (`epoch_frac = 1/7`) | Each epoch costs the same as the older stride-7 set-up, but every daily window can be used over the course of training. This was a v4.1 improvement |
| 4 | **Optimiser:** AdamW, learning rate 1e-3, weight decay 1e-2; batch 128; gradient clipping at norm 1.0 | Weight decay and a small network fight over-fitting; clipping prevents explosions on rare large changes |
| 5 | **Learning-rate schedule:** linear warm-up over the first 5 epochs, then ReduceLROnPlateau on the early-stopping loss (halve after 4 epochs without improvement, floor 1e-5) | Stable start; finer steps once progress stalls |
| 6 | **Early stopping** on the early-stopping block's masked pinball loss, patience 10 epochs, maximum 60. The best epoch's weights are restored | The stopping point is chosen on data the network does not train on. Best epochs in the v4.1 final run were 2–23 |
| 7 | **Predict** the calibration and test windows with the restored network; add to the seed sum | Step 8 of §8.2 |
| (8) | **Refit pass** (`train_fixed`, v4.1; *tested, not used*): retrain from scratch on training plus early-stopping windows for pass 1's best number of epochs (at least 5), replaying pass 1's learning-rate path | Meant to learn the most recent months. On the development folds it made the TFT worse (0.990 vs 0.979 vs SN), so the chosen set-up uses pass 1 |

### 8.6 Configuration (v4.1 final run)

| Setting | Value |
|---|---|
| d_model / d_select / GRN hidden factor | 32 / 8 / 0.5 |
| LSTM layers / attention heads / dropout | 1 / 4 / 0.25 |
| Categorical embedding dimension | 8 |
| Parameters | ~222,000 |
| Encoder / decoder length | 60 / 30 days |
| Optimiser | AdamW, learning rate 1e-3, 5-epoch warm-up, ReduceLROnPlateau (patience 4, factor 0.5, floor 1e-5), weight decay 1e-2, gradient clip 1.0 |
| Batch size / maximum epochs / early-stopping patience | 128 / 60 / 10 |
| Training sampling | all stride-1 windows; one-seventh drawn per epoch (`epoch_frac = 1/7`); sample-weight cap 1.5 |
| Seeds | 10 (42, 1337, 2026, 7, 99, 123, 555, 777, 2024, 31337), averaged |
| Districts modelled | 24 (tier D excluded) |

### 8.7 Target Formulation — How It Evolved

| Version | What the network predicts | What an output of 0 means | Why it changed |
|---|---|---|---|
| v1 | the absolute level (z-scored per district) | the district mean | The network spent its capacity re-learning the level and got it wrong by metres |
| v2 | the change from the anchor, Δ = gwl(t+h) − gwl(anchor), scaled by one change scale per district | **persistence** | Persistence becomes the floor; all groundwater inputs made relative to the anchor |
| v3 | the same, with a **separate scale for each of the 30 horizons** | persistence | With one shared scale, t+7 changes were small in scaled units and got almost no training signal |
| **v4 / v4.1** | the **residual on seasonal-naive**: Δ − SN change | **the seasonal-naive forecast** | v3's TFT was worse than SN (1.03–1.07); predicting only what SN gets wrong fixed this |

### 8.8 Post-Processing — Step by Step, and How It Evolved

After the networks and LightGBM have predicted, `postprocess_fold` does the following, in order, for each fold.

| Step | What happens | Why |
|---|---|---|
| 1 | **Long table** (`window_frame`): one row per (window, day ahead) with an observed, in-split target. Each row holds the actual change, the SN change, the TFT's q10/q50/q90, the LightGBM median, the scale and the season | Everything after this works on the same rows |
| 2 | **Build the pool**: this fold's calibration block plus every earlier fold's test rows. Rows from the test season get weight 1.0; other seasons get 0.3 | The calibration block alone is 5–6 calm weeks just before the test. Earlier folds add real out-of-sample errors from the same season. This is why v3's coverage undershot and v4's did not |
| 3 | **Shrinkage**: `mid = SN + λ · (TFT median − SN)` | Where the TFT's record is poor, its forecast is pulled back to the seasonal forecast |
| | – λ is searched on a grid from 0 to 1 in steps of 0.05, for each district and each 10-day band (days 1–10, 11–20, 21–30) | |
| | – the fit minimises weighted MAE on the pool | |
| | – each district's λ is pulled toward the band's overall λ, with a strength of 200 pool rows | A district with little data cannot pick an extreme λ |
| | – the interval keeps the TFT's spread around the new median | |
| 4 | **Stacking** (`fit_stack`): for t+7, t+14 and t+30, search every weight combination over {no change, SN, TFT median, LightGBM median} in steps of 0.1, with weights ≥ 0 that sum to 1, for the lowest weighted MAE on the pool. This is done globally, then per district, with each district pulled toward the global weights (prior 500 rows) | Each district learns how much to trust each forecaster. The simple baselines act as a safety net |
| | – *v4.2 option* `stack_guard`: if a district's shrunk weights are worse on its own pool (at least 50 weighted rows) than both SN alone and the TFT alone, that district uses the better of the two | The pull toward the global weights kept Pune on LightGBM, which is poor there. **Tested in v4.2 and rejected** (§9.8) |
| 5 | **Interval calibration** (`conformal_v4_online`): | Turns the intervals into ones with honest out-of-sample coverage, adapting as errors arrive in deployment order |
| | – on the pool, measure how far each actual fell below q10 and above q90, in units of the interval half-width (at least 0.25 × the change scale) | |
| | – for each (district, band, side), take a weighted quantile of those scores, pulled toward the district's tier with a pseudo-count of 100 | |
| | – walk through the test period in issue-date order and widen or narrow each side by that quantile × the half-width | |
| | – adaptive conformal inference (ACI): once a forecast's target date has passed, feed back whether it missed, and adjust the target miss rate by `α ← clip(α + 0.01·(0.10 − miss), 0.01, 0.30)` | |
| | – *v4.2 options*: feed back **one** outcome per (issue date, district, band) instead of one per cell (`aci_feedback='anchor'`); a symmetric α range such as 0–0.20; γ = 0 (no adaptation) | v4.1 fed the TFT about 10 correlated outcomes per issue date but the Ensemble only 1, and its α range left more room to narrow than to widen. **v4.2 chose γ = 0** (§9.8) |
| 6 | **Write the scored rows**: TFT (raw network), TFT (after shrinkage), TFT (calibrated), Ensemble (stack), Ensemble (calibrated), plus SN, ridge and LightGBM | Every stage can be judged separately. That is how the calibration problem in §10.9 was found |
| 7 | *v4.2*: **choose the reported interval** per model and horizon (`interval_choice`): the network's own interval or the calibrated one, decided on the development folds by a rule fixed in advance (§12.4) | Calibration helps some model–horizon pairs and hurts others; the choice is made once, never on the locked test |

How each step changed across versions:

| Step | v1 | v2 | v3 | v4 | v4.1 | v4.2 | v4.3 |
|---|---|---|---|---|---|---|---|
| **Shrinkage** `mid = c + λ(mid_TFT − c)`, λ per district and horizon band, fitted out of sample | – | – | toward persistence (c = 0), fitted on the calibration block | toward persistence, fitted on the pool | **toward seasonal-naive** (c = SN) | same | same |
| **Interval calibration** | in-sample | fitted on the test rows (invalid) | split conformal on the held-out calibration block | season-weighted pool, per-district normalised scores, each tail separately, ACI with delayed feedback (γ = 0.01, α clipped to [0.01, 0.3]) | same; **found to hurt the TFT's own intervals** (§10.9) | five variants compared on the development folds; **static conformal (γ = 0) chosen**; then **raw or calibrated chosen per model and horizon**: raw for the TFT and for the Ensemble at t+7 / t+14, calibrated for the Ensemble at t+30 | v4.2's + a **width cap** (3 × the raw distance); Ensemble 7-day now calibrated (evenness-aware rule D2) |
| **Combination** | – | – | 3-seed average | **stack**: simplex weights over {no change, SN, TFT, LightGBM} per district and horizon, fitted on the pool, pulled toward global weights (prior 500) | same, with the stronger LightGBM | same (a per-district **stack guard** was tested and rejected) | same + **LightGBM-only guard** (adopted; acted for Pune and Raigad; did not hold on the locked folds) |

**The Ensemble** is the final forecaster.
- Its median is the stacked blend.
- Its interval is the TFT's q10–q90 spread placed around that median. In v4.2 it is reported as is at t+7 and t+14, and calibrated (static conformal) at t+30, as chosen on the development folds.
- Both "TFT alone" and "Ensemble" are reported everywhere.

### 8.9 Changes to the Network Code, v1 → v3, and Why

Only two network files exist: `tft_model.py` (v1, v2) and `tft_model_v3.py` (v3, v4, v4.1). The network has not changed since v3. Every v4 and v4.1 improvement is in the data, the target, the training procedure or the post-processing.

| Change | Before (v1/v2, `tft_model.py`) | After (v3, `tft_model_v3.py`) | Why |
|---|---|---|---|
| Variable selection computed in batches | a Python loop: one `Linear(1 → d_model)` and one full GRN per input, per forward pass (about 114 small GPU operations per VSN call) | `BatchedFeatureProjection` and `BatchedGRN`: identical maths and weights, grouped into a few batched matrix operations | 3–5× faster. Together with the smaller network, a 6-fold × 3-seed run took 6.5 min instead of about 3 h, which made seed ensembles and screening affordable |
| Compression before selection (`d_select`) and a narrower per-variable GRN (`grn_hidden_factor`) | the selector saw all inputs at full width; its first layer alone (143k parameters) was larger than the rest of the network | each input compressed to 8 dimensions (`BatchedFeatureCompression`); per-variable GRN at 0.5× width | v2's real size was 1,231,159 parameters (it reported 205,173) for a few hundred independent windows. v3 really has about 222k |
| Monotone quantile head | three independent outputs; 3.2% of v1's predictions had crossing quantiles (q10 > q90) | median plus two softplus spreads, with a sensible initial width | Crossing becomes impossible; capacity is not wasted on ordering |
| Unused layers removed | three projection layers defined but never used (~17k parameters) | removed | Clean parameter count |
| Loss moved into the model file | the masked loss was defined in the notebook | `MaskedQuantileLoss(quantiles, q50_boost, horizon_weights)` | One tested implementation. Extra loss weight on t+7/14/30 (through the per-target weights) was tried in v4 and did not help |
| Configuration | v1: d_model 192, 3 LSTM layers, 8 heads, 120-day encoder (11.6M parameters). v2: d_model 64, 1 layer, 4 heads, 60-day encoder (1.23M) | d_model 32, `d_select` 8, 1 layer, 4 heads, 60-day encoder, dropout 0.25 (about 222k) | About 156–300 truly independent windows per fold. v1 memorised by epoch 1–2. v4.1 confirmed that a larger network (d_model 48) is worse |

Unchanged in every version: the overall TFT structure.
- the static covariate encoder;
- the LSTM encoder and decoder with gated skips;
- static enrichment;
- interpretable multi-head attention;
- the position-wise GRN;
- the quantile output on decoder days.

### 8.10 How the Run Is Organised and Cached (v4.1)

The pipeline caches its work in three stages.
- Cache keys include the data file's checksum and `PIPELINE_VERSION`, so stale results can never be reused silently.

| Stage | What it does | Cached? |
|---|---|---|
| 1a | Networks: training pass 1, plus the optional refit pass 2 | yes |
| 1b | Tabular models: LightGBM and ridge | yes, separately, so changing LightGBM never retrains the TFT |
| 2 | Post-processing: shrinkage, stacking, conformal, scoring | never cached; cheap to re-run |

This made it possible to screen 7 set-ups with only 3 network trainings, and to sweep post-processing settings for free.

**v4.2 uses the same caching more strictly.** The notebook trains development folds 1–4 once. Every post-processing variant is then a cache hit, and folds 5–6 are trained only after all choices are made. The v4.1 post-processing is re-run on the same networks as a reference, in case GPU training was not repeatable between Colab sessions.

**In the v4.2 run:**
- The re-trained networks and LightGBM models turned out to be **bit-for-bit identical** to v4.1's: all 220,215 forecast rows match exactly.
- The run took 31 min.

### 8.11 Proposed but Not Built (and why)

- **TFT + GAT spatial layer.** On hold, for three reasons:
  1. The per-district evidence points to data completeness as the limit (§4.3, §10.7).
  2. The 2026 HESS study found no systematic gain from globally trained groundwater models.
  3. There are only hundreds of independent training windows, so a graph layer would add capacity the data cannot support.

  The suggested cheaper experiment is to cluster districts by similar dynamics (correlation of their change series) and train per cluster.
- **Physics-informed loss** (penalising rises with no recharge): not implemented. It is described in `README.md` §4.4 as an optional variant.

---

## 9. Model Pipeline Versions v1 → v4.3 — What Changed in Each

### 9.1 Production v1 (14 Aug 2026) — "looked fine, was worse than doing nothing"

**Set-up:**
- `09_tft_colab_production.ipynb`.
- 11.6M parameters: d_model 192, 3 LSTM layers, 8 heads, 120-day encoder, stride 1.
- Absolute-level target.
- About 3 h on a T4.

**What it reported:** median RMSE 1.78–2.15 m and MAPE 18–23%, which looked plausible.

**What was actually true:** it lost to persistence in **15 of 18** fold-horizon cases.
- At t+7 in fold 2, persistence MAE was 0.56 m against v1's 2.91 m.
- Median MASE was 2.12 / 1.37 / 1.04.

**Root causes:**
1. The district series were not physically real (§5.3).
2. It predicted the level, not the change.
3. It had 11.6M parameters for about 156 independent windows. Three folds peaked at epoch 1–2, which is memorisation.
4. No metric had a baseline.

The v1 outputs are in `04_results/tft_prod_v1/`.

### 9.2 Production v2 (15 Aug 2026)

**Changes:**
- groundwater panel filter, per-well anomaly aggregation and drift filter (§5.3);
- change-from-anchor target;
- capacity cut: d_model 64, 1 layer, 4 heads, encoder 60, dropout 0.25, weight decay 1e-2, training stride 7;
- MASE added to every table.

**Result:**
- Median MASE 1.004 / 0.940 / 0.827: tied with persistence at t+7.
- 7 districts still lost to persistence at every horizon.

**Hidden problems, found only in the v3 audit:**
- The printed "205,173 parameters" was wrong; the real count was **1,231,159**, because the variable selector alone was larger than the rest of the network.
- Sentinel features leaked the future.
- ET and soil moisture were fabricated for half the evaluation.
- Conformal calibration was fitted on the test set.

### 9.3 Production v3 (15 Aug 2026)

**Changes:**
- ERA5-Land; causal Sentinel; data tiers (§5.4).
- Rewritten, **vectorised network** (`tft_model_v3.py`): batched GRNs and feature projections, `d_select` compression, a narrower GRN, and a monotone quantile head. It has a real ~222k parameters.
- Per-horizon change scaling.
- Shrinkage toward persistence, fitted on a held-out calibration block.
- Honest split conformal.
- 3 seeds.
- Baseline ladder (persistence, climatology, ridge, LightGBM).
- Diebold–Mariano test and bootstrap confidence intervals.
- Daily and 7-day-mean targets, both scored.

**Result (21 districts):**
- MASE **0.958 / 0.909 / 0.834**, all significant against persistence.
- Districts beating persistence: 16 / 19 / 19 of 21.
- All seven of v2's failing districts now tie or beat persistence.
- Run time 3 h → **6.5 min**.

**Honest caveats at the time:**
- LightGBM beat the TFT at t+7.
- Coverage was 70 / 66 / 63% against the 80% target.
- The weekly-mean target did not help.

**Hidden problems, found in v4:**
- On a fair comparison, v3's TFT was **worse than seasonal-naive** (1.03–1.07). v3 had only compared against persistence.
- The decoder-end-date windowing bug.

### 9.4 Production v4 (3 Oct 2026)

**Changes:**
1. Groundwater target rebuilt (§5.5).
2. Evaluation fixed: issue-date windows; every scored row carries `anchor_date` and `target_date`; all models are scored on identical keys, which the code asserts.
3. Seasonal-naive baseline.
4. Residual-on-SN target.
5. v4 calibration: pooled, asymmetric, ACI.
6. Stacked Ensemble.
7. Development and locked folds.
8. 5 seeds in the final run.

The network file is unchanged.

**Screening (development folds 1–4, rMAE vs seasonal-naive):**
- v3-style TFT: 1.008 / 1.032 / 1.040.
- Chosen v4 TFT: 0.996 / 0.989 / 0.976.
- Chosen v4 Ensemble: **0.973 / 0.966 / 0.951**.

**Final (locked folds 5–6):**
- Ensemble: 0.865 / 0.788 / 0.664 vs persistence; 0.967 / 0.952 / 0.945 vs SN; coverage 81%.
- TFT alone: tied with SN (1.017 / 1.008 / 0.992).

### 9.5 Production v4.1 (built 26 Oct, run 29 Oct 2026)

**Changes:**

| # | Change | Kept in the final set-up? |
|---|---|---|
| 1 | Shrink the TFT toward **SN** instead of toward persistence | yes (TFT-alone t+30 vs SN improved from 0.976 to 0.951 on development folds) |
| 2 | **Stronger LightGBM**: calendar, SN change, scale, level and year-on-year features; stride 1; trained fresh up to the day before the test period; 400 trees, learning rate 0.03, minimum 100 rows per leaf | **yes, the main gain** |
| 3 | **Stride-1 network sampling** (all daily windows, one-seventh per epoch) | **yes** |
| 4 | **Refit pass**: retrain on training plus early-stopping windows for pass 1's best epoch, replaying its learning-rate schedule | no (it made the TFT worse) |
| 5 | Larger network (d_model 48) | no (worse) |
| 6 | Sweeps of stacking strength {100, 250, 500, 1000} and ACI speed {0.005, 0.01, 0.02, 0.05} | defaults kept (500, 0.01) under the pre-registered rule |
| 7 | 10 seeds; three-stage caching; Winkler interval score | yes |
| 8 | Simple per-district outputs for **both** the Ensemble and the TFT (all 24 districts, with Beed and Latur as "no usable data"), plus `RESULTS_GUIDE.md` | yes |

**Selection** (pre-registered rule: lowest mean Ensemble rMAE vs SN on development folds; a simpler set-up wins within 0.002):

| Candidate | Ensemble mean vs SN | TFT mean vs SN |
|---|---|---|
| `ref` (v4 + change 1) | 0.963 | 0.979 |
| `tab2` (+ change 2) | 0.949 | 0.979 |
| `refit` (+ change 4) | 0.969 | 0.990 |
| `tab2_refit` | 0.946 | 0.990 |
| **`s1_tab2` (+ changes 2 and 3) — chosen** | **0.939** | 0.982 |
| `s1_tab2_refit` | 0.954 | 0.992 |
| `d48_tab2_refit` | 0.961 | 1.034 |

Results are in §10.

### 9.6 Everything That Changed, in One Table

| Area | v1 | v2 | v3 | v4 | v4.1 | v4.2 | v4.3 |
|---|---|---|---|---|---|---|---|
| Groundwater series | all stations, raw median | telemetric panel, anomaly aggregation, drift filter | same | **rebuilt**: placeholders, stuck runs, two-sided Hampel, steps and regimes, factor model, no fills | same | same | same |
| ET and soil moisture | NRSC (fabricated after 2025-06) | same | **ERA5-Land, 4 layers** | same | same | same | same |
| Sentinel | interpolated (future leak) | same | **causal** | same | same | same | same |
| Target | absolute level | Δ from anchor | Δ with per-horizon scale | **residual on SN** | same | same | same |
| Parameters | 11.6M | 1.23M (reported as 205k) | ~222k | ~222k | ~222k | ~222k | same |
| Training windows | stride 1 | stride 7 | stride 7 | stride 7 | **stride 1, 1/7 per epoch** | same | same |
| Seeds | 1 | 1 | 3 | 5 | **10** | 10 | same |
| Window keying | decoder end | decoder end | decoder end | **issue date** | issue date | issue date | same |
| Baselines | none | persistence | + climatology, ridge, LightGBM | **+ seasonal-naive** | + stronger LightGBM | same | same |
| Shrinkage | – | – | toward persistence | toward persistence | **toward SN** | same | same |
| Calibration | in-sample | on test (invalid) | held-out block | **pooled, asymmetric, ACI** | same | **raw or static conformal, chosen per model × horizon** (TFT raw; Ensemble raw at t+7 / t+14, calibrated at t+30) | as v4.2 + **width cap 3 ×**; Ensemble 7-day now calibrated (evenness-aware rule D2) |
| Combination | – | – | seed average | **stacked ensemble** | same | same (per-district guard tested, rejected) | same + **LightGBM-only guard** (acted for Pune and Raigad) |
| Significance | – | – | DM + bootstrap | cluster bootstrap + DM + Holm | same | same | same |
| Selection protocol | – | – | – | development folds 1–4, locked 5–6 | pre-registered rule | three rules fixed in advance; folds 5–6 computed only after them | rules G2, D2 fixed in advance; folds 5–6 computed only after them |
| Time per full run (T4) | ~3 h | ~3 h | 6.5 min | – | 54 min (screening, sweeps and a 10-seed final) | 31 min (one training; post-processing variants free) | 39 min (one training) + operational forecast |

### 9.7 Change Log — Every Change, When, Why, and Its Measured Effect

In date order. "Kept" means the change is part of the current system. Negative results are included on purpose.

| # | When (2026) | Version | Change | Why | Measured effect | Kept? |
|---|---|---|---|---|---|---|
| 1 | 4 Jul | data | Full daily calendar; date-true lags; district gate | Windows and lags were built over gappy rows | Window continuity 75–95% → 100% | yes |
| 2 | 4 Jul | data | Rainfall CWC stations → CHIRPS; district-name fixes | Station gaps; silent missing data for 4 districts | CHIRPS 36/36 districts; soil texture 35/36 | yes |
| 3 | Jul–Aug | evaluation | Six-fold walk-forward evaluation | One dry-season test window cannot show monsoon skill | Exposed the v1 failure | yes |
| 4 | 15 Aug | v2 | Telemetric panel filter, per-well anomaly aggregation, drift filter | Manual-survey wells created ~94 m false jumps | Persistence RMSE at t+7 2.60 → 1.07 m; worst district SD 30.5 → 10.2 m | yes (later replaced by #13) |
| 5 | 15 Aug | v2 | Predict the change from the anchor; anchor-relative inputs | Predicting the level lost to persistence | Median MASE about 2.1 → 1.004 at t+7 (with #4 and #6) | yes |
| 6 | 15 Aug | v2 | Capacity cut (d_model 192 → 64, 3 → 1 layers, encoder 120 → 60, stride 7, dropout and weight decay up) | 11.6M parameters for about 156 independent windows | No more epoch-1–2 memorisation | yes (further in #9) |
| 7 | 15 Aug | v2 | MASE in every table | No baseline anywhere | Made failure visible | yes |
| 8 | 15 Aug | v3 | ERA5-Land replaces NRSC; causal (month-end) Sentinel | Fabricated ET and soil moisture; 15-day future leak | Real inputs in every fold | yes |
| 9 | 15 Aug | v3 | Vectorised network with `d_select` compression and a monotone head (§8.9) | Real size 1.23M, not the reported 205k; quantiles could cross | About 222k parameters; 3 h → 6.5 min | yes |
| 10 | 15 Aug | v3 | Per-horizon change scale | t+7 got almost no training signal | t+7 MASE 1.004 → 0.958 (together with #8, #9 and #11) | yes |
| 11 | 15 Aug | v3 | Shrinkage toward persistence on a held-out block; honest split conformal; data tiers; baseline ladder; significance tests; 3 seeds | No floor under failing districts; coverage claim invalid | Worst district at t+30 1.896 → 1.017; all 7 failing districts fixed; honest coverage 63–70% | partly (shrinkage centre changed in #19; calibration replaced in #16) |
| 12 | 15 Aug | v3 | Weekly-mean target scored alongside the daily one | A ridge probe suggested it would help | Within 0.01–0.02 of daily; **no gain** | reported only |
| 13 | 3 Oct | v4 | Groundwater target rebuilt: placeholders, stuck loggers, two-sided Hampel filter, steps and regimes, factor-model aggregation, no fills | 376 false jumps > 2 m; 12 good wells lost | Jumps 329 → 193; development-fold t+7 MAE 0.57 → 0.51 m | yes |
| 14 | 3 Oct | v4 | Issue-date window keying; identical-key scoring | Some t+7 targets sat in validation; the last 23 days per season were never scored | Correct, complete scoring | yes |
| 15 | 3 Oct | v4 | Seasonal-naive baseline (tuned on its own development-fold skill) | Persistence is too easy a bar for a seasonal series | Showed the v3-style TFT was worse than SN (1.03–1.07) | yes |
| 16 | 3 Oct | v4 | Calibration v4: season-weighted pool, two-sided normalised scores, ACI | v3 coverage undershot (63–70%) | Ensemble coverage 78–81% | yes (but see #25) |
| 17 | 3 Oct | v4 | Stacked ensemble of no change, SN, TFT and LightGBM | No single model was best everywhere | Development folds: vs persistence 0.977 → 0.927 at t+7, 0.849 → 0.807 at t+30, same networks | yes |
| 18 | 3 Oct | v4 | Residual-on-SN target | The TFT was worse than SN | TFT vs SN: 1.031 / 1.055 / 1.050 → 0.996 / 0.989 / 0.976 | yes |
| (–) | 3 Oct | v4 | Extra loss weight on t+7/14/30 | Focus training on the scored horizons | Slightly worse (t+30 0.951 → 0.964 vs SN) | **no** |
| (–) | 3 Oct | v4 | Ablation: drop the ERA5 and Sentinel inputs | Check whether they carry information | Worse (mean vs SN 0.963 → 0.978), so they help a little | inputs kept |
| 19 | 26 Oct | v4.1 | Shrinkage centred on SN instead of on persistence | v4.0 shrank the seasonal part away | TFT t+30 vs SN 0.976 → 0.951 (development folds) | yes |
| 20 | 26 Oct | v4.1 | LightGBM v2 features, stride 1, fresh cut-offs, 400 trees | LightGBM could not see the time of year and was three months stale | Ensemble mean vs SN 0.963 → 0.949 | **yes, main gain** |
| 21 | 26 Oct | v4.1 | Stride-1 network sampling (one-seventh per epoch) | Use every daily window at the same cost | 0.949 → 0.939 | yes |
| (–) | 26 Oct | v4.1 | Refit pass on training plus early-stopping windows | Learn the most recent months | TFT 0.979 → 0.990 (worse) | **no** |
| (–) | 26 Oct | v4.1 | d_model 48 | Test more capacity | TFT 1.034 (worse) | **no** |
| 22 | 26 Oct | v4.1 | Sweeps of stacking strength and ACI speed | Evenness across districts; interval tails | Defaults (500, 0.01) were best or tied | defaults kept |
| 23 | 29 Oct | v4.1 | 10 seeds in the final run | Less seed noise | Locked Ensemble 0.865 / 0.788 / 0.664 → **0.839 / 0.763 / 0.616** (with #19–21) | yes |
| 24 | 26–29 Oct | v4.1 | Simple per-district outputs for both models; results guide | Results in the familiar v1 format | 24 districts plus 2 "no usable data" | yes |
| 25 | 29 Oct | analysis | Calibrated vs uncalibrated intervals compared on development folds | Check whether the calibration step earns its place | **It hurts the TFT's intervals** (coverage 80/77/71 → 72/68/64% on development folds) | to be fixed in v4.2 |
| 26 | 30 Oct | v4.2 | ACI fed one outcome per issue date and band; symmetric α range; static conformal; calibrated or raw interval chosen per model × horizon (rules C, D) | v4.1 fed the TFT ~10 correlated outcomes per issue date (the Ensemble 1), and the α range 0.01–0.30 around 0.10 favoured narrowing | **TFT:** locked coverage 74/70/66 → 80/78/77%, interval score −12/−11/−19%. **Ensemble:** +3/+3/+2% locked, +2/−9/−3% all folds. Static conformal (γ = 0) chosen; raw intervals reported except Ensemble t+30 | **yes** (TFT intervals fixed) |
| 27 | 30 Oct | v4.2 | Per-district stack guard (rule G) | Pune's weights were ≈ global, which leaned on LightGBM (MASE 1.30 at t+30 in Pune, vs 0.87 for SN and TFT) | Fixed Pune on development folds (t+30 1.27 → 0.87), but fired in 53 cases and made 22 district-horizons worse (Dhule +0.13); development mean vs SN 0.950 → 0.969 | **no** (rejected by rule G) |
| 28 | 31 Oct | v4.3 | LightGBM-only guard (rule G2): refit a district's stack without LightGBM where LightGBM alone is > 20% worse than SN on its pool | v4.2's whole-blend guard hurt 22 district-horizons | Adopted on development folds (mean vs SN 0.9501 → 0.9486; no district worse; acted only for Pune and Raigad). **On the locked folds Pune got worse** (0.89 / 0.79 / 0.77 → 0.94 / 0.91 / 0.92), moving the headline by +0.002 to +0.004 | yes (rule G2), mixed |
| 29 | 31 Oct | v4.3 | Width cap (3 ×) and the evenness-aware interval rule D2 | 90–108 m intervals; raw intervals uneven across districts. Designed after seeing v4.2's locked results | Ensemble 7-day now calibrated: districts covered 65–90% 86 → 100% (all folds), 57 → 76% (locked); intervals > 20 m 332 → 105 (max 34 m); interval score all folds −0.7 / −0.2 / −3.6% | yes |
| 30 | 31 Oct | v4.3 | Operational forecast from the latest reading (`forecast_latest`, `fold7_latest`) + `dashboard_export` | The dashboards need a forward forecast made exactly like an evaluated fold | Issued 4 Jun 2026 for 23 of 24 districts (Nanded skipped: 40% of the last 60 days observed). At t+30: 15 rising, 1 steady, 7 uncertain, 0 falling | yes |
| 31 | 31 Oct | dashboards | Farmer and government dashboards (`dev/`), en / mr / hi, site-guide chatbot, Gnani speech plan | The project's operational goal | 14 backend + 13 frontend tests; browser run with 0 console errors | yes |

### 9.8 Production v4.2 (built and run 30 Oct 2026)

**Post-processing only.** The networks and LightGBM use the v4.1 final set-up.
- The re-trained models reproduced v4.1 **bit-for-bit**: all 220,215 forecast rows are identical.
- So the v4.1 → v4.2 comparison isolates the post-processing change.
- The notebook trained development folds 1–4 once (10 seeds). It applied three rules fixed in advance on those folds, and only then trained folds 5–6. Total time 31 min.

| Rule | Question | Outcome on development folds 1–4 | Decision |
|---|---|---|---|
| **G** | Use a per-district stack guard (fall back to SN or TFT where the stack lost to both on the district's past errors)? | **Fixed Pune** (MASE 1.04 / 1.16 / 1.27 → 1.02 / 1.05 / 0.87). But it fired in 53 cases across 16 districts and made 22 district-horizons worse (Dhule +0.13, Jalna +0.09, Ratnagiri +0.09). Mean error ÷ SN error 0.950 → 0.969 | **rejected** |
| **C** | Which calibration variant? | Score = calibrated ÷ raw interval score (below 1 = calibration helps):<br>v4.1 1.131;<br>one ACI outcome per issue date 1.115;<br>+ symmetric α 1.129;<br>+ γ 0.03 1.166;<br>**static, γ = 0: 1.097**.<br>The per-issue-date fix lifted the TFT's calibrated coverage 72/68/64 → 76/77/73% and left the Ensemble unchanged, which confirms the diagnosis. But no variant beat the raw intervals | static conformal (γ = 0) |
| **D** | Report raw or calibrated, per model and horizon? | TFT: raw at all three horizons (raw coverage 80/77/71% vs 72/74/71% calibrated, with a lower interval score). Ensemble: raw at t+7 / t+14; at t+30 raw covered only 73.7%, calibrated 76.1% | TFT raw; Ensemble raw, raw, calibrated |

**Result** (§10.12):
- **Accuracy is unchanged:** the Ensemble median is v4.1's.
- **The TFT's intervals are fixed:** locked coverage 74 / 70 / 66% → 80 / 78 / 77%, interval score 11–19% better.
- **The Ensemble's intervals are about the same on average** (locked +1.5–3.3%, all folds +1.7 / −8.7 / −3.3%), but less even across districts.
- **Pune is still open.**

### 9.9 Production v4.3 (built 31 Oct, run 1 Nov 2026)

**Post-processing only, plus the operational forecast.**
- The networks and LightGBM again re-trained **bit-for-bit identical** to v4.1 and v4.2: TFT, LightGBM, ridge and seasonal-naive rows all identical.
- The Ensemble medians changed **only in Pune and Raigad**, where the new guard acted.
- 39 min on a T4.

| Rule | Outcome on development folds 1–4 | Decision |
|---|---|---|
| **G2**: refit a district's stack without LightGBM where LightGBM alone was > 20% worse than SN on its pool | The mean error ÷ SN error improved slightly (0.9501 → 0.9486); no district got worse. It acted only for Pune (folds 3–4) and Raigad (single well). Pune's dev MASE: 1.044 / 1.155 / 1.271 → 1.041 / 1.045 / 1.200 | **adopted** |
| **C**: calibration | Fixed in advance: v4.2's static conformal plus a width cap of 3 × the raw distance | – |
| **D2**: raw or calibrated, counting district evenness | Ensemble 7-day: both options were in 75–85% coverage with equal interval scores (2.766 vs 2.769), and calibrated was far more even (95% vs 81% of districts), so calibrated was chosen. The other choices stayed as in v4.2: TFT raw at every horizon; Ensemble raw at t+14 and calibrated at t+30 | Ensemble 7-day → calibrated |

**Results** (§10.13):
- **The intervals improved where intended.**
  - The 7-day Ensemble is now honest in **every** headline district over all folds (100%, from 86%), and in 76% on the locked test (from 57%).
  - Intervals wider than 20 m fell from 332 to 105; the widest is now 34 m, down from 94 m.
- **The Pune guard did not generalise.** Over all folds Pune improved at 14 days (1.050 → 1.004), but on the locked folds 5–6 it got worse at every horizon (t+30: 0.77 → 0.92). In those folds LightGBM happened to forecast Pune well, and the guard had removed it.
- **The locked headline moved by +0.002 to +0.004** (0.841 / 0.766 / 0.620), still significant at p = 0.003.
- **The rule's choice is kept and reported as is.** Reversing it because of the locked folds would turn the locked test into a selection set.

**The operational forecast** was issued from the last reading, 4 Jun 2026, for 23 of 24 districts. Nanded was skipped: its wells reported on 40% of the last 60 days, and 70% is needed. This is now what the dashboards show.

---

## 10. Results

Unless stated otherwise:
- numbers are pooled over the **21 headline districts**;
- they are t+7 / t+14 / t+30;
- below 1 is better for the ratios.

All v4.1 numbers come from `04_results/tft_v4_1/` (`RESULTS_TO_QUOTE.csv`, `FINAL_s1_tab2*/`, `simple/`). The v4.2 numbers come from `04_results/tft_v4_2/` (`RESULTS_TO_QUOTE.csv`, `v4_1_vs_v4_2.csv`, `FINAL_v4_2*/`, `simple/`). The two runs' forecasts are identical; only the reported intervals differ.

### 10.1 Scores of Every Version (as each was reported)

| Version | Model | Test set | MASE vs no-change | vs seasonal | MAE (m) | MAPE (%) | 80% coverage (%) |
|---|---|---|---|---|---|---|---|
| Phase 1 | LSTM (RMSE 3.27 / 4.18 / 3.54 m) | old data, Feb–Apr 2025 | not measured | – | 1.65 / 1.99 / 1.80 | – | – |
| **v1** | TFT, 11.6M params | 6 folds | **2.12 / 1.37 / 1.04** (median) | – | 1.28 / 1.37 / 1.53 | 18.5 / 19.5 / 23.2 | 52 / 52 / 53 (raw; the reported 80% was in-sample) |
| **v2** | TFT, change target | 6 folds | 1.004 / 0.940 / 0.827 (median) | – | – | – | 76 / 71 / 64 (raw; the reported 80% was invalid) |
| **v3** | TFT + shrinkage | 6 folds | 0.958 / 0.909 / 0.834 | – | 0.53 / 0.75 / 1.11 | 9.7 / 13.6 / 20.1 | 70 / 66 / 63 |
| v4 | TFT alone | 6 folds | 0.937 / 0.877 / 0.773 | 1.003 / 1.002 / 0.987 | 0.45 / 0.63 / 0.93 | 6.2 / 8.8 / 13.2 | 74 / 69 / 65 |
| v4 | Ensemble | 6 folds | 0.911 / 0.850 / 0.750 | 0.975 / 0.972 / 0.958 | 0.43 / 0.61 / 0.91 | 6.0 / 8.6 / 12.9 | 81 / 80 / 78 |
| v4 | Ensemble | **locked 5–6** | 0.865 / 0.788 / 0.664 | 0.967 / 0.952 / 0.945 | 0.32 / 0.42 / 0.59 | 4.4 / 5.7 / 7.4 | 81 / 82 / 81 |
| v4.1 | TFT alone | 6 folds | 0.935 / 0.876 / 0.764 | 1.000 / 1.002 / 0.976 | 0.44 / 0.63 / 0.92 | 6.2 / 8.7 / 13.0 | 73 / 69 / 64 |
| v4.1 | **Ensemble** | 6 folds | **0.903 / 0.830 / 0.712** | **0.967 / 0.949 / 0.909** | **0.43 / 0.60 / 0.86** | **6.0 / 8.4 / 12.4** | **80 / 78 / 78** |
| v4.1 | TFT alone | **locked 5–6** | 0.872 / 0.803 / 0.675 | 0.976 / 0.971 / 0.960 | 0.33 / 0.43 / 0.60 | 4.4 / 5.8 / 7.6 | 74 / 70 / 66 |
| v4.1 | Ensemble | locked 5–6 | 0.839 / 0.763 / 0.616 | 0.938 / 0.922 / 0.877 | 0.31 / 0.41 / 0.55 | 4.3 / 5.6 / 7.1 | 81 / 80 / 80 |
| v4.2 | TFT alone (raw intervals) | 6 folds | 0.935 / 0.876 / 0.764 | 1.000 / 1.002 / 0.976 | 0.44 / 0.63 / 0.92 | 6.2 / 8.7 / 13.0 | **80 / 77 / 73** |
| v4.2 | Ensemble | 6 folds | 0.903 / 0.830 / 0.712 | 0.967 / 0.949 / 0.909 | 0.43 / 0.60 / 0.86 | 6.0 / 8.4 / 12.4 | 81 / 79 / 77 |
| v4.2 | TFT alone (raw intervals) | **locked 5–6** | 0.872 / 0.803 / 0.675 | 0.976 / 0.971 / 0.960 | 0.33 / 0.43 / 0.60 | 4.4 / 5.8 / 7.6 | **80 / 78 / 77** |
| v4.2 | Ensemble | locked 5–6 | 0.839 / 0.763 / 0.616 | 0.938 / 0.922 / 0.877 | 0.31 / 0.41 / 0.55 | 4.3 / 5.6 / 7.1 | 82 / 81 / 79 |
| v4.3 | Ensemble | 6 folds | 0.903 / 0.829 / 0.712 | 0.967 / 0.948 / 0.909 | 0.43 / 0.60 / 0.86 | 6.0 / 8.4 / 12.3 | 77 / 79 / 77 |
| **v4.3** | **Ensemble (final)** | **locked 5–6** | **0.841 / 0.766 / 0.620** | **0.940 / 0.926 / 0.882** | **0.31 / 0.41 / 0.55** | **4.3 / 5.6 / 7.2** | **80 / 80 / 78** |

> **The versions were not all scored the same way.** The target data, folds and scoring changed between them, so treat Table 10.1 as history and §10.2 as the fair comparison.
> - v1 MASE was computed afterwards from its predictions.
> - Phase 1 used the old data and a different test period.
> - The v4.3 TFT alone is identical to v4.2 (same networks and intervals).
> - MAPE is not comparable across versions.

### 10.2 Like-for-Like Comparison (development folds 1–4, identical windows, scoring and baseline)

| Set-up | Model | vs no-change | vs seasonal | MAE (m) | Coverage (%) |
|---|---|---|---|---|---|
| v3 with fixed evaluation, **old** target | TFT | 0.982 / 0.951 / 0.889 | 1.008 / 1.032 / 1.040 | 0.57 / 0.83 / 1.31 | 74 / 69 / 64 |
| v3 with fixed evaluation, **new** target | TFT | 0.974 / 0.932 / 0.862 | 1.028 / 1.047 / 1.066 | 0.51 / 0.75 / 1.18 | 72 / 68 / 63 |
| + v4 calibration + ensemble | Ensemble | 0.927 / 0.876 / 0.807 | 0.978 / 0.985 / 0.998 | 0.48 / 0.70 / 1.10 | 78 / 77 / 73 |
| + residual-on-SN target (**v4**) | TFT | 0.944 / 0.880 / 0.789 | 0.996 / 0.989 / 0.976 | 0.49 / 0.71 / 1.08 | 74 / 69 / 65 |
| | Ensemble | 0.922 / 0.860 / 0.769 | 0.973 / 0.966 / 0.951 | 0.48 / 0.69 / 1.05 | 81 / 81 / 77 |
| + shrink toward SN (v4.1 `ref`) | TFT | 0.943 / 0.880 / 0.769 | 0.996 / 0.990 / 0.951 | – | 74 / 70 / 67 |
| + new LightGBM (`tab2`) | Ensemble | 0.924 / 0.848 / 0.744 | 0.976 / 0.953 / 0.919 | – | 82 / 80 / 77 |
| **+ stride-1 sampling (`s1_tab2`, v4.1)** | TFT | 0.937 / 0.880 / 0.783 | 0.989 / 0.990 / 0.967 | – | 73 / 69 / 64 |
| | **Ensemble** | **0.907 / 0.841 / 0.739** | **0.958 / 0.945 / 0.913** | – | **80 / 79 / 78** |

What each step contributed:

| Step | Contribution |
|---|---|
| Cleaner target | MAE −10% |
| Ensemble | the largest single gain over the v3-style TFT |
| Residual-on-SN target | brought the TFT from worse than SN to level with it |
| Stronger LightGBM and stride-1 sampling | the v4.1 gains |
| Loss weighting on t+7/14/30 (v4) | did not help |
| Refit pass (v4.1) | did not help |
| Larger network (v4.1) | did not help |

### 10.3 Final v4.1 Results (10 seeds)

| Test set | Model | Horizon | MAE (m) | RMSE (m) | vs no-change | vs seasonal | Coverage | below / above | Width (m) | Beat no-change |
|---|---|---|---|---|---|---|---|---|---|---|
| **Locked 5–6** | **Ensemble** | t+7 | 0.314 | 0.521 | **0.839** | **0.938** | 81.1 | 10.7 / 8.2 | 1.10 | 21/21 |
| | | t+14 | 0.406 | 0.628 | **0.763** | **0.922** | 80.1 | 12.7 / 7.1 | 1.52 | 21/21 |
| | | t+30 | 0.550 | 0.836 | **0.616** | **0.877** | 80.0 | 14.8 / 5.2 | 2.22 | 21/21 |
| | TFT alone | t+7 | 0.326 | 0.537 | 0.872 | 0.976 | 74.2 | 14.3 / 11.5 | 1.18 | 20/21 |
| | | t+14 | 0.428 | 0.656 | 0.803 | 0.971 | 70.0 | 18.8 / 11.2 | 1.49 | 20/21 |
| | | t+30 | 0.602 | 0.903 | 0.675 | 0.960 | 65.7 | 25.1 / 9.2 | 1.99 | 20/21 |
| **All 6 folds** | **Ensemble** | t+7 | 0.429 | 0.789 | 0.903 | 0.967 | 79.9 | 10.7 / 9.5 | 1.52 | 21/21 |
| | | t+14 | 0.595 | 1.105 | 0.830 | 0.949 | 78.4 | 12.8 / 8.8 | 2.50 | 20/21 |
| | | t+30 | 0.861 | 1.526 | 0.712 | 0.909 | 78.1 | 13.2 / 8.6 | 3.62 | 20/21 |
| | TFT alone | t+7 | 0.444 | 0.812 | 0.935 | 1.000 | 73.0 | 14.1 / 12.9 | 1.60 | 19/21 |
| | | t+14 | 0.629 | 1.157 | 0.876 | 1.002 | 68.6 | 17.2 / 14.2 | 2.26 | 19/21 |
| | | t+30 | 0.924 | 1.609 | 0.764 | 0.976 | 64.3 | 20.9 / 14.8 | 2.90 | 20/21 |

Nash–Sutcliffe efficiency (Ensemble): 0.959 / 0.943 / 0.910 on the locked test; 0.919 / 0.840 / 0.693 over all folds.

**Significance, locked test, Ensemble** (district-cluster bootstrap 95% CI; Diebold–Mariano statistic; all Holm-adjusted p = 0.003):

| Horizon | vs persistence | vs seasonal-naive |
|---|---|---|
| t+7 | 0.839 [0.796, 0.875], DM −9.25 | 0.938 [0.919, 0.954], DM −7.20 |
| t+14 | 0.763 [0.701, 0.819], DM −8.09 | 0.922 [0.892, 0.946], DM −5.60 |
| t+30 | 0.616 [0.554, 0.684], DM −8.34 | 0.877 [0.815, 0.927], DM −4.70 |

**TFT alone vs seasonal-naive:**
- Locked test: significant only at t+7 (p = 0.003). At t+14 and t+30, p = 0.063.
- All folds: tied at t+7 and t+14 (p = 0.92). 2.4% better at t+30 (p = 0.045).

**Development vs locked folds** (Ensemble vs SN): 0.958 / 0.945 / 0.913 on development folds and 0.938 / 0.922 / 0.877 on locked folds. There is no sign that the selection overfitted.

### 10.4 v4 → v4.1

| Test set | Horizon | MAE (m) | vs no-change | vs seasonal | Coverage (%) | Width (m) |
|---|---|---|---|---|---|---|
| Locked 5–6 | t+7 | 0.323 → **0.314** | 0.865 → **0.839** | 0.967 → **0.938** | 80.6 → 81.1 | 1.15 → **1.10** |
| | t+14 | 0.419 → **0.406** | 0.788 → **0.763** | 0.952 → **0.922** | 81.5 → 80.1 | 1.60 → **1.52** |
| | t+30 | 0.593 → **0.550** | 0.664 → **0.616** | 0.945 → **0.877** | 81.2 → 80.0 | 2.45 → **2.22** |
| All 6 folds | t+7 | 0.433 → **0.429** | 0.911 → **0.903** | 0.975 → **0.967** | 80.8 → 79.9 | 1.63 → **1.52** |
| | t+14 | 0.610 → **0.595** | 0.850 → **0.830** | 0.972 → **0.949** | 80.3 → 78.4 | 2.74 → **2.50** |
| | t+30 | 0.907 → **0.861** | 0.750 → **0.712** | 0.958 → **0.909** | 77.7 → 78.1 | 4.15 → **3.62** |

- Every error and ratio fell, and every interval narrowed, while coverage stayed near 80%.
- The gain grows with horizon. At t+30, MAE fell 7% on the locked test.
- On the locked test, all 21 districts now beat no-change at t+30; in v4, 20 of 21 did.

### 10.5 Baseline Ladder (v4.1 run, identical windows)

| Model | All folds: vs no-change | All folds: vs SN | Locked: vs no-change | Locked: vs SN | Coverage, locked (%) |
|---|---|---|---|---|---|
| Persistence | 1.000 | 1.070 / 1.143 / 1.277 | 1.000 | 1.118 / 1.209 / 1.423 | – |
| Seasonal-naive | 0.934 / 0.875 / 0.783 | 1.000 | 0.894 / 0.827 / 0.703 | 1.000 | 86 / 88 / 91 |
| Ridge | 1.120 / 1.147 / 0.962 | 1.199 / 1.311 / 1.229 | 0.983 / 0.946 / 0.790 | 1.099 / 1.144 / 1.124 | 85 / 86 / 87 |
| LightGBM | 0.903 / 0.832 / **0.703** | 0.967 / 0.951 / **0.897** | **0.833** / 0.770 / **0.612** | **0.932** / 0.931 / **0.871** | 75 / 72 / 67 |
| TFT (raw network) | 0.973 / 0.901 / 0.763 | 1.041 / 1.030 / 0.974 | 0.915 / 0.830 / 0.685 | 1.023 / 1.003 / 0.974 | 79 / 79 / 77 |
| TFT (shrunk toward SN) | 0.935 / 0.876 / 0.764 | 1.000 / 1.002 / 0.976 | 0.872 / 0.803 / 0.675 | 0.976 / 0.971 / 0.960 | 80 / 78 / 77 (before calibration) |
| **Ensemble** | **0.903 / 0.830** / 0.712 | **0.967 / 0.949** / 0.909 | 0.839 / **0.763** / 0.616 | 0.938 / **0.922** / 0.877 | **81 / 80 / 80** |

The persistence-vs-SN ratios are computed from the ladder MAEs: persistence MAE ÷ SN MAE.

### 10.6 By Season (Ensemble, all folds)

| Season (folds) | vs no-change | vs seasonal | Coverage (%) |
|---|---|---|---|
| Monsoon (1, 4) | 0.953 / 0.882 / 0.751 | 0.979 / 0.960 / 0.902 | 81 / 80 / 79 |
| Post-monsoon (2, 5) | 0.857 / 0.777 / 0.659 | 0.936 / 0.912 / 0.850 | 79 / 76 / 75 |
| Dry (3, 6) | 0.871 / 0.794 / 0.691 | 0.969 / 0.954 / 0.955 | 79 / 79 / 80 |

| Fold | vs no-change | MAE at t+30 (m) | Coverage at t+30 (%) |
|---|---|---|---|
| 1 Monsoon 2024 | 0.973 / 0.953 / 0.924 | 1.77 | 68 |
| 2 Post-monsoon 2024 | 0.888 / 0.804 / 0.704 | 0.68 | 67 |
| 3 Dry 2025 | 0.886 / 0.808 / 0.746 | 0.82 | 81 |
| 4 Monsoon 2025 | 0.931 / 0.802 / 0.572 | 0.89 | 87 |
| 5 Post-monsoon 2025 (locked) | 0.821 / 0.745 / 0.600 | 0.48 | 83 |
| 6 Dry 2026 (locked) | 0.851 / 0.774 / 0.625 | 0.60 | 78 |

- **The monsoon is the hardest season at t+7 and t+14**, because future rain is unknown. Monsoon 2024 (fold 1) is the weakest fold.
- At t+30 the model's gain over SN is actually larger in the monsoon (0.902) than in the dry season (0.955).
- Coverage varies by fold: 67–87% at t+30.

### 10.7 By District (MASE vs no-change, all 6 folds)

| District | Tier | Ensemble t+7 | Ensemble t+14 | Ensemble t+30 | TFT t+7 | TFT t+14 | TFT t+30 |
|---|---|---|---|---|---|---|---|
| Akola | B | 0.950 | 0.879 | 0.831 | 0.999 | 0.989 | 0.949 |
| Amravati | A | 0.785 | 0.649 | 0.546 | 0.807 | 0.666 | 0.525 |
| Aurangabad | C | 0.968 | 0.969 | 0.867 | 0.983 | 0.968 | 0.882 |
| Bhandara | C | 0.908 | 0.841 | 0.808 | 0.940 | 0.908 | 0.881 |
| Chandrapur | A | 0.822 | 0.706 | 0.652 | 0.858 | 0.764 | 0.679 |
| Dhule | C | 0.981 | 0.912 | 0.864 | 1.087 | 1.050 | 0.921 |
| Gadchiroli | C | 0.882 | 0.788 | 0.630 | 0.939 | 0.852 | 0.684 |
| Gondia | A | 0.897 | 0.793 | 0.654 | 0.927 | 0.853 | 0.753 |
| Jalgaon | B | 0.885 | 0.819 | 0.694 | 0.892 | 0.866 | 0.708 |
| Jalna | B | 0.916 | 0.848 | 0.715 | 0.914 | 0.878 | 0.805 |
| Kolhapur | C | 0.801 | 0.798 | 0.744 | 0.848 | 0.813 | 0.750 |
| Nagpur | A | 0.785 | 0.664 | **0.527** | 0.823 | 0.732 | 0.569 |
| Nandurbar | B | 0.843 | 0.740 | 0.541 | 0.863 | 0.761 | 0.603 |
| Nashik | B | 0.998 | 0.990 | 0.883 | 1.029 | 1.094 | 1.161 |
| Parbhani | B | 0.943 | 0.956 | 0.883 | 0.954 | 0.949 | 0.906 |
| **Pune** | C | 1.000 | **1.050** | **1.128** | 0.995 | 0.955 | 0.871 |
| Ratnagiri | A | 0.934 | 0.892 | 0.742 | 0.945 | 0.899 | 0.816 |
| Solapur | A | 0.950 | 0.912 | 0.834 | 0.978 | 0.945 | 0.910 |
| Wardha | A | 0.842 | 0.710 | 0.592 | 0.851 | 0.744 | 0.637 |
| Washim | A | 0.889 | 0.774 | 0.652 | 0.875 | 0.815 | 0.707 |
| Yavatmal | A | 0.832 | 0.701 | 0.576 | 0.856 | 0.728 | 0.581 |
| *Nanded (single well)* | C | 0.695 | 0.640 | 0.556 | 0.676 | 0.638 | 0.588 |
| *Raigad (single well)* | B | 0.586 | 0.565 | 0.649 | 0.639 | 0.620 | 0.551 |
| *Thane (single well)* | C | 0.700 | 0.687 | 0.662 | 0.806 | 0.703 | 0.598 |
| Beed | – | no usable data (stuck/dry sensor) | | | | | |
| Latur | – | no usable data (tier D) | | | | | |

**The gap between districts:**
- **Best at t+30:** Nagpur (0.527), Nandurbar, Amravati, Yavatmal, Wardha.
- **Weakest at t+30:** Pune, Nashik, Parbhani, Aurangabad, Dhule.
- **Pune** is the only headline district where the Ensemble loses to no-change over all folds.
  - LightGBM is poor there (MASE 1.30 at t+30), while the TFT alone does better (0.871 at t+30).
  - v4.2's whole-blend guard was rejected.
  - v4.3's LightGBM-only guard was adopted on the development folds and improved Pune at 14 days over all folds (1.050 → 1.004). But it made Pune worse on the locked folds (t+30: 0.77 → 0.92), where LightGBM happened to forecast Pune well.
  - **Pune remains the weakest district** (all folds: 1.012 / 1.004 / 1.121); §9.9.
- The worst-5 minus best-5 MASE gap **widened** from v4 to v4.1, because the best districts improved the most:
  - all folds: 0.144 / 0.236 / 0.343 → 0.175 / 0.290 / 0.368;
  - locked test: 0.237 / 0.308 / 0.390 → 0.264 / 0.336 / 0.457.
- The weakest five are mostly tier B/C districts with thin or gappy records, consistent with §4.3.

### 10.8 What the Ensemble Relies On

Average stacking weights over districts and folds:

| Horizon | No change | Seasonal-naive | TFT | LightGBM |
|---|---|---|---|---|
| t+7 | 0.24 | 0.03 | 0.29 | 0.45 |
| t+14 | 0.13 | 0.03 | 0.30 | 0.54 |
| t+30 | 0.12 | 0.04 | 0.19 | 0.64 |

- **In v4.1, LightGBM with seasonal features is about as accurate as the Ensemble on its own** (§10.5).
- Its intervals are badly too narrow: 61–75% coverage.
- The Ensemble keeps LightGBM's accuracy and gets honest intervals from the TFT's spread.
- So most of the v4.1 gain in point accuracy came from the improved LightGBM. The TFT contributes 20–30% of the median and the uncertainty estimate.

### 10.9 Prediction Intervals — a Finding From the v4.1 Run

Computed from `FINAL_s1_tab2/predictions_daily.csv` for the 21 headline districts. Each cell gives coverage at t+7 / t+14 / t+30, then the interval score.

| Model and rows | Development folds 1–4 | Locked 5–6 |
|---|---|---|
| TFT, before calibration | 80 / 77 / 71% · score 2.82 / 3.98 / 5.72 | 80 / 78 / 77% · score 1.78 / 2.21 / 2.89 |
| TFT, calibrated | 72 / 68 / 64% · score 3.07 / 4.67 / 6.87 | 74 / 70 / 66% · score 2.01 / 2.49 / 3.56 |
| Ensemble, before calibration | 81 / 78 / 74% · score 2.77 / 3.83 / 5.44 | 82 / 81 / 80% · score 1.73 / 2.12 / 2.74 |
| Ensemble, calibrated | 79 / 78 / 77% · score 2.74 / 4.32 / 6.55 | 81 / 80 / 80% · score 1.68 / 2.06 / 3.06 |

**For the TFT alone, the calibration step (pooled conformal with ACI) makes the intervals worse in every fold and at every horizon.**
- It shifts and mostly narrows them.
- It raises misses below q10 from 8–9% to 14–19% on the development folds.
- It costs 6–12 points of coverage.

**The low "TFT coverage" (65–74%) reported since v4 is therefore caused by this step, not by the network.** The network's own quantiles cover 77–80% on the locked test.

**For the Ensemble, the step helps coverage at t+30 on the development folds (74 → 77%), but at the cost of much wider intervals.**

The ACI-speed sweep shows coverage falling as adaptation speeds up, for *both* models. That points at the adaptive update, though the cause is not yet confirmed. Possible factors:
- it receives about 10 highly correlated daily outcomes per issue date;
- its α is clipped at 0.01;
- the calibration-block pool may under-represent test-period volatility.

**Resolved in v4.2** (§9.8, §10.12). The diagnosis was confirmed: the ACI update was fed about 10 correlated outcomes per issue date for the TFT. Feeding it one restored 4–9 points of coverage, but no calibration variant beat the network's own intervals. So:
- the TFT is now reported with its own intervals (locked coverage 77–80%);
- the Ensemble is reported with its own intervals at t+7 / t+14, and with static conformal at t+30.

**Remaining bias:** misses are more often below q10 than above q90 (Ensemble, locked t+30, v4.2: 16.8% vs 4.1%; v4.1: 14.8% vs 5.2%). The actual level more often ends up *deeper* than forecast, so declines are still under-forecast at long horizons.

### 10.10 Interpretability — What the TFT Weights Most

Variable-selection weights from the fold-6 network (`simple/tft_vsn_importance.csv`):

| Input type | Top inputs (weight) |
|---|---|
| Past (47 inputs) | gwl_max (0.091), gwl_lag_30 (0.080), **sm_l2 soil moisture 7–28 cm** (0.063), temp_max (0.053), **cumulative rainfall departure** (0.052), temp_min (0.034), sm_l1 (0.033), **NDVI** (0.031), antecedent precipitation (fast) (0.031), NDMI (0.028), 90-day rainfall (0.028), gwl_roll_std_30 (0.026) |
| Known future (8) | month_cos (0.465), season_id (0.166), is_monsoon (0.081), **sn_prior** (0.080), month_sin (0.075), horizon_frac (0.071) |
| Static | district + aquifer embedding (0.194), sand_10cm (0.177), annual_extraction_bcm (0.116), clay_0cm (0.107) |

These weights are physically sensible:
- groundwater history and seasonal timing come first;
- then soil moisture, long-window rainfall departure, temperature (an evaporative-demand proxy) and vegetation;
- among the static inputs, soil texture and extraction pressure.

They show what the network uses. They are not proof of physical causation.

### 10.11 Figures

All of these are in `04_results/tft_v4_2/simple/`, which shows the final, v4.2-reported intervals. v4.1's versions are in `04_results/tft_v4_1/simple/`; the forecasts are identical. The `tft/` folder has the same files for the TFT alone.

| Figure | File |
|---|---|
| Per-district MASE at t+7 / t+14 / t+30 | `ensemble/ensemble_per_district_mase_t7.png` (and `_t14`, `_t30`) |
| Per-district MAPE | `ensemble/ensemble_per_district_mape_t30.png` |
| Coverage heat map (district × horizon) | `ensemble/ensemble_coverage_heatmap.png`; for the TFT alone, `tft/tft_coverage_heatmap.png` |
| Forecast vs actual, all districts | `ensemble/ensemble_forecast_grid_t30.png` |
| One district, three horizons, with dates | `ensemble/quantile/quantile_nagpur.png` (best), `quantile_pune.png` (weakest), any district |
| Variable importance | `tft_vsn_importance.png` |
| Training curves | `tft_training_curves.png` |
| Before: the v1 run | `04_results/tft_prod_v1/per_district_mape_t30.png`, `04_results/tft_prod_v1/tft_coverage_heatmap.png` |

### 10.12 v4.1 → v4.2: the Interval Fix

Same forecasts (bit-for-bit), different reported intervals. 21 headline districts, t+7 / t+14 / t+30. Interval score = width + 10 × size of each miss; lower is better.

| Test set | Model | Interval reported | Coverage v4.1 → v4.2 (%) | Interval score v4.1 → v4.2 (m) | Change |
|---|---|---|---|---|---|
| locked 5–6 | **TFT** | raw / raw / raw | 74.2 / 70.0 / 65.7 → **80.1 / 78.5 / 77.3** | 2.010 / 2.490 / 3.557 → **1.778 / 2.209 / 2.888** | **−11.5 / −11.3 / −18.8%** |
| locked 5–6 | Ensemble | raw / raw / calibrated | 81.1 / 80.1 / 80.0 → 81.6 / 80.6 / 79.1 | 1.676 / 2.057 / 3.062 → 1.731 / 2.122 / 3.109 | +3.3 / +3.2 / +1.5% |
| all 6 folds | **TFT** | raw / raw / raw | 73.0 / 68.6 / 64.3 → **79.7 / 77.2 / 73.0** | 2.740 / 3.984 / 5.767 → **2.491 / 3.421 / 4.775** | **−9.1 / −14.1 / −17.2%** |
| all 6 folds | Ensemble | raw / raw / calibrated | 79.9 / 78.4 / 78.1 → 80.9 / 79.1 / 77.1 | 2.403 / 3.606 / 5.387 → 2.443 / 3.293 / 5.211 | +1.7 / −8.7 / −3.3% |

**Evenness across districts** (share of headline districts with coverage 65–90%):

| | t+7 | t+14 | t+30 |
|---|---|---|---|
| Ensemble, all folds | 100 → 86% | 100 → 81% | 95 → 86% |
| Ensemble, locked | 100 → 57% | 90 → 52% | 76 → 57% |
| TFT, all folds | 100 → 86% | 76 → 76% | 52 → 81% |
| TFT, locked | 100 → 57% | 81 → 48% | 57 → 57% |

**Coverage per fold at t+30** (folds 1–6):

| Model | v4.1 | v4.2 |
|---|---|---|
| TFT | 53 / 62 / 66 / 72 / 65 / 66 | 57 / 78 / 75 / 72 / 79 / 76 |
| Ensemble | 68 / 68 / 81 / 87 / 83 / 78 | 65 / 62 / 84 / 88 / 84 / 76 |

**Reading:**
- **The TFT's intervals are fixed.** They are better on every test set and horizon.
- **The Ensemble's intervals changed little on average, but are less even across districts.**
  - The conformal step had re-sized each district's interval. The raw intervals are honest overall but not per district.
  - The locked-test shares rest on about 300 forecasts per district and horizon, so they are noisy.
- **The misses are still seasonal** (Ensemble, all folds, t+30):
  - the monsoon misses above q90 in 15.5% of cases: rises are under-forecast. The TFT alone misses above in 31%;
  - the post-monsoon misses below q10 in 23.4%: declines are under-forecast.
- **A post-hoc observation, not used for any choice.** On the locked test, the raw Ensemble interval at t+30 would have scored better than the calibrated one rule D chose (2.74 vs 3.11). The rule was fixed beforehand on the development folds, where raw t+30 covered only 73.7%, so the choice stands.
- **Very wide calibrated intervals** occur in a few runs of issue dates, in both versions.
  - The worst reach 90–108 m: single-well Thane, and Dhule in fold 1.
  - 332 Ensemble t+30 rows are wider than 20 m (Dhule, Nandurbar, Thane), against a median width of 2.35 m.
  - These enter the reported Ensemble t+30 interval for those districts (Appendix A #24).

### 10.13 v4.2 → v4.3: Even Intervals, the Width Cap and the Pune Guard

Same networks (bit-for-bit). The Ensemble medians differ only in Pune and Raigad. 21 headline districts, t+7 / t+14 / t+30.

| Test set | Ensemble | v4.2 | v4.3 |
|---|---|---|---|
| locked 5–6 | error ÷ no-change | 0.839 / 0.763 / 0.616 | 0.841 / 0.766 / 0.620 |
| locked 5–6 | coverage (%) | 81.6 / 80.6 / 79.1 | 80.1 / 80.3 / 78.2 |
| locked 5–6 | interval score (m) | 1.731 / 2.122 / 3.109 | **1.681** / 2.123 / **3.086** |
| all 6 folds | error ÷ no-change | 0.903 / 0.830 / 0.712 | 0.903 / 0.829 / 0.712 |
| all 6 folds | coverage (%) | 80.9 / 79.1 / 77.1 | 77.4 / 79.2 / 77.0 |
| all 6 folds | interval score (m) | 2.443 / 3.293 / 5.211 | 2.425 / 3.287 / **5.022** |

| Districts with coverage 65–90% (% of 21) | v4.2 all folds | v4.3 all folds | v4.2 locked | v4.3 locked |
|---|---|---|---|---|
| Ensemble t+7 | 86 | **100** | 57 | **76** |
| Ensemble t+14 | 81 | 81 | 52 | 57 |
| Ensemble t+30 | 86 | 86 | 57 | 52 |

| | v4.2 | v4.3 |
|---|---|---|
| Reported intervals wider than 20 m (all districts) | 332 | 105 (Dhule 59, Thane 43, Nandurbar 3) |
| Widest Ensemble t+30 interval | 94 m | 34 m |

**Pune, Ensemble error ÷ no-change, by fold:**

| Version | Horizon | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 |
|---|---|---|---|---|---|---|---|
| v4.2 | t+14 | 1.460 | 0.690 | 1.350 | 0.898 | 0.959 | 0.717 |
| v4.3 | t+14 | 1.460 | 0.690 | **1.004** | 0.938 | 1.009 | 0.857 |
| v4.2 | t+30 | 0.852 | 0.798 | 2.220 | 1.028 | 0.878 | 0.722 |
| v4.3 | t+30 | 0.852 | 0.798 | 2.220 | **0.683** | 0.968 | 0.902 |

**Reading:**
- **The interval changes worked as designed.** 7-day evenness reached 100% of districts over all folds, the width cap removed two-thirds of the extreme intervals, and interval scores improved slightly.
- **The coverage trade-off.** The Ensemble's 7-day coverage over all folds fell to 77.4%. That is still within 75–85%, and the calibrated interval is honest *in each district* instead of only on average.
- **The Pune guard did not generalise.**
  - It helped in fold 3 (t+14) and fold 4 (t+30), both development folds.
  - It hurt in folds 5–6, where LightGBM did well for Pune.
  - A district's past component errors are a weak guide to the next season.
- **The decline bias remains** at t+30 on the locked test: 17.4% of outcomes fall below q10 and 4.4% above q90.

**The operational forecast** was issued on 4 Jun 2026 for 23 districts (Nanded skipped).

| Horizon | rising | steady | uncertain | falling |
|---|---|---|---|---|
| t+7 | 1 | 14 | 8 | 0 |
| t+14 | 2 | 6 | 15 | 0 |
| t+30 | 15 | 1 | 7 | 0 |

- **This is the expected monsoon-onset rise.** The largest median rises by 4 July are Thane +6.8 m, Nandurbar +5.0 m, Dhule +4.4 m and Kolhapur +2.6 m. Thane is single-well, with a 3–13 m range.
- **These target dates have passed, but there are no readings yet to check them,** because CGWB has published nothing after 4 Jun 2026. The forecasts will be scored when it does.

---

## 11. Discussion

### 11.1 Main Lessons

1. **A baseline-first evaluation changed the project more than any architecture choice.**
   - Without persistence and seasonal-naive in every table, a model three times worse than doing nothing looked acceptable (v1).
   - Adding them exposed, in turn: the v1 failure; v3's TFT losing to the seasonal forecast; and, in v4.1, that LightGBM carries most of the accuracy.
2. **The data was the main bottleneck.**
   - The largest improvements came from fixing the groundwater series: station composition, then placeholder and stuck readings, then aggregation.
   - Other data fixes mattered too: the calendar and windowing bug, fabricated ET and soil moisture, and Sentinel leakage.
   - Model capacity was *reduced* 50-fold along the way (11.6M → 222k parameters).
3. **How the target is framed matters.** Predicting the change, then only the residual on the seasonal forecast, means a model output of 0 equals a strong baseline. Anything the network learns is gain.
4. **Honest evaluation needs explicit rules:**
   - a locked test;
   - a selection rule fixed in advance;
   - identical forecast windows for every model;
   - out-of-sample calibration;
   - significance tests that resample whole districts.

   Several "results" in earlier versions turned out to be artefacts: v2's 80% coverage, v1's median RMSE and v2's parameter count.
5. **Per-district skill is limited by data completeness.**
   - Tier-A districts are mostly the best: 7 of 9 have a t+30 MASE between 0.53 and 0.65.
   - Thin or gappy districts are the weakest.
6. **The simplest honest interval can be the best one.**
   - The TFT's own quantiles, trained with the pinball loss, were already close to 80% coverage.
   - Every conformal variant tried in v4.2 scored worse on average on the development folds.
   - The adaptive step had been fed about 10 correlated outcomes per issue date, which made it over-react.
7. **Pooled coverage hides unevenness.** "80% overall" can mean districts at 50% and at 95%. Per-district coverage spread should be reported next to the pooled number.
8. **A rule fixed in advance can reject a fix that works where it was aimed.**
   - The v4.2 stack guard repaired Pune but harmed 22 other district-horizons, so the rule rejected it.
   - Without the rule, a Pune-only success could have been reported.
9. **A rule fixed in advance can also adopt a fix that does not generalise, and that must be reported too.**
   - v4.3's narrower guard passed its development-fold rule but made Pune worse on the locked folds.
   - Past per-district component errors are a weak guide to the next season, and the locked test is what shows it.

### 11.2 Is the Transformer Worth It?

Answered honestly:
- **On its own,** the TFT beats "no change" clearly and significantly. Against the seasonal forecast it is only 2–4% better on the locked test, and level with it over all folds.
- **In the Ensemble,** it provides the uncertainty estimate and 20–30% of the median. A LightGBM with good seasonal features is a strong point forecaster on this data.
- **Why the TFT is still useful:**
  - the ensemble's honest 80% intervals come from the TFT's spread;
  - the variable-selection weights give interpretable importance;
  - it is the only component that could use future-known weather inputs, such as rainfall forecasts, in a structured way. That is the most promising next step.

### 11.3 Limitations

1. Only three monsoons. The locked test (folds 5–6) contains no monsoon, so monsoon skill comes from folds 1 and 4, which were used for tuning.
2. Folds 5–6 were seen once before, in the v4.0 run. v4.1 and v4.2 were selected on folds 1–4 only.
3. No input knows future rain. This is the main remaining error source at t+30 in the monsoon.
4. Five weak districts remain (Pune, Nashik, Parbhani, Aurangabad, Dhule), and the gap between best and worst widened in v4.1. Pune still loses to no-change over all folds. Neither guard fixed it: v4.2's was rejected, and v4.3's did not hold on the locked folds.
5. The intervals are honest on average: on the locked test the Ensemble covers 79–82% and the TFT 77–80%. But they are uneven:
   - across districts, only 52–57% of districts are within 65–90% on the locked test;
   - by fold, the Ensemble at t+30 ranges from 62% to 88%;
   - they miss low more often than high at long horizons (§10.12).
6. 12 of 36 districts cannot be forecast for lack of data. Single-well districts are reported but are not true district signals.
7. District-level only. No well-level or sub-district forecasts.
8. A few calibrated intervals are still wide. After v4.3's cap the widest is 34 m (it was 94 m), and 105 rows are wider than 20 m (Dhule, Thane, Nandurbar).
9. Forecasts can only be issued from a groundwater reading. The latest CGWB reading is 4 Jun 2026, so the latest forecast covers dates that have already passed (§13).

---

## 12. Conclusion and Future Work

### 12.1 Summary

JalVaani now produces 7, 14 and 30-day groundwater forecasts with 80% prediction intervals for 24 Maharashtra districts.

**On the locked test (October 2025 – June 2026), the final Ensemble:**
- **beats "no change" by 16–38%** and **the seasonal forecast by 6–12%** (0.841 / 0.766 / 0.620 and 0.940 / 0.926 / 0.882; v4.3);
- has errors of 0.31 / 0.41 / 0.55 m;
- has intervals that contain the true level 78–80% of the time (the TFT alone: 77–80%), and its 7-day intervals are honest in every district over all folds;
- beats "no change" in all 21 headline districts at every horizon.

**Getting there involved:**
- rebuilding the groundwater target three times;
- replacing two data sources (rainfall with CHIRPS; ET and soil moisture with ERA5-Land);
- fixing six evaluation mistakes (§6.7) and a satellite-data leak;
- shrinking the network 50-fold;
- changing the prediction target twice;
- adding a seasonal baseline, a stacked ensemble and out-of-sample calibration;
- in v4.2, testing the calibration itself, which showed that the network's own intervals are the better choice in most cases.

The pipeline is reproducible: v4.2 and v4.3 re-trained every model on Colab and matched v4.1 bit-for-bit.
It also issues the operational forecast that the dashboards show (§13).

The first full TFT run was worse than doing nothing. The current system is significantly better than both naive benchmarks, and every number is reported against them.

### 12.2 Future Work (in priority order)

v4.2 completed the interval fix (priority 1 of the previous list). Its Pune guard was rejected, so Pune moves into the optional v4.3.

| Priority | Item | Why / expected effect |
|---|---|---|
| 1 | **Re-run when CGWB publishes readings after 2026-06-04**: collectors → preprocessing → the v4.3 notebook → `dev\scripts\build_data.ps1` | It gives current forecasts, scores the June–July 2026 forecast, and adds a monsoon-2026 locked fold. v4.3 itself is done (§9.9); Pune is still open |
| 2 | **Rainfall forecasts as known-future inputs** (IMD or ECMWF extended range) | Targets the largest remaining error source (t+30, monsoon) |
| 3 | **Re-run as data arrives:** new CGWB readings after 2026-06-04 (a monsoon-2026 test fold); final CHIRPS for September 2026 | A locked monsoon test; removes the provisional rainfall |
| 4 | **Reservoir storage** (`cwc_reservoir.py`, needs a data.gov.in key) | The main uncollected physical driver, especially for Marathwada |
| 5 | More stations for tier-C districts; any usable data for the 12 missing districts | Directly addresses the district gap |
| 6 | Cluster districts by similar dynamics and train per cluster | The cheaper alternative to a GAT suggested by the 2026 HESS study |
| 7 | TFT + GAT spatial layer | Only after 5 and 6, and only if the data supports it |
| 8 | **Deployment:** the farmer and government dashboards are **built and run locally** (§13). Remaining: hosting, a scheduled data refresh, the Gnani voice integration, user testing | The project's operational goal |

### 12.3 The Improvement Loop We Follow

Every version from v2 on came out of the same cycle (the full procedure is in §6.10):

```
results -> find a specific weakness -> diagnose it with a check or probe -> change it behind a switch
        -> screen on development folds 1-4 (3 seeds) -> pick by a rule fixed in advance
        -> final run (10 seeds, all folds) -> report locked folds 5-6 + significance -> document -> results
```

| Version | Weakness found | Diagnosis that found it | Change made |
|---|---|---|---|
| v2 | v1 worse than "no change" | persistence comparison | data panel and anomaly fix, change target, smaller network |
| v3 | v2 tied at t+7; 7 failing districts | parameter count, covariate reality table, ridge probe, conformal audit | ERA5-Land, causal Sentinel, vectorised network, per-horizon scale, shrinkage, honest calibration |
| v4 | v3 worse than the seasonal forecast; false jumps in the target | seasonal-naive baseline; target audit | target rebuild, issue-date windows, residual-on-SN target, pooled calibration, ensemble |
| v4.1 | TFT only tied SN; LightGBM could not see the season | stacking weights, feature review | shrink toward SN, LightGBM v2, stride-1 sampling |
| v4.2 | calibration hurts the TFT's intervals; Pune | calibrated vs uncalibrated comparison on development folds; reading the ACI code; Pune's stack weights and component errors | raw or calibrated interval per model and horizon (kept: TFT intervals fixed); static conformal; per-district stack guard (rejected) (§9.8) |
| v4.3 | Ensemble intervals uneven across districts; 90–108 m intervals; Pune; no forecast from the latest reading | per-district coverage spread; the v4.2 guard's side effects | evenness-aware interval rule (kept), width cap (kept), LightGBM-only guard (adopted, mixed), operational forecast (§9.9) |

### 12.4 Where the Project Stands, and What We Do Next

**Where we are (1 Nov 2026).**
- **The forecasting model is complete and frozen: v4.3.**
  - On the locked test it beats "no change" by 16–38% and the seasonal forecast by 6–12%, with p = 0.003.
  - Its 80% intervals cover 78–80%; the TFT alone covers 77–80%.
  - Its 7-day intervals are honest in every district over all folds.
- **It runs end to end.** Collected data → model on Colab → `dashboard_export` → farmer and government dashboards in three languages.
- **The whole pipeline is reproducible:** two Colab re-runs matched bit-for-bit.

**What v4.2 and v4.3 found** (§9.8–§9.9, §10.12–§10.13):
1. **The network's own intervals beat every calibration variant for the TFT.** For the Ensemble, the evenness-aware rule and the width cap made the intervals honest district by district.
2. **Two attempts to fix Pune failed in different ways.** v4.2's guard harmed other districts. v4.3's guard did not hold on the locked folds. Pune's problem is real, but per-district post-processing cannot predict it.
3. **Post-processing has reached its limit.** The remaining errors need information the model does not have.

**What we do next, in order:**
1. **Write-up and viva** on the frozen model: this report, `docs/history/TFT_VERSION_HISTORY.md`, `04_results/tft_v4_3/` with `docs/RESULTS_GUIDE.md`, and the dashboards.
2. **Re-run when CGWB publishes readings after 2026-06-04.** This gives:
   - current forecasts on the dashboards;
   - the first check of the June–July 2026 forecast;
   - a locked monsoon-2026 fold.

   It uses the same notebook: collectors → preprocessing → v4.3 notebook → `dev\scripts\build_data.ps1`.
3. **Dashboards:**
   - a native-speaker review of the Marathi, Hindi and Gujarati text;
   - choosing a Gnani Marathi voice;
   - short usability sessions with farmers and officials;
   - then hosting, with a scheduled data refresh.
4. **Possible research extensions:**
   - rainfall forecasts as an input (decided against for this project; future work);
   - a model that estimates *today's* level from rain since the last reading;
   - reservoir storage;
   - district clustering, then a GAT.

---

## 13. Dashboards and Decision Support

The project's operational goal is to turn the forecasts into decisions:
- a **farmer** view for one district;
- a **government** view for the state.

Both are built in `dev/` and run locally. They work in **English, Marathi, Hindi and Gujarati** (more Indian languages are planned), have light and dark themes, and include a site-guide chatbot. Their documentation is in `dev/README.md` and `dev/docs/`.

### 13.1 Who it is for, and what each view shows

| View | Users | Shows (all real data; see `dev/docs/DATA_CONTRACT.md`) |
|---|---|---|
| **Farmer** (phone first, plain words) | farmers, extension workers | **Three cards:**<br>1. *water now*: depth below ground, the 30-day and 1-year change, compared with the same weeks in 2023–25;<br>2. *next 7 / 14 / 30 days*: expected depth with a likely range and a reliability badge;<br>3. *rain and soil*: rain since 1 June vs the 2023–25 average, soil moisture.<br>**Then:** rule-based water-use tips, and a details chart (measured level + forecast fan) |
| **Government** | district and state officials, GSDA | **Overview:**<br>- headline counts (districts, wells, semi-critical extraction, falling forecasts, districts without data);<br>- a map with five layers (level vs usual, 30-day forecast change, stage of extraction, monitoring wells, reliability);<br>- a priority table, ranked by transparent flags with reasons, with CSV download;<br>- a "what we have / what we need" panel.<br>**District page:** measured level with past forecasts and ranges, accuracy, extraction history and use split, wells used |

### 13.2 How it is built
- **Batch model, file-based site.** The model runs on Colab (the v4.3 notebook) and writes `dashboard_export/`. A build script turns that, plus the project's data, into small JSON files, which a FastAPI backend serves to a React frontend.
- **No model server is needed.** Forecasts change only when CGWB publishes new readings.
- **Hosting later:** one small server, with the model still a batch job.
- **No mock data, and the model cannot be affected:**
  - the build only reads `data/` and `04_results/`, and writes only inside `dev/`;
  - a test checks that the project files are unchanged after a build;
  - where data is missing, the site says so. Examples:
    - groundwater readings end on 4 Jun 2026;
    - the latest forecast is "not yet produced" until the v4.3 run;
    - ten districts have no usable well data;
    - September 2026 rainfall is provisional.
- **The latest forecast** comes from v4.3's `forecast_latest`, issued on 4 Jun 2026 for 23 districts (§10.13). It is made exactly like an evaluated fold. A district with too few recent readings gets no forecast, and the site shows the reason (Nanded: readings on 40% of the last 60 days, 70% needed).
- **The site is honest about time.**
  - The last reading is 150 days old, so the farmer view shows a banner saying so.
  - The forecast card says its dates have passed.
  - Tips name the date of the reading they come from ("At the last reading (4 Jun 2026)…"), and give no "next month" advice from an expired forecast.
- **Two ranges from one forecast.** The 80% range answers "how bad could it be", but it is wide (for Akola at 14 days, 6.7–7.9 m). Next to it the site shows a "most likely" range: the same median ± the median size of that district's past errors at that horizon (Akola at 14 days: 7.0–7.6 m). It is not a new model and does not narrow the honest 80% range. Checked the honest way, with margins from the development folds only, it covered 63 / 65 / 68% of the locked-fold outcomes at 7 / 14 / 30 days, a little more than the 50% it aims for.

### 13.3 Human–machine interface
The design follows Nielsen's heuristics, Shneiderman's rules, Norman's principles and WCAG 2.1 AA. `dev/docs/HMI_CHECKLIST.md` maps each one to the screen that applies it. In particular:
- **Status is always visible.** A freshness bar on every page shows the date of every kind of data.
- **Units are real-world.** Depth is in metres below ground, on a reversed axis so that up = water nearer the surface.
- **Uncertainty is shown as words and ranges,** never a bare number: a darker "most likely" band inside a lighter "could be" band, with the numbers printed beside them.
- **Colour is never alone.** Every colour comes with an icon and a word; the palette is colour-blind safe.
- **Disclosure is progressive, in a bento grid.** The page uses the full screen width; tile size follows importance (water now and the forecast first, context tiles smaller); on phones the chart sits behind a button.
- **One visual language.** One icon set, no emoji, skeleton loading in the shape of each page, and short fades that switch off under reduced motion.
- **Recognition over recall.** Language, role and district are remembered.
- **The user stays in control.** The chatbot never navigates by itself; it offers a "Go there" button.
- **Light and dark themes.** Light, dark, or the device's setting. The dark colours, including the map's, are chosen for the dark surface and checked for colour-blind separation, not simply inverted.
- **Built for slow connections.** The farmer page loads about 99 kB of gzipped JavaScript (each extra language is a separate ~7 kB file); the map and charts load only when opened.

### 13.4 Languages, chatbot and voice
- **Languages.** English, Marathi, Hindi and Gujarati, chosen from a menu in the header. Every text is in `dev/frontend/src/i18n/<language>.json`.
  - **Natural wording.** The first Marathi and Hindi drafts read like word-for-word translations. They were rewritten in everyday wording: "अनुमान" rather than "पूर्वानुमान", the language's own sentence order, units in full, and the official CGWB/GSDA category names. Gujarati was written the same way. All three await a native speaker's review (`dev/docs/I18N_REVIEW.md`).
  - **Tests.** A test fails if a key or `{{placeholder}}` is missing, or if text is left in English. Another checks that every chat suggestion chip finds its answer in every language.
  - **District names** appear in every script, with the official new names.
  - **Planned next:** Kannada, Telugu, Tamil, Malayalam, Bengali, Punjabi and Odia.
- **Chatbot (site guide).**
  - It helps users navigate, teaches the site (guided tours, highlighting a card) and explains the numbers on screen.
  - It gives no farming, borewell or legal advice.
  - It works now as an offline three-language guide. With a Groq API key it becomes an LLM assistant under the same rules: answers only from the numbers on screen, with allowlisted actions (`dev/docs/CHATBOT.md`).
- **Voice (Gnani Vachana STT/TTS).** It is implemented with one API key (`dev/docs/SPEECH_PLAN.md`): push-to-talk, a transcript to confirm, spoken replies with captions, and "Listen" on every card. A live Marathi round trip (text → speech → text) returned the sentence word for word, in about 0.9 s each way.
- **The Groq assistant was also tested live.** It answered in Marathi from the on-screen numbers only, and referred a borewell question in Hindi to officials. It rotates between several API keys.

### 13.5 Verification
- **Backend tests (pytest):**
  - API values checked against the source files, including the "most likely" range against the back-test file;
  - the rules;
  - the chat guide in four languages (every suggestion chip) and the action allowlist;
  - speech returning 501 while not configured;
  - "the build never touches project data".
- **Frontend tests (vitest):** translation completeness and placeholders, and the formatting helpers.
- **The production build.**
- **A browser run (Playwright) on desktop and phone widths, in all four languages and in light and dark:**
  - the farmer and government flows;
  - a Marathi district search;
  - the chat guide highlighting a card;
  - a guided tour;
  - keyboard-only Tab order.

  The run produced no console errors.

### 13.6 Limits and next steps
- District level only.
- Groundwater readings end on 4 Jun 2026, until CGWB publishes newer ones.
- The latest forecast (4 Jun 2026) covers dates that have passed. Current forecasts need newer CGWB readings.
- **Still to do:** a native-language review, usability sessions with farmers and officials, the Gnani integration, and deployment with a scheduled data refresh.

---

## Appendix A — Every Bug Found and Fixed

| # | Bug | Found in | Impact | Fixed in |
|---|---|---|---|---|
| 1 | Days without groundwater dropped from the master; windows and lags built on row positions | prototype | Corrupted windows (only 75–92% continuous in gappy districts); "7-day" lags that were 7 rows | July rebuild |
| 2 | No district-name mapping in the soil collector; misspelled Gadchiroli in the Sentinel and CHIRPS collectors | prototype | Missing soil and satellite data for 4 districts | July rebuild |
| 3 | Validation used as the test; dry season only | prototype | Optimistic, monsoon-blind scores | Walk-forward evaluation |
| 4 | Manual-survey wells mixed with telemetric wells | v1 | False jumps of ~94 m; ~60% of apparent error | v2 |
| 5 | Absolute-level target | v1 | Worse than persistence | v2 |
| 6 | 11.6M parameters for ~156 independent windows | v1 | Memorisation by epoch 1–2 | v2 / v3 |
| 7 | No baseline in any metric | Phase 1 – v1 | Failure went undetected | v2 (MASE), v4 (SN) |
| 8 | Anchor taken from an interpolated series | v1 | Future data leaked into the anchor | v2 |
| 9 | Reported parameter count wrong by 6× | v2 | False "56× smaller" claim | v3 |
| 10 | Sentinel composites stamped at month-start and interpolated | v2 | Up to 15 days of future leakage | v3 |
| 11 | ET and soil moisture fabricated after 2025-06-30 | v2 | Folds 4–6 had 25%, 0% and 0% real data | v3 (ERA5-Land) |
| 12 | Conformal fitted on the test set | v2 | Meaningless 80% coverage | v3 |
| 13 | One change scale pooled over all horizons | v2 | Almost no training signal at t+7 | v3 |
| 14 | No floor under a failing district | v2 | Aurangabad MASE 2.84 | v3 (shrinkage) |
| 15 | Windows keyed by the decoder end date | v1–v3 | t+7 targets in the validation block; last 23 days per season never scored | v4 |
| 16 | Rainfall departure used the full-series mean | v3 | Small look-ahead | v4 |
| 17 | +1.000 placeholders; mixed-sign stations dropped whole | v3 target | 12 good wells lost | v4 target |
| 18 | Stuck loggers, dry-well plateaus, a short Hampel window, a wrong jump cap, backward fill | v3 target | 376 false jumps > 2 m; future fills | v4 target |
| 19 | District value = median of whichever wells reported | v3 target | 81–84% of large jumps caused by composition changes | v4 (factor model) |
| 20 | Rainfall zero-filled where CHIRPS was not yet published | Oct extension | False dry days from 2026-06-01 | re-fetched |
| 21 | Cached fold results not tied to the code version | v4 development | Risk of silently reusing stale results | `PIPELINE_VERSION` in cache keys |
| 22 | Conformal width blow-up: non-scored steps counted as misses; α unclipped | v4 development | Exploding intervals | v4 (NaN medians skipped, α clipped, asymmetric) |
| 23 | Calibration step hurts the TFT's own intervals (ACI fed ~10 correlated outcomes per issue date; lopsided α range) | v4.1 analysis | TFT coverage 66–74% instead of 77–80% | **v4.2**: the TFT is reported with its own intervals (77–80% locked). The conformal step still does not beat raw intervals on average |
| 24 | Occasional very wide calibrated intervals (90–108 m) | v4.2 analysis | Thane (single well), Dhule fold 1, Nandurbar | **reduced in v4.3** by the width cap: widest 94 → 34 m, rows > 20 m 332 → 105. The rest is still open (Dhule, Thane, Nandurbar) |

## Appendix B — Repository Map

The project is organised by what each thing is (code per pipeline stage, data, results, documents, archive).
Every folder and the full old-path → new-path table are in `docs/STRUCTURE.md`.

| Folder | What is in it |
|---|---|
| `01_data_collection/` | collectors: `groundwater_data.py`, `chirps_rainfall.py`, `era5_land.py`, `temperature_openmeteo.py`, `sentinel_indices.py`, `soil_type_gee.py` / `soil_type_api.py`, `cgwb_extraction.py`, `cwc_reservoir.py` (not run); extension tools `run_*_incremental.py`, `run_om_precip.py`, `chirps_prelim_fill.py`, `chirps_extend_assemble.py`; `legacy/` (superseded WRIS collectors), `progress/`, `logs/` |
| `02_preprocessing/` | `preprocess_groundwater.py`, `audit_groundwater_target.py`, `preprocess_env_features.py`, `merge_master.py`, `extract_station_geojson.py`; `legacy/` |
| `data/raw/` · `data/clean/` · `data/audit/` | collector outputs · preprocessing outputs · target-audit plots and tables |
| `data/master/` | frozen training sets: `master_dataset_v4target.csv` (current), `master_dataset_v3target.csv` (old target) |
| `03_models/baselines_phase1/` | Phase 1 ARIMA / XGBoost / LSTM notebooks and scripts |
| `03_models/tft/` | `tft_model_v3.py` (network), `tft_v4.py` (pipeline 4.7 = v4.3), `run_v4.py`, `tune_snaive.py`; `notebooks/09_tft_colab_production_v4_3.ipynb` (current); `notebooks/archive/` (production v1–v4.2, prototypes); `legacy/tft_model.py` (v1/v2 network) |
| `dev/` | the farmer and government dashboards (§13): `pipeline/` (data build), `backend/` (FastAPI, chat, speech), `frontend/` (React, en / mr / hi / gu, light / dark), `docs/` |
| `04_results/` | `tft_v4_3/` (**final results**, including `dashboard_export/`; read with `docs/RESULTS_GUIDE.md`), `tft_v4_2/` (v4.2 run), `tft_v4_1/` (the v4.1 run: identical forecasts, v4.1 intervals; its own guide is `RESULTS_GUIDE_as_shipped.md`), `tft_v4_0_dev/`, `tft_prod_v1/`, `phase1_baselines/`, `zips/` |
| `docs/` | this report, `PROJECT_STATUS.md`, `RESULTS_GUIDE.md`, `STRUCTURE.md`, `history/` (`TFT_VERSION_HISTORY.md` with §0 = scores of every version, `V3_PLAN.md`, `TFT_IMPROVEMENT_PLAN.md`) |
| `archive/` | backups and byte-identical duplicates, with `MANIFEST.csv`; not used by the pipeline |

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| Anchor / issue date | the last observed day before a forecast; forecasts are for anchor + h |
| Persistence / no change | forecast = today's level |
| Seasonal-naive (SN) | today's level + the district's typical change for this time of year (past data only) |
| MASE | model error ÷ persistence error; below 1 beats "no change" |
| vs seasonal | model error ÷ SN error; below 1 beats the seasonal forecast |
| Coverage | share of actual values inside the q10–q90 interval (target 80%) |
| Interval (Winkler) score | interval width plus 10 × the size of any miss; lower = sharp and honest |
| Walk-forward fold | a test period forecast by models trained only on earlier data |
| Development / locked folds | folds 1–4 used for choices / folds 5–6 used only to report |
| Data tier A/B/C/D | data quality from station count and completeness, never from results |
| Headline districts | the 21 modelled districts with more than one well |
| Shrinkage (λ) | pulls the TFT's forecast toward a baseline where its out-of-sample record is weak |
| Stacking / Ensemble | weighted blend of no change, SN, TFT and LightGBM, with weights learned on held-out errors |
| Conformal / ACI | interval correction from held-out errors / its adaptive, online version |
| VSN | variable selection network: the TFT's learned input weights |

## Appendix D — References

- Box, G. E. P. and Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control.* Holden-Day.
- Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD.*
- Diebold, F. X. and Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3).
- Funk, C. et al. (2015). The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes (CHIRPS). *Scientific Data*, 2.
- Gibbs, I. and Candès, E. (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS.*
- Gneiting, T. and Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*, 102(477).
- Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8).
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2).
- Hyndman, R. J. and Koehler, A. B. (2006). Another Look at Measures of Forecast Accuracy. *International Journal of Forecasting*, 22(4).
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS.*
- Kratzert, F. et al. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *HESS*, 22.
- Lim, B., Arık, S. Ö., Loeff, N. and Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting*, 37(4).
- Lundberg, S. M. and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS.*
- Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13.
- Romano, Y., Patterson, E. and Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS.*
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS.*
- Veličković, P. et al. (2018). Graph Attention Networks. *ICLR.*
- Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2).
- *Never Train a Deep Learning Model on a Single Well? Revisiting Training Strategies for Groundwater Level Prediction.* HESS 30, 2373 (2026). https://hess.copernicus.org/articles/30/2373/2026/

---

*End of Report — last updated 2026-11-01.*
*Status: Phase 1 (baselines) complete; Phase 2 (TFT) complete through production v4.3 (run 1 Nov 2026, final model); dashboards built in `dev/` and showing the v4.3 forecast (§13).*
*Detailed version history and every number: `docs/history/TFT_VERSION_HISTORY.md`. How to read the results files: `docs/RESULTS_GUIDE.md`.*
