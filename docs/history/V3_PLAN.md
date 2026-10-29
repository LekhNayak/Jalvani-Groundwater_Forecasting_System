# TFT Production v3 — Plan

**Written 2026-08-15.** Based on a read of `09_tft_colab_production_v2.ipynb`, `TFT_VERSION_HISTORY.md`,
the three preprocessing scripts, and **new diagnostics run on the current
`master_dataset.csv` (26 districts × 1,251 days, 2023-01-01 → 2026-06-04)**.

Target: **t+7 / t+14 / t+30 at q10 / q50 / q90**, accurate without overfitting, with an honest
account of which districts the data cannot support.

Everything below marked 📊 is a measured number from this repo's data, not an estimate.
Scripts used: `scratchpad/audit.py`, `audit2.py`, `probe.py`, `probe2.py`.

---

## 0. The one result that reframes everything

I ran an independent **predictability probe**: ridge regression on the same Δ-target, per district,
with lagged Δ (7/14/30/60 d), rainfall accumulations (7/30/90 d), a level anomaly and seasonal
terms — fitted train-only on the same six walk-forward folds.

📊 **Of the 7 districts where TFT v2 loses to persistence at t+30, 6 also lose with ridge:**

| District | ridge MASE t+30 | TFT v2 MASE t+30 | Verdict |
|---|---|---|---|
| Aurangabad | 1.371 | 1.896 | no signal for either model |
| Bhandara | 1.462 | 1.097 | no signal for either model |
| Nashik | 2.043 | 1.142 | no signal for either model |
| Jalgaon | 1.644 | 1.168 | no signal for either model |
| Dhule | 1.347 | 1.057 | no signal for either model |
| Kolhapur | 1.838 | 1.014 | no signal for either model |
| **Ratnagiri** | **0.787** | **1.084** | **signal exists — TFT is missing it** |

Two unrelated model families failing on exactly the same districts is evidence of a **data
limitation**, not a TFT bug. `TFT_VERSION_HISTORY.md` §7 caveat 3 records "no validated explanation" for
these seven — this is the explanation for six of them. Only **Ratnagiri** is a genuine model failure
and deserves debugging effort.

📊 Second result from the same probe — the TFT is genuinely better than a strong linear model.
On the 14 districts where both numbers exist, **TFT beats ridge in 11**. State-wide medians:

| | t+7 | t+14 | t+30 |
|---|---|---|---|
| ridge on Δ | 1.053 | 1.029 | 0.916 |
| TFT v2 | 1.004 | 0.940 | 0.827 |

*(Aggregations are not identical — ridge is median over district×fold, TFT v2 is median over
districts of the fold-pooled value. v3 must compute both in one table, same aggregation. But the
direction is not in doubt.)* This comparison was missing from the v2 report and it is one of the
strongest things you have: it answers "why a transformer and not regression?".

---

## 1. Data-pipeline shortcomings — fix before any runtime change

### D1 🔴 ET and soil moisture are 100% fabricated after 2025-06-30

📊 Last genuinely observed value for both: **2025-06-30**. **28.9%** of all master rows are
day-of-year climatology fill. Per test fold:

| Fold | et/sm observed |
|---|---|
| f1_monsoon2024 | 97% |
| f2_postmonsoon2024 | 100% |
| f3_dry2025 | 96% |
| **f4_monsoon2025** | **25%** |
| **f5_postmonsoon2025** | **0%** |
| **f6_dry2026** | **0%** |

`et_mm`, `soil_moisture` and `net_water_balance` (= rainfall − et) are all in `PAST_CONT`. In half
the walk-forward evaluation they carry **no information beyond `doy_sin`/`doy_cos`**, which the model
already has. `preprocess_env_features.py` is honest about this (it sets `*_observed = 0`), and those
flags are in `FLAG_COLS` — but the flag is constant-0 across an entire fold, so it is a degenerate
feature the model cannot act on.

**Options** (needs your call — see questions at the end):
1. Re-collect from WRIS/NRSC — the 2025-06-30 cut was a publication lag as of the last collection;
   14 months have passed.
2. Swap source: ERA5-Land evaporation + volumetric soil water, or MODIS/VIIRS ET. Near-real-time,
   no publication lag, global coverage. This is the durable fix.
3. Drop ET/SM/`net_water_balance` from the feature set entirely and state why.
4. Restrict the headline to folds 1–3. Wasteful — folds 4–6 have the best groundwater coverage.

Whatever is chosen, **the notebook must print the per-fold covariate-observed table** so this can
never be silently forgotten again.

### D2 🟠 Sentinel monthly → daily interpolation leaks ~15 days of future

`merge_master.py:230` stamps each monthly Sentinel composite at the **1st of the month** and then
linearly interpolates to daily. So NDVI on 15 January is a 50/50 blend of the January and
**February** composites. Every daily Sentinel value in the encoder contains up to half a month of
future information.

**Fix:** stamp at month-**end** (the composite summarises a month that has already happened) and use
step/`ffill` rather than linear interpolation, or lag the whole series by one month. ~5 lines.

### D3 🔴 2023 is a partial year, and unevenly so — this is the root of the district split

📊 State-wide `gwl_observed`: **2023 = 57%**, 2024 = 98%, 2025 = 94%, 2026 = 89%.
Per-district 2023 coverage ranges **0.25 → 0.81**. Longest single unobserved run:

| District | longest gap (days) | 2023 coverage |
|---|---|---|
| Aurangabad | 272 | 0.25 |
| Beed | 271 | 0.25 |
| Kolhapur | 269 | 0.26 |
| Gadchiroli | 265 | 0.27 |
| Dhule | 264 | 0.27 |
| Bhandara | 255 | 0.29 |
| … | | |
| Nagpur | 68 | 0.81 |
| Gondia | 67 | 0.81 |

📊 Consequence in training windows (stride 7, current gates), **fold1**:

| District | train windows |
|---|---|
| Beed, Aurangabad | **20** |
| Dhule, Gadchiroli, Kolhapur | 21 |
| Bhandara | 22 |
| … | |
| Nagpur, Chandrapur, Gondia | **49** |

A 2.4× imbalance — and `build_sampler()` **inverse-weights**, so Beed's 20 windows are sampled
~2.45× more often than Nagpur's 49. The model sees the same 20 windows over and over, in exactly
the districts that already have the least real history.

📊 Spearman correlation with ridge MASE @ t+30 (i.e. with how predictable a district is at all):

| Driver | ρ |
|---|---|
| 2023 observed rate | **−0.54** |
| overall observed rate | **−0.54** |
| n_stations | **−0.51** |
| median 30-day movement | −0.10 (nothing) |

**More wells and a longer complete record make a district predictable. Movement size does not.**
That is the headline data finding.

**Fixes:** (a) cap the sampler at 1.5× or drop it and test both — it is currently amplifying the
problem; (b) report a *training-record-length vs MASE* curve — it is a strong figure and it argues
for continued data collection; (c) consider starting the training span at 2024-01-01 for an ablation
— 2023 contributes few windows and mostly to the districts it corrupts.

### D4 🔴 Latur is not a usable district

📊 44% observed overall, **1 station**. Test-window coverage: f1 97%, f2 37%, f3 48%, **f4 3%,
f5 15%, f6 14%**. It produces **0 test windows in folds 4–6** and **0 validation windows in folds 5–6**.
It is `low_station_flag=1` so it is out of the headline — but it still contributes training windows
and sits in the sampler. **Exclude from training, or gate per-fold on observed rate.**

### D5 🟡 Small train/test leakage in the notebook's own preprocessing

- Cell 5: `df[SOIL_COLS].fillna(df[SOIL_COLS].median())` (same for `SENTINEL_COLS`, `CGWB_COLS`) —
  medians computed over **all** rows, including the test period.
- Cell 6: `df[roll_cols].transform(lambda s: s.fillna(s.median()))` — per-district median over the
  **full** series, used to fill the rolling-window warm-up NaNs.

Both are small in effect (statics + warm-up rows) and both are trivially fixable: move every fill
inside `normalize_for_fold()` and compute it on `date <= train_end` rows only. A reviewer *will*
find these, and the cost of fixing them is ten minutes.

### D6 🟠 Validation is 9 windows per district

📊 With `VAL_MONTHS=2` and `TRAIN_STRIDE=7`, each fold's validation set is **200–234 windows total,
~9 per district**. Early stopping *and* the `ReduceLROnPlateau` schedule both key off that. This is
why best-val epoch swung 3 → 17 across folds and why the validation curves are flat and noisy
(`TFT_VERSION_HISTORY.md` §7 caveat 2 attributes the flat curves to overfitting — part of it is simply a
tiny, noisy validation signal).

Worse: the two months **immediately before** the test window are used for validation and therefore
**never trained on** — the most relevant recent data is discarded in every fold.

### D7 🔴 Conformal calibration is computed on the set it calibrates

Cell 16 takes `all_test_preds[fold == name]`, computes the nonconformity quantile **from those same
rows**, then applies the margin **to those same rows**. "80% after calibration" is a mathematical
identity, not a result — you would get 80% from a random model. The reported interval widths
(2.98 / 3.68 / 4.90 m) are consequently not out-of-sample either.

This is the most serious *methodological* issue in v2 and the one most likely to be challenged.

---

## 2. Per-district data tiers — what to claim and what not to

Built from n_stations, 2023 coverage, per-fold test coverage, and whether *any* model finds signal.

The rule below is now **implemented in `merge_master.py`**, which writes a `data_tier` column into
the master and a scorecard to `data/clean/district_data_quality.csv`. It is deliberately a
statement about **data only** — station count and record completeness — and is never fitted to
model results, so it can be quoted as an independent explanation of them.

```
D  overall observed < 55%
A  >= 6 stations AND >= 65% of 2023 observed
C  <= 4 stations AND <  55% of 2023 observed
B  everything else
```

| Tier | n | Districts | What to say |
|---|---|---|---|
| **A — report as results** | 9 | Gondia, Nagpur, Chandrapur, Amravati, Wardha, Washim, Yavatmal, Ratnagiri, Solapur | 6–10 stations, 67–81% of 2023 observed. These carry the headline. |
| **B — report with caveat** | 7 | Jalgaon, Akola, Thane, Jalna, Parbhani, Raigad, Nandurbar | 1–4 stations but usable history. Note the station count. |
| **C — data-limited, not model failures** | 9 | Nanded, Nashik, Pune, Bhandara, Gadchiroli, Dhule, Kolhapur, Aurangabad, Beed | ≤4 stations **and** 25–55% of 2023 observed. Present as a **data-collection finding**. |
| **D — unusable** | 1 | Latur | 1 station, 44% observed, zero test windows in folds 4–6. Excluded from training. |

> **The rule and the empirical evidence do not agree perfectly, and that is worth showing.**
> Jalgaon and Akola land in tier B on coverage but have no learnable signal in the ridge probe;
> Nanded and Pune land in tier C but do have signal. The notebook therefore prints the
> **ridge MASE next to the TFT MASE for every district**, so a reader can distinguish
> "thin data" (a priori) from "no signal found" (a posteriori) rather than trusting one proxy.

Two Tier-C districts are worth calling out specifically. 📊 Correlation between 30-day rainfall and
subsequent Δ groundwater:

- **Nashik: −0.15** (rain associated with the water table *falling*)
- **Aurangabad: −0.04** (no relationship at all)

versus Latur 0.76, Kolhapur 0.49, Jalgaon 0.40, Nanded 0.40, Dhule 0.39.

A district whose water table does not respond to rainfall is not being measured by a
recharge-driven aquifer signal — either the 2-well panel is unrepresentative of a large district, or
those wells are pumping-dominated. **No model can fix that. Only more wells can.** This is exactly
the kind of shortcoming worth putting in the report.

---

## 3. Runtime / model changes for v3

Ordered so that Stage 1 changes are structurally incapable of degrading anything.

### M1 ⭐ Per-district × per-horizon shrinkage toward persistence — *fitted on validation*

```
Δ̂_final(d, h) = λ(d, h) · Δ̂(d, h),    λ ∈ [0, 1]
λ(d,h) = (n_d · λ̂_d + k · λ̂_global,h) / (n_d + k)      # hierarchical, k ≈ 20
```
λ̂ chosen to minimise **validation** MAE. λ = 0 reproduces persistence exactly; λ = 1 is the current
model. Because the current model is inside the search space, this is a one-parameter convex hull
that **cannot be worse than either endpoint on validation**, and it generalises well because it is
one number.

This is the single highest-value change. It caps Aurangabad's 2.84 MASE at ~1.0, protects every
Tier-C district, and directly satisfies "improving in all sectors and not degrading". It is the
standard damped/shrunk-combination-with-the-naive-forecast result from the forecasting literature.

### M2 ⭐ Per-horizon target scaling — the fix for t+7

`compute_delta_sd()` currently pools the change scale over h = 1…30 into **one** number per district.
In scaled units a 7-day change is therefore roughly half the size of a 30-day change, and the pinball
loss — averaged uniformly over the 30 decoder steps — is dominated by the long horizons.
**The model has almost no gradient incentive at t+7.** That is a mechanical explanation for
`TFT_VERSION_HISTORY.md` §7 caveat 1, alongside the physical one.

Use `delta_sd[d][h]` = std of h-step changes, train rows only, and scale each decoder step by its
own. Reconstruction becomes `level = anchor + out[h] · delta_sd[d][h]`; an output of 0 is still
exactly persistence, so the floor property is preserved.

### M3 ⭐ Fix validation (three parts, all cheap)

1. **Validation stride 1** instead of 7. Free — validation costs no gradient steps — and gives 7×
   more points, so early stopping stops being a coin flip.
2. **`VAL_MONTHS` 2 → 3**, or two disjoint validation blocks: one immediately prior + one
   same-season-as-test from the previous year. Currently fold1 tests on monsoon 2024 but early-stops
   on May–Jun 2024 (summer) — the wrong distribution.
3. **Refit on train+val for the selected number of epochs** and use that model for test. Recovers the
   two months of most-recent data currently thrown away in every fold. Report both if you want to be
   cautious.

### M4 ⭐ Honest conformal calibration

Compute nonconformity scores `max(q10 − y, y − q90)` on the **validation window**, per
(district, horizon) where n allows and pooled per horizon otherwise, then apply to test. Report raw
coverage *and* calibrated coverage. Expect calibrated coverage to land near-but-not-exactly 80% —
that number is the actual result. Note in the writeup that conformal's exchangeability assumption is
violated by temporal dependence, so the guarantee is approximate; that is a known and citable caveat.

### M5 Monotone quantiles

Currently q10/q50/q90 are three independent linear outputs and can cross. Emit
`q50`, `q10 = q50 − softplus(a)`, `q90 = q50 + softplus(b)`. Removes crossing, usually tightens
intervals slightly. ~10 lines in `tft_model.py`.

### M6 3-seed ensemble

Average reconstructed metre-space quantiles over seeds {42, 1337, 2024}. Reliable error reduction and
— more usefully here — it removes the "district X happened to fail this run" variance that makes the
per-district table hard to read. 3× compute.

### M7 ⭐ Add real baselines: seasonal-naive, ridge, LightGBM

v2 compares against persistence only. Add, on the identical folds and windows:
- seasonal naive (same day last year)
- climatology-of-change
- ridge on Δ (already written — `probe.py`)
- **LightGBM, one model per horizon, quantile objective for q10/q50/q90**

LightGBM runs in minutes on CPU. You need to know whether a gradient-boosted tree beats your TFT on
~2,900 training windows **before** the viva, not during it. If TFT > LGBM > ridge > persistence,
that is a complete and very defensible results story.

### M8 Feature additions (all past-only or genuinely known-future)

- **Antecedent Precipitation Index** with decay k ∈ {0.90, 0.95, 0.98} — the standard recharge proxy,
  strictly better than fixed 7/30-day sums.
- **Cumulative rainfall departure (CRD)** — the classic hard-rock-aquifer groundwater predictor.
- `rainfall_60d`, `rainfall_90d`, days-since-rain, standardised rainfall anomaly (SPI-like) at 1/3/6 months.
- **Explicit horizon index `h/30`** as a future-known feature — helps direct multi-horizon heads.
- **Climatological** rainfall/ET by day-of-year, computed train-only, as future-known decoder input.
  No leakage; gives the decoder something physical rather than pure calendar.
- Optional, clearly labelled: a *perfect-prog* ablation using actual future rainfall, as an upper
  bound on what integrating a rainfall forecast would buy. **Never in the headline.**

### M9 Capacity — ablate, don't guess

`TFT_VERSION_HISTORY.md` §10 item 2 proposes d_model 64→32 as a guess. Run the grid instead:
d_model ∈ {32, 64} × dropout ∈ {0.25, 0.35} × TRAIN_STRIDE ∈ {3, 7}, **on one mid-record fold
(fold4), selected on validation only**, then run the winner across all six. ~1 extra hour.

### M10 ⭐ Metrics — MAPE *and* MASE, plus the ones a reviewer expects

Report per horizon: **MAE (m)**, **MAPE (%)**, **MASE**, **Skill Score**, **NSE**, **pinball loss**,
**80% coverage** (plus separate q10/q90 exceedance), **mean interval width**.

> ⚠️ **The MAPE caveat you must state.** Groundwater depth is measured from an arbitrary datum, so
> MAPE rewards deep districts. A 5% error is **2.1 m in Dhule** (mean level −42.9 m) and **0.16 m in
> Raigad** (−3.25 m). MAPE will therefore rank districts almost *inversely* to real skill. Report it
> — it is easy to explain — but always next to MAE and MASE, with this sentence attached.

For the professor, the most explainable single number is the **skill score**:
`SS = 1 − MSE_model / MSE_persistence`, read as *"our forecast reduces squared error by 17% compared
to assuming nothing changes"*. It is the same information as MASE in a form that needs no defending,
and it is the Nash–Sutcliffe form with persistence as the reference — standard in hydrology.

Add **bootstrap confidence intervals** on median MASE (block bootstrap over district×fold) and a
**Diebold–Mariano test vs persistence per horizon**. "Beats persistence at t+30, p < 0.01" is worth a
great deal in a viva and costs about 20 lines.

### M11 Add a 7-day-mean target alongside the daily one

📊 Probe result: with the target defined as the centred 7-day mean level, districts where ridge beats
persistence at **t+7 goes 3 → 9**, and median t+30 MASE improves 1.078 → 1.044. The weekly-mean water
table is also the more meaningful quantity for a farmer/government dashboard — nobody acts on a
single day's district median from two wells.

Recommend reporting **both**: daily (strict, comparable to v2) and weekly-mean (operational). Do not
silently replace the daily target — that would look like moving the goalposts.

### M12 Anchor handling — recover windows in the gappy districts

v2 correctly drops a window when the last encoder day is unobserved (this fixed a real leak). But that
costs the most windows in exactly the Tier-C districts. Alternative matching operational reality:
anchor on the **most recent observed day within the last k days**, add `anchor_age` (0…k) as a past
feature, measure Δ from that anchor. Test as an ablation, not as a default.

### M13 Keep GAT parked 🛑

Confirmed externally: the HESS 2026 study over 2,951 German wells found globally-trained models had
**no systematic advantage** over per-well models (global better at 45.4% of wells, worse at 48.9%),
and that **dynamic similarity mattered far more than data volume** — filtering the training set for
correlated wells moved median NSE 0.47 → 0.72, while randomly removing 85% of wells changed almost
nothing. With 26 heterogeneous districts and flat validation curves, a spatial graph layer adds
capacity you cannot feed. The same paper suggests a better cheap experiment: **partition districts by
dynamic similarity** (correlation of their Δ series) and train per-cluster models.

### M14 Regression gate — the "not degrading" requirement, enforced

Add a cell that loads v2's `tft_metrics_per_district.csv` and prints a **per-district v2 → v3 MASE
delta table**, with any district degrading by >10% highlighted and required to be explained in the
writeup. Run v2 and v3 on the *identical* master dataset so the comparison is valid — v1→v2 was not
comparable because the data layer changed underneath, and you do not want that twice.

---

## 4. Sequencing

**Stage 0 — data (before any training run)**
D1 ET/SM decision · D2 Sentinel month-end · D4 gate Latur · D5 train-only fills ·
emit the per-district data-quality scorecard and the per-fold covariate-observed table as artifacts.

**Stage 1 — safe wins (structurally cannot degrade)**
M2 per-horizon scaling · M3 validation fixes · M4 honest conformal · M5 monotone quantiles ·
M1 shrinkage · M14 regression gate.

**Stage 2 — evidence for the report**
M7 baselines (seasonal-naive, ridge, LightGBM) · M10 full metric suite + significance tests ·
M11 weekly target · tier table from §2.

**Stage 3 — accuracy squeeze**
M6 seeds · M8 features · M9 capacity ablation · M12 anchor · D3 sampler cap.

Expected outcome if Stages 0–2 land: t+14 and t+30 hold or improve slightly, **t+7 moves off 1.004**
(M2 + M11), no district sits materially above MASE 1.0 (M1), and the uncertainty numbers become real
rather than self-fulfilling (M4). The report gains a baseline ladder, significance tests, and a
defensible per-district data story.

---

## 5. What was actually built (2026-08-15)

All of Stage 0, Stage 1 and Stage 2 are implemented and the whole notebook has been
**run end-to-end on CPU with a shrunk config** — every cell executes clean, so it will not
die three hours into a Colab session.

| File | Status | What it does |
|---|---|---|
| `01_data_collection/era5_land.py` | **new** | ERA5-Land daily ET, PET and 4-layer soil moisture via Earth Engine. Resumable, same pattern as `chirps_rainfall.py`. Replaces the WRIS/NRSC feed that stopped at 2025-06-30 |
| `02_preprocessing/preprocess_env_features.py` | rewritten | Consumes ERA5-Land when present, falls back to WRIS otherwise. Prints an **observed-rate-by-year table every run** and warns if any feature drops below 90% real |
| `02_preprocessing/merge_master.py` | patched | Sentinel month-**end** causal stamping (D2), `data_tier` + scorecard, ERA5 columns, `water_stress` |
| `03_models/tft/tft_model_v3.py` | **new** | Vectorised VSN (~2.4× faster on CPU, more on GPU — this is what pays for the 3-seed ensemble), monotone quantile head, and `d_select` / `grn_hidden_factor` capacity controls |
| `03_models/tft/notebooks/archive/09_tft_colab_production_v3.ipynb` | **new** | 29 cells, Stage 0–2 |
| `archive/duplicates/master_dataset_v3target__dup_models_tft.csv` | refreshed | Was a stale pre-rebuild copy — replaced with the current 55-column build |

### Two things found while building it

**1. v2's parameter count is wrong by ~6×.** The notebook reports `205,173`; that exact
configuration instantiates **1,231,159** parameters. The dominant block is the variable-selection
GRN, whose first layer is `Linear(n_past × d_model, d_model)` — 143K parameters on its own, plus a
78K skip connection. **The variable selector was larger than the rest of the network**, on a
problem with ~290 genuinely independent training windows. So `TFT_VERSION_HISTORY.md` §7 caveat 2 —
"it remains overparameterised" — is far more true than it says, and item 2 in §10 ("try d_model
64→32") was aimed at the wrong knob.

v3 defaults to `d_model=32, d_select=8, grn_hidden_factor=0.5` ≈ **222K real parameters**,
a genuine 5.5× reduction, and prints the params-per-independent-window ratio every fold.

**2. The baseline ladder does its job immediately.** Even in the deliberately crippled 2-epoch
smoke run, **LightGBM beat the TFT at t+7**. That is exactly the question a professor will ask,
and now the notebook answers it in a table instead of leaving it open.

### Decisions taken

| Question | Decision |
|---|---|
| ET / soil moisture | **Swap to ERA5-Land.** Also gains PET and 4 soil-moisture depths — `sm_l3` (28–100 cm) and `sm_l4` (100–289 cm) are on the actual recharge path, so this is an upgrade, not just a patch |
| Compute | 3 seeds × 6 folds = 18 runs, `MAX_EPOCHS=60`, `PATIENCE=10`. The VSN vectorisation is what keeps this affordable. Per-fold results cache to disk, so a disconnect costs one fold, not the run. Set `SEEDS=[42]` for a ~1/3-length single-seed pass |
| t+7 target | **Both** daily and 7-day-mean reported. Daily keeps the v2 comparison honest; weekly-mean is what the dashboards should show |
| Refit on validation | **Dropped.** It conflicts with honest calibration — the validation span is split instead (60% early stopping, 40% λ + conformal). Clean uncertainty numbers were judged worth more than ~2 months of extra training data |

---

## 6. How to run

```bash
# ── Stage 0: data (do this first) ────────────────────────────────────────────
cd 01_data_collection
python era5_land.py                  # needs: earthengine authenticate (one-time)

cd ../02_preprocessing
python preprocess_groundwater.py
python preprocess_env_features.py    # check the observed-rate-by-year table it prints
python merge_master.py               # writes master_dataset.csv + district_data_quality.csv
```

Then on Colab (**Runtime → Change runtime type → T4 GPU** first — the notebook hard-stops
without one, and a runtime switch gives a fresh VM so upload afterwards):

1. Upload `tft_model_v3.py` and the rebuilt `master_dataset.csv`.
2. Optionally also upload v2's `outputs/tft_metrics_per_district.csv`, renamed to
   `tft_metrics_per_district_v2.csv`, to switch on the regression gate.
3. Run all cells. Confirm `Device: cuda` prints.
4. **Read the data-quality report cell before the results cells.** If the covariate reality table
   shows any feature below 50% in any fold, stop and fix the data — that is the whole point of
   putting it first.

---

## 7. Minor code notes

- `metrics()` groups by a one-element list (`['horizon']`); pandas ≥ 2 returns a 1-tuple key there,
  so `row['Horizon'] = k` can become `('t+7',)`. Harmless on the version Colab ran, brittle across
  versions — group by the scalar.
- `evaluate()` reconstructs the window index as `bi*BATCH_SIZE + i`. Correct only while the loader's
  batch size equals the global `BATCH_SIZE` and `shuffle=False`. Return the index from the Dataset
  instead — one line, removes a silent-corruption trap.
- `station_count` varies 0.6–8.2 per day *within* a district. The anomaly aggregation fixes level
  offsets but not the change in noise variance when the panel size changes. Worth either weighting
  the loss by `station_count` or documenting.

---

## Sources consulted

- [Never Train a Deep Learning Model on a Single Well? Revisiting Training Strategies for Groundwater Level Prediction — HESS 30, 2373 (2026)](https://hess.copernicus.org/articles/30/2373/2026/)
- [Multi-site deep learning for groundwater level prediction across global datasets — J. Hydroinformatics 27(10)](https://iwaponline.com/jh/article/27/10/1632/109550/Multi-site-deep-learning-for-groundwater-level)
- [Benchmarking ML and DL Models for Groundwater Level Prediction in Karst Aquifers — Water 18(8), 939](https://www.mdpi.com/2073-4441/18/8/939)
- [Delta-Based Target Reformulation for Short-Term Forecasting with LSTM and Transformer Models](https://arxiv.org/pdf/2606.17692)
- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting — Lim et al.](https://arxiv.org/abs/1912.09363)
- [A Gentle Introduction to Conformal Time Series Forecasting](https://arxiv.org/html/2511.13608)
- [Predictive inference for time series: why is split conformal effective despite temporal dependence?](https://arxiv.org/pdf/2510.02471)
- [Solving the Forecast Combination Puzzle — Lee & Lee](https://economics.ucr.edu/repec/ucr/wpaper/202514.pdf)
- [Addressing Challenges in Time Series Forecasting: A Comprehensive Comparison of ML Techniques](https://arxiv.org/pdf/2503.20148)
