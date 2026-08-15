# TFT Production v1 → v2 → v3: What Changed and Why

**v1** = `09_tft_colab_production.ipynb` — full 6-fold run on Colab T4, ~3 hours, completed 2026-08-14
**v2** = `09_tft_colab_production_v2.ipynb` — rebuilt 2026-08-15 after the v1 post-mortem; full 6-fold T4 run completed
**v3** = `09_tft_colab_production_v3.ipynb` — rebuilt 2026-08-15 after a full audit of v2; full 6-fold, 3-seed T4 run completed in **6.5 minutes**

*v1 and v2 ran on different master datasets (v2's groundwater layer was rebuilt), so raw RMSE is not
directly comparable between them. v2 and v3 ran on the **same district set and the same headline
21-district exclusion list**, so their numbers are a fair apples-to-apples comparison — see §11.*

---

## 1. TL;DR

v1's reported numbers looked reasonable (median RMSE 1.78–2.15 m, MAPE 18–23%). They were not.
Benchmarking against a **persistence baseline** revealed v1 lost to persistence in **15 of 18**
fold-horizon combinations. v2 fixed the data layer, the prediction target, and the model size, and
added MASE. v2 was a large real improvement but was still tied with persistence at t+7, still
overfitting, and still had **7 districts that lost to persistence at every horizon** with "no
validated explanation" for six of them.

v3 answers that open question, fixes v2's largest hidden bugs (a mislabeled parameter count, a
self-referential conformal calibration, a future-leaking satellite feature), and adds a shrinkage
mechanism that puts a floor under every district. Result: **all three horizons now beat persistence
on the headline, all three are statistically significant against persistence (Diebold–Mariano
p < 0.05, bootstrap 95% CI excludes MASE 1.0), and all seven of v2's persistently-failing districts
now tie or beat persistence at every horizon.**

| | v1 (real GPU) | v2 (real GPU) | v3 (real GPU, 3-seed) |
|---|---|---|---|
| **Median MASE t+7**  | ~2.96 *(post-hoc, pooled)* | 1.004 (tied) | **0.958** |
| **Median MASE t+14** | ~2.15 *(post-hoc, pooled)* | 0.940 | **0.909** |
| **Median MASE t+30** | ~1.56 *(post-hoc, pooled)* | 0.827 | 0.834 *(flat, see §11)* |
| Districts beating persistence @ t+7  | — | 12/24 (50%) | **16/21 (76%)** |
| Districts beating persistence @ t+14 | — | 17/24 (71%) | **19/21 (90%)** |
| Districts beating persistence @ t+30 | — | 17/24 (71%) | **19/21 (90%)** |
| Worst-district MASE @ t+30 | — | **1.896** (Aurangabad) | **1.017** (Solapur) |
| Model parameters (as configured) | 11,597,735 | **1,231,159** *(v2 reported 205,173 — wrong, see §7)* | **~222,000** *(correctly reported)* |
| Statistically significant vs persistence? | not tested | not tested | **yes, all 3 horizons, p < 0.05** |
| Conformal calibration | none | self-referential (invalid, see §9) | fitted on a held-out block (honest, see §9) |
| Baseline ladder | none | persistence only | **persistence, climatology, ridge, LightGBM** |
| 3-seed ensemble | no | no | **yes** |
| Wall time (6 folds) | ~3 h | ~3 h | **6.5 min** |

**Verdict: v3 closes the "seven failing districts" question, makes t+7 a genuine (if modest) win
instead of a tie, and adds the statistical and calibration rigor v2 was missing.** t+30's pooled
headline number is essentially flat — that is explained, not hidden, in §11.

---

## 2. The discovery that triggered the v1→v2 rebuild

Groundwater is a near-random-walk: today's level tells you almost everything about next week's. So
the honest question is not "what is the RMSE?" but "does the model beat doing nothing?"

**Persistence vs v1 TFT — test RMSE (m), real GPU run:**

| Fold | Horizon | Persistence | v1 TFT | Winner |
|---|---|---|---|---|
| fold1_monsoon2024 | t+7 | **2.31** | 6.97 | Persistence |
| fold2_postmonsoon2024 | t+7 | **1.40** | 4.86 | Persistence |
| fold3_dry2025 | t+7 | **3.30** | 3.72 | Persistence |
| fold4_monsoon2025 | t+7 | 4.45 | **2.94** | v1 TFT |
| fold5_postmonsoon2025 | t+7 | **2.40** | 4.10 | Persistence |
| fold6_dry2026 | t+7 | **1.71** | 3.16 | Persistence |
| *(t+14 and t+30 follow the same pattern — v1 won only on fold4)* | | | | |

In MAE terms the gap was worse: fold2 t+7 persistence **0.56 m** vs v1 TFT **2.91 m** — 5×.

**For scale:** the median real GWL change is **0.29 m over 7 days** and **0.90 m over 30 days**.
v1's errors (1.7–5.1 m MAE) were several times larger than the movement it was supposed to predict.
It had learned district average levels, not dynamics.

### Why the v1 headline hid this
v1 reported the **median across (district × fold)**, which was 1.78–2.15 m RMSE — while the
**pooled per-fold** RMSE was 2.9–9.1 m. The median suppressed the bad districts, and with no
baseline column there was nothing to reveal that even the good numbers were worse than doing
nothing.

---

## 3. Root cause 1 (v1→v2) — the district series weren't physically real

**This was the largest single problem, and it was in the data, not the model.**

The raw CGWB feed mixes two completely different measurement programmes:

| | Count | Coverage | What they are |
|---|---|---|---|
| Telemetric stations | ~125 | 20–90% of days | Automatic loggers, continuous |
| Manual-survey wells | ~1,254 | **exactly 2 readings each** (Jan + May 2023) | Shallow dug wells, one-off survey |

Coverage is sharply bimodal — nothing sits between ~1% and ~20%. v1 aggregated **all** of them
together with a raw median, so when the manual wells dropped out after May 2023 the "district
level" jumped to whatever the telemetric boreholes read.

**Aurangabad in v1:**
```
2023-01    -4.94 m   (40 manual dug wells)
2023-06   -98.88 m   ( 1 deep telemetric borehole)   ← ~94 m "change" that never happened
2024-09    -1.28 m   ( 3 stations)
```
That is not the water table moving. It is *which wells were measured* changing. Aurangabad's std
was 30.5 m, its train→test level shift 19.8 m, its MAPE 301%, its prediction interval 17.7 m — one
district was distorting every aggregate metric in the report.

### The v1→v2 fix — three stages in `preprocess_groundwater.py`

1. **Panel filter** — keep only stations reporting on ≥10% of days (the clean gap in the bimodal
   distribution). 1,379 → ~109 stations.
2. **Per-station anomaly aggregation** — instead of aggregating raw depths, each station is
   converted to its deviation from its *own* long-run mean; anomalies are aggregated and
   re-referenced to a fixed per-district baseline. A station joining or leaving now contributes
   "how far above/below its own normal it is", so composition changes can no longer masquerade as
   water-table movement.
3. **Drift-artifact filter** — drop wells whose record is dominated by a single monotonic trend
   (trend R² ≥ 0.6 **and** range > 20 m). These are newly-commissioned boreholes still filling:
   Aurangabad's `CGWBNG0515` (100 m deep) started at −98.8 m — essentially the bottom of the hole —
   and climbed steadily to −5 m over three years, i.e. **102 m of pure drift**. Only 3 stations
   state-wide match; all 26 districts survive.

> ⚠️ **Do not filter on range alone.** Dhule and Nandurbar legitimately swing 30–50 m *seasonally*
> (trend R² 0.004–0.09) — they are genuinely deep hard-rock aquifers. A naive `range > 30 m` rule
> would have deleted 3 of Dhule's 4 wells and wiped out a district. Monotonic **drift** is the
> artifact; large seasonal **oscillation** is real signal.

### Measured effect of the data fix alone (no model change)

| Metric | v1 data | v2/v3 data |
|---|---|---|
| Worst district std | 30.47 m (Aurangabad) | **10.24 m** (Dhule) |
| Worst 1-day jump | 29.10 m | **14.75 m** |
| Implausible jumps NaN'd | 405 | **89** |
| Overall `gwl_mean` missing | 24.2% | **12.3%** |
| Persistence RMSE t+7 (mean over folds) | 2.60 m | **1.07 m** |
| Persistence RMSE t+14 / t+30 | 3.04 / 3.64 m | **1.49 / 2.33 m** |

**~60% of what looked like "prediction error" was station-composition noise.** Aurangabad now
shows a textbook seasonal cycle (summer decline to −7 m, monsoon recovery to −1.8 m).

Note this makes the task *harder*, not easier: the bar to beat is now 1.07 m, not 2.60 m.

---

## 4. Root cause 2 (v1→v2) — predicting the level instead of the change

**v1:** predicted the absolute (per-district z-scored) GWL level. Because groundwater barely moves,
the model spent all its capacity re-deriving the level — the easy part that persistence gives for
free — and got it wrong by metres, swamping any skill at predicting the actual change.

**v2/v3:** predicts `Δ = gwl(t+h) − gwl(anchor)`, where `anchor` is the last encoder day.
Additionally **every GWL encoder feature is made anchor-relative** (`(value − anchor) / delta_sd`),
so the network never sees an absolute depth and the whole problem is translation-invariant.

Reconstruction is `level = anchor + output × delta_sd`, which has a useful property:

> **An output of exactly 0 reproduces persistence.** Persistence becomes the model's floor rather
> than its competitor, so anything it learns is upside.

This mirrors published practice — the hybrid TFN–LSTM groundwater model for Rhode Island and the
STRPM_r water-level model both train on residuals rather than absolute levels, and both report
better accuracy and greater stability at long horizons.

---

## 5. Root cause 3 (v1→v2) — 11.6M parameters for ~156 independent samples

v1 reported "19,535 training windows" for fold6. With `stride=1`, consecutive windows share **149
of 150 days** — they are near-duplicates. The number of genuinely independent 150-day windows is
roughly `1034 days ÷ 150 × 26 districts ≈ 156`.

That is **~74,000 parameters per independent sample**. The consequence was visible in the v1
training logs: three folds peaked at **epoch 1–2**, then train loss fell ~6× while validation
flatlined — textbook memorisation.

### v2 capacity changes

| Setting | v1 | v2 | Reason |
|---|---|---|---|
| `d_model` | 192 | 64 | capacity |
| `lstm_layers` | 3 | 1 | capacity |
| `n_heads` | 8 | 4 | capacity |
| `ENCODER_LEN` | 120 | 60 | shorter context ⇒ ~2× more usable windows |
| `dropout` | 0.15 | 0.25 | regularisation |
| `weight_decay` | 1e-4 | 1e-2 | regularisation |
| **train stride** | 1 | 7 | kills near-duplicate windows |

v2's notebook printed "205,173 parameters" and claimed a 56× reduction from v1. **That number was
wrong — see §7.** The real reduction only happened in v3.

---

## 6. Root cause 4 (v1→v2) — metrics that couldn't detect failure

v1 had no baseline comparison in any metric. v2 added `Persist RMSE`/`Persist MAE` to every table
and introduced **MASE** = model MAE ÷ persistence MAE, with a headline verdict that explicitly
instructs the reader to stop and revisit the framing rather than tune hyperparameters if MASE ≥ 1.

**Why MAPE was misleading:** it explodes when actual GWL is near 0 m. Correlation between a
district's MAPE and its fraction of readings within 1 m of the surface was **r = 0.59** in v1.
MAPE is retained in v2/v3 (it is the easiest number to explain to a non-technical audience) but is
never reported without MASE alongside it, and v3 adds the explicit datum caveat — see §9 of
`V3_PLAN.md`.

---

## 7. What the v2→v3 audit found

Before writing a line of v3 code, the running v2 notebook and pipeline were audited end-to-end.
Four real problems were found, none of which showed up in v2's own printed output:

### 7.1 v2's reported parameter count was wrong by ~6×

v2 printed **"Model parameters: 205,173"** and claimed a 56× cut from v1's 11.6M. Rebuilding the
exact same configuration and counting parameters directly gives **1,231,159** — six times larger
than reported. The dominant block was the variable-selection GRN: with `d_model=64` and ~35 past
features, its first layer alone (`Linear(n_past × d_model, d_model)`) is 143,360 parameters, plus a
78,400-parameter skip connection — **the variable selector was larger than the rest of the network
combined**, on a problem with ~290 genuinely independent training windows (stride-7 windows still
share 83 of 90 days). §5's "capacity was cut 56×" claim was true of the label, not the model.

**v3 fix:** the VSN was rewritten (`tft_model_v3.py`) with a `d_select` compression step (each
variable's embedding is projected down before the selector sees it — default 64→8) and a
`grn_hidden_factor` that narrows the per-variable GRN bottleneck (default 0.5×). Default v3 config
genuinely builds **~222,000 parameters** — a real 5.5× reduction — and the notebook prints
params-per-independent-window every fold so this can't silently drift again.

### 7.2 Sentinel features leaked ~15 days of the future

`merge_master.py` stamped each monthly satellite composite (NDVI/NDWI/NDMI/VV/VH) at the **1st** of
its month, then linearly interpolated to daily. NDVI on 15 January was therefore a 50/50 blend of
the January **and February** composites — up to half a month of future information in every daily
value, in a model whose entire purpose is forecasting 7–30 days ahead.

**v3 fix:** composites are now stamped at month-**end** and held constant (`ffill`) until the next
one lands — strictly causal. The cost is a lag of up to ~30 days, which is honest: that is genuinely
when the information becomes available.

### 7.3 ET and soil moisture were 100% fabricated for half the evaluation

The WRIS/NRSC ET and soil-moisture feed stopped publishing at **2025-06-30**. Everything after that
was per-district day-of-year climatology, correctly flagged (`*_observed=0`) but still fed to the
model as an ordinary input feature:

| Fold | et/soil-moisture genuinely observed |
|---|---|
| fold1_monsoon2024 | 97% |
| fold2_postmonsoon2024 | 100% |
| fold3_dry2025 | 96% |
| **fold4_monsoon2025** | **25%** |
| **fold5_postmonsoon2025** | **0%** |
| **fold6_dry2026** | **0%** |

In half the walk-forward evaluation, `et_mm`/`soil_moisture`/`net_water_balance` carried no
information beyond what `doy_sin`/`doy_cos` already gave the model.

**v3 fix:** re-sourced to **ERA5-Land** (`Data Collection/era5_land.py`, ~5-day latency, no
publication-lag problem) via Google Earth Engine. This also adds potential evapotranspiration and
four soil-moisture depths instead of one — `sm_l3` (28–100 cm) and `sm_l4` (100–289 cm) sit on the
actual recharge pathway to the water table, which a single surface reading cannot represent.

### 7.4 Conformal calibration was fitted on the set it calibrated

v2's conformal step computed the nonconformity quantile from the test fold's own rows, then applied
the resulting margin to those same rows. "80% coverage after calibration" was therefore a
mathematical identity — any model, however bad, reaches 80% this way. It was not a result.

**v3 fix:** the validation span is now split into an early-stopping block and a separate
**calibration block**, and both the shrinkage weights (§8) and the conformal margins are fitted on
the calibration block only, then applied to test. See §9 for what the honest number turned out to
be — it is not 80%, and that is the more useful finding.

### 7.5 An independent predictability probe explained six of v2's seven "unexplained" failures

`PROD_V1_vs_V2.md` §10 (v2) listed diagnosing "Aurangabad, Bhandara, Nashik, Jalgaon, Kolhapur,
Dhule, Ratnagiri" — the seven districts that lost to persistence at every horizon — as priority 1,
noting that the leading hypothesis (inflated `delta_sd` scale) had been tested and rejected, with
"no validated explanation."

A ridge regression was fitted on the identical Δ-target, per district, on the identical
walk-forward folds, completely independent of the TFT. **Six of the seven — Aurangabad, Bhandara,
Nashik, Jalgaon, Dhule, Kolhapur — also lost to persistence with ridge.** Two unrelated model
families failing on exactly the same districts is a **data limitation, not a model bug**. All six
sit in the new data-tier scheme (below, §8) as tier B or C: thin station panels combined with
2023 records that are 25–72% incomplete.

**Only Ratnagiri — a tier-A district with 9 stations and 70% coverage in 2023 — had a real ridge
signal (MASE 0.79) that the TFT was failing to capture (MASE 1.08).** That was the one genuine
modelling bug in v2, as opposed to a data ceiling. §11 shows it is now fixed in v3.

---

## 8. v3 additions

### 8.1 Data tiers, assigned from station count and coverage only — never from model results

`merge_master.py` now writes a `data_tier` column and a `district_data_quality.csv` scorecard,
computed **before any model runs**, so it is available as an independent explanation of the
per-district results rather than a post-hoc excuse:

```
D  overall observed < 55%                                    -> excluded from training
A  >= 6 stations AND >= 65% of 2023 observed                  -> headline results
C  <= 4 stations AND <  55% of 2023 observed                  -> data-collection finding
B  everything else                                             -> usable, reported with caveat
```

| Tier | n | Districts |
|---|---|---|
| A | 9 | Amravati, Chandrapur, Gondia, Nagpur, Ratnagiri, Solapur, Wardha, Washim, Yavatmal |
| B | 7 | Akola, Jalgaon, Jalna, Nandurbar, Parbhani, Raigad, Thane |
| C | 9 | Aurangabad, Beed, Bhandara, Dhule, Gadchiroli, Kolhapur, Nanded, Nashik, Pune |
| D | 1 | Latur (44% observed, 1 station, zero test windows in 3 of 6 folds — excluded from training) |

Spearman correlation with ridge MASE @ t+30: 2023 coverage **−0.54**, overall coverage **−0.54**,
station count **−0.51**, size of typical movement **−0.10 (nothing)**. Predictability is driven by
data completeness, not by how much a district's water table moves.

### 8.2 Per-horizon target scaling

v2 pooled the h-step change scale (`delta_sd`) over h = 1…30 into a single number per district. In
scaled units a 7-day change was therefore roughly half the size of a 30-day change, and the pinball
loss — averaged uniformly over the 30 decoder steps — was dominated by the long horizons. **The
model had almost no gradient incentive at t+7.** That is a mechanical explanation, on top of the
physical one, for v2's t+7 MASE landing at exactly 1.004.

v3 computes `delta_sd[district][horizon]` separately for every one of the 30 decoder steps, so each
horizon contributes comparably to the loss. Reconstruction stays `level = anchor + out[h]·sd[h]`, so
an output of 0 is still exactly persistence at every horizon.

### 8.3 Shrinkage toward persistence (the mechanism that removes catastrophic districts)

```
level_hat = anchor + λ(district, horizon-band) · delta_hat        λ ∈ [0, 1]
```

λ = 0 reproduces persistence exactly; λ = 1 is the raw model. λ is fitted on the calibration block
by grid search to minimise MAE, then pulled toward a global (district-independent) value in
proportion to how little calibration data supports it — a district cannot pick an extreme λ off a
handful of points. Because the unshrunk model is inside the search space, this is a one-parameter
convex hull that cannot be worse than either endpoint **on the calibration data** — it is what stops
any district from repeating Aurangabad's v2 outcome (MASE 2.84 at t+7, no floor).

**What the real run's λ values show:** early folds, with less training history, produce very low λ
(fold1 global λ = 0.05–0.15 — "don't trust this model yet, fall back toward no-change"); by fold6,
with 2.75× more training months, λ climbs to 0.8–1.0 at the longer horizons. The calibration step is
automatically detecting how much the model should be trusted at each stage of the walk-forward
schedule, with no manual tuning.

### 8.4 Monotone quantile head

v2's q10/q50/q90 were three independent linear outputs and could cross. v3 emits `q50` plus two
non-negative spreads via softplus (`q10 = q50 − softplus(a)`, `q90 = q50 + softplus(b)`), so
crossing is impossible by construction.

### 8.5 Baseline ladder

v2 compared only to persistence. v3 adds seasonal climatology-of-change, ridge regression, and
LightGBM — all fitted on the identical windows and identical Δ-target, so MASE is directly
comparable across the ladder. See §11.

### 8.6 Vectorised model (`tft_model_v3.py`)

The v1/v2 Variable Selection Network ran a Python loop — one `Linear(1→d_model)` projection plus one
full GRN *per feature*, per forward pass. `BatchedGRN`/`BatchedFeatureProjection` compute the
identical math as batched matrix operations instead. Combined with the smaller default capacity,
this is what makes a 6-fold × 3-seed run (18 total trainings) finish in 6.5 minutes on a T4, versus
v2's ~3 hours for a single seed.

### 8.7 Daily and weekly-mean targets, both reported

For several districts the daily district median is two wells averaged on one day, and the median
real 7-day movement is 0.06–0.14 m — smaller than the measurement noise. v3 additionally scores a
7-day-mean target (the mean level over the week ending at the horizon) alongside the daily one. See
§11 for what this did and did not change.

### 8.8 Statistical significance

v3 adds a **Diebold–Mariano test** (HAC-corrected for the autocorrelation that overlapping forecast
windows introduce) against persistence at each horizon, and a **district-level bootstrap 95%
confidence interval** on the median MASE. Neither existed in v1 or v2.

---

## 9. Results — v3, real GPU run (6 folds × 3 seeds, T4, 6.5 minutes)

### Headline (21 non-low-station districts, pooled over all folds, daily target)

| Horizon | MAE (m) | Persist MAE (m) | **MASE** | Skill % | MAPE (%) | Coverage (%) | n |
|---|---|---|---|---|---|---|---|
| t+7  | 0.532 | 0.555 | **0.958** | +8.0%  | 9.73  | 71.8 | 13,905 |
| t+14 | 0.753 | 0.829 | **0.909** | +13.7% | 13.62 | 69.7 | 13,780 |
| t+30 | 1.112 | 1.333 | **0.834** | +22.2% | 20.10 | 67.1 | 13,758 |

**Beats persistence (MASE < 1) at 3/3 horizons — the first v-generation where t+7 is also a genuine
win rather than a tie.**

### Statistical significance vs persistence

| Horizon | Diebold–Mariano stat | p-value | Verdict |
|---|---|---|---|
| t+7  | −6.749 | 1.5×10⁻¹¹ | TFT better, p < 0.05 |
| t+14 | −8.000 | 1.3×10⁻¹⁵ | TFT better, p < 0.05 |
| t+30 | −9.558 | ~0 | TFT better, p < 0.05 |

| Horizon | Median MASE | 95% bootstrap CI (resampling districts) | Excludes 1.0? |
|---|---|---|---|
| t+7  | 0.971 | [0.940, 0.982] | **YES** |
| t+14 | 0.929 | [0.874, 0.949] | **YES** |
| t+30 | 0.862 | [0.796, 0.885] | **YES** |

This is not noise — at every horizon, every plausible resample of the district set still beats
persistence.

### Baseline ladder — identical windows, identical Δ-target

| Model | t+7 MASE | t+14 MASE | t+30 MASE |
|---|---|---|---|
| Persistence | 1.000 | 1.000 | 1.000 |
| Climatology-of-change | 1.052 | 1.018 | 0.940 |
| Ridge | 1.218 | 1.178 | 1.196 |
| **LightGBM** | **0.949** | 0.937 | 0.857 |
| TFT (unshrunk) | 0.989 | 0.909 | **0.801** |
| **TFT (shrunk, headline)** | 0.958 | **0.909** | 0.834 |

**Honest reading:** LightGBM edges the TFT at t+7 (0.949 vs 0.958) — worth reporting plainly rather
than hiding it. At t+7 even the winning model barely beats persistence, which is physically expected
(recharge lags are 3–21 days, so a week is not enough time for environmental drivers to act). The
TFT wins clearly at t+14 and t+30, and both linear models (ridge, climatology) lose to persistence
outright at the short-to-medium horizons — evidence that the sequence model is earning its
complexity where the horizon is long enough to matter.

### Coverage — the honest number, not the self-fulfilling one

| | t+7 | t+14 | t+30 |
|---|---|---|---|
| v2 raw (before any calibration) | 76.2% | 71.2% | 63.7% |
| v2 "calibrated" (self-referential — see §7.4) | 80.0% | 80.0% | 80.0% |
| v2 calibrated interval width | 5.9–6.3 m | | |
| v3 raw (before shrinkage/conformal) | 70.8% | 69.3% | 68.3% |
| **v3 calibrated (honest — fitted on a held-out block)** | **70.1%** | **66.1%** | **63.2%** |
| v3 calibrated interval width | 1.54 m | 2.16 m | 2.98 m |

The legitimately-calibrated v3 coverage **undershoots the 80% target**, and at t+14/t+30 is even
slightly *below* the pre-calibration number — the calibration block (the 5–6 calmer weeks
immediately before each test fold) under-represents the volatility of the full test window that
follows it, a real distribution-shift effect rather than a bug. This is a materially more honest
result than v2's number, and interval widths are 2–4× narrower even while under-covering — the
model's uncertainty is tighter than v2's but still somewhat overconfident. Documented as future
work in `V3_PLAN.md` rather than papered over.

---

## 10. Results — v3 per-district, with data tier alongside

Districts beating persistence by tier (MASE < 1):

| Horizon | Tier A (9) | Tier B (7 total, 5 headline*) | Tier C (9 total, 7 headline*) |
|---|---|---|---|
| t+7  | 8/9 (median 0.964) | 4/5 (median 0.975) | 4/7 (median 0.973) |
| t+14 | 8/9 (median 0.894) | 5/5 (median 0.943) | 6/7 (median 0.939) |
| t+30 | 8/9 (median 0.813) | 5/5 (median 0.867) | 6/7 (median 0.884) |

*Beed, Nanded, Raigad, Thane are additionally `low_station_flag`-excluded from the headline and
reported separately (§10 of the notebook) — they are not the same exclusion as data tier.

**Even tier-C districts (thin panel, gappy 2023) win at t+14/t+30 in 6 of 7 cases** — the shrinkage
mechanism (§8.3) is doing real work here, pulling the point forecast toward persistence exactly
where the calibration data says the signal is weak, rather than letting a thin-data district's
model output run free.

---

## 11. v2 → v3: the seven persistently-failing districts, resolved

v2 identified seven districts that lost to persistence **at every horizon**, and left the root
cause "open rather than guessed at" for six of them (§7.5 above explains why: it's a data
limitation, confirmed independently by ridge). Here is the same seven districts' MASE, v2 vs v3:

| District | Tier | v2 MASE t+30 | v3 MASE t+7 | v3 MASE t+14 | v3 MASE t+30 |
|---|---|---|---|---|---|
| Aurangabad | C | 1.896 | 1.006 | 0.969 | **0.884** |
| Jalgaon    | B | 1.168 | 0.975 | 0.969 | **0.867** |
| Nashik     | C | 1.142 | 1.010 | 0.979 | **0.980** |
| Bhandara   | C | 1.097 | 0.973 | 0.929 | **0.888** |
| Ratnagiri  | A | 1.084 | 0.982 | 0.949 | **0.885** |
| Dhule      | C | 1.057 | 0.940 | 0.939 | **0.884** |
| Kolhapur   | C | 1.014 | 0.850 | 0.812 | **0.698** |

**All seven now tie (Aurangabad, Nashik at t+7 — 1.006 and 1.010, statistically indistinguishable
from persistence) or clearly beat persistence at every horizon.** Aurangabad alone went from
"nearly 3× worse than doing nothing" (v2 §7 caveat 3: MASE 2.841 at t+7) to essential parity at
t+7 and a clear win at t+14/t+30 — driven by the shrinkage mechanism (§8.3) putting a floor under
what the raw model was doing, combined with the per-horizon scaling fix (§8.2) and the causal
Sentinel/ERA5 data fixes (§7.2–7.3) removing noise from its inputs.

**Ratnagiri is the one case worth flagging separately.** It is tier A (9 stations, 70% 2023
coverage) — the ridge probe found real signal there in v2 (MASE 0.79) that the TFT was not
capturing (MASE 1.08), i.e. it was v2's one genuine model bug rather than a data ceiling. v3 fixes
it (0.982 / 0.949 / 0.885) — most plausibly via the per-horizon delta scaling and the corrected
capacity, since those are the two v3 changes that touch model *mechanics* rather than upstream data.

---

## 12. Honest caveats carried into v3

1. **t+30's pooled headline is essentially flat** (v2 0.827 → v3 0.834, both on the identical
   21-district headline set). This is not a regression to be worried about: the worst-district MASE
   at t+30 fell from 1.896 to 1.017, and 19/21 districts beat persistence versus 17/24 (using v2's
   own denominator) — the pooled median is a less informative summary than the per-district
   distribution here, which is exactly why §10's tier breakdown and §11's before/after table are the
   results worth quoting, not the single pooled number.

2. **The weekly-mean target did not repeat the ridge probe's improvement.** An independent ridge
   probe (documented in `V3_PLAN.md` §0) found that switching from a daily to a 7-day-mean target
   took districts-with-skill at t+7 from 3 to 9. In the real TFT run this did not materialise: daily
   vs weekly-mean MASE are within 0.01–0.02 of each other at every horizon, and the weekly target
   actually wins one *fewer* district at t+7 (15/21 vs 16/21). Reported plainly rather than
   cherry-picked — the daily target, combined with the rolling/anchor features already in the
   encoder, appears to already capture most of what the weekly aggregation would add for this model
   (unlike for the simpler ridge probe, which had no such features).

3. **LightGBM beats the TFT at t+7.** See §9's baseline ladder discussion — expected given how
   little exploitable signal exists at that horizon, and reported rather than hidden.

4. **Calibrated coverage undershoots 80%** (§9) — a genuine, now-honestly-measured limitation rather
   than v2's manufactured 80%. Left as documented future work.

5. **v1 MASE figures are post-hoc and pooled per fold** (v1 never computed MASE); v2 and v3 use the
   same per-district-then-pooled aggregation on the same 21-district headline set, so the v2↔v3
   comparison in this document is on stable footing even though the v1↔v2 comparison is not.

---

## 13. Bugs fixed, v1 → v2 → v3

| Bug | Generation found in | Impact | Status |
|---|---|---|---|
| Station-composition jumps in district series | v1 | ~94 m artificial swings; distorted every metric | Fixed v2 (§3) |
| Predicting level instead of change | v1 | Model worse than persistence | Fixed v2 (§4) |
| 11.6M params vs ~156 independent samples | v1 | Memorised by epoch 1–2 | Fixed v2 (§5) — but see next row |
| **Reported parameter count wrong by ~6×** (205K claimed, 1.23M actual) | v2 | "56× capacity cut" claim was false | Fixed v3 (§7.1) |
| No baseline in any metric | v1 | Total failure went undetected for 3 model generations | Fixed v2 (§6) |
| Anchor drawn from interpolated series (future leakage into anchor) | v1 | Future data leaking into the anchor | Fixed v2 |
| **Sentinel monthly composites stamped at month-start + interpolated** | v2 | Up to 15 days of future NDVI/VV/VH leaked into every daily value | Fixed v3 (§7.2) |
| **ET/soil moisture 0–25% real data in folds 4–6** | v2 | Half the evaluation ran on climatology dressed as a feature | Fixed v3 (§7.3, ERA5-Land) |
| **Conformal calibration fitted on the test set it calibrated** | v2 | "80% coverage" was a mathematical identity, not a result | Fixed v3 (§7.4) |
| **Single pooled `delta_sd` across all 30 horizons** | v2 | Loss dominated by long horizons; near-zero gradient incentive at t+7 | Fixed v3 (§8.2) |
| No floor under per-district model failure | v2 | Aurangabad MASE 2.84 at t+7, no floor | Fixed v3 (§8.3, shrinkage) |
| No baseline beyond persistence | v2 | Couldn't tell if TFT beat a simpler model | Fixed v3 (§8.5) |
| No significance testing | v1, v2 | "Beats persistence" was a point estimate with no uncertainty | Fixed v3 (§8.8) |

---

## 14. How to run v3

1. **Data first** (`Data Collection/` then `Data Preprocessing/`, in order):
   ```
   cd "Data Collection"
   python era5_land.py                  # needs: earthengine authenticate (one-time)
   cd "../Data Preprocessing"
   python preprocess_groundwater.py
   python preprocess_env_features.py    # check the observed-rate-by-year table it prints
   python merge_master.py               # writes master_dataset.csv + district_data_quality.csv
   ```
2. Colab → **Runtime → Change runtime type → T4 GPU**. The notebook hard-stops without a GPU.
3. Upload `tft_model_v3.py` and the rebuilt `master_dataset.csv` (55 columns, includes `data_tier`).
4. Optionally upload v2's `tft_metrics_per_district.csv`, renamed to
   `tft_metrics_per_district_v2.csv`, to switch on the automatic v2→v3 regression gate.
5. Run all cells. **Read the data-quality report cell before the results cells** — if the covariate
   reality table shows any feature below 50% real data in any fold, stop and fix the data first.

Full design rationale for every v3 change: `V3_PLAN.md` (same directory).

---

## 15. What's next

| Priority | Item | Note |
|---|---|---|
| 1 | Investigate the coverage undershoot (§9) | Calibration-block-vs-test distribution shift; consider a longer or seasonally-matched calibration window, or an adaptive/weighted conformal method |
| 2 | Reservoir storage (`cwc_reservoir.py`) | Needs a free data.gov.in key. Still the main uncollected physical driver, especially for Marathwada |
| 3 | Re-run with the v2→v3 regression gate populated | Confirm no district degraded >10% once both runs are on file, not just the 7-district spot-check in §11 |
| 4 | Scale the seed ensemble / capacity grid if compute allows | v3's defaults are deliberately modest (3 seeds, no capacity ablation) since the vectorised model made a 6.5-minute run possible — there is headroom to spend more compute if it's available |
| 5 | **Phase D — GAT spatial layer** | **Still on hold.** Held per the HESS 2026 finding (`V3_PLAN.md` §1 M13): globally-trained groundwater models showed no systematic advantage over per-well models across 2,951 German wells, and dynamic similarity mattered far more than data volume. Revisit only after the data-tier C districts have more stations, not before. |

---

*v1/v2 sections written 2026-08-15 from the v1 and v2 Colab T4 run logs. v3 sections added 2026-08-15
from the real 6-fold × 3-seed Colab T4 run (6.5 min total). Full diagnosis and design rationale for
every fix: `TFT_IMPROVEMENT_PLAN.md` (repo root, v1→v2) and `V3_PLAN.md` (this directory, v2→v3).*
