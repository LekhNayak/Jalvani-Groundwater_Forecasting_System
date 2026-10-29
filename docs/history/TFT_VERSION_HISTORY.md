# TFT Production v1 → v2 → v3 → v4: What Changed and Why

**v1** = `09_tft_colab_production.ipynb` — full 6-fold run on Colab T4, ~3 hours, completed 2026-08-14
**v2** = `09_tft_colab_production_v2.ipynb` — rebuilt 2026-08-15 after the v1 post-mortem; full 6-fold T4 run completed
**v3** = `09_tft_colab_production_v3.ipynb` — rebuilt 2026-08-15 after a full audit of v2; full 6-fold, 3-seed T4 run completed in **6.5 minutes**
**v4** = `09_tft_colab_production_v4.ipynb` + `tft_v4.py` — rebuilt target, fixed evaluation, seasonal-naive baseline, TFT-led ensemble; Colab T4 run 2026-10-03 (v4 section near the end)
**v4.1** = `09_tft_colab_production_v4_1.ipynb` — improvement run + simple per-district results; Colab T4 run 2026-10-29 (54 min), **best version so far** (v4.1 section at the end; results in `04_results/tft_v4_1/`)

*v1 and v2 ran on different master datasets (v2's groundwater layer was rebuilt), so raw RMSE is not
directly comparable between them. v2 and v3 ran on the **same district set and the same headline
21-district exclusion list**, so their numbers are a fair apples-to-apples comparison — see §11.*

---

## 0. Scores by version (summary, updated 2026-10-29)

**How to read.** MASE is model error ÷ the error of "no change" (persistence). "vs seasonal" is model
error ÷ the error of the seasonal-naive forecast (today's level + the district's usual change for this
time of year, from past data only). Below 1 is better for both. Coverage of the 80% interval should be
about 80%. All values are t+7 / t+14 / t+30.

**The versions were not all measured the same way.** The target data, test folds and scoring changed
between them. Table A is the history as each version reported it. Table B is the only like-for-like
comparison.

### Table A — every version as originally reported

| Version | Model | Test set | MASE vs no-change | vs seasonal | MAE (m) | MAPE (%) | 80% coverage (%) |
|---|---|---|---|---|---|---|---|
| **v1** (`04_results/tft_prod_v1/`) | TFT, 11.6M params, absolute-level target | old target, 6 folds | **2.12 / 1.37 / 1.04** (median); 2.78 / 2.07 / 1.53 (pooled) | – | 1.28 / 1.37 / 1.53 (median) | 18.5 / 19.5 / 23.2 (median) | 52 / 52 / 53 raw; the reported "80%" was fitted on the test set itself, so invalid |
| **v2** | TFT, change-from-anchor target | rebuilt target, 6 folds | 1.004 / 0.940 / 0.827 (median) | – | – | – | 76 / 71 / 64 raw; the reported "80%" was invalid (same reason) |
| **v3** | TFT + shrinkage toward no-change | same target, 6 folds, 21 districts | 0.958 / 0.909 / 0.834 | – | 0.53 / 0.75 / 1.11 | 9.7 / 13.6 / 20.1 | 70 / 66 / 63 |
| v3 run | LightGBM (v3 baseline ladder) | same | 0.949 / 0.937 / 0.857 | – | – | – | – |
| **v4** | **TFT alone** | new target, all 6 folds | 0.937 / 0.877 / 0.773 | 1.003 / 1.002 / 0.987 | 0.45 / 0.63 / 0.93 | 6.2 / 8.8 / 13.2 | 74 / 69 / 65 |
| **v4** | **Ensemble** (TFT + LightGBM + seasonal-naive + no-change) | new target, all 6 folds | **0.911 / 0.850 / 0.750** | **0.975 / 0.972 / 0.958** | **0.43 / 0.61 / 0.91** | **6.0 / 8.6 / 12.9** | **81 / 80 / 78** |
| **v4, locked test** | TFT alone | folds 5-6 only | 0.910 / 0.834 / 0.697 | 1.017 / 1.008 / 0.992 | 0.34 / 0.44 / 0.62 | 4.6 / 6.0 / 7.7 | 74 / 70 / 67 |
| **v4, locked test** | **Ensemble** | folds 5-6 only | **0.865 / 0.788 / 0.664** | **0.967 / 0.952 / 0.945** | **0.32 / 0.42 / 0.59** | **4.4 / 5.7 / 7.4** | **81 / 82 / 81** |
| v4 run | seasonal-naive baseline | all 6 folds | 0.934 / 0.875 / 0.783 | 1.000 | – | – | – |
| v4 run | LightGBM alone | all 6 folds | 0.914 / 0.864 / 0.759 | 0.978 / 0.987 / 0.970 | – | – | – |
| **v4.1** | **TFT alone** | new target, all 6 folds | 0.935 / 0.876 / 0.764 | 1.000 / 1.002 / 0.976 | 0.44 / 0.63 / 0.92 | 6.2 / 8.7 / 13.0 | 73 / 69 / 64 |
| **v4.1** | **Ensemble** | new target, all 6 folds | **0.903 / 0.830 / 0.712** | **0.967 / 0.949 / 0.909** | **0.43 / 0.60 / 0.86** | **6.0 / 8.4 / 12.4** | **80 / 78 / 78** |
| **v4.1, locked test** | TFT alone | folds 5-6 only | 0.872 / 0.803 / 0.675 | 0.976 / 0.971 / 0.960 | 0.33 / 0.43 / 0.60 | 4.4 / 5.8 / 7.6 | 74 / 70 / 66 |
| **v4.1, locked test** | **Ensemble (best so far)** | folds 5-6 only | **0.839 / 0.763 / 0.616** | **0.938 / 0.922 / 0.877** | **0.31 / 0.41 / 0.55** | **4.3 / 5.6 / 7.1** | **81 / 80 / 80** |
| v4.1 run | LightGBM alone (new features) | all 6 folds | 0.903 / 0.832 / 0.703 | 0.967 / 0.951 / 0.897 | 0.43 / 0.60 / 0.85 | 6.1 / 8.6 / 12.3 | 73 / 69 / 61 |
| v4.1 run | LightGBM alone (new features) | folds 5-6 only | 0.833 / 0.770 / 0.612 | 0.932 / 0.931 / 0.871 | 0.31 / 0.41 / 0.55 | 4.3 / 5.7 / 7.2 | 75 / 72 / 67 |

Notes on Table A:
- **v1** was scored on the old, noisy groundwater series. Its MASE was computed afterwards from the `04_results/tft_prod_v1/` predictions. §1 below quotes ~2.96 / 2.15 / 1.56 from a slightly different post-hoc calculation; both say v1 lost to "no change".
- **v1 and v2 coverage:** the reported 80% was fitted on the same test rows it was applied to, so it is not a result.
- **v3** also had a window-end-date bug: some t+7 test targets fell inside the validation period, and the last 23 days of each season were never scored. No v2 or v3 output files exist on disk.
- **MAPE across versions is not comparable.** v1's figure is a median over districts on the old target; v3 and v4 use pooled MAPE with a 0.5 m floor.
- The old ARIMA / XGBoost / LSTM baselines in `03_models/baselines_phase1/` used an older dataset and a single Feb-Apr 2025 test window, so they cannot go in this table.

### Table B — like-for-like: dev folds 1-4, identical windows, scoring and seasonal-naive baseline (v4 Colab screening)

| Set-up | Model | MASE vs no-change | vs seasonal | MAE (m) | Coverage (%) |
|---|---|---|---|---|---|
| v3 with fixed evaluation, **old** target | TFT | 0.982 / 0.951 / 0.889 | 1.008 / 1.032 / 1.040 | 0.57 / 0.83 / 1.31 | 74 / 69 / 64 |
| v3 with fixed evaluation, **new** target | TFT | 0.974 / 0.932 / 0.862 | 1.028 / 1.047 / 1.066 | 0.51 / 0.75 / 1.18 | 72 / 68 / 63 |
| + v4 calibration + ensemble | TFT | 0.977 / 0.938 / 0.849 | 1.031 / 1.055 / 1.050 | 0.51 / 0.75 / 1.16 | 71 / 66 / 63 |
| | Ensemble | 0.927 / 0.876 / 0.807 | 0.978 / 0.985 / 0.998 | 0.48 / 0.70 / 1.10 | 78 / 77 / 73 |
| **+ residual-on-seasonal target (v4 chosen)** | TFT | 0.944 / 0.880 / 0.789 | 0.996 / 0.989 / 0.976 | 0.49 / 0.71 / 1.08 | 74 / 69 / 65 |
| | **Ensemble** | **0.922 / 0.860 / 0.769** | **0.973 / 0.966 / 0.951** | **0.48 / 0.69 / 1.05** | **81 / 81 / 77** |
| + loss weighting on t+7/14/30 | Ensemble | 0.921 / 0.862 / 0.780 | 0.972 / 0.969 / 0.964 | 0.48 / 0.69 / 1.07 | 82 / 81 / 78 |
| without ERA5 and Sentinel inputs | Ensemble | 0.926 / 0.872 / 0.789 | 0.978 / 0.980 / 0.975 | 0.48 / 0.70 / 1.08 | 80 / 78 / 77 |
| **v4.1 screening** (same folds and windows; MAE not saved): | | | | | |
| v4 set-up + shrink toward seasonal-naive (`ref`) | TFT | 0.943 / 0.880 / 0.769 | 0.996 / 0.990 / 0.951 | – | 74 / 70 / 67 |
| | Ensemble | 0.922 / 0.860 / 0.769 | 0.973 / 0.966 / 0.951 | – | 81 / 81 / 77 |
| + new LightGBM (`tab2`) | Ensemble | 0.924 / 0.848 / 0.744 | 0.976 / 0.953 / 0.919 | – | 82 / 80 / 77 |
| + refit pass (`tab2_refit`) | Ensemble | 0.920 / 0.845 / 0.743 | 0.971 / 0.950 / 0.918 | – | 79 / 78 / 78 |
| **+ stride-1 network sampling (`s1_tab2`, v4.1 chosen)** | TFT | 0.937 / 0.880 / 0.783 | 0.989 / 0.990 / 0.967 | – | 73 / 69 / 64 |
| | **Ensemble** | **0.907 / 0.841 / 0.739** | **0.958 / 0.945 / 0.913** | – | **80 / 79 / 78** |
| stride-1 + refit (`s1_tab2_refit`) | Ensemble | 0.932 / 0.851 / 0.746 | 0.983 / 0.956 / 0.922 | – | 79 / 78 / 77 |
| d_model 48 + refit (`d48_tab2_refit`) | TFT | 0.986 / 0.940 / 0.814 | 1.041 / 1.057 / 1.006 | – | 73 / 69 / 65 |

### What the scores say
- **Best so far: the v4.1 Ensemble.** On the locked test it is 16-38% better than "no change" and 6-12% better than the seasonal forecast. Every one of those six comparisons is significant (p = 0.003, Holm-adjusted). The 80% intervals cover 80-81% and are narrower than v4's.
- **v4 to v4.1:** better at every horizon on both test sets. The largest gain is at t+30, where the locked score vs seasonal moved from 0.945 to 0.877.
- **v1 to v4.1:** from worse than doing nothing (2.12 at t+7) to clearly better (0.90 over all folds, 0.84 on the locked test).
- **The TFT alone:**
  - on the locked test it now beats the seasonal forecast by 2-4%, though this is significant only at t+7;
  - over all 6 folds it still ties the seasonal forecast;
  - on a fair comparison, v3's TFT was worse than the seasonal forecast (1.03-1.07).
- **Most of the v4.1 gain came from the new LightGBM.** On its own it is now about as accurate as the Ensemble. The Ensemble's advantage is its honest intervals (80% coverage against LightGBM's 61-75%). The v4.1 section explains this.
- **Quote MASE (and "vs seasonal"), not MAPE,** when comparing versions.

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

**v3 fix:** re-sourced to **ERA5-Land** (`01_data_collection/era5_land.py`, ~5-day latency, no
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

`TFT_VERSION_HISTORY.md` §10 (v2) listed diagnosing "Aurangabad, Bhandara, Nashik, Jalgaon, Kolhapur,
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

1. **Data first** (`01_data_collection/` then `02_preprocessing/`, in order):
   ```
   cd 01_data_collection
   python era5_land.py                  # needs: earthengine authenticate (one-time)
   cd ../02_preprocessing
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
every fix: `docs/history/TFT_IMPROVEMENT_PLAN.md` (repo root, v1→v2) and `V3_PLAN.md` (this directory, v2→v3).*

---

# v4 (2026-10-03) — cleaner target, honest evaluation, seasonal baseline, TFT-led ensemble

**v4** = `09_tft_colab_production_v4.ipynb` + `tft_v4.py` (+ `tft_model_v3.py`, unchanged network).
Run on a Colab T4: dev screening on folds 1-4 (3 seeds), final run on all 6 folds (5 seeds),
folds 5-6 (Oct 2025 -> 4 Jun 2026) reported separately as the untouched test.
Data: `master_dataset_v4target.csv` (25 districts, 31,275 rows, 64 cols).

> **Note on `04_results/tft_prod_v1/`:** that folder holds the **v1** run (11.6M-parameter, absolute-level
> target, pre-fix groundwater series; median MASE 2.12 / 1.37 / 1.04). It is not the best model.

## What changed vs v3

| # | Change | Where |
|---|---|---|
| 1 | **Groundwater target rebuilt.** Placeholder (+1.000) and stuck-logger readings removed; two-sided time-based Hampel; dry-well plateau check; step detection against the last accepted value; 12 wells wrongly discarded by the sign rule recovered; district value = median of offset/amplitude-adjusted wells so a well joining or leaving no longer moves the series; no ffill/bfill. Jumps > 2 m: 329 -> 193. | `02_preprocessing/preprocess_groundwater.py`, `audit_groundwater_target.py` |
| 2 | **Evaluation fixed.** Windows keyed by issue date; a window is in a split if any target date is; every scored row has `anchor_date` and `target_date`; all models scored on identical keys (asserted). Removes v3's window-end-date bug (some t+7 targets sat in the val block; the last 23 days of each season were never scored). | `tft_v4.py` |
| 3 | **Seasonal-naive (SN) baseline** - anchor + the district's typical h-day change for this day of year (past data only; bandwidth 30 d, shrinkage 20, damping 1.0 tuned on SN's own dev-fold skill). Every result is now reported against persistence **and** SN. | `tft_v4.py`, `tune_snaive.py` |
| 4 | **Residual-on-SN target.** The TFT predicts what SN gets wrong; output 0 = SN. | `target_mode='snaive'` |
| 5 | **Calibration.** Season-weighted pool of earlier folds' out-of-sample errors, per-district normalised scores, each tail calibrated separately, adaptive conformal with delayed feedback. | `calib_mode='v4'` |
| 6 | **Ensemble.** Simplex-weighted blend of persistence, SN, TFT and LightGBM, weights fitted on the pool, intervals from the TFT's spreads. Both "TFT" and "Ensemble" are reported. | `combine='stack'` |

The network (`tft_model_v3.py`, ~222k parameters) is unchanged.

## Dev-screening results (folds 1-4, identical windows; pooled rMAE; lower is better)

rMAE vs persistence / rMAE vs seasonal-naive / 80%-interval coverage (%), t+7 | t+14 | t+30:

| Run | Model | vs persistence | vs seasonal-naive | coverage |
|---|---|---|---|---|
| v3-fixed, **old** target | TFT | 0.982 / 0.951 / 0.889 | 1.008 / 1.032 / 1.040 | 74 / 69 / 64 |
| v3-fixed, new target | TFT | 0.974 / 0.932 / 0.862 | 1.028 / 1.047 / 1.066 | 72 / 68 / 63 |
| + v4 calibration + ensemble | TFT | 0.977 / 0.938 / 0.849 | 1.031 / 1.055 / 1.050 | 71 / 66 / 63 |
| | Ensemble | 0.927 / 0.876 / 0.807 | 0.978 / 0.985 / 0.998 | 78 / 77 / 73 |
| + residual-on-SN target (**chosen**) | TFT | 0.944 / 0.880 / 0.789 | 0.996 / 0.989 / 0.976 | 74 / 69 / 65 |
| | **Ensemble** | **0.922 / 0.860 / 0.769** | **0.973 / 0.966 / 0.951** | **81 / 81 / 77** |
| + loss weight on t+7/14/30 | Ensemble | 0.921 / 0.862 / 0.780 | 0.972 / 0.969 / 0.964 | 82 / 81 / 78 |
| - ERA5 and Sentinel inputs | Ensemble | 0.926 / 0.872 / 0.789 | 0.978 / 0.980 / 0.975 | 80 / 78 / 77 |

What each step contributed (dev folds):
* **Cleaning the target:** small gain vs persistence (t+30 0.889 -> 0.862); MAE fell 0.57 -> 0.51 m at t+7 and 1.31 -> 1.18 m at t+30 (a cleaner series also strengthens persistence, so the ratio understates it).
* **The v3-style TFT was worse than the seasonal baseline** (rMAE vs SN 1.03-1.07). This was previously hidden because v3 only compared against persistence.
* **Ensemble alone** (same networks): 0.977 -> 0.927 at t+7, 0.938 -> 0.876 at t+14, 0.849 -> 0.807 at t+30, and coverage 71/66/63 -> 78/77/73.
* **Residual-on-SN target:** the TFT alone goes 0.977 -> 0.944 (t+7), 0.938 -> 0.880 (t+14), 0.849 -> 0.789 (t+30) and now ties/just beats SN; for the Ensemble the extra gain is smaller (it already contains SN).
* **Loss weighting** did not help. **Dropping ERA5/Sentinel hurt slightly** (mean rMAE vs SN 0.978 vs 0.963), so those inputs carry a small amount of information - an earlier ridge probe suggesting otherwise did not carry over to the TFT.
* Differences between the top three candidates (0.963 / 0.968 / 0.978) are small relative to seed noise (3 seeds); the choice rule (lowest mean rMAE vs SN) was fixed in advance.

## Final run (set-up `snaive` + Ensemble, 5 seeds, new target)

Pooled over the 21 headline districts (single-well Beed, Latur, Nanded, Raigad, Thane excluded; Latur is tier D). Headline = **Ensemble (calibrated)**, with the TFT alone beside it.

| Test set | Model | Horizon | MAE (m) | MAPE (%)* | vs persistence | vs seasonal | 80% coverage | districts < 1 |
|---|---|---|---|---|---|---|---|---|
| **Locked, folds 5-6** | Ensemble | t+7 | 0.323 | 4.4 | **0.865** | **0.967** | 80.6 | 21/21 |
| | | t+14 | 0.419 | 5.7 | **0.788** | **0.952** | 81.5 | 21/21 |
| | | t+30 | 0.593 | 7.4 | **0.664** | **0.945** | 81.2 | 20/21 |
| | TFT alone | t+7 | 0.340 | 4.6 | 0.910 | 1.017 | 74.0 | 20/21 |
| | | t+14 | 0.444 | 6.0 | 0.834 | 1.008 | 70.1 | 20/21 |
| | | t+30 | 0.623 | 7.7 | 0.697 | 0.992 | 66.8 | 20/21 |
| **All 6 folds** | Ensemble | t+7 | 0.433 | 6.0 | 0.911 | 0.975 | 80.8 | 20/21 |
| | | t+14 | 0.610 | 8.6 | 0.850 | 0.972 | 80.3 | 20/21 |
| | | t+30 | 0.907 | 12.9 | 0.750 | 0.958 | 77.7 | 20/21 |
| | TFT alone | t+7 | 0.445 | 6.2 | 0.937 | 1.003 | 74.1 | 20/21 |
| | | t+14 | 0.629 | 8.8 | 0.877 | 1.002 | 69.4 | 19/21 |
| | | t+30 | 0.934 | 13.2 | 0.773 | 0.987 | 65.0 | 20/21 |

\* MAPE with |actual| floored at 0.5 m; pooled, not a median over districts.

Significance (district-cluster bootstrap, Holm-adjusted; p = 0.003 is the floor of a 2,000-draw bootstrap):
the Ensemble beats persistence at all 3 horizons and the seasonal baseline at all 3 horizons on the locked
test (t+30 vs seasonal p = 0.018); the **TFT alone does not beat the seasonal baseline significantly**.

Dev-fold ratios (Ensemble, 0.973 / 0.966 / 0.951 vs SN) held up on the locked folds (0.967 / 0.952 / 0.945): no sign of tuning on the test set.

## Honest caveats

1. **The TFT alone ties the seasonal baseline** (rMAE vs SN about 1.00). The gain over SN (3-5%) comes from the ensemble. The TFT is the primary model and supplies the interval width; the Ensemble is the TFT-led final forecaster. Report both.
2. **The best-vs-worst district gap did not shrink** (worst-5 minus best-5 mean rMAE vs persistence, dev folds: v3-fixed 0.120 / 0.186 / 0.286 -> Ensemble 0.146 / 0.243 / 0.371), because the best districts improved most. Worst districts still only tie persistence (about 0.94-0.98).
3. **Folds 5-6 contain no monsoon**; the locked numbers are for the easier seasons. The monsoon fold 1 is the weakest (Ensemble 0.915 at t+30).
4. **Intervals still miss low** (11-17% of targets below q10 vs 7% above q90): the actual level often ends up *deeper* than forecast, i.e. dry-season / post-monsoon **declines are under-forecast** (an earlier draft said 'under-predicts recharge rises' - that was the wrong direction). Part of the cause for the TFT-alone rows: v4.0's shrinkage pulled the residual-on-SN forecast toward no-change, removing part of the seasonal decline (fixed in v4.1).
5. The TFT alone has weak coverage (65-74%); only the Ensemble reaches about 80%.
6. The ranking among the top three candidates is within seed noise.

*v4 section written 2026-10-03 from the Colab run's `ablations.csv`, `dev_screening_summary.csv` and `RESULTS_TO_QUOTE.csv`.*

---

# Appendix — model-file history (`tft_model.py` -> `tft_model_v3.py`) and what is in each file

Only **two** network files exist. The architecture was changed once (v1 -> v3); v2 and v4 are
configuration / pipeline changes around an existing network file.

| Version | Network file | Code change to the network? | What differs |
|---|---|---|---|
| v1 | `tft_model.py` (487 lines) | original | Variable-selection network (VSN) is a Python loop: one `Linear(1 -> d_model)` and one full GRN per feature, per forward pass. q10 / q50 / q90 are three independent linear outputs and can cross (3.2% of v1 predictions did). Plain `QuantileLoss`; the masked loss was defined in the notebook. Three layers (`static_proj`, `past_proj`, `future_proj`, ~17k parameters) are defined but never used. Config: d_model 192, 3 LSTM layers, 8 heads, ~11.6M parameters. |
| v2 | `tft_model.py` (same) | **none** | Notebook config only: d_model 64, 1 LSTM layer, 4 heads, dropout 0.25, weight decay 1e-2, encoder 60. The notebook printed "205,173 parameters"; the real count was 1,231,159 (the VSN GRN was larger than the rest of the network). |
| v3 | `tft_model_v3.py` (475 lines) | **yes** | (1) `BatchedGRN`, `BatchedFeatureProjection`, `BatchedFeatureCompression`: the VSN math is vectorised (identical computation, batched; 6-fold run ~3 h -> 6.5 min). (2) `d_select` (8): each variable is compressed before the selector sees it, plus `grn_hidden_factor` (0.5) narrowing the per-variable GRN - the real parameter count at d_model 32 is ~222k (5.5x below v2's real size, ~50x below v1). (3) `monotone_quantiles=True`: q50 plus two softplus spreads, so q10 <= q50 <= q90 by construction. (4) the three unused projection layers are removed. (5) `MaskedQuantileLoss(quantiles, q50_boost, horizon_weights)` now lives in the model file. |
| v4 | `tft_model_v3.py` (same) | **none** | Imported unchanged by the new `tft_v4.py`. All v4 changes are around the network: target (change from the last reading, then the residual on seasonal-naive), cleaned groundwater data, windowing / evaluation, calibration and the ensemble. Details in the v4 section above. |

Unchanged in every version: the TFT structure itself - static covariate encoder, LSTM encoder and
decoder with gated skip connections, static enrichment, interpretable multi-head attention, the
position-wise GRN and the quantile head on decoder positions.

Pipeline files by version (training / evaluation code, not network code):

| Version | Notebook / code | Key characteristics |
|---|---|---|
| v1 | `09_tft_colab_production.ipynb` (= `09_tft_colab.ipynb`) | absolute-level target, stride 1, in-sample conformal, no baselines, windows keyed by decoder-end date |
| v2 | `09_tft_colab_production_v2.ipynb` | delta-from-anchor target, panel + anomaly + drift groundwater filter, stride 7, MASE vs persistence |
| v3 | `09_tft_colab_production_v3.ipynb` + `tft_model_v3.py` | per-horizon scaling, shrinkage toward persistence, honest calibration block, monotone head, baseline ladder, significance tests, 3 seeds; still windows keyed by decoder-end date |
| v4 | `09_tft_colab_production_v4.ipynb` + `tft_v4.py` (+ `run_v4.py`, `tune_snaive.py`) | issue-date windows, seasonal-naive baseline and residual target, rebuilt groundwater target, season-weighted adaptive conformal, ensemble, locked folds 5-6 |

*Appendix verified against the code on 2026-10-03.*

---

# v4.1 (built 2026-10-26, Colab T4 run 2026-10-29) — improvement run + simple results

Notebook `09_tft_colab_production_v4_1.ipynb`, pipeline `tft_v4.py` v4.5, guide `RESULTS_GUIDE.md`.
The v4 notebook is unchanged.

What v4.1 changes:

| # | Change | Why |
|---|---|---|
| 1 | The TFT's shrinkage pulls toward **seasonal-naive** instead of no-change (residual-on-SN target) | v4.0 shrank the seasonal part of the forecast too, biasing it high (TFT-alone rows only; the Ensemble uses the raw TFT) |
| 2 | **Stronger LightGBM** (`tab_features='v2'`): + day-of-year, SN change, change scale, level and year-on-year features; stride-1 windows; the cal-block model is trained to the end of the early-stopping block and the test model **up to the day before the test period** | LightGBM is the strongest single component but could not see the time of year and was trained 3 months stale |
| 3 | **Refit pass** (`refit=True`): each network is retrained on train + early-stopping windows for pass 1's best epoch, replaying pass 1's learning-rate path | Networks stopped learning 3 months before each test period |
| 4 | Candidates: stride-1 network sampling; `d_model` 48 | Not tested before |
| 5 | Sweeps of per-district stacking strength and interval adaptivity (pre-registered pick rules) | Evenness across districts; interval tails |
| 6 | 10 seeds in the final run | Seed averaging |
| 7 | Networks, tabular models and post-processing cached separately | LightGBM / post-processing changes never retrain the networks; 7 candidates need only 3 network trainings |
| 8 | **Simple v1-style outputs** for all 24 districts, for both the Ensemble and the TFT (`export_simple`, `export_shared`), plus Beed/Latur "no usable data" rows, training curves, VSN importance, optional v1 comparison | Results in the familiar format |

Selection uses dev folds 1-4 only; folds 5-6 were viewed once in v4.0, which is stated in the write-up.

## v4.1 results (Colab T4, 2026-10-29, 54 min)

Files: `04_results/zips/tft_v4_1_outputs.zip`, extracted to `04_results/tft_v4_1/` (guide: `RESULTS_GUIDE.md`).
Pipeline 4.5, data `master_dataset_v4target.csv` (md5 e035dc26), 10 seeds.

### How the set-up was chosen (dev folds 1-4, 3 seeds, pre-registered rule)
Rule: the lowest mean Ensemble error ÷ seasonal-naive error. A simpler candidate wins if it is within 0.002.

| Candidate | What it adds | Ensemble mean vs seasonal | TFT mean vs seasonal |
|---|---|---|---|
| `ref` | v4 set-up + shrink toward seasonal-naive | 0.963 | 0.979 |
| `tab2` | + new LightGBM | 0.949 | 0.979 |
| `refit` | + refit pass | 0.969 | 0.990 |
| `tab2_refit` | + new LightGBM + refit | 0.946 | 0.990 |
| **`s1_tab2`** | **+ new LightGBM + stride-1 network sampling** | **0.939 (chosen)** | **0.982** |
| `s1_tab2_refit` | + all three | 0.954 | 0.992 |
| `d48_tab2_refit` | bigger network (d_model 48) | 0.961 | 1.034 |

What the screening shows:
- **The new LightGBM helped most.** It took the Ensemble from 0.963 to 0.949.
- **Stride-1 sampling helped further** (0.939). Each epoch draws 1/7 of all daily windows, so the cost is the same as stride 7.
- **The refit pass did not help.** It made the TFT worse (0.990 vs 0.979) and did not improve the Ensemble, so the chosen set-up uses pass 1.
- **The bigger network was worse.**
- **Shrinking toward seasonal-naive (`ref`) improved the TFT alone at t+30**, from 0.976 to 0.951 vs seasonal, compared with the v4 set-up in Table B.

Post-processing sweeps on `s1_tab2`, using the same rule:
- **Stacking strength.** `stack_prior_n` 100 / 250 / 500 / 1000 gave 0.943 / 0.939 / 0.939 / 0.939, so the default of 500 was kept.
- **Interval adaptivity.** `aci_gamma` 0.005 / 0.01 / 0.02 / 0.05 leaves the medians unchanged.
  - Ensemble t+30 coverage was 78.4 / 78.4 / 75.0 / 68.1%.
  - The rule keeps 0.01. 0.005 would have given a slightly better interval score (4.57 vs 4.64).

### Final run (all 6 folds, 10 seeds; 21 headline districts)

| Test set | Model | Horizon | MAE (m) | vs no-change | vs seasonal | Coverage | below q10 / above q90 | districts beating no-change |
|---|---|---|---|---|---|---|---|---|
| **Locked 5-6** | **Ensemble** | t+7 | 0.314 | **0.839** | **0.938** | 81.1 | 10.7 / 8.2 | 21/21 |
| | | t+14 | 0.406 | **0.763** | **0.922** | 80.1 | 12.7 / 7.1 | 21/21 |
| | | t+30 | 0.550 | **0.616** | **0.877** | 80.0 | 14.8 / 5.2 | 21/21 |
| | TFT alone | t+7 | 0.326 | 0.872 | 0.976 | 74.2 | 14.3 / 11.5 | 20/21 |
| | | t+14 | 0.428 | 0.803 | 0.971 | 70.0 | 18.8 / 11.2 | 20/21 |
| | | t+30 | 0.602 | 0.675 | 0.960 | 65.7 | 25.1 / 9.2 | 20/21 |
| **All 6 folds** | **Ensemble** | t+7 | 0.429 | 0.903 | 0.967 | 79.9 | 10.7 / 9.5 | 21/21 |
| | | t+14 | 0.595 | 0.830 | 0.949 | 78.4 | 12.8 / 8.8 | 20/21 |
| | | t+30 | 0.861 | 0.712 | 0.909 | 78.1 | 13.2 / 8.6 | 20/21 |
| | TFT alone | t+7 | 0.444 | 0.935 | 1.000 | 73.0 | 14.1 / 12.9 | 19/21 |
| | | t+14 | 0.629 | 0.876 | 1.002 | 68.6 | 17.2 / 14.2 | 19/21 |
| | | t+30 | 0.924 | 0.764 | 0.976 | 64.3 | 20.9 / 14.8 | 20/21 |

**Significance** (district-cluster bootstrap and Diebold–Mariano, Holm-adjusted):
- **The Ensemble beats no-change and seasonal-naive at all three horizons on the locked test**, with p = 0.003, the floor of the bootstrap. Its locked 95% confidence interval vs seasonal at t+30 is 0.815–0.927.
- **The TFT alone beats seasonal-naive on the locked test only at t+7** (p = 0.003). At t+14 and t+30 it does not (p = 0.063). Over all 6 folds it does not beat seasonal-naive at t+7 or t+14.

**Dev vs locked folds** (no sign of overfitting the selection): the Ensemble scored 0.958 / 0.945 / 0.913 vs seasonal on the dev folds and **0.938 / 0.922 / 0.877** on the locked folds.

### v4.0 -> v4.1 (Ensemble, calibrated)

| Test set | Horizon | MAE (m) | vs no-change | vs seasonal | Coverage | Interval width (m) |
|---|---|---|---|---|---|---|
| Locked 5-6 | t+7 | 0.323 -> **0.314** | 0.865 -> **0.839** | 0.967 -> **0.938** | 80.6 -> 81.1 | 1.15 -> **1.10** |
| | t+14 | 0.419 -> **0.406** | 0.788 -> **0.763** | 0.952 -> **0.922** | 81.5 -> 80.1 | 1.60 -> **1.52** |
| | t+30 | 0.593 -> **0.550** | 0.664 -> **0.616** | 0.945 -> **0.877** | 81.2 -> 80.0 | 2.45 -> **2.22** |
| All 6 folds | t+7 | 0.433 -> **0.429** | 0.911 -> **0.903** | 0.975 -> **0.967** | 80.8 -> 79.9 | 1.63 -> **1.52** |
| | t+14 | 0.610 -> **0.595** | 0.850 -> **0.830** | 0.972 -> **0.949** | 80.3 -> 78.4 | 2.74 -> **2.50** |
| | t+30 | 0.907 -> **0.861** | 0.750 -> **0.712** | 0.958 -> **0.909** | 77.7 -> 78.1 | 4.15 -> **3.62** |

- **Accuracy and sharpness improved everywhere.** Every MAE and ratio is lower and every interval is narrower, while coverage stays close to 80%.
- **The gain grows with horizon.** At t+30, MAE fell 7% on the locked test and 5% over all folds.
- **The TFT alone also improved on the locked test:** vs seasonal went from 1.017 / 1.008 / 0.992 to 0.976 / 0.971 / 0.960.

### Per season (Ensemble, all 6 folds, `simple/ensemble/ensemble_metrics_per_season.csv`)

| Season (folds) | vs no-change t+7 / t+14 / t+30 | vs seasonal t+7 / t+14 / t+30 |
|---|---|---|
| Monsoon (1, 4) | 0.953 / 0.882 / 0.751 | 0.979 / 0.960 / 0.902 |
| Post-monsoon (2, 5) | 0.857 / 0.777 / 0.659 | 0.936 / 0.912 / 0.850 |
| Dry (3, 6) | 0.871 / 0.794 / 0.691 | 0.969 / 0.954 / 0.955 |

- **The monsoon is still the hardest season at t+7 and t+14**, because future rain is unknown. At t+30, though, the monsoon gain over seasonal-naive (0.902) is larger than the dry season's (0.955).
- **Fold 1 (monsoon 2024) is the weakest fold:** 0.973 / 0.953 / 0.924 vs no-change.

### Per district (Ensemble MASE, all 6 folds, `simple/ensemble/ensemble_mase_by_district.csv`)

**Best at t+30:**

| District | MASE |
|---|---|
| Nagpur | 0.527 |
| Nandurbar | 0.541 |
| Amravati | 0.546 |
| Yavatmal | 0.576 |
| Wardha | 0.592 |

**Weakest at t+30:**

| District | MASE | Note |
|---|---|---|
| Pune | 1.128 | the only district that loses to no-change, here and at t+14 (1.050); tier C |
| Nashik | 0.883 | |
| Parbhani | 0.883 | |
| Aurangabad | 0.867 | |
| Dhule | 0.864 | |

**On the locked test, all 21 headline districts beat no-change at every horizon.** In v4, 20/21 did at t+30.

**The gap between the best and worst districts widened again:**

| Test set | v4 worst-5 minus best-5 MASE (t+7 / t+14 / t+30) | v4.1 |
|---|---|---|
| All folds | 0.144 / 0.236 / 0.343 | 0.175 / 0.290 / 0.368 |
| Locked | 0.237 / 0.308 / 0.390 | 0.264 / 0.336 / 0.457 |

The best districts improved most, so the gap grew. The worst five now average 0.87–0.93 on the locked test.

### What drives the v4.1 gain: LightGBM, not the network

Average stacking weights over districts (`FINAL_s1_tab2/stack_weights.csv`):

| Horizon | No-change | Seasonal-naive | TFT | LightGBM |
|---|---|---|---|---|
| t+7 | 0.24 | 0.03 | 0.29 | 0.45 |
| t+14 | 0.13 | 0.03 | 0.30 | 0.54 |
| t+30 | 0.12 | 0.04 | 0.19 | 0.64 |

- **LightGBM alone, with the new features, is about as accurate as the Ensemble:**
  - locked test: 0.833 / 0.770 / 0.612 vs no-change, against the Ensemble's 0.839 / 0.763 / 0.616;
  - all folds: 0.903 / 0.832 / 0.703, against 0.903 / 0.830 / 0.712.
- **But LightGBM's intervals are badly too narrow** (61-75% coverage). The Ensemble keeps LightGBM's accuracy and adds honest intervals, which are the TFT's spread around the blended median.
- **The TFT's share is 20-30%** and its spread sets the interval width.

So, honestly:
- The Ensemble is the forecaster to quote.
- Its point accuracy now comes mainly from the gradient-boosted model fed with seasonal features.
- The TFT mainly contributes the uncertainty (quantile) estimate and a 20-30% share of the median.

### Interval calibration hurts the TFT (found in this run)

Computed from `FINAL_s1_tab2/predictions_daily.csv` (21 headline districts). The interval score is the Winkler score for the 80% interval: width plus 10 × the size of each miss. Lower is better.

| Model | Rows | Dev folds 1-4: coverage / interval score (t+7, t+14, t+30) | Locked 5-6: coverage / interval score |
|---|---|---|---|
| TFT | `TFT` (λ only, no conformal) | 80 / 77 / 71 · 2.82 / 3.98 / 5.72 | 80 / 78 / 77 · 1.78 / 2.21 / 2.89 |
| TFT | `TFT (calibrated)` | 72 / 68 / 64 · 3.07 / 4.67 / 6.87 | 74 / 70 / 66 · 2.01 / 2.49 / 3.56 |
| Ensemble | `Ensemble` (no conformal) | 81 / 78 / 74 · 2.77 / 3.83 / 5.44 | 82 / 81 / 80 · 1.73 / 2.12 / 2.74 |
| Ensemble | `Ensemble (calibrated)` | 79 / 78 / 77 · 2.74 / 4.32 / 6.55 | 81 / 80 / 80 · 1.68 / 2.06 / 3.06 |

**For the TFT, the conformal + ACI step is worse in every fold and at every horizon.**
- It shifts the intervals and mostly narrows them.
- It raises the misses below q10 (on the dev folds, from 8-9% to 14-19%).
- It cuts coverage by 6-12 points, to 64-74%.

So **the TFT-alone's "65-74% coverage" reported since v4 is caused by the calibration step, not by the network.** The network's own quantiles cover 77-80% on the locked test.

**For the Ensemble the step is mixed.**
- At t+30 it lifts dev-fold coverage from 74% to 77%, but at the cost of much wider intervals (mean width 4.3 m vs 2.8 m).
- On the dev folds its interval score is worse at t+14 and t+30.

**Evidence on the cause (not yet confirmed):** in the `aci_gamma` sweep, faster adaptation lowers coverage for *both* models. That points at the adaptive (ACI) update, which:
- is fed about 10 highly correlated daily outcomes per anchor;
- is clipped at alpha ≥ 0.01;
- may be working from a cal-block pool whose TFT errors are smaller than the test errors.

**What to quote now.** This decision rests on the dev folds alone:
- for the TFT alone, quote the un-calibrated `TFT` rows;
- for the Ensemble, keep `Ensemble (calibrated)`.

**Fix for a v4.2 (post-processing only, no retraining needed):**
- add `calib_mode='none'` per model;
- feed ACI one outcome per anchor and band;
- choose calibration on/off per model and horizon on dev folds by interval score.

### Caveats for the write-up
1. Folds 5-6 were viewed once before, in v4.0. v4.1 was chosen on folds 1-4 only, and the dev-to-locked ratios held up.
2. Folds 5-6 contain no monsoon.
3. Most of the v4.1 accuracy gain is from LightGBM, as described above. Report the TFT alone alongside it.
4. The district gap widened, and Pune still loses to no-change at t+14 and t+30 over all folds.
5. Intervals still miss low more often than high (Ensemble, locked t+30: 14.8% below q10, 5.2% above q90). Declines are still under-forecast at long horizons.
6. The refit pass and the d_model-48 network did not help. Both are recorded as negative results.

*v4.1 results section written 2026-10-29 from `RUN_SUMMARY.md`, `RESULTS_TO_QUOTE.csv`, `v4_vs_v4_1.csv`, `dev_screening_summary.csv`, `post_sweep_summary.csv`, `FINAL_s1_tab2*/` and `simple/`.*
