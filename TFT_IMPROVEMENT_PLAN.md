# TFT Generalization — Diagnosis & Improvement Plan

**Created:** 2026-07-04 · **Phases A & B completed:** 2026-07-04 · **v3 post-mortem + v4 rebuild:** 2026-08-15
**Goal:** Make TFT accuracy uniformly good across districts, not just on the easy ones.

> ## 🔴 2026-08-15 — v3 POST-MORTEM: the model was losing to a naive baseline
>
> The first full GPU run of v3 completed (6 folds, ~3 h on a T4). Headline numbers looked
> plausible (median RMSE ~1.8-2.2 m, MAPE ~19-23%) — but benchmarking against a
> **persistence baseline** (`gwl(t+h) = gwl(t)`, i.e. "nothing changes") showed:
>
> **TFT lost to persistence in 15 of 18 fold-horizon combinations.**
> Persistence RMSE 1.4-5.8 m vs TFT 2.9-9.1 m; at t+7 fold2, persistence MAE 0.56 m vs
> TFT 2.91 m (5× worse). For scale, the median real GWL change is **0.29 m over 7 days**
> and **0.90 m over 30 days** — TFT's errors were several times larger than the signal.
>
> **Lesson for all future work: report MASE (MAE ÷ persistence MAE) in every table.**
> MASE < 1 is the only meaningful definition of success here. RMSE/MAPE alone hid this
> completely for three model generations.
>
> ### Root causes and the v4 fixes
> | # | Root cause | Evidence | Fix |
> |---|---|---|---|
> | 1 | **Predicted absolute level, not change.** GWL is near-random-walk, so predicting the level from scratch wasted capacity on the part persistence gives free, and mis-estimated it by metres. | Errors ≫ actual movement | Predict `Δ` from an anchor (last encoder day); make all encoder GWL features anchor-relative. Model output 0 ⇒ persistence, so persistence becomes the floor. |
> | 2 | **District series weren't physically real.** Raw CGWB mixes ~125 telemetric stations (20-90% coverage) with ~1250 manual-survey wells having *exactly 2 readings each* (Jan+May 2023). When the manual wells dropped out, the district median jumped ~95 m. Plus 3 newly-commissioned wells still equilibrating. | Aurangabad: −4.9 → −98.9 → −1.3 m; std 30.5 m; 19.8 m train/test shift | 10% coverage panel filter + per-station **anomaly** aggregation + drift filter (trend R² ≥ 0.6 **and** range > 20 m). |
> | 3 | **11.6M params vs ~156 independent windows.** stride=1 windows share 149/150 days, so "19,535 windows" was an illusion. | 3 folds hit best val at **epoch 1-2**; train loss 0.08 vs val 0.19-0.51 | d_model 192→64, layers 3→1, encoder 120→60, dropout 0.25, wd 1e-2, **stride 7** for train/val (test stays stride 1). |
> | 4 | **Coverage ~50% vs 80% target** — a *symptom* of (3), not a separate problem: collapsed train residuals ⇒ quantiles too narrow. | — | Follows from fixing (3); conformal calibration retained as a guarantee. |
>
> ⚠️ **Do not use "range" alone to filter wells.** Dhule and Nandurbar legitimately swing
> 30-50 m *seasonally* (trend R² 0.004-0.09). Only monotonic **drift** (high trend R²)
> indicates a commissioning artifact. A naive range>30 m rule would have deleted 3 of
> Dhule's 4 wells and an entire district.
>
> ### Measured effect of the data fix alone (before any model change)
> | Metric | Before | After |
> |---|---|---|
> | Worst district std | 30.47 m (Aurangabad) | 10.24 m (Dhule) |
> | Worst 1-day jump | 29.10 m | 14.75 m |
> | Implausible jumps NaN'd | 405 | 89 |
> | Overall gwl missing | 24.2% | 12.3% |
> | **Persistence RMSE t+7 (mean over folds)** | **2.60 m** | **1.07 m** |
>
> Most of what looked like "prediction error" was station-composition noise. The bar the
> model must now clear is persistence RMSE ≈ **1.07 m (t+7) / 1.49 m (t+14) / 2.33 m (t+30)**.
>
> **Implemented in:** `Models/TFT/09_tft_colab_production_v2.ipynb` (v4) and the rewritten
> `Data Preprocessing/preprocess_groundwater.py`. v3 notebooks kept for reference.
**TL;DR:** The per-district performance gap is primarily a **data-pipeline problem**, not an architecture problem. The master dataset is stale, half-migrated, and — critically — the TFT window builder assumes contiguous daily rows while the master has calendar holes. Fix the master (Phase A), fix windowing + evaluation (Phase B), then retrain (Phase C). Architecture changes come last.

**Status: Phase A and Phase B are implemented and locally validated. Phase C (actually retraining on Colab GPU) is next — see §5.**

---

## 1. Documentation vs Reality (audit of 2026-07-04)

PROJECT_STATUS.md (last updated 2026-05-14) no longer matches what is on disk:

| Claim in docs | Actual state on disk |
|---|---|
| Rainfall = CWC station data, mean across stations | `rainfall_clean.csv` **is now CHIRPS satellite gridded rainfall** (34 districts, through **2025-12-31**, `station_count=0`). Docs never mention CHIRPS. |
| Master = 42 cols, 24 districts | Master = **47 cols, 28 districts** (CGWB extraction merge added 5 cols). Includes 5 junk districts with only 30–45 rows (Bhandara, Hingoli, Mumbai Suburban, Osmanabad, Sangli). |
| TFT v2 trained on this master (results include Satara, Gadchiroli) | Satara & Gadchiroli are **not in the current master** — TFT v2 results came from an older master. Reported numbers are not reproducible from current files. |
| GW cleaning = active filter, zero filter, [-100,0] range | `preprocess_groundwater.py` (uncommitted) now does **sign normalisation, mixed-sign station drop, Hampel filter, 5 m/day jump cap, district median (not mean), aquifer/well-depth/station-count metadata** |
| Data ends 2025-04-22 | **Raw groundwater goes to 2025-12-31. CHIRPS to 2025-12-31. ET/soil-moisture raw to 2025-06-30.** Only Open-Meteo temp/RH (END_DATE hardcoded) and the clean files stop at 2025-04-22. |
| — | `master_dataset.csv` (built 16-05 01:37) predates the rebuilt `groundwater_clean.csv` (16-05 17:08). **Master is stale**: built from the old (mean-aggregated, 28-district) groundwater file. |
| — | New `groundwater_clean.csv` has `aquifer_type`, `well_depth_m`, `n_stations` — but `merge_master.py` **silently drops them** (not in its `ordered_cols` list). |
| — | `cwc_reservoir.py` written (reservoir storage collector, needs data.gov.in API key) but never run — no output CSV. |
| — | Sentinel now covers **33 districts** (GAUL name mapping fixed Beed/Gondia/Raigad/Gadchiroli) but only through ~Apr 2025. |

**Districts with no usable CGWB data at all** (not a preprocessing bug — nothing to collect): Ahmednagar, Buldhana, Latur, Palghar, Sindhudurg (never in raw), plus Satara (1 station, mixed-sign, correctly dropped by new cleaning). Near-zero data: Mumbai City (12 readings), Mumbai Suburban (34), Hingoli (49), Osmanabad (76), Sangli (80).

---

## 2. Root Cause — why TFT is great for some districts, terrible for others

### 2.1 The windowing bug (biggest modelling defect)

`merge_master.py` drops every row where `gwl_mean` is NaN, so each district's series has **calendar holes**. The TFT notebook (`GWLDataset`, cell 9) then slides a 120+30-row window over **row positions**, assuming row i+1 = next day. Consequences, measured on the current master:

- Raigad: only **75%** of windows are truly 150 consecutive days; Nanded 77%; Dhule/Solapur ~89%; Nashik/Kolhapur ~92%. The rest silently span up to months longer — the model is fed "120 days of history" that is actually 5–8 months with invisible seams, and asked to predict "t+7" targets that may be weeks away.
- `gwl_lag_7 = shift(7)` in `merge_master.py` is a **7-row** lag, not 7 days — wrong across every gap. Same for lag_14/30, the change features, and the notebook's rolling 30/60/90 stats.
- Districts with the most gaps get the most corrupted windows **and** the fewest of them → they are simultaneously under-represented in training and trained on corrupted samples. This is exactly the good/bad district split.

### 2.2 Data coverage is the district-quality axis

Per-district GWL missingness (new clean file) lines up with TFT v2 error almost monotonically:

| District | Missing % | Stations | TFT v2 t+7 result |
|---|---|---|---|
| Nagpur / Yavatmal / Amravati | 4–6% | 77–96 | excellent (<0.3 m RMSE) |
| Nashik | 15% | 69 | bad (3.3 m) — high seasonal amplitude |
| Jalgaon | 21% | **1** | terrible (5.6 m) — "district mean" is one deep well |
| Aurangabad / Beed / Kolhapur | 27–28% | 40–51 | bad |
| Dhule | **49%** | 32 | worst (6.9 m) |

Three separate failure modes:
1. **Gap-corrupted windows** (Dhule, Aurangabad, Beed, Kolhapur, Raigad, Nanded) — fixable by Phase A/B below.
2. **Single/few-station districts** (Jalgaon 1, Akola 2–3) — the target itself is one noisy well, not a district signal. No model fixes this; needs honest reporting, station-level weighting, or exclusion from headline metrics.
3. **High-amplitude deep aquifers** (Nashik, Nandurbar, Jalgaon) — genuinely harder; helped by more training data (2nd monsoon cycle), aquifer statics, and later GAT.

### 2.3 Evaluation is misleading

- **No test set.** `TRAIN_END='2025-01-01'`; validation = Jan–Apr 2025, used both for early stopping *and* reported as final metrics → optimistic.
- **Validation covers only the dry season.** The model has never been evaluated on monsoon/post-monsoon dynamics — the operationally critical period.
- Lag features imputed with per-district **median** (cell 5) injects flat fake history into precisely the gappy districts.
- Sentinel/soil NaNs filled with **global** medians — a coastal district's NDVI hole gets filled with the state-wide median.

### 2.4 We are training on 60% of the data we already collected

Clean data stops 2025-04-22, but raw groundwater + CHIRPS extend to **2025-12-31** (ET/SM to 2025-06-30, likely further if re-fetched now). Extending to 2025-12-31 adds ~8 months (+30% rows) including a **second full monsoon → post-monsoon cycle** — the single largest generalization lever available. With only ~1.3 monsoon cycles currently in training, seasonal generalization is data-starved by construction.

---

## 3. The Plan

### Phase A — Rebuild the Master Dataset (data foundation) — ✅ DONE (2026-07-04)

> **Decision (2026-07-04):** training data stays at 2023-01-01 → 2025-04-22 (no extension of training size).
> The May–Dec 2025 raw data (groundwater + CHIRPS already on disk) remains available as an optional
> **pure held-out test set** — never trained on — but was not collected/built this round; walk-forward
> evaluation (B2) was implemented instead and covers the monsoon-blind-spot concern directly.

**A1. (Optional, test-only) Collect covariates for the May–Dec 2025 holdout window — NOT DONE, still optional/future.**

**A2. Fixed `preprocess_groundwater.py`:** added `gwl_observed` (0/1) mask column, computed before the
short-gap ffill/bfill so it correctly flags true CGWB readings vs. gap-filled days. Sign fix, Hampel
filter, median aggregation, aquifer/well-depth/station-count metadata (already present) kept as-is.

**A3. Rewrote `merge_master.py`** (full rewrite, not just patched):
- Full calendar grid preserved — **no NaN-gwl rows dropped**. `gwl_observed` carried through from groundwater_clean.
- Lags/changes (`gwl_lag_7/14/30`, `gwl_7d_change`, `gwl_30d_change`) computed by `shift()` on the
  now-gap-free calendar — verified date-true (asserted + spot-checked, diff=0.0 vs. manual 7-day-earlier lookup).
- `aquifer_type` (→ `aquifer_type_id`), `well_depth_m`, `n_stations`, `low_station_flag` passed through.
- CGWB extraction/recharge merge folded into `merge_master.py` directly (no more separate overwrite step).
- Canonical district-name map added (`MUMBAI CITY` → `Mumbai City`).
- **District gate implemented:** districts ≥60% GWL missing excluded and logged to
  `excluded_districts.csv` with reasons (Bhandara, Hingoli, Mumbai City, Mumbai Suburban, Osmanabad,
  Sangli — all 95–96% missing). `low_station_flag` set for Jalgaon (1 station) and Akola (2 stations) —
  kept in the master, reported separately, never silently blended into headline metrics.
- **Result: 24 districts, 843 days each, 20,232 rows, 52 columns.** Validated: 100% calendar contiguity
  (every district exactly 843 contiguous daily rows, 0 duplicates), 100% window contiguity (every
  150-day window is genuinely 150 consecutive days — up from 75–95% before), 0 unexpected nulls.

**Also fixed while rebuilding (found via the district gate + null audit, not originally scoped):**
- `soil_type_gee.py` had **no GAUL name mapping at all** — silently returned NaN soil texture for
  Beed, Gondia, Raigad, Gadchiroli (whose GAUL spellings differ: Bid/Gondiya/Raigarh/Garhchiroli).
  Added the mapping; re-ran; now 35/36 districts populated (only Palghar null, unused in master).
- `sentinel_indices.py` and `chirps_rainfall.py` had a wrong/stale GAUL spelling for Gadchiroli
  (`"Gadtchiroli"` — a typo — vs. the real GAUL name `"Garhchiroli"`, confirmed by querying
  `ADM2_NAME` directly). Fixed; cleared stale progress-file entries (which had cached the old
  silent-failure "done" state); re-ran. CHIRPS rainfall now 36/36 districts; Sentinel now 34/34.

**A4. (Optional, cheap win)** `cwc_reservoir.py` (reservoir storage, needs data.gov.in API key) — still not run.

### Phase B — Fix the TFT data pipeline & evaluation — ✅ DONE (2026-07-04)

New notebook: `Models/TFT/09_tft_colab.ipynb` (v3). Old v2 notebook backed up as
`Models/TFT/09_tft_colab_v2_backup.ipynb`.

**B1. Calendar-aware windowing — implemented.**
- Windows now built on the full calendar grid (guaranteed by Phase A) — every window is exactly
  `ENCODER_LEN + DECODER_LEN` consecutive days.
- GWL split into a **raw target** (`gwl_mean_raw`, true NaN preserved) used only for loss/eval, and an
  **interpolated encoder feature** (`gwl_mean`, linearly interpolated per district, NaN-free) used only
  as model input. `gwl_observed` added as a raw (unscaled) 0/1 past feature.
- `MaskedWeightedQuantileLoss`: pinball loss averaged only over `mask==1` positions — unobserved
  decoder days contribute zero gradient instead of being trained against a fabricated value.
- Window gate: encoder slice needs ≥70% observed days, decoder slice ≥50%, else the window is dropped
  (not the whole district). Verified on real data: even Dhule (45% overall observed) still yields 168
  training windows under this gate — no district is zeroed out.
- Evaluation is mask-aware too: unobserved target positions are skipped when scoring, not counted.

**B2. Walk-forward (rolling-origin) evaluation — implemented.**
- 3 folds: `fold1_monsoon` (test Jul–Sep 2024), `fold2_postmonsoon` (test Oct–Dec 2024), `fold3_dry`
  (test Jan–Apr 2025). Each fold trains on everything before its test window and early-stops on the
  last `VAL_MONTHS=2` months of its own training span — the test window is never seen during training
  or early stopping.
- Headline = median RMSE/MAE/MAPE across (fold × district), **low-station districts (Jalgaon, Akola)
  excluded from the headline and reported in their own table.**
- Per-season breakdown reported separately (this *is* the monsoon evaluation the old val-only split lacked).
- Coverage analysis and conformal calibration adapted to run on the pooled genuine-test predictions
  across all 3 folds (legitimate now — every row is out-of-fold).
- A1 holdout (May–Dec 2025) was not built this round; walk-forward's monsoon fold covers the same gap.

**B2b. Window economy — partially implemented.**
- `ENCODER_LEN` kept as a single config constant with a comment recommending the 90-day ablation;
  not run both ways by default (would 2x Colab compute) — left as a documented manual toggle.
- 3-seed ensembling documented as a future extension, not implemented by default (would 3x compute
  across 3 folds = 9x total runs) — reassess after the v3.0 single-seed result is in.

**B3. Sampling & imputation fixes — implemented.**
- Removed the old per-district-median imputation of lag columns entirely — replaced by the
  interpolation approach in B1 (a smoother, non-fabricated proxy signal).
- Soil/Sentinel median-fill kept only as a defensive safety net (should be a no-op now — Phase A's
  GAUL fixes already brought both to 100% coverage on the districts used).
- `build_sampler()`: inverse per-district train-window-count sample weights, capped at 3× the
  best-covered district's weight, applied via `WeightedRandomSampler` on the train loader only.

**Validation performed (all local, CPU, before handing off to Colab):**
1. Standalone script `design_windows.py` confirmed window counts/drop rates per district per fold
   against the rebuilt master before writing any model code.
2. Full pipeline prototype (data prep → masked dataset → masked loss → model forward/backward) run
   locally with real data; caught and fixed two real bugs: (a) `gwl_observed` was accidentally being
   z-scored by the `StandardScaler` alongside continuous features, corrupting the window-quality gate;
   (b) a Windows-specific sklearn/torch import-order segfault (fixed by importing `torch` before
   `sklearn`, harmless but also safer on Colab).
3. Full walk-forward driver (all 3 folds, tiny model, 2 epochs) dry-run end-to-end with zero crashes:
   correct window counts per fold, correct early-stopping-only validation split, correct masked
   evaluation, correct headline/per-season/low-station aggregation. Numbers from this dry-run are
   meaningless (deliberately crippled model) — only the code path was being validated.

### Phase C — Retrain TFT (v3) — NEXT STEP (not yet run; needs Colab GPU)

- Upload `tft_model.py` (unchanged) + `09_tft_colab.ipynb` + the rebuilt `master_dataset.csv` to Colab.
- Run as-is first (d_model=192, encoder=120, full epoch budget) — this isolates the Phase A+B data/pipeline
  effect from any architecture change, per the original "change one thing at a time" principle.
- Then, as documented ablations: encoder=90, d_model 128–160, 3-seed ensemble — compare against the v3.0 baseline.
- Success criteria vs v2 (recomputed honestly, since v2's own numbers weren't from a real test set):
  - Median t+30 MAPE ≤ 8% **and** worst non-flagged district t+30 RMSE < 3 m, on the walk-forward test folds.
  - q10–q90 empirical coverage 75–85% (post conformal calibration).
  - Gap between best-5 and worst-5 (non-flagged) districts shrinks by ≥50% vs. the old per-district spread.

### Phase D — Only then, architecture (GAT etc.)

Unchanged from the roadmap: district adjacency + basin graph, GAT between encoder and decoder. Do it
**after** C — spatial modelling fitted on the old corrupted windows / dry-season-only val would have
produced noise, not signal.

### Phase E — Documentation & hygiene

- [ ] Commit the Phase A/B work (`preprocess_groundwater.py`, `merge_master.py`, `sentinel_indices.py`,
  `chirps_rainfall.py`, `soil_type_gee.py`, `09_tft_colab.ipynb`, `cwc_reservoir.py`) — real progress,
  currently unversioned.
- [ ] Update PROJECT_STATUS.md: CHIRPS switch, new GW cleaning + `gwl_observed`, CGWB features, the
  actual 24-district list + exclusion log, actual date ranges, and mark TFT v2 results as
  "superseded — non-reproducible from current master, pending v3 Colab run" — done as part of this update, see PROJECT_STATUS.md §9.
- [ ] One `config.py` (or YAML) holding START/END dates and the district exclusion list, imported by
  every script — the 2025-04-22 cap was hardcoded in 3+ places (still true; not addressed this round).

---

## 4. Expected impact ranking

| Fix | Effort | Expected impact on bad districts | Status |
|---|---|---|---|
| Calendar-aware windows + masked loss (B1) | Medium | **Very high** — removes corrupted training signal for exactly the failing districts | ✅ Done |
| Walk-forward eval incl. monsoon fold (B2) | Low | **High** — trustworthy metrics from fixed data; old numbers were dry-season val used for early stopping | ✅ Done |
| District gate + separate reporting (A3) | Low | High (headline honesty) — stops 1-station wells masquerading as districts | ✅ Done |
| Date-based lags, no median lag imputation (A3/B3) | Low | High for gappy districts | ✅ Done |
| Weighted sampling (B3) | Low | Medium | ✅ Done |
| Aquifer/well-depth/station statics (C) | Low | Medium | ✅ Passed through in master; not yet used in a retrain |
| Encoder 90 ablation + smaller model + seed ensemble (B2b) | Low | Medium–High — more windows per district, less overfit | Config knob added; not run |
| May–Dec 2025 as pure holdout test (A1, optional) | Low | High (trust) — honest monsoon eval without changing training size | Not done (walk-forward covers this) |
| Reservoir storage (A4) | Medium | Medium (Marathwada districts) | Not done |
| GAT (D) | High | Unknown until C is clean | Not started (Phase D) |
