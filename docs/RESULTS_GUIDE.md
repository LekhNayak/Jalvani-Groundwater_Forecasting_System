# JalVaani v4.3: guide to the results zip

This guide comes with `tft_v4_3_outputs.zip`, produced by `09_tft_colab_production_v4_3.ipynb`. It says what each file is, which ones to use, and how to read them.

The v4.1 zip has its own copy of its guide: `04_results/tft_v4_1/RESULTS_GUIDE_as_shipped.md`. The v4.2 version of this guide is in git history (commit `ac2cdf2`).

**What v4.3 changed.** Only post-processing. The TFT networks and LightGBM are the v4.1 ones; v4.2 showed they re-train bit-for-bit.
1. **Pune (rule G2).** Where LightGBM alone was more than 20% worse than seasonal-naive on a district's own past errors, that district's blend is refitted without LightGBM.
2. **Width cap.** A calibrated interval may reach at most 3 × the network's own distance from the median.
3. **Even intervals (rule D2).** Raw or calibrated is chosen per model and horizon with the district evenness of coverage counted.
4. **The latest forecast.** A forecast issued from each district's last groundwater reading, plus `dashboard_export/`, the files the dashboards (`dev/`) read.

**Honesty note.** Changes 2 and 3 were designed after seeing v4.2's results, including the locked folds 5–6. So v4.3's locked-fold numbers are a refinement check, not a fresh hold-out. All rules still use dev folds 1–4 only.

## 1. The words used everywhere

| Term | Meaning | Good value |
|---|---|---|
| **t+7 / t+14 / t+30** | forecast made today for the water level 7, 14 or 30 days ahead | – |
| **No-change (persistence)** | the simplest forecast: "the level stays where it is today" | – |
| **Seasonal-naive (SN)** | today's level + the district's usual change for this time of year (past data only). Hard to beat | – |
| **MASE** (`rMAE_P`, "MASE (vs no-change)") | our error ÷ the no-change error | **below 1** = better than no-change |
| **vs seasonal** (`rMAE_SN`) | our error ÷ the seasonal-naive error | **below 1** = better than the seasonal forecast |
| **MAE / RMSE (m)** | typical / large-error-weighted miss, in metres | smaller is better |
| **Coverage (%)** | how often the real level fell inside the 80% interval (q10 to q90) | **about 80** (75–85) |
| **Below q10 / Above q90 (%)** | misses on each side | **about 10 each** (5–15) |
| **Interval score (m)** | width plus 10 × the size of every miss (Winkler score for an 80% interval) | smaller is better |
| **Districts 65–90 (%)** | share of the 21 headline districts whose own coverage is between 65% and 90% (evenness) | higher is better |
| **raw / calibrated / reported interval** | raw = the network's own q10–q90 around the median; calibrated = conformal (static, capped in v4.3); reported = the one chosen per model and horizon by rule D2. The `interval` column says which | – |
| **Ensemble** | the final forecast: a weighted blend of no-change, seasonal-naive, TFT and LightGBM, weights learned per district | – |
| **Dev folds 1–4 / locked folds 5–6** | used for every choice / Oct 2025 → 4 Jun 2026 | – |
| **Headline districts (21)** | the 24 modelled districts minus the single-well ones (Nanded, Raigad, Thane) | – |
| **Latest forecast** | issued from each district's last reading (`fold7_latest`), made exactly like an evaluated fold; no actuals yet | – |

## 2. What is in the zip

```
tft_v4_3_outputs.zip
├── RESULTS_GUIDE.md, RUN_SUMMARY.md     this file; the decisions (G2, D2) with their tables, run time, comparisons
├── RESULTS_TO_QUOTE.csv                 *** v4.3 headline numbers (both models, locked + all folds) ***
├── RESULTS_TO_QUOTE_ref_v4_2_postproc.csv   the same with v4.2's post-processing on the SAME networks
├── v4_2_vs_v4_3.csv                     published v4.2 | v4.2 post-processing on the same networks | v4.3
├── district_evenness_v4_2_vs_v4_3.csv   share of districts with coverage 65-90%, both versions, all / locked
├── wide_intervals_v4_2_vs_v4_3.csv      reported intervals wider than 20 m, both versions
├── pune_v4_2_vs_v4_3.csv                Pune's Ensemble, both versions
├── guard_decision.csv, guard_cases_dev.csv     rule G2: dev MASE per district guard off / on; where it acted
├── interval_choice_table.csv, interval_choice_decisions.csv   rule D2 inputs (incl. evenness) and picks
├── ablations.csv
├── dashboard_export/                    *** what the dashboards read (dev/pipeline/build_dashboard_data.py) ***
│   ├── forecast_latest.csv              latest forecast: district, model (Ensemble/TFT), horizon, issue / target date,
│   │                                    level_now, q10, q50, q90, interval, seasonal_naive
│   ├── forecast_latest_tft_path.csv     the TFT's 30 daily steps from the issue date
│   ├── forecast_latest_skipped.csv      modelled districts with no latest forecast, and why
│   ├── backtest_ensemble.csv / backtest_tft.csv   every evaluated forecast (reported intervals, all folds)
│   ├── metrics_ensemble_per_district.csv / metrics_tft_per_district.csv   all folds and locked folds
│   ├── districts.csv                    tier, single-well flag, modelled, headline
│   └── export_manifest.json             run tag, pipeline version, data md5, issue date, settings
├── simple/                              v1-style tables and plots, reported intervals (as in v4.2)
├── FINAL_v4_3/ (+ _locked/)             detailed results of the final run (as in v4.2; coverage_reported*.csv etc.)
├── REF_v4_2_postproc/ (+ _locked/)      metric files of the same-net reference
└── dev_ref/                             dev-fold metric files of the reference
```

## 3. Which files to use

| You need | Use |
|---|---|
| **The numbers for the report / viva** | `RESULTS_TO_QUOTE.csv`, locked rows first, then all folds |
| **What v4.3 changed** | `v4_2_vs_v4_3.csv` (compare `v4.3` with `same-net v4.2-post`), `district_evenness_v4_2_vs_v4_3.csv`, `wide_intervals_v4_2_vs_v4_3.csv`, `pune_v4_2_vs_v4_3.csv` |
| How the choices were made | `RUN_SUMMARY.md`, `guard_decision.csv`, `interval_choice_decisions.csv` |
| **The latest forecast / the dashboards** | `dashboard_export/`. Unzip to `04_results/tft_v4_3/`, then run `dev\scripts\build_data.ps1` |
| Tables and figures | `simple/` (as before) |
| Statistical proof | `FINAL_v4_3_locked/ensemble_significance.csv` |

## 4. The v4.3 rules (dev folds 1–4 only; fixed before the run)

| Rule | Question | Decision |
|---|---|---|
| **G2** | Use the LightGBM-only guard? | Yes, if the mean Ensemble error ÷ seasonal-naive error (21 headline districts, 3 horizons) does not get worse **and** no district's Ensemble MASE rises by more than 0.02 at any horizon |
| **C** | Calibration | Fixed: v4.2's static conformal (γ = 0) + width cap 3 (not tuned) |
| **D2** | Raw or calibrated, per model and horizon? | Keep options with coverage 75–85%; drop any whose interval score is > 5% worse than the best of those; of the rest take the higher share of districts at 65–90% coverage (within 5 points: lower interval score, then raw). If neither is in 75–85%: lower interval score |

## 5. How to analyse the results
1. **Headline:** `RESULTS_TO_QUOTE.csv`, locked Ensemble rows. Medians change only where the guard acted (Pune), so expect the v4.2 numbers almost exactly.
2. **v4.3 vs v4.2:** in `v4_2_vs_v4_3.csv`, compare the interval score and coverage; then evenness and wide intervals; then Pune.
3. **The latest forecast:** `dashboard_export/forecast_latest.csv`. Check the issue date (each district's last reading) and that `forecast_latest_skipped.csv` explains any missing district.
4. Then the per-district, per-season and interval checks as in earlier versions (`simple/`).

## 6. Caveats to state with the results
- **Folds 5–6:**
  - they contain no monsoon;
  - they have been seen in earlier versions' results;
  - v4.3's design was informed by v4.2's locked results, so they are a refinement check for v4.3.
- **The latest forecast is issued from the last CGWB reading (4 Jun 2026)** and has no actuals yet. It is shown as such on the dashboards.
- **No input knows future rain.** That is the main remaining error source (monsoon, t+30).
- **Single-well and limited data.** Single-well districts (Nanded, Raigad, Thane) are reported but kept out of the headline. Beed and Latur have no usable data.
- **A district can miss the latest forecast.** It does so if its wells reported on fewer than 70% of the last 60 days. The same rule applies to every evaluated window.
- **LightGBM carries most of the Ensemble's accuracy;** the TFT supplies most of its interval spread.
- **(Found in the 2026-11-01 run.) The Pune guard (rule G2) was adopted on the dev folds but made Pune worse on the locked folds** (t+30: 0.77 → 0.92). The rule's choice is kept and reported; the headline moved by +0.002 to +0.004.
- **(Found in the 2026-11-01 run.) The latest forecast was issued on 4 Jun 2026** (the last CGWB reading), so its target dates (11 Jun – 4 Jul 2026) have passed without readings to check them.
