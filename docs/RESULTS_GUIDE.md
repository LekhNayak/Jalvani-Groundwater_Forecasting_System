# JalVaani v4.2: guide to the results zip

This guide comes with `tft_v4_2_outputs.zip`, produced by `09_tft_colab_production_v4_2.ipynb`. It says what each file is, which ones to use, and how to read them.

(The v4.1 zip has its own copy of the guide it shipped with: `04_results/tft_v4_1/RESULTS_GUIDE_as_shipped.md`.)

**What v4.2 changed.** Only what happens *after* the models have predicted:
- how the 80% intervals are calibrated;
- which interval is reported for each model;
- a guard against a district's ensemble leaning on a model that is poor there (Pune).

The TFT networks and LightGBM use the v4.1 settings. They were re-trained on Colab, so the medians differ from v4.1 only by small GPU run-to-run noise. **For a like-for-like comparison, use the "same-net" columns** (see §3).

## 1. The words used everywhere

| Term | Meaning | Good value |
|---|---|---|
| **t+7 / t+14 / t+30** | forecast made today for the water level 7, 14 or 30 days ahead | – |
| **No-change (persistence)** | the simplest forecast: "the level stays where it is today" | – |
| **Seasonal-naive (SN)** | today's level + the district's usual change for this time of year (past data only). Hard to beat | – |
| **MASE** (`rMAE_P`, "MASE (vs no-change)") | our error ÷ the no-change error | **below 1** = better than no-change. 0.75 = 25% less error |
| **vs seasonal** (`rMAE_SN`) | our error ÷ the seasonal-naive error | **below 1** = better than the seasonal forecast |
| **MAE / RMSE (m)** | typical / large-error-weighted miss, in metres | smaller is better |
| **MAPE (%)** | error as a % of the depth. The simple files use the old v1 formula; `MAPE floored` divides by at least 0.5 m. On this data the two are almost identical | smaller is better; never quote alone |
| **Coverage (%)** | how often the real level fell inside the 80% interval (q10 to q90) | **about 80** (75–85). Below = intervals too narrow; above = too wide |
| **Below q10 / Above q90 (%)** | misses on each side | **about 10 each** (5–15) |
| **Interval (m)** / **Width (m)** | average width of the 80% interval | narrower is better, at the right coverage |
| **Interval score (m)** | width plus 10 × the size of every miss (Winkler score for an 80% interval) | smaller is better; one number for "sharp *and* honest" |
| **TFT** | the Temporal Fusion Transformer alone (the project's main model) | – |
| **Ensemble** | the final forecast: a weighted blend of no-change, seasonal-naive, TFT and LightGBM, with weights learned per district on past held-out errors | – |
| **raw interval** | the interval the model itself gives (the TFT's own q10–q90 spread around the median). In `predictions_daily.csv` these are the `TFT` and `Ensemble` rows | – |
| **calibrated interval** | the same median with the interval re-sized by conformal calibration from past errors: the `TFT (calibrated)` and `Ensemble (calibrated)` rows | – |
| **reported interval** (new in v4.2) | for each model and horizon, either the raw or the calibrated interval, **chosen on the dev folds by rule D** (§4). Every file in `simple/`, `RESULTS_TO_QUOTE.csv` and `coverage_reported*.csv` uses the reported interval. The `interval` column says which one | – |
| **ACI** | adaptive conformal inference: the calibration widens or narrows the intervals as recent misses come in | – |
| **Stack guard** (new in v4.2) | if a district's blend was worse on its own past errors than both seasonal-naive alone and the TFT alone, that district uses the better of the two | – |
| **Folds** | six test periods, each forecast by models trained only on data before it: monsoon 2024, post-monsoon 2024, dry 2025, monsoon 2025, post-monsoon 2025, dry 2026 | – |
| **Dev folds 1–4** | used for every choice in v4.2 | – |
| **Locked folds 5–6** | Oct 2025 → 4 Jun 2026. In v4.2 they are not even computed until every choice has been made. **The honest test** (seen in earlier versions' results, but never used to choose anything since v4.0) | – |
| **Headline districts (21)** | the 24 modelled districts minus the single-well ones (Nanded, Raigad, Thane) | – |
| **Data tier A / B / C** | data quality: A = 6+ wells and well observed; C = thin or gappy. Assigned from the data only, never from results | – |
| **Beed, Latur** | listed as "no usable data" (Beed: its only well is a stuck/dry sensor; Latur: tier D) | – |

## 2. What is in the zip

```
tft_v4_2_outputs.zip
├── RESULTS_GUIDE.md                       this file
├── RUN_SUMMARY.md                         the three decisions (rules G, C, D) with their tables, seeds, run time,
│                                          v4.1 -> v4.2, Pune
├── RESULTS_TO_QUOTE.csv                   *** the v4.2 headline numbers (both models, locked + all folds) ***
│                                          incl. `interval` (raw / calibrated) and `interval score (m)`
├── RESULTS_TO_QUOTE_ref_v4_1_postproc.csv the same table with v4.1's post-processing on the SAME networks
├── v4_1_vs_v4_2.csv                       published v4.1 | v4.1 post-processing on the v4.2 networks ("same-net") | v4.2
├── pune_v4_1_vs_v4_2.csv                  Pune's Ensemble, same-net v4.1 post-processing vs v4.2
├── guard_decision.csv                     rule G: every headline district x horizon, dev MASE with the guard off / on
├── guard_cases_dev.csv                    which (district, horizon, fold) the guard switched on the dev folds, and to what
├── conformal_variant_summary.csv          rule C: the five calibration variants tried, with their scores
├── interval_choice_table.csv              rule D inputs: dev coverage, tails, width, interval score of raw and calibrated
├── interval_choice_decisions.csv          rule D: the pick per model x horizon and the reason
├── ablations.csv                          one row per reported run with its settings and key numbers
│
├── simple/                                *** v1-style results, reported intervals - start here ***
│   ├── summary_tft_vs_ensemble.csv        TFT vs Ensemble, 21 headline districts, all folds + locked (`interval` column)
│   ├── tft_training_curves.png            train / early-stopping loss per fold and seed
│   ├── tft_vsn_importance.png/.csv        which inputs the TFT weighted most (fold 6 network)
│   ├── comparison_v1_vs_v4_2_per_district.csv   only if the old v1 file was uploaded (not like-for-like)
│   ├── ensemble/                          the final forecast
│   │   ├── ensemble_metrics_headline.csv          per horizon, 21 headline and all 24 districts, all folds
│   │   ├── ensemble_metrics_headline_locked.csv   the same, locked folds 5-6 only
│   │   ├── ensemble_metrics_per_district.csv      every district x horizon (Beed/Latur = "no usable data")
│   │   ├── ensemble_metrics_per_season.csv        per fold, and grouped: monsoon / post-monsoon / dry
│   │   ├── ensemble_mase_by_district.csv          district x horizon table of MASE
│   │   ├── ensemble_mape_by_district.csv          district x horizon table of MAPE
│   │   ├── ensemble_coverage_by_district.csv      district x horizon table of coverage
│   │   ├── ensemble_predictions_all_folds.csv     every forecast: district, date, horizon, actual, q10, q50, q90,
│   │   │                                          fold, issue_date, no_change, seasonal_naive
│   │   ├── ensemble_per_district_mase_t7/t14/t30.png   bar chart per district (dashed line = 1)
│   │   ├── ensemble_per_district_mape_t7/t14/t30.png
│   │   ├── ensemble_coverage_heatmap.png          red = intervals too narrow, blue = too wide
│   │   ├── ensemble_forecast_grid_t7/t14/t30.png  every district on one page
│   │   └── quantile/quantile_<district>.png       one page per district: t+7, t+14, t+30 with real dates
│   └── tft/                               exactly the same files for the TFT alone (tft_ prefix)
│
├── FINAL_v4_2/                            detailed results of the final run, all 6 folds
│   ├── metrics_headline.csv / ensemble_metrics_headline.csv     full metric set (TFT / Ensemble); the coverage
│   │                                      columns in these two are of the RAW interval
│   ├── coverage_reported.csv / ensemble_coverage_reported.csv   coverage of the reported interval (+ `_by_district`)
│   ├── coverage_calibrated.csv / ensemble_coverage_calibrated.csv   the conformal interval, always (diagnostics)
│   ├── baseline_ladder.csv                every model vs both baselines (seasonal-naive, ridge, LightGBM, TFT, Ensemble)
│   ├── significance.csv / ensemble_significance.csv   bootstrap + Diebold-Mariano p-values (Holm-adjusted)
│   ├── metrics_per_fold.csv, metrics_per_district.csv, district_gap.csv
│   ├── stack_weights.csv                  how much each district's ensemble trusts no-change / SN / TFT / LightGBM
│   │                                      (a guarded district shows 1.0 on SN or TFT)
│   ├── stack_guard.csv                    every guarded (district, horizon, fold) with the pool errors that triggered it
│   ├── best_epochs.csv, windows.csv, config.json
│   ├── predictions_daily.csv              TFT (raw), TFT, TFT (calibrated), Ensemble, Ensemble (calibrated) rows
│   ├── predictions_base.csv               seasonal-naive, ridge and LightGBM forecasts
│   ├── predictions_weekly.csv             7-day-mean target
│   └── model_<fold>_lastseed.pt           one trained network per fold (last seed)
├── FINAL_v4_2_locked/                     the same metric files restricted to folds 5-6
├── REF_v4_1_postproc/ (+ _locked/)        metric files of the same-net reference (v4.1 post-processing)
└── dev_ref/                               dev-fold metric files of the reference (v4.1 post-processing)
```

## 3. Which files to use

| You need | Use |
|---|---|
| **The numbers for the report / viva** | `RESULTS_TO_QUOTE.csv`, locked rows first, then all folds |
| **What v4.2 changed** | `v4_1_vs_v4_2.csv`. Compare the `v4.2` columns with the `same-net v4.1-post` columns, which use the same networks, so only post-processing differs. The `v4.1` columns are the published v4.1 run and differ also by GPU noise |
| How the three choices were made | `RUN_SUMMARY.md`, then `guard_decision.csv`, `conformal_variant_summary.csv`, `interval_choice_decisions.csv` |
| Pune | `pune_v4_1_vs_v4_2.csv`, `simple/ensemble/quantile/quantile_pune.png` |
| A results table like the old one | `simple/ensemble/ensemble_metrics_headline.csv` (and the `tft/` version) |
| Per-district results | `simple/ensemble/ensemble_metrics_per_district.csv` or the `*_by_district.csv` tables |
| Figures | `simple/ensemble/ensemble_per_district_mase_t30.png`, `ensemble_coverage_heatmap.png`, `simple/tft/tft_coverage_heatmap.png`, `ensemble_forecast_grid_t30.png`, `quantile/quantile_nagpur.png` (or any district), `simple/tft_vsn_importance.png`, `simple/tft_training_curves.png` |
| "Is the transformer worth it?" | `FINAL_v4_2/baseline_ladder.csv` and `simple/summary_tft_vs_ensemble.csv` |
| Statistical proof | `FINAL_v4_2_locked/ensemble_significance.csv` |
| Raw forecasts for your own analysis or a dashboard | `simple/ensemble/ensemble_predictions_all_folds.csv` |

## 4. The three v4.2 rules (fixed before the run; dev folds 1–4 only)

| Rule | Question | Decision |
|---|---|---|
| **G** | Use the stack guard? | Yes, if the mean Ensemble error ÷ seasonal-naive error (21 headline districts, 3 horizons) does not get worse **and** no district's Ensemble MASE rises by more than 0.02 at any horizon |
| **C** | Which calibration variant? (V0 = v4.1; V1 = one ACI outcome per issue date and band; V2 = V1 + symmetric α range 0–0.20; V3 = V2 with γ 0.03; V4 = no adaptation) | The lowest mean of (calibrated interval score ÷ raw interval score) over TFT and Ensemble × 3 horizons. V0 is kept unless beaten by more than 0.01 |
| **D** | For each model and horizon, report the raw or the calibrated interval? | If exactly one has coverage within 75–85%, take it. Otherwise take the lower interval score. Within 1%, take raw |

## 5. How to analyse the results (in this order)

1. **Headline.** Open `RESULTS_TO_QUOTE.csv` and look at the *locked folds 5-6* Ensemble rows.
   - MASE (`rMAE vs persistence`) below 1 at every horizon, with p < 0.05, means it beats doing nothing.
   - `rMAE vs seasonal` below 1 with p < 0.05 means it also beats the seasonal forecast. This is the harder and more convincing claim.
2. **Did v4.2 help?** In `v4_1_vs_v4_2.csv`, compare v4.2 with same-net:
   - interval score (lower is better);
   - coverage (closer to 80 is better);
   - vs no-change (should be equal, except where the guard acted).
3. **TFT vs Ensemble.** Compare the two models in `simple/summary_tft_vs_ensemble.csv`. If the TFT alone is about 1.0 against seasonal-naive, say so: the ensemble is where the gain over seasonality comes from.
4. **Across districts.** Open `ensemble_per_district_mase_t30.png`.
   - Bars left of the dashed line beat no-change.
   - Compare the colours: if tier C (thin data) districts sit at the bottom, the limit is the data, not the model.
   - `district_gap.csv` gives the worst-5 minus best-5 spread.
5. **Across seasons.** Use `ensemble_metrics_per_season.csv`. The monsoon is the hard season, because future rain is unknown. Expect it to be the weakest.
6. **Intervals.** Open `ensemble_coverage_heatmap.png` and `tft_coverage_heatmap.png`.
   - White means close to 80%, which is right. Red means too narrow; blue means too wide.
   - Check the overall coverage and the two tails in `RESULTS_TO_QUOTE.csv`, aiming for about 80 overall and about 10 per tail.
7. **Look at the forecasts.** Open two or three `quantile_<district>.png` pages: a good district, a tier-C one and Pune. Check how the forecast (blue) tracks the actual (black) through the monsoon rise and the dry-season decline, against the no-change line (orange dotted).

## 6. Caveats to state with the results

- The groundwater record starts in 2023, so the walk-forward test covers only three monsoons.
- Folds 5-6 contain no monsoon, so the locked test covers post-monsoon and dry seasons only. Monsoon skill comes from folds 1 and 4, which are development folds.
- Folds 5-6 have been seen in earlier versions' results, but nothing in v4.1 or v4.2 was chosen on them.
- The networks were re-trained, and GPU training is not bit-for-bit repeatable, so the medians move slightly between runs. That is why v4.2 is compared with v4.1's post-processing on the *same* networks.
- **No input knows future rain.** That is the main remaining source of error, especially at t+30 in the monsoon.
- The single-well districts (Nanded, Raigad, Thane) are reported but kept out of the headline. Beed and Latur have no usable data.
- The VSN plot shows what the network weights. It is not proof of physical causation.
- Most of the Ensemble's accuracy comes from LightGBM: on its own it is about as accurate, but its intervals are too narrow. Check `baseline_ladder.csv`.
- **Pune was identified as a problem from v4.1's all-fold results.** Its failure is visible in dev folds 1 and 3 alone, and the guard rule uses only each district's own past errors. It is not tuned to Pune.
- The v1 comparison file (if present) is **not like-for-like**: v1 used the old, noisier groundwater series and a different scoring protocol.
