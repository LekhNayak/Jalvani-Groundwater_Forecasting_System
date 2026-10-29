# JalVaani v4.1 — guide to the results zip

This guide comes with `tft_v4_outputs.zip`, produced by `09_tft_colab_production_v4_1.ipynb`. It says
what each file is, which ones to use, and how to read them.

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
| **Interval (m)** | average width of the 80% interval | narrower is better, at the right coverage |
| **Interval score (m)** | width plus a penalty for every miss (Winkler) | smaller is better; one number for "sharp *and* honest" |
| **TFT** | the Temporal Fusion Transformer alone (the project's main model) | – |
| **Ensemble** | the final forecast: a weighted blend of no-change, seasonal-naive, TFT and LightGBM, with weights learned per district on past held-out errors | – |
| **(calibrated)** | the same median forecast with intervals corrected so coverage is honest. Use these rows for the **Ensemble**. For the **TFT alone**, the un-calibrated `TFT` rows have the better intervals (see §5) | – |
| **Folds** | six test periods, each forecast by models trained only on data before it: monsoon 2024, post-monsoon 2024, dry 2025, monsoon 2025, post-monsoon 2025, dry 2026 | – |
| **Dev folds 1–4** | used to choose the set-up | – |
| **Locked folds 5–6** | Oct 2025 → 4 Jun 2026; nothing in v4.1 was tuned on them. **The honest test** (it was looked at once before, in v4.0) | – |
| **Headline districts (21)** | the 24 modelled districts minus the single-well ones (Nanded, Raigad, Thane) | – |
| **Data tier A / B / C** | data quality: A = 6+ wells and well observed; C = thin or gappy. Assigned from the data only, never from results | – |
| **Beed, Latur** | listed as "no usable data" (Beed: its only well is a stuck/dry sensor; Latur: tier D) | – |

## 2. What is in the zip

```
tft_v4_outputs.zip
├── RESULTS_GUIDE.md               this file
├── RUN_SUMMARY.md                 which set-up was chosen and why, seeds, run times, v4 -> v4.1 comparison
├── RESULTS_TO_QUOTE.csv           *** the headline numbers (both models, locked + all folds) ***
├── v4_vs_v4_1.csv                 v4.0 vs v4.1 headline side by side
├── dev_screening_summary.csv      every candidate set-up on dev folds 1-4 (how the choice was made)
├── post_sweep_summary.csv         the stacking / interval settings tried on the chosen set-up
├── ablations.csv                  one row per run with its settings and key numbers
│
├── simple/                        *** v1-style results - start here ***
│   ├── summary_tft_vs_ensemble.csv     TFT vs Ensemble, 21 headline districts, all folds + locked
│   ├── tft_training_curves.png         train / early-stopping loss per fold and seed
│   ├── tft_vsn_importance.png/.csv     which inputs the TFT weighted most (fold 6 network)
│   ├── comparison_v1_vs_v4_1_per_district.csv   only if the old v1 file was uploaded (not like-for-like)
│   ├── ensemble/                       the final forecast
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
│   └── tft/                            exactly the same files for the TFT alone (tft_ prefix)
│
├── FINAL_<setup>/                 detailed results of the final run, all 6 folds
│   ├── metrics_headline.csv / ensemble_metrics_headline.csv     full metric set (TFT / Ensemble)
│   ├── baseline_ladder.csv        every model vs both baselines (seasonal-naive, ridge, LightGBM, TFT, Ensemble)
│   ├── significance.csv / ensemble_significance.csv   bootstrap + Diebold-Mariano p-values (Holm-adjusted)
│   ├── metrics_per_fold.csv, metrics_per_district.csv, district_gap.csv, coverage_calibrated*.csv
│   ├── stack_weights.csv          how much each district's ensemble trusts no-change / SN / TFT / LightGBM
│   ├── best_epochs.csv, windows.csv, config.json
│   ├── predictions_daily.csv      all models' forecasts (raw, shrunk, calibrated) in one long table
│   ├── predictions_base.csv       seasonal-naive, ridge and LightGBM forecasts
│   ├── predictions_weekly.csv     7-day-mean target
│   └── model_<fold>_lastseed.pt   one trained network per fold (last seed)
├── FINAL_<setup>_locked/          the same metric files restricted to folds 5-6
├── dev_<candidate>/               metrics of each screening candidate (dev folds only)
└── sweep_<setting>/               metrics of each post-processing setting tried
```

## 3. Which files to use

| You need | Use |
|---|---|
| **The numbers for the report / viva** | `RESULTS_TO_QUOTE.csv`, locked rows first, then all folds |
| A results table like the old one | `simple/ensemble/ensemble_metrics_headline.csv` (and the `tft/` version) |
| Per-district results | `simple/ensemble/ensemble_metrics_per_district.csv` or the `*_by_district.csv` tables |
| Figures | `simple/ensemble/ensemble_per_district_mase_t30.png`, `ensemble_coverage_heatmap.png`, `ensemble_forecast_grid_t30.png`, `quantile/quantile_nagpur.png` (or any district), `simple/tft_vsn_importance.png`, `simple/tft_training_curves.png` |
| "Is the transformer worth it?" | `FINAL_<setup>/baseline_ladder.csv` and `simple/summary_tft_vs_ensemble.csv` |
| Statistical proof | `FINAL_<setup>_locked/ensemble_significance.csv` |
| How the set-up was chosen | `RUN_SUMMARY.md`, `dev_screening_summary.csv`, `post_sweep_summary.csv` |
| Raw forecasts for your own analysis or a dashboard | `simple/ensemble/ensemble_predictions_all_folds.csv` |

## 4. How to analyse the results (in this order)

1. **Headline.** Open `RESULTS_TO_QUOTE.csv` and look at the *locked folds 5-6* Ensemble rows.
   - MASE (`rMAE vs persistence`) below 1 at every horizon, with p < 0.05, means it beats doing nothing.
   - `rMAE vs seasonal` below 1 with p < 0.05 means it also beats the seasonal forecast. This is the harder and more convincing claim.
2. **TFT vs Ensemble.** Compare the two models in `simple/summary_tft_vs_ensemble.csv`. If the TFT alone is about 1.0 against seasonal-naive, say so: the ensemble is where the gain over seasonality comes from.
3. **Across districts.** Open `ensemble_per_district_mase_t30.png`.
   - Bars left of the dashed line beat no-change.
   - Compare the colours: if tier C (thin data) districts sit at the bottom, the limit is the data, not the model.
   - `district_gap.csv` gives the worst-5 minus best-5 spread.
4. **Across seasons.** Use `ensemble_metrics_per_season.csv`. The monsoon is the hard season, because future rain is unknown. Expect it to be the weakest.
5. **Intervals.** Open `ensemble_coverage_heatmap.png`.
   - White means close to 80%, which is right. Red means too narrow; blue means too wide.
   - Check the overall coverage and the two tails in `RESULTS_TO_QUOTE.csv`, aiming for about 80 overall and about 10 per tail.
6. **Look at the forecasts.** Open two or three `quantile_<district>.png` pages: a good district, a tier-C one and a single-well one. Check how the forecast (blue) tracks the actual (black) through the monsoon rise and the dry-season decline, against the no-change line (orange dotted).

## 5. Caveats to state with the results

- The groundwater record starts in 2023, so the walk-forward test covers only three monsoons.
- Folds 5-6 contain no monsoon, so the locked test covers post-monsoon and dry seasons only. Monsoon skill comes from folds 1 and 4, which were tuning folds.
- Folds 5-6 were viewed once before (v4.0). v4.1 was chosen on folds 1-4 only.
- **No input knows future rain.** That is the main remaining source of error, especially at t+30 in the monsoon.
- The single-well districts (Nanded, Raigad, Thane) are reported but kept out of the headline. Beed and Latur have no usable data.
- The VSN plot shows what the network weights. It is not proof of physical causation.
- **(Found in the 2026-10-29 run.) For the TFT alone, the calibration step makes the intervals worse.**
  - On the locked folds, the un-calibrated `TFT` rows in `FINAL_<setup>/predictions_daily.csv` cover 77-80%; the `TFT (calibrated)` rows cover 66-74%.
  - The dev folds show the same.
  - So the "TFT alone" coverage in `RESULTS_TO_QUOTE.csv` and `simple/tft/` (which use the calibrated rows) understates the network.
  - Details: `PROD_V1_vs_V2.md`, v4.1 results section.
- In v4.1 most of the Ensemble's accuracy comes from LightGBM: on its own it is about as accurate, but its intervals are too narrow. Check `baseline_ladder.csv`.
- The v1 comparison file (if present) is **not like-for-like**: v1 used the old, noisier groundwater series and a different scoring protocol.
