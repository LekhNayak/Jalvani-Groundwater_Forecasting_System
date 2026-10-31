# Data contract: what the dashboards show and where each value comes from

`dev/pipeline/build_dashboard_data.py` builds everything below from the project's files, read-only. `dev/backend/data/manifest.json` records:
- the md5 of every source file;
- the model run the data came from;
- the freshness dates shown in the top bar.

## Rule: no mock data

Every number shown comes from a real file. Where a value does not exist, the site says so and shows nothing in its place:

| Situation | What the site does |
|---|---|
| Groundwater readings end on 2026-06-04 (Latur: 2025-05-19); CGWB has published nothing newer | It shows the date, with "newer CGWB readings not yet published" |
| No operational forecast yet (no v4.3 export) | The forecast card says "not produced yet". No placeholder numbers appear anywhere |
| The last reading is old (4 Jun 2026; 150 days on 1 Nov 2026), so the forecast dates have passed | A banner when the last reading is more than 30 days old. The forecast card says its dates have passed. Tips quote the reading date, and an expired forecast gives no "next month" advice (`advisories.forecast_is_past`) |
| A district has too few recent readings for a forecast (for example Nanded: readings on 40% of the last 60 days, below the 70% the model needs) | It shows that reason |
| 10 districts have no usable CGWB well data; Beed and Latur have too few | They are greyed, with "no monitoring data". Their rain and extraction are still shown |
| September 2026 rainfall is preliminary, and ET and soil moisture after 2026-09-24 are carried forward | Rain is tagged "provisional", with a day count. Soil moisture uses only observed days (as of 2026-09-24) |
| There is no long-term rainfall-normal file | "Usual" = the average of the same calendar window in 2023–2025, labelled exactly so |
| Village or taluka level | Not available. District level only, stated on every farmer page. District forecasts are never split up |

## Fields

| Shown | Districts | Source | How it is computed |
|---|---|---|---|
| Level now (m below ground) | 24 modelled | `data/clean/groundwater_clean.csv` | Mean `gwl_mean` over observed days in the 7 days up to the last reading (`gwl_mean` < 0 = below ground; the site shows depth = −level). Labelled "Average of the week to …". Where the last single day differs from it by 0.2 m or more, the tile also shows that day's level, because the forecast starts from it (5 districts differ by 0.4–1.3 m: Solapur, Akola, Washim, Gadchiroli, Nashik) |
| Change over 30 days and vs a year ago | 24 | same | The same 7-day mean, 30 days and 1 year earlier |
| Compared with usual | 24 | same | Observed levels within ±15 days of the day of year, in other years. "About" if within max(0.5 m, 0.5 SD); otherwise "deeper" or "shallower" |
| Forecast t+7 / 14 / 30 (q10, q50, q90) | 23–24 | `04_results/tft_v4_3/dashboard_export/forecast_latest.csv` (Ensemble rows; reported intervals) | Issued from each district's last reading; change = q50 − the level on the issue day |
| "Most likely" range (about 50%) | same | `forecast_latest.csv` + `backtest_ensemble.csv` | q50 ± the median of \|actual − q50\| over all back-tests (folds 1–6) for that district and horizon, clipped to the 80% range (this clips Gondia and Washim at 30 days, by 9 cm or less). Same forecast, same middle value; the 80% range stays as "could be". Honest check: margins from folds 1–4 only cover 63.4 / 65.0 / 68.3% (7 / 14 / 30 days) of the locked folds 5–6 in the 21 headline districts, so the band is a little wider than 50%, which errs on the safe side. Fields `likely_margin_m`, `likely_lo`, `likely_hi`; method and check in `manifest.json` → `most_likely` |
| Forecast path (30 days) | same | `forecast_latest_tft_path.csv` | TFT daily steps. Kept in the history JSON for analysts but not drawn: the farmer chart draws the Ensemble 7 / 14 / 30-day values (joined by straight lines, both ranges), so the chart always matches the forecast tile |
| Direction words | same | backend `advisories.forecast_direction` | falling = the whole 80% range below today; rising = the whole range above; steady = median within ±0.1 m; otherwise uncertain |
| Reliability (good / fair / limited) | 24 | `metrics_ensemble_per_district.csv` (all folds; fallback: v4.2 `ensemble_metrics_per_district.csv`) | **Good:** MASE ≤ 0.85, coverage 70–90%, tier A/B, more than 1 well. **Fair:** MASE < 1, more than 1 well. **Limited:** anything else (including single-well districts) |
| Past forecasts (government chart) | 24 | `backtest_ensemble.csv` (fallback: v4.2 `ensemble_predictions_all_folds.csv`) | One row per issue date and horizon, with actual, q10, q50, q90 |
| Rain since 1 June, last 7 / 30 days, vs usual | 36 | `data/clean/rainfall_clean.csv` (`rainfall_observed` flags) | Sums over the window; usual = the mean of the same window in 2023–2025 |
| Soil moisture (top layers) vs usual | 36 | `data/clean/env_features_clean.csv` (`sm_l1`, `sm_l2`; observed days only) | 7-day mean vs the same window in 2023–2025; "about" within ±10% |
| Stage of extraction and category, 2023–2025 | 36 | `data/raw/cgwb_extraction_maharashtra.csv` | As published by CGWB. Safe < 70%, semi-critical 70–90%, critical 90–100%, over-exploited > 100% |
| Irrigation / industry / domestic split | 36 | `data/raw/mh_gw_2025.csv` (ha·m) | Share of total annual extraction. Names mapped: Yawatmal → Yavatmal, Mumbai → Mumbai City, Mumbai Sub → Mumbai Suburban |
| Wells, completeness, tier | 26 | `data/clean/district_data_quality.csv` | As computed by `02_preprocessing/merge_master.py` |
| Wells used (list, map) | 122 wells | `data/clean/groundwater_station_daily.csv` + `groundwater_stations.geojson` (coordinates for 117) | First and last reading, days |
| District boundaries | 36 | `data/raw/maharashtra_district_geoms.geojson` | Simplified (0.004°) for the web |
| Names in Marathi / Hindi | 36 | `dev/frontend/src/i18n/districts.json` | Drafted; native review pending (`I18N_REVIEW.md`). The official new names (Ahilyanagar, Chhatrapati Sambhajinagar, Dharashiv) are shown alongside |

## Government priority flags (backend `priorities.py`; the weights are shown on the Help page)

| Flag | Rule | Weight | Action |
|---|---|---|---|
| No well readings | status `no_data` | 3 | improve monitoring |
| Too few readings | status `insufficient_data` | 2 | improve monitoring |
| Only one well | modelled district with one well | 1 | improve monitoring |
| Extraction critical | stage ≥ 90% | 3 | manage demand |
| Extraction semi-critical | stage 70–90% | 2 | manage demand |
| Forecast decline | t+30 median falls ≥ 0.5 m, and the whole range is below today | 2 | watch closely |
| Deeper than usual | "compared with usual" = deeper | 1 | watch closely |
| Rain deficit | season rain < 75% of usual | 1 | watch closely |
| Reliability limited | t+30 reliability = limited | 1 | improve monitoring |

The score is the sum of the weights. Each row lists its flags, so the ranking can always be explained.

## Farmer tips (backend `advisories.py`)

Each tip is a translation key plus the values it quotes, chosen by the same transparent rules:
- falling, rising or uncertain forecast;
- deeper than usual;
- rain deficit (< 75%) or surplus (> 125%);
- extraction not "Safe";
- limited reliability;
- no groundwater data.

At most three tips are shown. A warning always comes with "talk to your local agriculture office or the GSDA". The tips are general water-use guidance, never crop, borewell, scheme or legal advice.
