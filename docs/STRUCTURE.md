# Project Structure

The project was reorganised on **2026-10-29**. Before that, code, data, results and documents were mixed together inside `Data Collection/`, `Data Preprocessing/`, `Models/`, `Latest results/` and `result zips/`.

**Nothing was deleted.**
- All 737 files were moved. Their checksums before and after the move are identical (`archive/MANIFEST_before_md5.txt`).
- The only things removed were `__pycache__` folders. These hold compiled Python bytecode, which Python regenerates automatically.

The complete file-by-file log is `archive/MOVES_2026-10-29.csv`.

## The idea

| Folder | Holds | Rule |
|---|---|---|
| `01_data_collection/` | code that downloads data | writes only to `data/raw/` (plus its own `progress/` and `logs/`) |
| `02_preprocessing/` | code that cleans and merges | reads `data/raw/`, writes `data/clean/` and `data/audit/` |
| `data/` | **all data**, one live copy of each file | `raw/` = downloaded, `clean/` = produced, `master/` = frozen training sets with versioned names |
| `03_models/` | model code and notebooks | the current version sits at the top level; older versions in `notebooks/archive/` or `legacy/` |
| `04_results/` | **all results**, one folder per run | a run's files stay together, exactly as the run wrote them |
| `docs/` | every document | the report, the status log, the results guide, this file; `history/` holds the design history |
| `archive/` | backups and duplicates | never read by the pipeline; `MANIFEST.csv` lists each file's checksum and origin |

Each stage's scripts find their files relative to their own location, so they can be started from any folder.

## Where things went (folder level)

| Old location | New location |
|---|---|
| `Data Collection/*.py` (active collectors) | `01_data_collection/` |
| `Data Collection/` WRIS collectors that were replaced (`CWC_Rainfall`, `ET_data`, `soil_moisture_data`, `humidity_data`, `temperature_data`, `collect_dataset`, `test.py`) | `01_data_collection/legacy/` |
| `Data Collection/datasets/` | `data/raw/` |
| `Data Collection/cgwb_pdfs/` | `data/raw/cgwb_pdfs/` |
| `Data Collection/txt files/` | `01_data_collection/progress/` |
| `Data Collection/log/` | `01_data_collection/logs/` |
| `Data Collection/backup_pre_2026_extend/` (`*.bak`) | `archive/backups/2026-04_pre_extend/` |
| `Data Collection/backup_pre_2026_10_extend/` | `archive/backups/2026-10_pre_extend/` |
| `Data Preprocessing/*.py` | `02_preprocessing/` (`preprocess_rainfall.py` and `merge_cgwb_into_master.py` → `legacy/`) |
| `Data Preprocessing/clean datasets/` | `data/clean/` |
| `Data Preprocessing/clean datasets/v3_target_backup/` | `data/master/district_data_quality_v3target.csv`, `archive/backups/2026-10_v3_target/`, `archive/duplicates/` |
| `Data Preprocessing/unclean datasets/` (copies of the raw files) | `archive/duplicates/`. Preprocessing now reads `data/raw/` directly, which removes the manual copy step |
| `Data Preprocessing/audit/` | `data/audit/` |
| `Models/baselines/` notebooks · `scripts/` · `outputs/` | `03_models/baselines_phase1/notebooks/` · `scripts/` · `04_results/phase1_baselines/` |
| `Models/TFT/tft_model_v3.py`, `tft_v4.py`, `run_v4.py`, `tune_snaive.py` | `03_models/tft/` |
| `Models/TFT/09_tft_colab_production_v4_1.ipynb` | `03_models/tft/notebooks/` |
| `Models/TFT/` older notebooks (production v1–v4, prototypes) | `03_models/tft/notebooks/archive/` |
| `Models/TFT/tft_model.py` | `03_models/tft/legacy/` |
| `Models/TFT/master_dataset_v4target.csv`, `master_dataset_v3target.csv` | `data/master/` |
| `Models/TFT/outputs/v4/` | `04_results/tft_v4_0_dev/` |
| `Models/TFT/PROD_V1_vs_V2.md` | `docs/history/TFT_VERSION_HISTORY.md` (renamed: the old name said "v1 vs v2" but the file covers v1 → v4.1) |
| `Models/TFT/V3_PLAN.md`, `TFT_IMPROVEMENT_PLAN.md` | `docs/history/` |
| `Models/TFT/RESULTS_GUIDE.md` | `docs/RESULTS_GUIDE.md` |
| `REPORT.md`, `PROJECT_STATUS.md` | `docs/` |
| `Latest results/` (the production v1 run) | `04_results/tft_prod_v1/` (its `README.md` → `RUN_README.md`) |
| `result zips/v4_1/` | `04_results/tft_v4_1/` (the guide copy shipped in the zip → `RESULTS_GUIDE_as_shipped.md`) |
| `result zips/tft_v4_1_outputs.zip` | `04_results/zips/` |
| `.ipynb_checkpoints/` (Jupyter autosaves) | `archive/notebook_checkpoints/` |

**After the restructure (2026-10-30, v4.2):**
- `09_tft_colab_production_v4_1.ipynb` moved to `03_models/tft/notebooks/archive/`.
- The current notebook is `03_models/tft/notebooks/09_tft_colab_production_v4_2.ipynb`.
- Its results zip goes to `04_results/zips/` and is extracted to `04_results/tft_v4_2/`.
- `docs/RESULTS_GUIDE.md` now describes the v4.2 zip. The v4.1 zip keeps the guide it shipped with (`04_results/tft_v4_1/RESULTS_GUIDE_as_shipped.md`).

## Duplicates and same-named files

**Byte-identical extra copies** went to `archive/duplicates/`, renamed with a `__dup_<where it was>` suffix:
- the six raw CSVs that were copied into `unclean datasets/`;
- two extra copies of the v3-target master;
- an extra `district_data_quality` file;
- `09_tft_colab.ipynb`, which was identical to `09_tft_colab_production.ipynb`.

**Backups** were renamed with a `__pre_2026_04_extend`, `__pre_2026_10_extend` or `__v3_target` suffix. This means no backup shares a name with a live file.

**Three different datasets used to be called `master_dataset.csv`.** They now have distinct names:

| File | What it is |
|---|---|
| `data/clean/master_dataset.csv` | the latest output of `merge_master.py` |
| `data/master/master_dataset_v4target.csv` | **current** frozen training set (v4 / v4.1 results); currently identical to the file above |
| `data/master/master_dataset_v3target.csv` | old-target training set (v3-fixed comparisons). The TFT pipeline's default used to point at this one by mistake. It now points at the v4 target |
| `archive/backups/2026-10_pre_extend/master_dataset__pre_2026_10_extend.csv` | the master before the October 2026 data extension |

**Same content, different roles, kept on purpose:**
- `data/raw/rainfall_chirps_maharashtra_23_25.csv` (collector output) and `data/clean/rainfall_clean.csv` (preprocessing input).

**Names that repeat by convention only:**
- **Standard output files inside separate run folders**, such as `config.json` or `metrics_headline.csv` in each `04_results/<run>/<setup>/`. The folder identifies the run, and `docs/RESULTS_GUIDE.md` relies on these names.
- **Tool files:** `README.md` (root) and `.claude/settings.local.json`.

## What could not be recovered

The **May 2026 master dataset** used by the Phase 1 baselines (16,785 rows × 36 columns) had already been overwritten in July 2026. It was never in git, because `*.csv` is ignored.
- The Phase 1 scripts now point at `data/clean/master_dataset.csv`, so a re-run uses today's data and will not reproduce the May numbers exactly.
- The original Phase 1 outputs are preserved in `04_results/phase1_baselines/`.
