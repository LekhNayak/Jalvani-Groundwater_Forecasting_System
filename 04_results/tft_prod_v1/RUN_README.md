# 04_results/tft_prod_v1 (formerly `Latest results/`) — this is the v1 run (superseded)

These files come from the **v1** pipeline (`03_models/tft/notebooks/archive/09_tft_colab_production.ipynb` + `tft_model.py`,
11.6M parameters, absolute-level target) scored against the **old, pre-fix groundwater series**.
Against a "nothing changes" forecast it does not win (median MASE 2.12 / 1.37 / 1.04 at t+7/t+14/t+30),
and its headline MAPE is a median across districts that hides this.

Do not quote these as the project's results. The current results are the v4 run:
`docs/history/TFT_VERSION_HISTORY.md` (v4 section) and `03_models/tft/notebooks/archive/09_tft_colab_production_v4.ipynb`.
