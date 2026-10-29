"""
Unified evaluation table for all baseline models.
Loads prediction CSVs from whichever models have been run and prints a comparison table.

Designed to be extended — Phase 2 (TFT) and Phase 3 (TFT+GAT) prediction CSVs
will be picked up automatically when they exist in the same directory.

Run from: 03_models/baselines_phase1/
    python evaluate_baselines.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("../../../04_results/phase1_baselines")

# Map: model name → prediction CSV filename
# Add TFT and GAT+TFT prediction files here when ready
MODEL_FILES = {
    "SARIMA":   "arima_predictions.csv",
    "XGBoost":  "xgboost_predictions.csv",
    "LSTM":     "lstm_predictions.csv",
    # "TFT":      "tft_predictions.csv",        # Phase 2
    # "TFT+GAT":  "tft_gat_predictions.csv",    # Phase 3
}

# Expected columns: district, date (optional), gwl_actual, gwl_predicted or gwl_pred, split
ACTUAL_COL = "gwl_actual"
PRED_COLS  = ["gwl_predicted", "gwl_pred"]   # either column name is accepted


def load_predictions(path):
    df = pd.read_csv(path)
    for col in PRED_COLS:
        if col in df.columns:
            df = df.rename(columns={col: "gwl_pred"})
            break
    if "gwl_pred" not in df.columns:
        raise ValueError(f"No prediction column found in {path}. Expected one of: {PRED_COLS}")
    return df


def compute_metrics(actual, pred):
    actual = np.array(actual, dtype=float)
    pred   = np.array(pred,   dtype=float)
    valid  = ~(np.isnan(actual) | np.isnan(pred))
    actual, pred = actual[valid], pred[valid]
    if len(actual) == 0:
        return np.nan, np.nan, np.nan
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mae  = np.mean(np.abs(actual - pred))
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100 if mask.any() else np.nan
    return rmse, mae, mape


def main():
    print("=" * 68)
    print("JalVaani — Baseline Model Comparison")
    print("=" * 68)

    all_metrics    = []
    per_dist_data  = {}

    for model_name, filename in MODEL_FILES.items():
        fpath = OUT_DIR / filename
        if not fpath.exists():
            print(f"  [{model_name}] — prediction file not found: {filename}  (skipped)")
            continue

        df = load_predictions(fpath)

        val_df   = df[df["split"] == "val"]
        test_df  = df[df["split"] == "test"]

        val_rmse,  val_mae,  val_mape  = compute_metrics(val_df[ACTUAL_COL],  val_df["gwl_pred"])
        test_rmse, test_mae, test_mape = compute_metrics(test_df[ACTUAL_COL], test_df["gwl_pred"])

        all_metrics.append({
            "Model":     model_name,
            "Val RMSE":  round(val_rmse,  4) if not np.isnan(val_rmse)  else "N/A",
            "Val MAE":   round(val_mae,   4) if not np.isnan(val_mae)   else "N/A",
            "Val MAPE":  round(val_mape,  2) if not np.isnan(val_mape)  else "N/A",
            "Test RMSE": round(test_rmse, 4) if not np.isnan(test_rmse) else "N/A",
            "Test MAE":  round(test_mae,  4) if not np.isnan(test_mae)  else "N/A",
            "Test MAPE": round(test_mape, 2) if not np.isnan(test_mape) else "N/A",
        })

        # Per-district test metrics
        if "district" in test_df.columns:
            dist_metrics = {}
            for dist, grp in test_df.groupby("district"):
                r, m, _ = compute_metrics(grp[ACTUAL_COL].values, grp["gwl_pred"].values)
                dist_metrics[dist] = round(r, 4)
            per_dist_data[model_name] = dist_metrics

    if not all_metrics:
        print("\nNo prediction files found. Run the baseline scripts first.")
        return

    # ── Aggregate table ───────────────────────────────────────────────────────
    summary = pd.DataFrame(all_metrics)
    print("\nAggregate Results:")
    print("-" * 68)
    print(summary.to_string(index=False))

    # ── Per-district table ────────────────────────────────────────────────────
    if per_dist_data:
        all_districts = sorted(set(d for m in per_dist_data.values() for d in m))
        rows = []
        for dist in all_districts:
            row = {"District": dist}
            for model_name in per_dist_data:
                row[f"{model_name} RMSE"] = per_dist_data[model_name].get(dist, "—")
            rows.append(row)
        dist_table = pd.DataFrame(rows)
        print("\nPer-District Test RMSE (metres):")
        print("-" * 68)
        print(dist_table.to_string(index=False))
        dist_table.to_csv(OUT_DIR / "comparison_per_district.csv", index=False)
        print(f"\nPer-district table saved -> comparison_per_district.csv")

    summary.to_csv(OUT_DIR / "comparison_summary.csv", index=False)
    print(f"Summary table saved -> comparison_summary.csv")
    print("=" * 68)


if __name__ == "__main__":
    main()
