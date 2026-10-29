"""
Multi-Horizon Evaluation (JalVaani)

Loads prediction CSVs from all models that have been run and builds:
  1. Aggregate comparison table: model × horizon × {RMSE, MAE, MAPE}
  2. Per-district breakdown for each horizon
  3. Feature importance crossover analysis (which SHAP features dominate at each horizon)

Designed to be extended — TFT and TFT+GAT rows will be picked up automatically
when their prediction files exist.

Run from: 03_models/baselines_phase1/scripts/
    python evaluate_multihorizon.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("../../../04_results/phase1_baselines")

HORIZONS = [7, 14, 30]

MODEL_FILES = {
    "ARIMA":   "arima_multihorizon_t{h}.csv",
    "XGBoost": "xgboost_multihorizon_t{h}.csv",
    "LSTM":    "lstm_multihorizon_t{h}.csv",
    # "TFT":     "tft_multihorizon_t{h}.csv",       # Phase 2
    # "TFT+GAT": "tft_gat_multihorizon_t{h}.csv",   # Phase 3
}

SHAP_FILES = {
    7:  OUT_DIR / "xgboost_shap_t7.csv",
    14: OUT_DIR / "xgboost_shap_t14.csv",
    30: OUT_DIR / "xgboost_shap_t30.csv",
}


def compute_metrics(actual, pred):
    a, p = np.array(actual, float), np.array(pred, float)
    valid = ~(np.isnan(a) | np.isnan(p))
    a, p  = a[valid], p[valid]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae  = np.mean(np.abs(a - p))
    mask = a != 0
    mape = np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100 if mask.any() else np.nan
    return rmse, mae, mape


def main():
    # ── Load predictions ──────────────────────────────────────────────────────
    loaded = {}  # (model, horizon) → DataFrame
    for model, tmpl in MODEL_FILES.items():
        for h in HORIZONS:
            path = OUT_DIR / tmpl.format(h=h)
            if path.exists():
                df = pd.read_csv(path)
                loaded[(model, h)] = df
                print(f"  ✅ {model:<10} t+{h:<2d} — {len(df):,} rows")
            else:
                print(f"  ❌ {model:<10} t+{h:<2d} — not found ({path.name})")

    if not loaded:
        print("\nNo prediction files found. Run multi-horizon baseline scripts first.")
        return

    # ── Aggregate metrics table ───────────────────────────────────────────────
    rows = []
    for (model, h), df in loaded.items():
        val_df  = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
        vr, vm, vp = compute_metrics(val_df["gwl_actual"],  val_df["gwl_pred"])
        tr, tm, tp = compute_metrics(test_df["gwl_actual"], test_df["gwl_pred"])
        rows.append({
            "Model":     model,
            "Horizon":   f"t+{h}",
            "Val RMSE":  round(vr, 4),
            "Val MAE":   round(vm, 4),
            "Test RMSE": round(tr, 4),
            "Test MAE":  round(tm, 4),
            "Test MAPE": round(tp, 2),
        })

    summary = pd.DataFrame(rows).sort_values(["Horizon", "Test RMSE"])
    summary.to_csv(OUT_DIR / "multihorizon_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("  Multi-Horizon Comparison — Test RMSE / MAE / MAPE")
    print("=" * 70)
    for h in HORIZONS:
        sub = summary[summary["Horizon"] == f"t+{h}"]
        if len(sub) == 0:
            continue
        print(f"\n  Horizon t+{h}:")
        print("  " + sub[["Model", "Test RMSE", "Test MAE", "Test MAPE"]]
              .to_string(index=False).replace("\n", "\n  "))

    # ── Crossover analysis ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Crossover Analysis: At which horizon does LSTM beat XGBoost?")
    print("=" * 70)
    for h in HORIZONS:
        xgb_row  = summary[(summary["Model"] == "XGBoost") & (summary["Horizon"] == f"t+{h}")]
        lstm_row = summary[(summary["Model"] == "LSTM")    & (summary["Horizon"] == f"t+{h}")]
        if xgb_row.empty or lstm_row.empty:
            continue
        xgb_rmse  = float(xgb_row["Test RMSE"].values[0])
        lstm_rmse = float(lstm_row["Test RMSE"].values[0])
        winner    = "LSTM" if lstm_rmse < xgb_rmse else "XGBoost"
        delta     = abs(lstm_rmse - xgb_rmse)
        print(f"  t+{h:<2d}: XGBoost={xgb_rmse:.4f}m  LSTM={lstm_rmse:.4f}m  "
              f"→ {winner} wins by {delta:.4f}m")

    # ── SHAP crossover ────────────────────────────────────────────────────────
    shap_dfs = {}
    for h, path in SHAP_FILES.items():
        if path.exists():
            shap_dfs[h] = pd.read_csv(path).set_index("feature")["mean_abs_shap"]

    if shap_dfs:
        print("\n" + "=" * 70)
        print("  SHAP Feature Importance Crossover (XGBoost, top-5 per horizon)")
        print("=" * 70)
        for h, shap in shap_dfs.items():
            top5 = shap.nlargest(5)
            print(f"\n  t+{h}:")
            for feat, val in top5.items():
                print(f"    {feat:<30} {val:.4f}")

        # Lag vs environmental features ratio
        lag_feats = ["gwl_lag_7", "gwl_lag_14", "gwl_lag_30", "gwl_7d_change", "gwl_30d_change"]
        env_feats = ["rainfall_mm", "rainfall_7d", "rainfall_30d", "et_mm",
                     "soil_moisture", "net_water_balance", "temp_mean", "humidity_mean"]
        print("\n  Lag vs Environmental SHAP ratio:")
        for h, shap in sorted(shap_dfs.items()):
            lag = sum(shap.get(f, 0) for f in lag_feats)
            env = sum(shap.get(f, 0) for f in env_feats)
            total = lag + env
            if total > 0:
                print(f"  t+{h:<2d}: lag={lag/(total)*100:.1f}%  env={env/(total)*100:.1f}%")

    # ── Per-district breakdown ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Per-District Test RMSE (best horizon = t+7 shown)")
    print("=" * 70)
    h_show = 7
    dist_rows = {}
    for (model, h), df in loaded.items():
        if h != h_show:
            continue
        test_df = df[df["split"] == "test"]
        if "district" not in test_df.columns:
            continue
        for dist, grp in test_df.groupby("district"):
            r, _, _ = compute_metrics(grp["gwl_actual"], grp["gwl_pred"])
            if dist not in dist_rows:
                dist_rows[dist] = {}
            dist_rows[dist][model] = round(r, 4)

    if dist_rows:
        dist_table = pd.DataFrame(dist_rows).T.reset_index().rename(columns={"index": "District"})
        dist_table.to_csv(OUT_DIR / "multihorizon_per_district.csv", index=False)
        print("  " + dist_table.to_string(index=False).replace("\n", "\n  "))

    print(f"\nSaved → multihorizon_summary.csv")
    print("Done.")


if __name__ == "__main__":
    main()
