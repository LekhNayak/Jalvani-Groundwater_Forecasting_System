"""
XGBoost baseline — single model, all 36 districts, all available features.

Unlike LSTM, XGBoost handles mixed static/dynamic features naturally.
No need to repeat static features — just include them as regular columns.
Soil texture and district_id are included as regular features.

Key addition: SHAP analysis after training to understand which features
drive groundwater prediction. These findings directly inform TFT architecture.

Split (by time, never random):
    Train : 2023-01-01 → 2024-09-30
    Val   : 2024-10-01 → 2025-01-31
    Test  : 2025-02-01 → 2025-04-22

Run from: Models/baselines/
    python xgboost_baseline.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MASTER  = Path("../../Data Preprocessing/clean datasets/master_dataset.csv")
OUT_DIR = Path(".")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_END = "2024-09-30"
VAL_END   = "2025-01-31"

FEATURE_COLS = [
    # Time-series / dynamic
    "gwl_lag_7", "gwl_lag_14", "gwl_lag_30",
    "gwl_7d_change", "gwl_30d_change",
    "rainfall_mm", "rainfall_7d", "rainfall_30d",
    "et_mm", "soil_moisture", "net_water_balance",
    "temp_max", "temp_min", "temp_mean", "humidity_mean", "vpd_approx",
    "month", "day_of_year", "is_monsoon",
    # Static
    "district_id",
    "sand_0cm", "clay_0cm", "silt_0cm",
    "sand_10cm", "clay_10cm", "silt_10cm",
    "sand_30cm", "clay_30cm", "silt_30cm",
]
TARGET = "gwl_mean"


def compute_metrics(actual, pred):
    actual = np.array(actual, dtype=float)
    pred   = np.array(pred,   dtype=float)
    rmse   = np.sqrt(np.mean((actual - pred) ** 2))
    mae    = np.mean(np.abs(actual - pred))
    mask   = actual != 0
    mape   = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
    return rmse, mae, mape


def main():
    print("Loading master dataset ...")
    df = pd.read_csv(MASTER, parse_dates=["date"])
    print(f"  {len(df):,} rows | {df['district'].nunique()} districts")

    # ── Use only rows where all feature cols are present ──────────────────────
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    df_model = df.dropna(subset=available_features + [TARGET]).copy()
    print(f"  Rows after dropping NaN features/target: {len(df_model):,}")

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df = df_model[df_model["date"] <= TRAIN_END]
    val_df   = df_model[(df_model["date"] > TRAIN_END) & (df_model["date"] <= VAL_END)]
    test_df  = df_model[df_model["date"] > VAL_END]
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    X_train = train_df[available_features].values
    y_train = train_df[TARGET].values
    X_val   = val_df[available_features].values
    y_val   = val_df[TARGET].values
    X_test  = test_df[available_features].values
    y_test  = test_df[TARGET].values

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining XGBoost ...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse",
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_rmse,  val_mae,  val_mape  = compute_metrics(y_val,  val_pred)
    test_rmse, test_mae, test_mape = compute_metrics(y_test, test_pred)

    print(f"\nResults:")
    print(f"  Val  RMSE={val_rmse:.4f}m  MAE={val_mae:.4f}m  MAPE={val_mape:.2f}%")
    print(f"  Test RMSE={test_rmse:.4f}m  MAE={test_mae:.4f}m  MAPE={test_mape:.2f}%")

    # ── Per-district metrics ───────────────────────────────────────────────────
    dist_map   = dict(zip(df["district_id"], df["district"]))
    test_df_cp = test_df.copy()
    test_df_cp["gwl_pred"] = test_pred
    test_df_cp["gwl_actual"] = y_test

    print("\nPer-district test RMSE:")
    dist_metrics = []
    for dist_id, grp in test_df_cp.groupby("district_id"):
        r, m, mp = compute_metrics(grp["gwl_actual"].values, grp["gwl_pred"].values)
        dist_metrics.append({
            "district": dist_map.get(dist_id, dist_id),
            "rmse": round(r, 4), "mae": round(m, 4), "mape": round(mp, 2)
        })
    dm = pd.DataFrame(dist_metrics).sort_values("rmse", ascending=False)
    print(dm.to_string(index=False))
    dm.to_csv(OUT_DIR / "xgboost_metrics.csv", index=False)

    # ── Save predictions ───────────────────────────────────────────────────────
    val_df_cp = val_df.copy()
    val_df_cp["gwl_pred"]   = val_pred
    val_df_cp["gwl_actual"] = y_val
    val_df_cp["split"] = "val"
    test_df_cp["split"] = "test"

    preds_df = pd.concat([
        val_df_cp[["district", "date", "gwl_actual", "gwl_pred", "split"]],
        test_df_cp[["district", "date", "gwl_actual", "gwl_pred", "split"]],
    ], ignore_index=True)
    preds_df.to_csv(OUT_DIR / "xgboost_predictions.csv", index=False)
    print(f"\nPredictions saved -> xgboost_predictions.csv")

    # ── Feature importance ────────────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature":    available_features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nTop 15 features by XGBoost gain importance:")
    print(importance.head(15).to_string(index=False))
    importance.to_csv(OUT_DIR / "xgboost_feature_importance.csv", index=False)

    # ── SHAP analysis ──────────────────────────────────────────────────────────
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("\nRunning SHAP analysis (sample of 2000 rows) ...")
        sample_idx = np.random.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
        X_shap     = X_test[sample_idx]

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_shap,
            feature_names=available_features,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.title("XGBoost — SHAP Feature Importance (test set sample)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "xgboost_shap_summary.png", dpi=150)
        plt.close()
        print("SHAP summary plot saved -> xgboost_shap_summary.png")

        # Also save mean |SHAP| per feature
        shap_mean = pd.DataFrame({
            "feature":    available_features,
            "mean_shap":  np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_shap", ascending=False)
        shap_mean.to_csv(OUT_DIR / "xgboost_shap_values.csv", index=False)
        print("SHAP values saved -> xgboost_shap_values.csv")
        print("\nTop 10 features by mean |SHAP|:")
        print(shap_mean.head(10).to_string(index=False))

    except ImportError:
        print("\n[NOTE] shap not installed — skipping SHAP analysis.")
        print("  Install with: pip install shap")

    print("\nDone.")


if __name__ == "__main__":
    main()
