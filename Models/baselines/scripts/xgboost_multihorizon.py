"""
XGBoost Multi-Horizon Baseline (JalVaani)

Direct h-step forecasting: one separate XGBoost model per horizon.
Target for horizon h = gwl_mean shifted -h days (per district to avoid cross-district leakage).

SHAP analysis is run for each horizon to show how feature importance evolves:
  - t+7:  lag features expected to dominate (mostly autoregressive)
  - t+14: lag features weaken, environmental features begin to emerge
  - t+30: seasonal + rainfall features expected to dominate

Split (by time, never random):
    Train : data up to 2024-09-30
    Val   : 2024-10-01 → 2025-01-31
    Test  : 2025-02-01 → 2025-04-22

Run from: Models/baselines/scripts/
    python xgboost_multihorizon.py
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

MASTER    = Path("../../Data Preprocessing/clean datasets/master_dataset.csv")
OUT_DIR   = Path("../outputs")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

TRAIN_END = "2024-09-30"
VAL_END   = "2025-01-31"
HORIZONS  = [7, 14, 30]

FEATURE_COLS = [
    # Dynamic / time-series
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

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric="rmse",
    verbosity=0,
)


def compute_metrics(actual, pred):
    a, p = np.array(actual, float), np.array(pred, float)
    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae  = np.mean(np.abs(a - p))
    mask = a != 0
    mape = np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100 if mask.any() else np.nan
    return rmse, mae, mape


def main():
    print("Loading master dataset ...")
    df = pd.read_csv(MASTER, parse_dates=["date"])
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    print(f"  {len(df):,} rows | {df['district'].nunique()} districts")

    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    print(f"  Features available: {len(available_feats)}/{len(FEATURE_COLS)}")

    # Create shifted targets per district (no cross-district contamination)
    for h in HORIZONS:
        df[f"target_t{h}"] = df.groupby("district")["gwl_mean"].shift(-h)

    # Split flag
    df["split"] = "train"
    df.loc[df["date"] > TRAIN_END, "split"] = "val"
    df.loc[df["date"] > VAL_END,   "split"] = "test"

    # Drop rows missing features
    df_clean = df.dropna(subset=available_feats)
    print(f"  Rows after feature NaN drop: {len(df_clean):,}")

    # Summary storage for final table
    summary_rows = []

    for h in HORIZONS:
        print(f"\n{'='*50}")
        print(f"  Horizon t+{h}")
        print(f"{'='*50}")

        target_col = f"target_t{h}"
        df_h = df_clean.dropna(subset=[target_col]).copy()

        train = df_h[df_h["split"] == "train"]
        val   = df_h[df_h["split"] == "val"]
        test  = df_h[df_h["split"] == "test"]

        X_tr, y_tr = train[available_feats], train[target_col]
        X_va, y_va = val[available_feats],   val[target_col]
        X_te, y_te = test[available_feats],  test[target_col]

        print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}  Test: {len(X_te):,}")

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=50,
        )
        best_iter = model.best_iteration
        print(f"  Best iteration: {best_iter}")

        val_pred  = model.predict(X_va)
        test_pred = model.predict(X_te)

        val_r,  val_m,  val_mp  = compute_metrics(y_va.values, val_pred)
        test_r, test_m, test_mp = compute_metrics(y_te.values, test_pred)

        print(f"  Val  RMSE={val_r:.4f}  MAE={val_m:.4f}  MAPE={val_mp:.2f}%")
        print(f"  Test RMSE={test_r:.4f}  MAE={test_m:.4f}  MAPE={test_mp:.2f}%")

        summary_rows.append({
            "model": "XGBoost", "horizon": h,
            "val_rmse": round(val_r, 4), "val_mae": round(val_m, 4),
            "test_rmse": round(test_r, 4), "test_mae": round(test_m, 4),
        })

        # Per-district test metrics
        test_cp = test.copy()
        test_cp["gwl_pred"]   = test_pred
        test_cp["gwl_actual"] = y_te.values
        dist_rows = []
        for dist, grp in test_cp.groupby("district"):
            r, m, _ = compute_metrics(grp["gwl_actual"], grp["gwl_pred"])
            dist_rows.append({"district": dist, "rmse": round(r, 4), "mae": round(m, 4)})
        dist_df = pd.DataFrame(dist_rows).sort_values("rmse", ascending=False)
        print(f"\n  Per-district test RMSE (t+{h}):")
        print("  " + dist_df.to_string(index=False).replace("\n", "\n  "))

        # Save predictions
        val_cp = val.copy()
        val_cp["gwl_pred"]   = val_pred
        val_cp["gwl_actual"] = y_va.values
        val_cp["split"]      = "val"
        test_cp["split"]     = "test"

        pred_df = pd.concat([
            val_cp[["district", "date", "gwl_actual", "gwl_pred", "split"]],
            test_cp[["district", "date", "gwl_actual", "gwl_pred", "split"]],
        ], ignore_index=True)
        pred_df["horizon"] = h
        pred_path = OUT_DIR / f"xgboost_multihorizon_t{h}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"\n  Predictions saved → {pred_path.name}")

        # SHAP analysis
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            print(f"  Running SHAP (2000-row sample) ...")
            rng     = np.random.default_rng(42)
            idx     = rng.choice(len(X_te), size=min(2000, len(X_te)), replace=False)
            X_shap  = X_te.iloc[idx]

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)

            mean_abs = pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=available_feats,
            ).sort_values(ascending=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, max(5, len(available_feats) * 0.28)))
            mean_abs.plot(kind="barh", ax=ax, color="#377eb8")
            ax.set_title(f"XGBoost SHAP Feature Importance — t+{h}")
            ax.set_xlabel("mean |SHAP value|  (metres)")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"xgboost_shap_t{h}.png", dpi=150)
            plt.close()

            shap_df = mean_abs.reset_index()
            shap_df.columns = ["feature", "mean_abs_shap"]
            shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
            shap_df.to_csv(OUT_DIR / f"xgboost_shap_t{h}.csv", index=False)

            top5 = shap_df["feature"].head(5).tolist()
            print(f"  Top-5 SHAP: {top5}")
            print(f"  SHAP plot saved → plots/xgboost_shap_t{h}.png")

        except ImportError:
            print("  [SHAP not installed — skipping]  pip install shap")

    # Final summary table
    print("\n" + "=" * 60)
    print("  XGBoost Multi-Horizon Summary")
    print("=" * 60)
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "xgboost_multihorizon_summary.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
