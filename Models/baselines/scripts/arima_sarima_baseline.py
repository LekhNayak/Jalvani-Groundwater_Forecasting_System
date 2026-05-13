"""
ARIMA / SARIMA baseline — one model per district, univariate GWL only.

This is the statistical floor: no environmental features, no spatial context.
Purpose: show what pure time-series statistics achieves so TFT+GAT improvement
is clearly attributable to the model architecture and multi-source features.

Approach:
- Aggregate daily gwl_mean to monthly mean per district (SARIMA seasonal period = 12)
- Fit SARIMA(1,1,1)(1,1,0)[12] per district
- Train: 2023-01 → 2024-09  |  Val: 2024-10 → 2025-01  |  Test: 2025-02 → 2025-04

Run from: Models/baselines/
    python arima_sarima_baseline.py
"""

import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MASTER    = Path("../../Data Preprocessing/clean datasets/master_dataset.csv")
OUT_DIR   = Path(".")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_END = "2024-09-30"
VAL_END   = "2025-01-31"
SPARSE_THRESHOLD = 0.40   # districts with ≥40% missing GWL are skipped

# ─── SARIMA ORDER ─────────────────────────────────────────────────────────────
ORDER         = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 0, 12)


def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))

def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def mape(actual, pred):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100


def main():
    print("Loading master dataset ...")
    df = pd.read_csv(MASTER, parse_dates=["date"])
    print(f"  {len(df):,} rows | {df['district'].nunique()} districts")

    # ── Monthly aggregation ────────────────────────────────────────────────────
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby(["district", "year_month"])["gwl_mean"]
        .mean()
        .reset_index()
    )
    monthly["date"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly.sort_values(["district", "date"])

    all_preds = []
    metrics   = []
    skipped   = []

    districts = sorted(monthly["district"].unique())
    print(f"\nFitting SARIMA{ORDER}x{SEASONAL_ORDER} per district ...")
    print(f"  Seasonal period : 12 months")
    print(f"  Train end       : {TRAIN_END}")
    print(f"  Val end         : {VAL_END}")

    for dist in districts:
        grp = monthly[monthly["district"] == dist].set_index("date")["gwl_mean"]

        # Check sparsity on the daily data
        dist_daily = df[df["district"] == dist]
        missing_pct = dist_daily["gwl_mean"].isna().mean()
        if missing_pct >= SPARSE_THRESHOLD:
            print(f"  [SKIP] {dist:<20} — {missing_pct:.0%} missing (SPARSE)")
            skipped.append(dist)
            continue

        train = grp[grp.index <= TRAIN_END]
        val   = grp[(grp.index > TRAIN_END) & (grp.index <= VAL_END)]
        test  = grp[grp.index > VAL_END]

        if len(train) < 12:
            print(f"  [SKIP] {dist:<20} — insufficient training months ({len(train)})")
            skipped.append(dist)
            continue

        try:
            model  = SARIMAX(train, order=ORDER, seasonal_order=SEASONAL_ORDER,
                             enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit(disp=False)

            # ── Validation predictions ─────────────────────────────────────────
            val_preds = []
            history   = list(train)
            for _ in range(len(val)):
                m = SARIMAX(history, order=ORDER, seasonal_order=SEASONAL_ORDER,
                            enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit(disp=False)
                val_preds.append(r.forecast(steps=1)[0])
                history.append(val.iloc[len(val_preds) - 1])

            val_actual = val.values
            val_pred   = np.array(val_preds)

            # ── Test predictions (roll forward through val before test) ────────
            history = list(train) + list(val)
            test_preds = []
            for _ in range(len(test)):
                m = SARIMAX(history, order=ORDER, seasonal_order=SEASONAL_ORDER,
                            enforce_stationarity=False, enforce_invertibility=False)
                r = m.fit(disp=False)
                test_preds.append(r.forecast(steps=1)[0])
                history.append(test.iloc[len(test_preds) - 1])

            test_actual = test.values
            test_pred   = np.array(test_preds)

            val_rmse  = rmse(val_actual, val_pred)
            val_mae   = mae(val_actual, val_pred)
            val_mape  = mape(val_actual, val_pred)
            test_rmse = rmse(test_actual, test_pred)
            test_mae  = mae(test_actual, test_pred)
            test_mape = mape(test_actual, test_pred)

            print(f"  {dist:<20} val RMSE={val_rmse:.3f}m  test RMSE={test_rmse:.3f}m")

            metrics.append({
                "district":  dist,
                "val_rmse":  round(val_rmse,  4),
                "val_mae":   round(val_mae,   4),
                "val_mape":  round(val_mape,  4),
                "test_rmse": round(test_rmse, 4),
                "test_mae":  round(test_mae,  4),
                "test_mape": round(test_mape, 4),
            })

            # Store predictions for val + test combined
            for idx, (actual, pred) in enumerate(zip(list(val_actual) + list(test_actual),
                                                     list(val_pred) + list(test_pred))):
                split = "val" if idx < len(val_actual) else "test"
                date  = (list(val.index) + list(test.index))[idx]
                all_preds.append({
                    "district":     dist,
                    "date":         date,
                    "gwl_actual":   actual,
                    "gwl_predicted": pred,
                    "split":        split,
                })

        except Exception as e:
            print(f"  [FAIL] {dist:<20} — {e}")
            skipped.append(dist)

    # ─── SAVE PREDICTIONS ─────────────────────────────────────────────────────
    preds_df = pd.DataFrame(all_preds)
    preds_df.to_csv(OUT_DIR / "arima_predictions.csv", index=False)
    print(f"\nPredictions saved -> arima_predictions.csv")

    # ─── METRICS SUMMARY ──────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics)
    print("\n" + "="*70)
    print("SARIMA — Per-District Results")
    print("="*70)
    print(metrics_df.to_string(index=False))

    print("\n" + "="*70)
    print("SARIMA — Aggregate (mean across districts)")
    print("="*70)
    agg = metrics_df[["val_rmse", "val_mae", "val_mape", "test_rmse", "test_mae", "test_mape"]].mean()
    print(agg.to_string())

    if skipped:
        print(f"\n[NOTE] Skipped {len(skipped)} districts: {skipped}")

    metrics_df.to_csv(OUT_DIR / "arima_metrics.csv", index=False)
    print("\nMetrics saved -> arima_metrics.csv")


if __name__ == "__main__":
    main()
