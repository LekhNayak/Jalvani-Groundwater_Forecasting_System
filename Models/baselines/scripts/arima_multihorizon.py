"""
ARIMA Multi-Horizon Baseline (JalVaani)

Fits ARIMA(7,1,1) per district on DAILY gwl_mean data.
Uses monthly rolling refit (every 30 steps) for efficiency.
At each step evaluates h-step-ahead forecasts for h = 7, 14, 30 days.

Rolling forecast design:
  - Model is fit on all history up to the last refit point.
  - At step i with steps_since_refit=s, forecast[s+h] = h-step ahead prediction
    from the current date (see inline comments for derivation).
  - This gives a proper rolling evaluation that avoids data leakage.

Split (by time, never random):
    Train : data up to 2024-09-30
    Val   : 2024-10-01 → 2025-01-31
    Test  : 2025-02-01 → 2025-04-22

Run from: Models/baselines/scripts/
    python arima_multihorizon.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MASTER    = Path("../../Data Preprocessing/clean datasets/master_dataset.csv")
OUT_DIR   = Path("../outputs")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_END    = pd.Timestamp("2024-09-30")
VAL_END      = pd.Timestamp("2025-01-31")
HORIZONS     = [7, 14, 30]
ORDER        = (7, 1, 1)          # AR(7) with differencing + MA(1) for daily data
MIN_TRAIN    = 90                 # skip districts with < 90 valid training obs
REFIT_EVERY  = 30                 # monthly rolling refit


def compute_metrics(actual, pred):
    a, p = np.array(actual, float), np.array(pred, float)
    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae  = np.mean(np.abs(a - p))
    mask = a != 0
    mape = np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100 if mask.any() else np.nan
    return rmse, mae, mape


def fill_series(s):
    """Linear interpolation + boundary fill for ARIMA-compatible series."""
    return s.interpolate("linear").bfill().ffill()


def main():
    print("Loading master dataset ...")
    df = pd.read_csv(MASTER, parse_dates=["date"])
    print(f"  {len(df):,} rows | {df['district'].nunique()} districts")

    districts = sorted(df["district"].unique())
    all_rows  = []

    for dist in tqdm(districts, desc="ARIMA multi-horizon"):
        sub = (
            df[df["district"] == dist][["date", "gwl_mean"]]
            .set_index("date")
            .sort_index()
            .asfreq("D")           # ensure daily frequency, gaps become NaN
        )

        train_s = sub.loc[sub.index <= TRAIN_END, "gwl_mean"]
        val_s   = sub.loc[(sub.index > TRAIN_END) & (sub.index <= VAL_END), "gwl_mean"]
        test_s  = sub.loc[sub.index > VAL_END, "gwl_mean"]

        if train_s.notna().sum() < MIN_TRAIN:
            tqdm.write(f"  Skip {dist}: only {train_s.notna().sum()} valid train obs")
            continue

        # Fill gaps in training for model fit
        history = list(fill_series(train_s).values)

        # ── Initial ARIMA fit ──────────────────────────────────────────────────
        try:
            cur_model = ARIMA(
                np.array(history), order=ORDER
            ).fit(method_kwargs={"warn_convergence": False})
        except Exception as e:
            tqdm.write(f"  Fail {dist} (initial fit): {e}")
            continue

        last_refit_step = 0
        fc_cache        = None   # cached forecast array from last refit

        # ── Rolling evaluation over val + test ─────────────────────────────────
        combined = pd.concat([val_s, test_s])

        for i, (date, obs) in enumerate(combined.items()):
            split_name = "val" if date <= VAL_END else "test"
            steps_since_refit = i - last_refit_step

            # Compute / refresh forecast cache
            if fc_cache is None:
                fc_len   = REFIT_EVERY + max(HORIZONS) + 1
                try:
                    fc_cache = cur_model.forecast(steps=fc_len)
                except Exception:
                    fc_cache = np.full(fc_len, np.nan)

            # Record prediction for each horizon
            # Derivation: model was fit on history ending `last_refit_step` steps ago.
            # fc_cache[0] = prediction 1 step after last refit.
            # Current date is `steps_since_refit` steps after last refit.
            # h-step ahead from current date = steps_since_refit + h steps from last refit.
            # → fc_cache index = steps_since_refit + h  (0-indexed fc_cache[k] = k+1 step ahead)
            # Wait: fc_cache[0] = 1-step prediction, so fc_cache[k] = (k+1)-step prediction.
            # h-step from current = forecast of position last_refit + steps_since_refit + h.
            # That is (steps_since_refit + h) steps ahead from last refit.
            # → fc_cache[steps_since_refit + h - 1] ... but step 0 → fc[0] is 1-step ahead.
            # Resolve: fc[k] is the (k+1)-step prediction from end of fitted history.
            # (steps_since_refit + h)-step prediction → fc[steps_since_refit + h - 1].
            # For step i=0 (right at refit), steps_since_refit=0 → fc[h-1] = h-step ahead ✓
            # For step i=5, steps_since_refit=5 → fc[5+h-1] = (5+h)-step ahead from refit,
            # which equals h-step ahead from current ✓
            for h in HORIZONS:
                fc_idx = steps_since_refit + h - 1
                if fc_idx >= len(fc_cache):
                    continue
                pred_val    = float(fc_cache[fc_idx])
                future_date = date + pd.Timedelta(days=h)
                if future_date in sub.index:
                    actual_h = sub.loc[future_date, "gwl_mean"]
                    if not pd.isna(actual_h) and not np.isnan(pred_val):
                        all_rows.append({
                            "district":   dist,
                            "date":       future_date,
                            "horizon":    h,
                            "gwl_actual": float(actual_h),
                            "gwl_pred":   pred_val,
                            "split":      split_name,
                        })

            # Append observation to history
            obs_val = float(obs) if not pd.isna(obs) else history[-1]
            history.append(obs_val)

            # Monthly rolling refit
            if (i - last_refit_step + 1) >= REFIT_EVERY:
                try:
                    cur_model = ARIMA(
                        np.array(history), order=ORDER
                    ).fit(
                        start_params=cur_model.params,
                        method_kwargs={"warn_convergence": False},
                    )
                except Exception:
                    try:
                        cur_model = ARIMA(np.array(history), order=ORDER).fit(
                            method_kwargs={"warn_convergence": False}
                        )
                    except Exception:
                        pass
                last_refit_step = i + 1
                fc_cache        = None   # invalidate; will recompute next step

        tqdm.write(f"  ✓ {dist}")

    out_df = pd.DataFrame(all_rows)
    print(f"\nTotal prediction records: {len(out_df):,}")

    # ── Save per-horizon CSVs ─────────────────────────────────────────────────
    for h in HORIZONS:
        sub_h    = out_df[out_df["horizon"] == h]
        out_path = OUT_DIR / f"arima_multihorizon_t{h}.csv"
        sub_h.to_csv(out_path, index=False)
        print(f"  Saved {out_path.name}  ({len(sub_h):,} rows)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== ARIMA Multi-Horizon — Test Metrics ===")
    print(f"{'Horizon':<10} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
    print("-" * 38)
    for h in HORIZONS:
        sub_h = out_df[(out_df["horizon"] == h) & (out_df["split"] == "test")]
        if len(sub_h) == 0:
            continue
        r, m, mp = compute_metrics(sub_h["gwl_actual"], sub_h["gwl_pred"])
        print(f"  t+{h:<7d} {r:>8.4f} {m:>8.4f} {mp:>7.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
