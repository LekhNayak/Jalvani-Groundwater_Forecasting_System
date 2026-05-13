"""
LSTM Multi-Horizon Baseline (JalVaani)

Direct h-step forecasting: one separate model per horizon (h = 7, 14, 30).
Each model sees a 30-day lookback window and predicts gwl_mean exactly h days
after the last observed day in the window.

Architecture (same as t+1 baseline for fair comparison):
    ts_seq[t-30:t]    → LSTM(hidden=64, layers=2, dropout=0.2)
    static_soil ×9    → broadcast across all 30 time steps
    district_embed(8) → broadcast across all 30 time steps
    → Linear(64 → 1) → gwl_mean[t + h]

Sliding window construction:
    Window i: input = data[i-30:i], target = gwl_mean[i + h - 1]
    Split by TARGET date (not input date) to avoid leakage.

Split (by time, never random):
    Train : target date ≤ 2024-09-30
    Val   : 2024-10-01 ≤ target date ≤ 2025-01-31
    Test  : target date > 2025-01-31

Run from: Models/baselines/scripts/
    python lstm_multihorizon.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MASTER    = Path("../../Data Preprocessing/clean datasets/master_dataset.csv")
OUT_DIR   = Path("../outputs")
OUT_DIR.mkdir(exist_ok=True)

# ─── HYPER-PARAMETERS ─────────────────────────────────────────────────────────
LOOKBACK   = 30
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT    = 0.2
EMBED_DIM  = 8
BATCH_SIZE = 128
LR         = 1e-3
MAX_EPOCHS = 100
PATIENCE   = 15
LR_PAT     = 5

TRAIN_END = "2024-09-30"
VAL_END   = "2025-01-31"
HORIZONS  = [7, 14, 30]

TIME_SERIES_COLS = [
    "gwl_lag_7", "gwl_lag_14", "gwl_lag_30",
    "gwl_7d_change", "gwl_30d_change",
    "rainfall_mm", "rainfall_7d", "rainfall_30d",
    "et_mm", "soil_moisture", "net_water_balance",
    "temp_max", "temp_min", "temp_mean", "humidity_mean", "vpd_approx",
    "month", "day_of_year", "is_monsoon",
]
STATIC_COLS = [
    "sand_0cm", "clay_0cm", "silt_0cm",
    "sand_10cm", "clay_10cm", "silt_10cm",
    "sand_30cm", "clay_30cm", "silt_30cm",
]
TARGET = "gwl_mean"


# ─── DATASET ──────────────────────────────────────────────────────────────────
class GWLDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ts_seq, static, dist_id, target = self.windows[idx][:4]
        return (
            torch.tensor(ts_seq,  dtype=torch.float32),
            torch.tensor(static,  dtype=torch.float32),
            torch.tensor(dist_id, dtype=torch.long),
            torch.tensor(target,  dtype=torch.float32),
        )


def build_windows(df, ts_cols, static_cols, target_col, lookback, horizon,
                  train_end, val_end):
    """
    Build sliding windows split by TARGET date.
    Returns (train_windows, val_windows, test_windows).
    Each window = (ts_seq[lookback, ts_dim], static[9], dist_id, target_norm, date_str).
    """
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts   = pd.Timestamp(val_end)

    train_w, val_w, test_w = [], [], []

    for dist, grp in df.groupby("district"):
        grp  = grp.sort_values("date").reset_index(drop=True)
        ts   = grp[ts_cols].values.astype(float)   # [N, ts_dim]
        stat = grp[static_cols].values.astype(float)
        dids = grp["district_id"].values.astype(int)
        tgts = grp[target_col].values.astype(float)
        dates = grp["date"].values

        for i in range(lookback, len(grp) - horizon + 1):
            ts_seq = ts[i - lookback: i]
            if np.isnan(ts_seq).any():
                continue
            target_val  = tgts[i + horizon - 1]
            if np.isnan(target_val):
                continue
            static_vec  = stat[i].copy()
            if np.isnan(static_vec).any():
                static_vec = np.nan_to_num(static_vec, nan=0.0)
            dist_id     = int(dids[i])
            target_date = pd.Timestamp(dates[i + horizon - 1])

            w = (ts_seq, static_vec, dist_id, target_val, str(target_date.date()),
                 dist)

            if target_date <= train_end_ts:
                train_w.append(w)
            elif target_date <= val_end_ts:
                val_w.append(w)
            else:
                test_w.append(w)

    return train_w, val_w, test_w


# ─── MODEL ────────────────────────────────────────────────────────────────────
class VanillaLSTM(nn.Module):
    def __init__(self, ts_input_dim, static_dim, n_districts, embed_dim,
                 hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(n_districts, embed_dim)
        lstm_in    = ts_input_dim + static_dim + embed_dim
        self.lstm  = nn.LSTM(lstm_in, hidden_dim, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head  = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, ts_seq, static_vec, dist_id):
        B, T, _ = ts_seq.shape
        emb      = self.embed(dist_id)
        stat_exp = static_vec.unsqueeze(1).expand(-1, T, -1)
        emb_exp  = emb.unsqueeze(1).expand(-1, T, -1)
        x        = torch.cat([ts_seq, stat_exp, emb_exp], dim=-1)
        out, _   = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ─── METRICS ──────────────────────────────────────────────────────────────────
def compute_metrics(actual, pred):
    a, p = np.array(actual, float), np.array(pred, float)
    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae  = np.mean(np.abs(a - p))
    mask = a != 0
    mape = np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100 if mask.any() else np.nan
    return rmse, mae, mape


# ─── NORMALISATION ────────────────────────────────────────────────────────────
def fit_scaler(df, cols):
    means = df[cols].mean()
    stds  = df[cols].std().replace(0, 1)
    return means, stds


def apply_scaler(df, cols, means, stds):
    df = df.copy()
    df[cols] = (df[cols] - means) / stds
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading master dataset ...")
    raw = pd.read_csv(MASTER, parse_dates=["date"])
    raw = raw.sort_values(["district", "date"]).reset_index(drop=True)
    print(f"  {len(raw):,} rows | {raw['district'].nunique()} districts")

    n_districts = raw["district_id"].nunique()
    ts_dim      = len(TIME_SERIES_COLS)
    static_dim  = len(STATIC_COLS)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Normalise time-series + target on TRAIN statistics
    train_raw   = raw[raw["date"] <= TRAIN_END]
    norm_cols   = TIME_SERIES_COLS + [TARGET]
    ts_means, ts_stds = fit_scaler(train_raw, norm_cols)

    df_norm = apply_scaler(raw, norm_cols, ts_means, ts_stds)

    # Normalise static cols
    st_means, st_stds = fit_scaler(train_raw, STATIC_COLS)
    for col in STATIC_COLS:
        df_norm[col] = (df_norm[col].fillna(float(st_means[col])) - float(st_means[col])) / float(st_stds[col])

    target_mean = float(ts_means[TARGET])
    target_std  = float(ts_stds[TARGET])

    summary_rows = []

    for h in HORIZONS:
        print(f"\n{'='*55}")
        print(f"  Horizon t+{h}")
        print(f"{'='*55}")

        print(f"  Building sliding windows (lookback={LOOKBACK}, horizon={h}) ...")
        train_w, val_w, test_w = build_windows(
            df_norm, TIME_SERIES_COLS, STATIC_COLS, TARGET, LOOKBACK, h,
            TRAIN_END, VAL_END,
        )
        print(f"  Train: {len(train_w):,}  Val: {len(val_w):,}  Test: {len(test_w):,}")

        if len(val_w) == 0 or len(test_w) == 0:
            print(f"  Skip t+{h}: empty val or test split")
            continue

        train_loader = DataLoader(GWLDataset(train_w), BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(GWLDataset(val_w),   BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader  = DataLoader(GWLDataset(test_w),  BATCH_SIZE, shuffle=False, num_workers=0)

        model = VanillaLSTM(ts_dim, static_dim, n_districts, EMBED_DIM,
                            HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler  = ReduceLROnPlateau(optimizer, patience=LR_PAT, factor=0.5, verbose=False)
        criterion  = nn.MSELoss()

        best_val   = float("inf")
        pat_ctr    = 0
        best_state = None

        print(f"  Training (max {MAX_EPOCHS} epochs, early stop patience={PATIENCE}) ...")
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            tr_loss = 0.0
            for ts_seq, stat, did, tgt in train_loader:
                ts_seq, stat, did, tgt = (ts_seq.to(device), stat.to(device),
                                          did.to(device),   tgt.to(device))
                optimizer.zero_grad()
                pred    = model(ts_seq, stat, did)
                loss    = criterion(pred, tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * len(tgt)
            tr_loss /= len(train_w)

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for ts_seq, stat, did, tgt in val_loader:
                    ts_seq, stat, did, tgt = (ts_seq.to(device), stat.to(device),
                                              did.to(device),   tgt.to(device))
                    va_loss += criterion(model(ts_seq, stat, did), tgt).item() * len(tgt)
            va_loss /= len(val_w)
            scheduler.step(va_loss)

            if epoch % 10 == 0 or epoch == 1:
                marker = " ←" if va_loss < best_val else ""
                print(f"    Epoch {epoch:3d} | train={tr_loss:.5f} | val={va_loss:.5f}{marker}")

            if va_loss < best_val:
                best_val   = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                pat_ctr    = 0
            else:
                pat_ctr += 1
                if pat_ctr >= PATIENCE:
                    print(f"    Early stop at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        ckpt = OUT_DIR / f"lstm_multihorizon_t{h}.pt"
        torch.save(best_state, ckpt)
        print(f"  Checkpoint saved → {ckpt.name}")

        # ── Evaluate and collect predictions ──────────────────────────────────
        def collect_preds(loader, windows, split_name):
            model.eval()
            records = []
            batch_offset = 0
            with torch.no_grad():
                for ts_seq, stat, did, tgt in loader:
                    ts_seq, stat, did = (ts_seq.to(device), stat.to(device), did.to(device))
                    preds = model(ts_seq, stat, did).cpu().numpy()
                    for k in range(len(tgt)):
                        actual_orig = float(tgt[k]) * target_std + target_mean
                        pred_orig   = float(preds[k]) * target_std + target_mean
                        w           = windows[batch_offset + k]
                        records.append({
                            "district":   w[5],
                            "date":       w[4],
                            "gwl_actual": actual_orig,
                            "gwl_pred":   pred_orig,
                            "split":      split_name,
                        })
                    batch_offset += len(tgt)
            return pd.DataFrame(records)

        val_preds  = collect_preds(val_loader,  val_w,  "val")
        test_preds = collect_preds(test_loader, test_w, "test")
        preds_df   = pd.concat([val_preds, test_preds], ignore_index=True)
        preds_df["horizon"] = h

        out_path = OUT_DIR / f"lstm_multihorizon_t{h}.csv"
        preds_df.to_csv(out_path, index=False)

        # Metrics
        val_r,  val_m,  _  = compute_metrics(val_preds["gwl_actual"],  val_preds["gwl_pred"])
        test_r, test_m, tm = compute_metrics(test_preds["gwl_actual"], test_preds["gwl_pred"])
        print(f"  Val  RMSE={val_r:.4f}  MAE={val_m:.4f}")
        print(f"  Test RMSE={test_r:.4f}  MAE={test_m:.4f}")
        print(f"  Predictions saved → {out_path.name}")

        summary_rows.append({
            "model": "LSTM", "horizon": h,
            "val_rmse": round(val_r, 4), "val_mae": round(val_m, 4),
            "test_rmse": round(test_r, 4), "test_mae": round(test_m, 4),
        })

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LSTM Multi-Horizon Summary")
    print("=" * 60)
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "lstm_multihorizon_summary.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
