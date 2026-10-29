"""
Vanilla LSTM baseline — all districts, district embedding + soil texture static features.

Architecture:
    Input per timestep = [time-series features (19)] + [soil texture (9)]
                        concatenated at each of T=30 time steps
    District embedding (dim=8) also concatenated at each time step
    2-layer LSTM, hidden_dim=64, dropout=0.2
    Linear(64 → 1) → predicted gwl_mean at t+1

Static data handling:
    Soil texture (9 values, one per district) and district embedding output
    are REPEATED across all 30 time steps and concatenated with the
    time-series input at each step. This mirrors the TFT's static context
    injection, keeping the comparison fair.

Split (by time, never random):
    Train : 2023-01-01 → 2024-09-30
    Val   : 2024-10-01 → 2025-01-31
    Test  : 2025-02-01 → 2025-04-22

Run from: 03_models/baselines_phase1/
    python lstm_baseline.py
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
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MASTER    = Path("../../../data/clean/master_dataset.csv")
OUT_DIR   = Path("../../../04_results/phase1_baselines")
OUT_DIR.mkdir(exist_ok=True)

# ─── HYPER-PARAMETERS ─────────────────────────────────────────────────────────
LOOKBACK        = 30
HIDDEN_DIM      = 64
NUM_LAYERS      = 2
DROPOUT         = 0.2
EMBED_DIM       = 8
BATCH_SIZE      = 128
LR              = 1e-3
MAX_EPOCHS      = 100
PATIENCE        = 15      # early stopping on val MSE
LR_PATIENCE     = 5

TRAIN_END = "2024-09-30"
VAL_END   = "2025-01-31"

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
    """Sliding-window dataset. Each sample = 30-day lookback → next-day gwl_mean."""

    def __init__(self, windows):
        """windows: list of (ts_seq [T, ts_dim], static [9], district_id int, target float)"""
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ts_seq, static, dist_id, target = self.windows[idx]
        return (
            torch.tensor(ts_seq,  dtype=torch.float32),
            torch.tensor(static,  dtype=torch.float32),
            torch.tensor(dist_id, dtype=torch.long),
            torch.tensor(target,  dtype=torch.float32),
        )


def build_windows(df, ts_cols, static_cols, target_col, lookback):
    """Build sliding windows from sorted (district, date) dataframe."""
    windows = []
    for dist, grp in df.groupby("district"):
        grp = grp.sort_values("date").reset_index(drop=True)
        ts    = grp[ts_cols].values          # shape [N, ts_dim]
        stat  = grp[static_cols].values      # shape [N, 9] — same per district row
        dids  = grp["district_id"].values    # shape [N]
        tgts  = grp[target_col].values       # shape [N]

        for i in range(lookback, len(grp)):
            ts_seq = ts[i - lookback: i]     # [lookback, ts_dim]
            if np.isnan(ts_seq).any():
                continue
            static_vec = stat[i, :]          # [9] — same values repeated across T in model
            if np.isnan(static_vec).any():
                # Fall back to column mean if soil texture missing for this district
                static_vec = np.nan_to_num(static_vec, nan=0.0)
            dist_id = int(dids[i])
            target  = float(tgts[i])
            if np.isnan(target):
                continue
            windows.append((ts_seq, static_vec, dist_id, target))
    return windows


# ─── MODEL ────────────────────────────────────────────────────────────────────
class VanillaLSTM(nn.Module):
    def __init__(self, ts_input_dim, static_dim, n_districts, embed_dim,
                 hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(n_districts, embed_dim)
        # Each time step input = ts_features + static_features + embed
        lstm_input_dim = ts_input_dim + static_dim + embed_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ts_seq, static_vec, dist_id):
        """
        ts_seq    : [B, T, ts_dim]
        static_vec: [B, static_dim]
        dist_id   : [B]
        """
        B, T, _ = ts_seq.shape
        emb     = self.embed(dist_id)                          # [B, embed_dim]
        stat    = static_vec.unsqueeze(1).expand(-1, T, -1)   # [B, T, static_dim]
        emb_exp = emb.unsqueeze(1).expand(-1, T, -1)          # [B, T, embed_dim]
        x       = torch.cat([ts_seq, stat, emb_exp], dim=-1)  # [B, T, lstm_input_dim]
        out, _  = self.lstm(x)                                 # [B, T, hidden_dim]
        last    = out[:, -1, :]                                # [B, hidden_dim]
        return self.head(last).squeeze(-1)                     # [B]


# ─── METRICS ──────────────────────────────────────────────────────────────────
def compute_metrics(actual, pred):
    actual = np.array(actual)
    pred   = np.array(pred)
    rmse   = np.sqrt(np.mean((actual - pred) ** 2))
    mae    = np.mean(np.abs(actual - pred))
    mask   = actual != 0
    mape   = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
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
    df = pd.read_csv(MASTER, parse_dates=["date"])
    print(f"  {len(df):,} rows | {df['district'].nunique()} districts")

    # ── Split by time ─────────────────────────────────────────────────────────
    train_df = df[df["date"] <= TRAIN_END].copy()
    val_df   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
    test_df  = df[df["date"] > VAL_END].copy()
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # ── Normalise time-series cols using train statistics ─────────────────────
    # Static soil texture stays as raw percentages (already in 0-100 range)
    # Target gwl_mean also normalised — we denormalise for metrics
    norm_cols = TIME_SERIES_COLS + [TARGET]
    means, stds = fit_scaler(train_df, norm_cols)

    train_df = apply_scaler(train_df, norm_cols, means, stds)
    val_df   = apply_scaler(val_df,   norm_cols, means, stds)
    test_df  = apply_scaler(test_df,  norm_cols, means, stds)

    # Normalise static cols with train stats too
    static_means, static_stds = fit_scaler(train_df, STATIC_COLS)
    for col in STATIC_COLS:
        for d in [train_df, val_df, test_df]:
            d[col] = (d[col].fillna(static_means[col]) - static_means[col]) / static_stds[col]

    n_districts = df["district_id"].nunique()
    ts_dim      = len(TIME_SERIES_COLS)
    static_dim  = len(STATIC_COLS)

    print(f"\nBuilding sliding windows (lookback={LOOKBACK}) ...")
    train_windows = build_windows(train_df, TIME_SERIES_COLS, STATIC_COLS, TARGET, LOOKBACK)
    val_windows   = build_windows(val_df,   TIME_SERIES_COLS, STATIC_COLS, TARGET, LOOKBACK)
    test_windows  = build_windows(test_df,  TIME_SERIES_COLS, STATIC_COLS, TARGET, LOOKBACK)
    print(f"  Train: {len(train_windows):,}  Val: {len(val_windows):,}  Test: {len(test_windows):,}")

    train_loader = DataLoader(GWLDataset(train_windows), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(GWLDataset(val_windows),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(GWLDataset(test_windows),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    model = VanillaLSTM(
        ts_input_dim=ts_dim,
        static_dim=static_dim,
        n_districts=n_districts,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer   = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler   = ReduceLROnPlateau(optimizer, patience=LR_PATIENCE, factor=0.5, verbose=False)
    criterion   = nn.MSELoss()

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for up to {MAX_EPOCHS} epochs (early stopping patience={PATIENCE}) ...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for ts_seq, static_vec, dist_id, target in train_loader:
            ts_seq, static_vec, dist_id, target = (
                ts_seq.to(device), static_vec.to(device),
                dist_id.to(device), target.to(device)
            )
            optimizer.zero_grad()
            pred = model(ts_seq, static_vec, dist_id)
            loss = criterion(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(target)
        train_loss /= len(train_windows)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ts_seq, static_vec, dist_id, target in val_loader:
                ts_seq, static_vec, dist_id, target = (
                    ts_seq.to(device), static_vec.to(device),
                    dist_id.to(device), target.to(device)
                )
                pred     = model(ts_seq, static_vec, dist_id)
                val_loss += criterion(pred, target).item() * len(target)
        val_loss /= len(val_windows)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
                break

    # ── Load best model ────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save(best_state, OUT_DIR / "lstm_checkpoint.pt")
    print(f"\nBest model saved -> lstm_checkpoint.pt")

    # ── Evaluate and collect predictions ──────────────────────────────────────
    target_mean = float(means[TARGET])
    target_std  = float(stds[TARGET])

    def eval_split(loader, windows, split_name):
        model.eval()
        all_actual, all_pred = [], []
        with torch.no_grad():
            for ts_seq, static_vec, dist_id, target in loader:
                ts_seq, static_vec, dist_id = (
                    ts_seq.to(device), static_vec.to(device), dist_id.to(device)
                )
                pred = model(ts_seq, static_vec, dist_id).cpu().numpy()
                all_actual.extend(target.numpy())
                all_pred.extend(pred)

        # Denormalise
        actual_orig = np.array(all_actual) * target_std + target_mean
        pred_orig   = np.array(all_pred)   * target_std + target_mean
        rmse, mae, mape = compute_metrics(actual_orig, pred_orig)
        print(f"  {split_name:<6} RMSE={rmse:.4f}m  MAE={mae:.4f}m  MAPE={mape:.2f}%")
        return actual_orig, pred_orig

    print("\nEvaluation:")
    val_actual,  val_pred  = eval_split(val_loader,  val_windows,  "Val")
    test_actual, test_pred = eval_split(test_loader, test_windows, "Test")

    # ── Per-district metrics ───────────────────────────────────────────────────
    dist_map = dict(zip(df["district_id"], df["district"]))

    def per_district_metrics(loader, windows, split_name):
        model.eval()
        records = []
        with torch.no_grad():
            for ts_seq, static_vec, dist_id, target in loader:
                ts_seq, static_vec, did = (
                    ts_seq.to(device), static_vec.to(device), dist_id.to(device)
                )
                pred = model(ts_seq, static_vec, did).cpu().numpy()
                for i in range(len(target)):
                    records.append({
                        "district_id": int(dist_id[i]),
                        "gwl_actual":  float(target[i]) * target_std + target_mean,
                        "gwl_pred":    float(pred[i])   * target_std + target_mean,
                        "split":       split_name,
                    })
        return pd.DataFrame(records)

    val_df_pred  = per_district_metrics(val_loader,  val_windows,  "val")
    test_df_pred = per_district_metrics(test_loader, test_windows, "test")
    preds_df = pd.concat([val_df_pred, test_df_pred], ignore_index=True)
    preds_df["district"] = preds_df["district_id"].map(dist_map)
    preds_df = preds_df[["district", "district_id", "split", "gwl_actual", "gwl_pred"]]
    preds_df.to_csv(OUT_DIR / "lstm_predictions.csv", index=False)
    print(f"\nPredictions saved -> lstm_predictions.csv")

    # Per-district table
    print("\nPer-district test RMSE:")
    dist_metrics = []
    for dist_id, grp in preds_df[preds_df["split"] == "test"].groupby("district_id"):
        r, m, mp = compute_metrics(grp["gwl_actual"].values, grp["gwl_pred"].values)
        dist_metrics.append({"district": dist_map.get(dist_id, dist_id), "rmse": round(r, 4), "mae": round(m, 4)})
    dm = pd.DataFrame(dist_metrics).sort_values("rmse", ascending=False)
    print(dm.to_string(index=False))

    dm.to_csv(OUT_DIR / "lstm_metrics.csv", index=False)
    print("\nMetrics saved -> lstm_metrics.csv")


if __name__ == "__main__":
    main()
