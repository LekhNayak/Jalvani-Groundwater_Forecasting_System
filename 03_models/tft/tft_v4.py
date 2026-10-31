"""
JalVaani TFT v4 pipeline — importable module, runs locally (CPU) or on Colab (GPU).

Built from `09_tft_colab_production_v3.ipynb` (cells 4-22). The model itself is
unchanged (`tft_model_v3.py`); what changes is the evaluation protocol, the
baselines, and (behind config switches) the target, calibration and combination.

Phase 0 fixes relative to v3 (always on):
  * Windows are keyed by ANCHOR (issue) date. A window belongs to a split if any
    of its target dates fall inside the split, and its mask is obs x in_split.
    v3 keyed and split on the END of the 30-day decoder, so the first 23 days of
    t+7 targets in every test fold actually sat in the val block and the last
    23 days of each season were never scored.
  * Every scored row carries anchor_date and target_date (= anchor + h).
  * Seasonal-naive (SN) baseline: anchor + the district's typical h-day change
    for this time of year, fitted on past data only. It nearly matches v3's TFT,
    so every result is reported against persistence AND SN.
  * CRD (cumulative rainfall departure) uses the TRAINING-period mean rainfall
    (v3 used the full-series mean, a small look-ahead).
  * Constant columns (the *_observed flags are all 1.0 in the ERA5 master) and
    `soil_moisture` (identical to sm_l2) are dropped.
  * Run modes: dev = folds 1-4 (all tuning), final = folds 5-6 (run once), all.

Config switches for Phase 2 (defaults reproduce "v3-fixed"):
  target_mode  'persistence' (v3: predict gwl(t+h)-anchor) | 'snaive' (predict the
               residual on top of seasonal-naive, so output 0 == SN)
  calib_mode   'v3' (split conformal on the cal block) | 'v4' (season-weighted pool
               + per-district normalised scores + adaptive conformal)
  combine      'lambda' (v3 shrinkage toward persistence) | 'stack' (simplex blend of
               persistence / SN / TFT / LightGBM fitted on the cal pool) | 'none'

v4.2 switches (post-processing only; the defaults reproduce v4.1 exactly):
  aci_feedback     'cell' (v4.1: every scored cell updates alpha — about 10 correlated outcomes
                   per anchor for the TFT, 1 for the Ensemble) | 'anchor' (one mean outcome per
                   anchor, district and band)
  aci_alpha_min/max  clip range of the adaptive miscoverage (v4.1: 0.01-0.30 around a 0.10
                   target, i.e. more room to narrow than to widen)
  stack_guard      per (district, horizon): if the shrunk stack is worse on that district's own
                   pool than both seasonal-naive alone and the TFT alone, use the better of the two
  interval_choice  {model: {horizon: 'raw' | 'calibrated'}}: which interval is REPORTED for that
                   model and horizon (missing = 'calibrated', as in v4.1). Both are always computed.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# torch BEFORE sklearn (BLAS/OpenMP DLL conflict on some Windows installs)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:  # pragma: no cover
    HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingRegressor

from tft_model_v3 import MaskedQuantileLoss, TemporalFusionTransformer

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent


def _repo_root():
    """Project root (the folder holding data/master and 04_results), or None when this file runs
    outside the repository (e.g. uploaded flat to /content on Colab)."""
    for p in HERE.parents:
        if (p / "data" / "master").is_dir() and (p / "04_results").is_dir():
            return p
    return None


REPO = _repo_root()
DEFAULT_DATA = (REPO / "data" / "master" if REPO else HERE) / "master_dataset_v4target.csv"
DEFAULT_OUT = REPO / "04_results" / "tft_runs" if REPO else HERE / "outputs" / "v4"
# Part of every cache key — bump whenever code that changes fold results is edited,
# so a cached fold computed by older code can never be silently reused.
PIPELINE_VERSION = "4.6"

# ─── Folds ────────────────────────────────────────────────────────────────────
FOLDS = [
    {"name": "fold1_monsoon2024",     "test_start": "2024-07-01", "test_end": "2024-09-30"},
    {"name": "fold2_postmonsoon2024", "test_start": "2024-10-01", "test_end": "2024-12-31"},
    {"name": "fold3_dry2025",         "test_start": "2025-01-01", "test_end": "2025-05-31"},
    {"name": "fold4_monsoon2025",     "test_start": "2025-06-01", "test_end": "2025-09-30"},
    {"name": "fold5_postmonsoon2025", "test_start": "2025-10-01", "test_end": "2025-12-31"},
    {"name": "fold6_dry2026",         "test_start": "2026-01-01", "test_end": "2026-06-04"},
]
RUN_FOLDS = {"dev": FOLDS[:4], "final": FOLDS[4:], "all": FOLDS}

# Frozen headline set: 26 modelled districts minus tier D (Latur) minus the
# single-well districts of the v3 master. Frozen so that a target rebuild that
# changes station counts cannot silently change what the headline averages over.
HEADLINE_EXCLUDE = ["Beed", "Latur", "Nanded", "Raigad", "Thane"]

EVAL_STEPS = [6, 13, 29]                    # 0-indexed -> t+7, t+14, t+30
HNAME = {6: "t+7", 13: "t+14", 29: "t+30"}
HORDER = ["t+7", "t+14", "t+30"]

SOIL_COLS = ["sand_0cm", "clay_0cm", "silt_0cm", "sand_10cm", "clay_10cm", "silt_10cm",
             "sand_30cm", "clay_30cm", "silt_30cm"]
SENTINEL_COLS = ["ndvi_mean", "ndwi_mean", "ndmi_mean", "vv_db", "vh_db", "vh_vv_ratio"]
CGWB_COLS = ["annual_recharge_bcm", "annual_extraction_bcm",
             "extractable_resource_bcm", "stage_of_extraction_pct"]
ERA5_ALL = ["pet_mm", "water_stress", "sm_l1", "sm_l2", "sm_l3", "sm_l4"]
RAIN_EXTRA = ["api_fast", "api_mid", "api_slow", "rainfall_60d", "rainfall_90d", "crd",
              "days_since_rain"]
LEVEL_COLS = ["gwl_mean", "gwl_min", "gwl_max", "gwl_lag_7", "gwl_lag_14", "gwl_lag_30",
              "gwl_roll_mean_30", "gwl_roll_mean_60", "gwl_roll_mean_90"]
DIFF_COLS = ["gwl_7d_change", "gwl_30d_change", "gwl_roll_std_30", "gwl_roll_std_60",
             "gwl_roll_std_90"]
STATIC_CAT = ["district_id", "aquifer_category_id"]
TARGET, LEVEL, OBS_COL = "gwl_mean_raw", "gwl_mean", "gwl_observed"


# ─── Config ───────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # data / run
    data_path: str = str(DEFAULT_DATA)
    out_root: str = str(DEFAULT_OUT)
    tag: str = "v3fixed"
    mode: str = "dev"
    seeds: list = field(default_factory=lambda: [42, 1337, 2026])
    # windows
    encoder_len: int = 60
    decoder_len: int = 30
    train_stride: int = 7
    val_stride: int = 2
    test_stride: int = 1
    enc_obs_min: float = 0.70
    dec_obs_min: float = 0.50
    # capacity
    d_model: int = 32
    d_select: int = 8
    grn_hidden_factor: float = 0.5
    n_heads: int = 4
    lstm_layers: int = 1
    dropout: float = 0.25
    cat_embed_dim: int = 8
    weight_decay: float = 1e-2
    # optimisation
    batch_size: int = 128
    eval_batch: int = 512
    max_epochs: int = 60
    lr: float = 1e-3
    warmup_epochs: int = 5
    patience: int = 10
    grad_clip: float = 1.0
    lr_patience: int = 4
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    sample_weight_cap: float = 1.5
    # validation split
    val_months: int = 3
    val_es_frac: float = 0.60
    # calibration / shrinkage
    target_coverage: float = 0.80
    lambda_prior_n: int = 200
    conf_min_n: int = 50
    band_size: int = 10
    exclude_tiers: list = field(default_factory=lambda: ["D"])
    # seasonal-naive — tuned on SN's own dev-fold skill (tune_snaive.py, 2026-10-02):
    # bw 30 / k 20 / damping 1.0 -> rMAE vs persistence 0.948 / 0.892 / 0.807 (t+7/14/30)
    sn_bandwidth: float = 30.0      # days, circular Gaussian kernel on day-of-year
    sn_prior_k: float = 20.0        # pseudo-count pulling the climatological change toward 0
    sn_damping: float = 1.0         # multiply the climatological change by this
    # Phase 2 switches
    target_mode: str = "persistence"   # 'persistence' | 'snaive'
    calib_mode: str = "v3"             # 'v3' | 'v4'
    combine: str = "lambda"            # 'lambda' | 'stack' | 'none'
    feature_groups: list = field(default_factory=lambda: ["gwl", "rain", "met", "era5", "sentinel"])
    loss_weighting: str = "none"       # 'none' | 'eval_steps' | 'frac_real' | 'both'
    aci_gamma: float = 0.01
    stack_prior_n: float = 500.0       # pseudo-rows pulling per-district stack weights to global
    season_weight_other: float = 0.3   # pool weight of rows from a different season (v4 calib/stack)
    with_tabular: bool = True
    # ── v4.1 ──
    # post-processing (no retraining)
    lambda_center: str = "auto"        # shrink TFT toward: 'auto' (= SN in snaive mode, else 0) | 'zero' | 'sn'
    use_refit: bool = True             # use the pass-2 (refit) networks when they were trained
    # network training
    refit: bool = False                # pass 2: retrain on train + early-stopping block for pass 1's best epoch
    refit_min_epochs: int = 5
    epoch_frac: float = 1.0            # samples per epoch = epoch_frac * windows (stride-1 sampling uses 1/7)
    # tabular models (LightGBM / ridge) - own cache stage, never retrain the TFT
    tab_features: str = "v1"           # 'v1' = v4 snapshot | 'v2' = + calendar, SN change, scale, level, year-on-year
    tab_stride: int = 7
    tab_fresh: bool = False            # cal model trained to es_end, test model trained to train_end (else fit_end)
    gbm_n_estimators: int = 300
    gbm_lr: float = 0.05
    gbm_min_child: int = 30
    gbm_leaves: int = 15
    # ── v4.2 ── post-processing only (defaults = v4.1)
    aci_feedback: str = "cell"         # 'cell' | 'anchor' (one mean outcome per anchor, district, band)
    aci_alpha_min: float = 0.01
    aci_alpha_max: float = 0.30
    stack_guard: bool = False          # fall back to the better of SN / TFT where the stack loses to both
    stack_guard_min_n: float = 50.0    # weighted pool rows a district needs before the guard may act
    interval_choice: dict = field(default_factory=dict)   # {model: {horizon: 'raw'|'calibrated'}}
    only_folds: list = field(default_factory=list)   # smoke tests: restrict to these fold names

    # fields that do not change per-fold results (excluded from the cache key)
    _NON_RESULT = ("out_root", "tag", "mode", "only_folds")

    def key(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in self._NON_RESULT}
        d["data_md5"] = file_md5(self.data_path)
        d["pipeline_version"] = PIPELINE_VERSION
        return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def file_md5(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Data ─────────────────────────────────────────────────────────────────────
def _api(s, k):
    out = np.zeros(len(s), dtype=np.float64)
    acc = 0.0
    for i, v in enumerate(s):
        acc = acc * k + (v if np.isfinite(v) else 0.0)
        out[i] = acc
    return out


def load_data(cfg: Config, verbose=True):
    """Cells 5-7 of v3: load, interpolate encoder series, engineer features, define roles."""
    df = pd.read_csv(cfg.data_path, parse_dates=["date"]).sort_values(["district", "date"]).reset_index(drop=True)
    if verbose:
        print(f"Master: {df.shape} | {df['district'].nunique()} districts | "
              f"{df['date'].min().date()} -> {df['date'].max().date()}")

    df["aquifer_category_id"] = df["aquifer_category_id"].fillna(0).astype(int)
    df["season_id"] = df["season"].map({"winter": 0, "summer": 1, "monsoon": 2, "post_monsoon": 3}).astype(int)
    if "data_tier" not in df.columns:
        raise ValueError("master has no data_tier column — re-run merge_master.py")

    # raw target (NaN when unobserved) vs interpolated encoder series
    df[TARGET] = df[LEVEL].where(df[OBS_COL] == 1)
    for col in ["gwl_mean", "gwl_min", "gwl_max"]:
        df[col] = df.groupby("district")[col].transform(lambda s: s.interpolate("linear", limit_direction="both"))
    for lag in (7, 14, 30):
        df[f"gwl_lag_{lag}"] = df.groupby("district")["gwl_mean"].shift(lag)
    df["gwl_7d_change"] = df["gwl_mean"] - df["gwl_lag_7"]
    df["gwl_30d_change"] = df["gwl_mean"] - df["gwl_lag_30"]

    # calendar
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # rainfall features (past-only)
    for c in ["rainfall_mm", "rainfall_7d", "rainfall_30d"]:
        df[c + "_log"] = np.log1p(df[c].clip(lower=0))
    parts = []
    for _, g in df.groupby("district", sort=False):
        g = g.sort_values("date").copy()
        r = g["rainfall_mm"].fillna(0.0).values
        for k, name in [(0.90, "api_fast"), (0.95, "api_mid"), (0.98, "api_slow")]:
            g[name] = _api(r, k)
        g["rainfall_60d"] = np.log1p(g["rainfall_mm"].rolling(60, min_periods=1).sum().clip(lower=0))
        g["rainfall_90d"] = np.log1p(g["rainfall_mm"].rolling(90, min_periods=1).sum().clip(lower=0))
        # CRD is finished per fold in normalize_for_fold (needs the TRAIN mean rainfall)
        g["_cumrain"] = np.cumsum(r)
        g["_tidx"] = np.arange(1, len(g) + 1, dtype=np.float64)
        dsr, run = np.zeros(len(g)), 0
        for i, w in enumerate(r > 1.0):
            run = 0 if w else run + 1
            dsr[i] = run
        g["days_since_rain"] = np.minimum(dsr, 120)
        parts.append(g)
    df = pd.concat(parts).sort_index()
    df["crd"] = 0.0

    for w in (30, 60, 90):
        df[f"gwl_roll_mean_{w}"] = df.groupby("district")["gwl_mean"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=5).mean())
        df[f"gwl_roll_std_{w}"] = df.groupby("district")["gwl_mean"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=5).std())
    roll_cols = [c for c in df.columns if c.startswith("gwl_roll_")]

    # ── feature roles ───────────────────────────────────────────────────────
    def present_varying(cols):
        return [c for c in cols if c in df.columns and df[c].nunique(dropna=True) > 1]

    era5 = present_varying(ERA5_ALL)
    groups = {
        "gwl": LEVEL_COLS + DIFF_COLS + present_varying(["gwl_observed", "gwl_frac_real"])
               + present_varying(["station_count", "n_active", "gwl_days_since_obs"]),
        "rain": ["rainfall_mm_log", "rainfall_7d_log", "rainfall_30d_log"] + RAIN_EXTRA,
        "met": ["et_mm", "net_water_balance", "temp_max", "temp_min", "temp_mean", "humidity_mean", "vpd_approx"],
        "era5": era5,
        "sentinel": SENTINEL_COLS,
    }
    # `soil_moisture` duplicates sm_l2 in the ERA5 master; keep it only on the old feed
    if "sm_l2" not in era5 and "soil_moisture" in df.columns:
        groups["met"].append("soil_moisture")
    past = []
    for gname in cfg.feature_groups:
        past += [c for c in groups[gname] if c in df.columns and c not in past]
    flag_cols = [c for c in past if c in ("gwl_observed", "gwl_frac_real")]

    future = ["month_sin", "month_cos", "doy_sin", "doy_cos", "is_monsoon", "season_id", "horizon_frac"]
    if cfg.target_mode == "snaive":
        future.append("sn_prior")          # SN delta for this decoder step (scaled), written per window
    df["horizon_frac"] = 0.0
    df["sn_prior"] = 0.0

    static_cont = [c for c in SOIL_COLS + CGWB_COLS if c in df.columns]
    tier = df.drop_duplicates("district").set_index("district")["data_tier"].to_dict()
    low = df.drop_duplicates("district").set_index("district")["low_station_flag"].to_dict()
    drop = sorted(d for d, t in tier.items() if t in cfg.exclude_tiers)
    df = df[~df["district"].isin(drop)].reset_index(drop=True)

    meta = dict(
        past=past, future=future, static_cont=static_cont, flag_cols=flag_cols,
        level_cols=[c for c in LEVEL_COLS if c in past], diff_cols=[c for c in DIFF_COLS if c in past],
        fill_cols=[c for c in SOIL_COLS + SENTINEL_COLS + CGWB_COLS + roll_cols + era5 if c in df.columns],
        n_districts=int(df["district_id"].max()) + 1, n_categories=int(df["aquifer_category_id"].max()) + 1,
        tier=tier, low_station=low, dropped=drop,
    )
    if verbose:
        print(f"Past={len(past)} Future={len(future)} Static={len(static_cont)}+{len(STATIC_CAT)}cat | "
              f"modelling {df['district'].nunique()} districts (dropped tier {cfg.exclude_tiers}: {drop})")
    return df, meta


# ─── Seasonal-naive ───────────────────────────────────────────────────────────
def fit_snaive_table(df, cutoff, cfg: Config, exclude_year=None):
    """Past-only climatological change table, per district: (366 doy, H) in metres.

    For every anchor day t whose t and t+h are both observed and t+h <= cutoff,
    d_h(t) = y(t+h) - y(t). The table at day-of-year j is a circular-Gaussian
    kernel average of d_h over anchors near j (bandwidth cfg.sn_bandwidth days),
    shrunk toward 0 (= persistence) by a pseudo-count, then damped.
    `exclude_year` leaves one calendar year of anchors out (cross-fitting for
    training windows in target_mode='snaive').
    """
    cutoff = pd.Timestamp(cutoff)
    H = cfg.decoder_len
    doy_grid = np.arange(1, 367)
    out = {}
    for dist, g in df.groupby("district"):
        g = g.sort_values("date")
        y = g[TARGET].values.astype(np.float64)
        dates = g["date"].values
        doy = g["day_of_year"].values
        yr = g["date"].dt.year.values
        n = len(y)
        tab = np.zeros((366, H))
        diff = np.abs(doy_grid[:, None] - doy[None, :]).astype(np.float64)
        diff = np.minimum(diff, 365.25 - diff)
        w_full = np.exp(-0.5 * (diff / cfg.sn_bandwidth) ** 2)          # (366, n), anchors as columns
        for h in range(1, H + 1):
            if n <= h:
                continue
            d = y[h:] - y[:-h]
            ok = np.isfinite(d) & (dates[h:] <= np.datetime64(cutoff))
            if exclude_year is not None:
                ok &= yr[:-h] != exclude_year
            if ok.sum() < 5:
                continue
            w = w_full[:, :n - h]
            num = w @ np.where(ok, d, 0.0)
            den = w @ ok.astype(np.float64)
            tab[:, h - 1] = cfg.sn_damping * num / (den + cfg.sn_prior_k + 1e-12)
        out[dist] = tab
    return out


def sn_lookup(tables, district, anchor_doy, H):
    t = tables.get(district)
    if t is None:
        return np.zeros(H)
    return t[int(anchor_doy) - 1, :H]


# ─── Normalisation (train-only) ───────────────────────────────────────────────
def compute_delta_scales(d, fit_end, cfg: Config, sn_tables=None):
    """Per-district, per-horizon scale of the quantity the model predicts.

    persistence mode: SD of the observed h-step change (v3 M2).
    snaive mode     : SD of the observed h-step change minus the SN change.
    """
    tr = d[d["date"] <= pd.Timestamp(fit_end)]
    H = cfg.decoder_len
    sd_h, sd_enc = {}, {}
    for dist, g in tr.groupby("district"):
        g = g.sort_values("date")
        s = g[TARGET].values.astype(np.float64)
        doy = g["day_of_year"].values
        per_h = []
        for h in range(1, H + 1):
            if len(s) <= h:
                per_h.append(np.nan)
                continue
            v = s[h:] - s[:-h]
            if sn_tables is not None and dist in sn_tables:
                v = v - sn_tables[dist][doy[:-h] - 1, h - 1]
            v = v[np.isfinite(v)]
            per_h.append(float(np.std(v)) if len(v) > 10 else np.nan)
        per_h = pd.Series(per_h).interpolate().bfill().ffill().values
        per_h = np.maximum(np.nan_to_num(per_h, nan=1.0), 1e-3)
        sd_h[dist] = per_h.astype(np.float64)
        sd_enc[dist] = float(np.sqrt(np.mean(per_h ** 2)))
    return sd_enc, sd_h


def normalize_for_fold(d, fit_end, cfg: Config, meta):
    fit_end = pd.Timestamp(fit_end)
    tr = d["date"] <= fit_end
    dn = d.copy()

    # CRD against the TRAINING-period mean daily rainfall (v3 used the full series)
    mean_rain = dn.loc[tr].groupby("district")["rainfall_mm"].mean()
    dn["crd"] = dn["_cumrain"] - dn["district"].map(mean_rain).fillna(0.0) * dn["_tidx"]

    for c in meta["fill_cols"]:
        med = dn.loc[tr].groupby("district")[c].median()
        dn[c] = dn[c].fillna(dn["district"].map(med)).fillna(dn.loc[tr, c].median()).fillna(0.0)

    scaled_past = [c for c in meta["past"] if c not in meta["level_cols"] + meta["diff_cols"] + meta["flag_cols"]]
    scaled_fut = [c for c in meta["future"] if c not in ("horizon_frac", "sn_prior")]
    feat = [c for c in scaled_past + meta["static_cont"] + scaled_fut if c in dn.columns]
    dn[feat] = dn[feat].fillna(0.0)
    scaler = StandardScaler().fit(dn.loc[tr, feat])
    dn[feat] = scaler.transform(dn[feat])
    return dn


# ─── Dataset ──────────────────────────────────────────────────────────────────
class GWLDataset(Dataset):
    """Calendar-true windows keyed by ANCHOR date (the last encoder day).

    target step s (0-indexed) is the day anchor + (s+1).
    A window is part of the split if ANY target date lies in [split_start, split_end];
    mask = observed x in_split, so only in-split, observed targets are trained on
    or scored. `base` is what an output of 0 reconstructs to:
        persistence mode: base = 0                -> level = anchor + out*sd
        snaive mode     : base = SN delta (metres) -> level = anchor + base + out*sd
    """

    def __init__(self, d, cfg: Config, meta, sd_enc, sd_h, split_start=None, split_end=None,
                 stride=1, sn_tables=None, sn_tables_by_year=None, dec_gate=True, keep_tensors=True):
        E, H = cfg.encoder_len, cfg.decoder_len
        past_cols, fut_cols = meta["past"], meta["future"]
        lvl_idx = [past_cols.index(c) for c in meta["level_cols"]]
        diff_idx = [past_cols.index(c) for c in meta["diff_cols"]]
        hf_idx = fut_cols.index("horizon_frac")
        sn_idx = fut_cols.index("sn_prior") if "sn_prior" in fut_cols else None
        horizon_frac = ((np.arange(H) + 1) / H).astype(np.float32)
        s0 = np.datetime64(pd.Timestamp(split_start)) if split_start is not None else None
        s1 = np.datetime64(pd.Timestamp(split_end)) if split_end is not None else None
        snaive = cfg.target_mode == "snaive"

        P, F, T, M, IS, SC, SX, TAB = [], [], [], [], [], [], [], []
        self.district, self.anchor_date, self.anchor, self.sd_mat = [], [], [], []
        self.actual, self.did, self.base, self.sn, self.anchor_doy, self.yoy = [], [], [], [], [], []
        self.n_drop_enc = self.n_drop_dec = self.n_drop_anchor = 0

        for dist, grp in d.groupby("district"):
            grp = grp.sort_values("date").reset_index(drop=True)
            n = len(grp)
            obs = grp[OBS_COL].values
            lvl = grp[LEVEL].values.astype(np.float64)
            past = grp[past_cols].values.astype(np.float64)
            fut = grp[fut_cols].values.astype(np.float32)
            tgt = grp[TARGET].values.astype(np.float64)
            dates = grp["date"].values
            doy = grp["day_of_year"].values
            years = grp["date"].dt.year.values
            sc = grp.loc[0, STATIC_CAT].values.astype(np.int64)
            sx = grp.loc[0, meta["static_cont"]].values.astype(np.float32)
            se = float(sd_enc.get(dist, 1.0))
            sh = sd_h.get(dist, np.ones(H))
            did = int(grp.loc[0, "district_id"])

            for i in range(E, n - H + 1, stride):
                tdates = dates[i:i + H]
                ins = np.ones(H, bool)
                if s0 is not None:
                    ins &= tdates >= s0
                if s1 is not None:
                    ins &= tdates <= s1
                if not ins.any():
                    continue
                if obs[i - E:i].mean() < cfg.enc_obs_min:
                    self.n_drop_enc += 1
                    continue
                if dec_gate and obs[i:i + H][ins].mean() < cfg.dec_obs_min:
                    self.n_drop_dec += 1
                    continue
                if obs[i - 1] != 1:
                    self.n_drop_anchor += 1
                    continue
                a = lvl[i - 1]
                if not np.isfinite(a):
                    continue

                pc = past[i - E:i].copy()
                if lvl_idx:
                    pc[:, lvl_idx] = (pc[:, lvl_idx] - a) / se
                if diff_idx:
                    pc[:, diff_idx] = pc[:, diff_idx] / se
                fc = fut[i:i + H].copy()
                fc[:, hf_idx] = horizon_frac

                tabs = sn_tables
                if sn_tables_by_year is not None:
                    tabs = sn_tables_by_year.get(int(years[i - 1]), sn_tables)
                sn = sn_lookup(tabs, dist, doy[i - 1], H) if tabs is not None else np.zeros(H)
                base = sn if snaive else np.zeros(H)
                if sn_idx is not None:
                    fc[:, sn_idx] = (sn / sh).astype(np.float32)

                raw = tgt[i:i + H]
                tg = (raw - a - base) / sh
                mk = (obs[i:i + H] == 1) & ins & np.isfinite(raw)
                if not np.isfinite(pc).all() or not np.isfinite(fc).all():
                    continue

                if keep_tensors:                # tabular-only datasets (LightGBM) skip the big tensors
                    P.append(pc.astype(np.float32)); F.append(fc)
                    T.append(np.nan_to_num(tg, nan=0.0).astype(np.float32))
                M.append(mk.astype(np.float32))
                IS.append(ins)
                SC.append(sc); SX.append(sx)
                TAB.append(np.concatenate([pc[-1], pc[-7:].mean(0), pc[-30:].mean(0)]).astype(np.float32))
                self.district.append(dist)
                self.anchor_date.append(dates[i - 1])
                self.anchor.append(float(a)); self.sd_mat.append(sh.astype(np.float32))
                self.actual.append(raw.astype(np.float64)); self.did.append(did)
                self.base.append(base.astype(np.float64)); self.sn.append(sn.astype(np.float64))
                self.anchor_doy.append(int(doy[i - 1]))
                # change vs the same day last year, in district change units (past-only)
                yo = (a - lvl[i - 1 - 365]) / se if i - 1 - 365 >= 0 and np.isfinite(lvl[i - 1 - 365]) else np.nan
                self.yoy.append(float(yo))

        nP, nF = len(past_cols), len(fut_cols)
        self.P = np.asarray(P, np.float32) if P else np.zeros((0, E, nP), np.float32)
        self.F = np.asarray(F, np.float32) if F else np.zeros((0, H, nF), np.float32)
        self.T = np.asarray(T, np.float32) if T else np.zeros((0, H), np.float32)
        self.M = np.asarray(M, np.float32) if M else np.zeros((0, H), np.float32)
        self.W = self.M.copy()                      # loss weights (mask x optional weighting)
        self.insplit = np.asarray(IS, bool) if IS else np.zeros((0, H), bool)
        self.SC = np.asarray(SC, np.int64) if SC else np.zeros((0, len(STATIC_CAT)), np.int64)
        self.SX = np.asarray(SX, np.float32) if SX else np.zeros((0, len(meta["static_cont"])), np.float32)
        self.tab = np.asarray(TAB, np.float32) if TAB else np.zeros((0, 3 * nP), np.float32)
        self.anchor = np.asarray(self.anchor, np.float64)
        self.sd_mat = np.asarray(self.sd_mat, np.float32) if self.sd_mat else np.zeros((0, H), np.float32)
        self.actual = np.asarray(self.actual, np.float64) if self.actual else np.zeros((0, H))
        self.base = np.asarray(self.base, np.float64) if self.base else np.zeros((0, H))
        self.sn = np.asarray(self.sn, np.float64) if self.sn else np.zeros((0, H))
        self.did = np.asarray(self.did, np.int64)
        self.anchor_doy = np.asarray(self.anchor_doy, np.int64)
        self.yoy = np.asarray(self.yoy, np.float64)
        self.district = np.asarray(self.district)
        self.anchor_date = pd.to_datetime(np.asarray(self.anchor_date)) if len(self.anchor_date) \
            else pd.DatetimeIndex([])
        self.H = H

    def target_dates(self):
        return self.anchor_date.values[:, None] + (np.arange(1, self.H + 1) * np.timedelta64(1, "D"))[None, :]

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, i):
        return (torch.from_numpy(self.SC[i]), torch.from_numpy(self.SX[i]),
                torch.from_numpy(self.P[i]), torch.from_numpy(self.F[i]),
                torch.from_numpy(self.T[i]), torch.from_numpy(self.W[i]), i)

    def counts(self):
        return Counter(self.district.tolist())


def build_sampler(ds, cap, n_samples=None):
    counts = ds.counts()
    raw = {d: 1.0 / c for d, c in counts.items()}
    mn = min(raw.values())
    capped = {d: min(w, mn * cap) for d, w in raw.items()}
    return WeightedRandomSampler([capped[d] for d in ds.district], num_samples=int(n_samples or len(ds)),
                                 replacement=True)


def apply_loss_weighting(ds, cfg: Config, frac_real=None):
    """Optional per-target loss weights (Phase 2.7). Mask stays the scoring mask."""
    W = ds.M.copy()
    if cfg.loss_weighting in ("eval_steps", "both"):
        w = np.ones(cfg.decoder_len, np.float32)
        for s in EVAL_STEPS:
            w[max(0, s - 2):s + 3] = 2.0
        W *= w[None, :]
    if cfg.loss_weighting in ("frac_real", "both") and frac_real is not None:
        W *= np.sqrt(np.clip(frac_real, 0.0, 1.0)).astype(np.float32)
    ds.W = W


# ─── Model + training ─────────────────────────────────────────────────────────
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: Config, meta, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    return TemporalFusionTransformer(
        n_static_cat=len(STATIC_CAT), cat_vocab_sizes=[meta["n_districts"], meta["n_categories"]],
        cat_embed_dim=cfg.cat_embed_dim, n_static_cont=len(meta["static_cont"]),
        n_past=len(meta["past"]), n_future=len(meta["future"]),
        encoder_len=cfg.encoder_len, decoder_len=cfg.decoder_len,
        d_model=cfg.d_model, n_heads=cfg.n_heads, lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout, quantiles=[0.1, 0.5, 0.9], monotone_quantiles=True,
        d_select=cfg.d_select, grn_hidden_factor=cfg.grn_hidden_factor,
    ).to(device)


def _run_epoch(loader, model, criterion, device, cfg, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for sc, sx, px, fx, y, w, _ in loader:
            sc, sx, px, fx, y, w = [t.to(device, non_blocking=True) for t in (sc, sx, px, fx, y, w)]
            preds, _, _ = model(sc, sx, px, fx)
            loss = criterion(preds, y, w)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def train_one(cfg: Config, meta, train_ds, es_ds, seed, device, tag="", verbose=False):
    """Pass 1: train with early stopping on the ES block. Returns (model, train_hist, val_hist, lr_trace)."""
    model = build_model(cfg, meta, device, seed)
    criterion = MaskedQuantileLoss([0.1, 0.5, 0.9]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = build_sampler(train_ds, cfg.sample_weight_cap, round(len(train_ds) * cfg.epoch_frac))
    sampler.generator = g
    tl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
    vl = DataLoader(es_ds, batch_size=cfg.eval_batch, shuffle=False, num_workers=0)

    tr_hist, va_hist, lr_trace = [], [], []
    best, wait, best_state = float("inf"), 0, None
    for ep in range(1, cfg.max_epochs + 1):
        if ep <= cfg.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * ep / cfg.warmup_epochs
        lr_trace.append(optimizer.param_groups[0]["lr"])
        tr = _run_epoch(tl, model, criterion, device, cfg, optimizer)
        va = _run_epoch(vl, model, criterion, device, cfg)
        tr_hist.append(tr)
        va_hist.append(va)
        if ep > cfg.warmup_epochs:
            plateau.step(va)
        if va < best:
            best, wait = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    if verbose:
        print(f"    [{tag}] stopped ep {ep}, best val {best:.4f} at ep {int(np.argmin(va_hist)) + 1}")
    return model, tr_hist, va_hist, lr_trace


def train_fixed(cfg: Config, meta, ds, seed, device, lr_trace):
    """Pass 2 (refit, v4.1): retrain from scratch on train + early-stopping windows, replaying
    pass 1's learning-rate path for exactly len(lr_trace) epochs (pass 1's best epoch). No early
    stopping - nothing inside this data is held out; the calibration block after it is."""
    model = build_model(cfg, meta, device, seed)
    criterion = MaskedQuantileLoss([0.1, 0.5, 0.9]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = build_sampler(ds, cfg.sample_weight_cap, round(len(ds) * cfg.epoch_frac))
    sampler.generator = g
    tl = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
    hist = []
    for lr in lr_trace:
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        hist.append(_run_epoch(tl, model, criterion, device, cfg, optimizer))
    return model, hist


@torch.no_grad()
def predict_scaled(ds, model, cfg: Config, device):
    model.eval()
    out = np.zeros((len(ds), cfg.decoder_len, 3), np.float32)
    for sc, sx, px, fx, _, _, idx in DataLoader(ds, batch_size=cfg.eval_batch, shuffle=False, num_workers=0):
        p, _, _ = model(sc.to(device), sx.to(device), px.to(device), fx.to(device))
        out[idx.numpy()] = p.cpu().numpy()
    return out


# ─── Reconstruction ───────────────────────────────────────────────────────────
def to_delta_m(ds, qs):
    """scaled model output -> quantile DELTAS from the anchor, in metres. (N, H, 3)"""
    return qs * ds.sd_mat[:, :, None] + ds.base[:, :, None]


def _bands(cfg):
    return np.array([min(s // cfg.band_size, (cfg.decoder_len - 1) // cfg.band_size)
                     for s in range(cfg.decoder_len)])


# ─── Baselines ────────────────────────────────────────────────────────────────
def baseline_snaive(ds, train_ds):
    """SN quantiles: median = the SN delta stored on each window; spread = training
    residual quantiles of (actual delta - SN delta), per district and step."""
    qd = np.zeros((len(ds), ds.H, 3), np.float32)
    qd[:, :, 1] = ds.sn
    res = np.where(train_ds.M > 0, train_ds.actual - train_ds.anchor[:, None] - train_ds.sn, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glo = np.nanquantile(res, [0.1, 0.9], axis=0)
        for dist in np.unique(ds.district):
            r = res[train_ds.district == dist]
            q = np.nanquantile(r, [0.1, 0.9], axis=0) if len(r) >= 20 else glo
            q = np.where(np.isfinite(q), q, glo)
            rows = ds.district == dist
            qd[rows, :, 0] = ds.sn[rows] + q[0]
            qd[rows, :, 2] = ds.sn[rows] + q[1]
    qd[:, :, 0] = np.minimum(qd[:, :, 0], qd[:, :, 1])
    qd[:, :, 2] = np.maximum(qd[:, :, 2], qd[:, :, 1])
    return qd


def _tab_features(ds, s, cfg: Config):
    """Features for the tabular models at eval step s.

    v1 (v4.0): anchor-day snapshot (last day, 7-day and 30-day means of the encoder inputs) + district id.
    v2 (v4.1): + WHEN the forecast is issued (anchor day-of-year sin/cos), the seasonal-naive change for
    this step (metres and in district change units), the step's change scale, the anchor level itself and
    the change since the same day last year. v1 could not see the time of year it forecasts, so the
    seasonal part of a residual-on-SN target was unlearnable for it. All past-only or calendar.
    """
    X = [ds.tab, ds.did[:, None].astype(np.float32)]
    if cfg.tab_features == "v2":
        ang = 2 * np.pi * ds.anchor_doy / 365.25
        sd = np.maximum(ds.sd_mat[:, s], 1e-3)
        X += [np.sin(ang)[:, None], np.cos(ang)[:, None], ds.sn[:, s, None], (ds.sn[:, s] / sd)[:, None],
              sd[:, None], ds.anchor[:, None], ds.yoy[:, None]]
    return np.hstack(X).astype(np.float32)


def fit_predict_tabular(train_ds, dss, kind, cfg: Config):
    """Ridge / LightGBM, one model per EVAL step, predicting the same residual the TFT predicts
    (delta, or delta - SN). Fitted once, applied to every dataset in `dss` (a single dataset is also
    accepted). Returns quantile deltas in metres, (N, H, 3) per dataset."""
    single = not isinstance(dss, (list, tuple))
    dss = [dss] if single else list(dss)
    outs = []
    for ds in dss:
        qd = np.zeros((len(ds), ds.H, 3), np.float32)
        qd[:, :, 1] = ds.base
        outs.append(qd)
    if len(train_ds) < 50:
        return outs[0] if single else outs
    for s in EVAL_STEPS:
        mtr = train_ds.M[:, s] > 0
        if mtr.sum() < 50:
            continue
        ytr = train_ds.actual[mtr, s] - train_ds.anchor[mtr] - train_ds.base[mtr, s]
        Xtr = _tab_features(train_ds, s, cfg)[mtr]
        if kind == "ridge":
            Xtr = np.nan_to_num(Xtr)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            m = Ridge(alpha=10.0).fit((Xtr - mu) / sd, ytr)
            res = np.quantile(np.abs(ytr - m.predict((Xtr - mu) / sd)), 0.8)
            for ds, qd in zip(dss, outs):
                if len(ds):
                    mid = m.predict((np.nan_to_num(_tab_features(ds, s, cfg)) - mu) / sd)
                    qd[:, s, 1], qd[:, s, 0], qd[:, s, 2] = mid, mid - res, mid + res
        else:
            for qi, q in enumerate([0.1, 0.5, 0.9]):
                if HAS_LGB:
                    mdl = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=cfg.gbm_n_estimators,
                                            learning_rate=cfg.gbm_lr, num_leaves=cfg.gbm_leaves,
                                            min_child_samples=cfg.gbm_min_child, subsample=0.8, subsample_freq=1,
                                            colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
                else:
                    mdl = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=cfg.gbm_n_estimators,
                                                        learning_rate=cfg.gbm_lr, max_leaf_nodes=cfg.gbm_leaves,
                                                        min_samples_leaf=cfg.gbm_min_child)
                mdl.fit(Xtr, ytr)
                for ds, qd in zip(dss, outs):
                    if len(ds):
                        qd[:, s, qi] = mdl.predict(_tab_features(ds, s, cfg))
        for ds, qd in zip(dss, outs):
            qd[:, s, :] += ds.base[:, s, None]
            qd[:, s, 0] = np.minimum(qd[:, s, 0], qd[:, s, 1])
            qd[:, s, 2] = np.maximum(qd[:, s, 2], qd[:, s, 1])
    return outs[0] if single else outs


# ─── Row-frame post-processing (shrinkage / stacking / conformal) ─────────────
# Everything after the networks have predicted works on a long frame with one row
# per (window, decoder step) whose target is observed and in-split. Earlier folds'
# TEST frames are carried forward as an out-of-sample "pool" (Phase 2.1): it is the
# only source of same-season errors (fold 4 monsoon inherits fold 1's monsoon), and
# it is what the v3 calibration block — 5-6 calm weeks just before each test fold —
# lacked, which is why v3's honest coverage undershot (70/66/63%).
SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer",
                   6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "post", 11: "post"}


def window_frame(ds, qd, gbm=None):
    """Long frame of all masked (window, step) cells. Deltas are metres from the anchor."""
    w_idx, s_idx = np.nonzero(ds.M > 0)
    ad = ds.anchor_date.values[w_idx]
    F = pd.DataFrame({
        "district": ds.district[w_idx], "anchor_date": ad,
        "target_date": ad + (s_idx + 1) * np.timedelta64(1, "D"), "step": s_idx,
        "y": ds.actual[w_idx, s_idx] - ds.anchor[w_idx], "anchor": ds.anchor[w_idx],
        "sn": ds.sn[w_idx, s_idx], "sd": ds.sd_mat[w_idx, s_idx].astype(np.float64),
        "tft_lo": qd[w_idx, s_idx, 0], "tft_mid": qd[w_idx, s_idx, 1], "tft_hi": qd[w_idx, s_idx, 2],
    })
    F["gbm_mid"] = np.nan
    if gbm is not None:
        ev = np.isin(s_idx, EVAL_STEPS)
        F.loc[ev, "gbm_mid"] = gbm[w_idx[ev], s_idx[ev], 1]
    F["season"] = pd.to_datetime(F["target_date"]).dt.month.map(SEASON_OF_MONTH)
    return F


def _wmae(err, w):
    return float(np.sum(np.abs(err) * w) / max(np.sum(w), 1e-12))


def _lambda_center(cfg: Config):
    if cfg.lambda_center == "auto":
        return "sn" if cfg.target_mode == "snaive" else "zero"
    return cfg.lambda_center


def fit_lambda_frame(P, cfg: Config, bands):
    """Shrinkage per (district, band): mid = c + lambda * (tft_mid - c), fitted by weighted MAE
    (v3 M1), pulled toward the band's global lambda with strength cfg.lambda_prior_n.

    c is the forecast the network falls back to: 0 (= persistence) for the v3 delta target, and
    the seasonal-naive change for the residual-on-SN target (v4.1). Until v4.0, lambda < 1 pulled
    a residual-on-SN TFT toward persistence, i.e. AWAY from the forecast it was trained around.
    """
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 3)
    b = bands[P["step"].values]
    center = _lambda_center(cfg)
    c = P["sn"].values if center == "sn" else 0.0
    y, p, w = P["y"].values - c, P["tft_mid"].values - c, P["w"].values
    out = {"_global": {}, "_center": center}
    for bb in np.unique(bands):
        m = b == bb
        out["_global"][int(bb)] = float(grid[np.argmin([_wmae(y[m] - g * p[m], w[m]) for g in grid])]) \
            if m.sum() >= 10 else 1.0
    for (dist, bb), idx in pd.Series(np.arange(len(P))).groupby([P["district"].values, b]).groups.items():
        idx = np.asarray(list(idx))
        g = out["_global"][int(bb)]
        if len(idx) < 10:
            out[(dist, int(bb))] = g
            continue
        lam_d = float(grid[np.argmin([_wmae(y[idx] - gg * p[idx], w[idx]) for gg in grid])])
        n = float(w[idx].sum())
        out[(dist, int(bb))] = float(np.clip((n * lam_d + cfg.lambda_prior_n * g) / (n + cfg.lambda_prior_n), 0, 1))
    return out


def apply_lambda_frame(F, lam, bands):
    b = bands[F["step"].values]
    L = np.array([lam.get((d, int(bb)), lam["_global"].get(int(bb), 1.0)) for d, bb in zip(F["district"].values, b)])
    c = F["sn"].values if lam.get("_center") == "sn" else 0.0
    mid = c + L * (F["tft_mid"].values - c)
    return mid, mid - (F["tft_mid"].values - F["tft_lo"].values), mid + (F["tft_hi"].values - F["tft_mid"].values)


STACK_COMPONENTS = ["zero", "sn", "tft_mid", "gbm_mid"]   # persistence, seasonal-naive, TFT, LightGBM


def _simplex_grid(k, step=0.1):
    n = int(round(1 / step))
    pts = []

    def rec(prefix, left, depth):
        if depth == k - 1:
            pts.append(prefix + [left])
            return
        for i in range(left + 1):
            rec(prefix + [i], left - i, depth + 1)
    rec([], n, 0)
    return np.array(pts, dtype=np.float64) / n


def fit_stack(P, cfg: Config, guarded=None):
    """Simplex weights over STACK_COMPONENTS per eval step, fitted on the pool (weighted MAE).
    Per-district weights are pulled toward the global ones with strength cfg.stack_prior_n.

    v4.2 stack_guard: the pull toward the global weights can leave a district on a component that is
    poor there (Pune: LightGBM, MASE 1.30 at t+30, while seasonal-naive and the TFT are ~0.87). If the
    shrunk weights are worse on the district's own pool than both SN alone and the TFT alone, that
    district uses the better of the two. Each guarded case is appended to `guarded` (if given)."""
    W = _simplex_grid(len(STACK_COMPONENTS), 0.1)
    out = {}
    for s in EVAL_STEPS:
        Q = P[(P["step"] == s) & P["gbm_mid"].notna()]
        if len(Q) < 30:
            continue
        C = np.column_stack([np.zeros(len(Q)), Q["sn"], Q["tft_mid"], Q["gbm_mid"]])
        E = np.abs(Q["y"].values[:, None] - C @ W.T)            # (n, n_grid)
        w = Q["w"].values
        g = W[np.argmin((E * w[:, None]).sum(0))]
        out[("_global", s)] = g
        for dist, idx in Q.groupby("district").indices.items():
            n = float(w[idx].sum())
            wd = W[np.argmin((E[idx] * w[idx, None]).sum(0))]
            ws = (n * wd + cfg.stack_prior_n * g) / (n + cfg.stack_prior_n)
            if cfg.stack_guard and n >= cfg.stack_guard_min_n:
                y_, w_ = Q["y"].values[idx], w[idx]
                e_stack = _wmae(y_ - C[idx] @ ws, w_)
                e_sn, e_tft = _wmae(y_ - C[idx, 1], w_), _wmae(y_ - C[idx, 2], w_)
                if e_stack > min(e_sn, e_tft):
                    pick = 1 if e_sn <= e_tft else 2
                    ws = np.eye(len(STACK_COMPONENTS))[pick]
                    if guarded is not None:
                        guarded.append({"district": dist, "horizon": HNAME[s], "pool_n": round(n, 1),
                                        "stack MAE": round(e_stack, 4), "SN MAE": round(e_sn, 4),
                                        "TFT MAE": round(e_tft, 4), "uses": STACK_COMPONENTS[pick]})
            out[(dist, s)] = ws
    return out


def apply_stack(F, stack):
    """Stacked delta median at eval steps (NaN elsewhere); intervals keep the TFT's spreads."""
    mid = np.full(len(F), np.nan)
    for s in EVAL_STEPS:
        m = (F["step"].values == s) & F["gbm_mid"].notna().values
        if not m.any() or ("_global", s) not in stack:
            continue
        C = np.column_stack([np.zeros(m.sum()), F.loc[m, "sn"], F.loc[m, "tft_mid"], F.loc[m, "gbm_mid"]])
        Wd = np.array([stack.get((d, s), stack[("_global", s)]) for d in F.loc[m, "district"].values])
        mid[m] = (C * Wd).sum(1)
    lo = mid - (F["tft_mid"].values - F["tft_lo"].values)
    hi = mid + (F["tft_hi"].values - F["tft_mid"].values)
    return mid, lo, hi


def _wquantile(v, w, q):
    if len(v) == 0:
        return 0.0
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w) / w.sum()
    return float(np.interp(min(max(q, 0.0), 1.0), cw, v))


def conformal_v3(P, mid, lo, hi, cfg: Config, bands):
    """v3 split conformal: absolute scores from the CAL block only, per (district, band)."""
    score = np.maximum(lo - P["y"].values, P["y"].values - hi)
    ok = np.isfinite(score)
    b = bands[P["step"].values]
    out = {"_global": {}}

    def marg(v):
        if len(v) < 5:
            return None
        lvl = min(np.ceil((len(v) + 1) * cfg.target_coverage) / len(v), 1.0)
        return float(np.quantile(v, lvl))
    for bb in np.unique(bands):
        out["_global"][int(bb)] = marg(score[ok & (b == bb)]) or 0.0
    for dist in np.unique(P["district"].values):
        dm = P["district"].values == dist
        for bb in np.unique(bands):
            m = ok & dm & (b == bb)
            out[(dist, int(bb))] = marg(score[m]) if m.sum() >= cfg.conf_min_n else out["_global"][int(bb)]
    return out


def apply_conformal_v3(F, lo, hi, margins, bands):
    b = bands[F["step"].values]
    M = np.array([margins.get((d, int(bb)), margins["_global"].get(int(bb), 0.0)) if margins else 0.0
                  for d, bb in zip(F["district"].values, b)])
    return lo - M, hi + M


def _norm_width(mid, lo, hi, sd):
    return np.maximum((hi - lo) / 2.0, 0.25 * sd)


class _WQ:
    """Weighted empirical quantile with the sort done once."""

    def __init__(self, v, w):
        o = np.argsort(v)
        self.v = v[o]
        cw = np.cumsum(w[o])
        self.cw = cw / cw[-1] if len(cw) else cw
        self.n = float(w.sum()) if len(w) else 0.0

    def q(self, level):
        if len(self.v) == 0:
            return 0.0
        return float(np.interp(min(max(level, 0.0), 1.0), self.cw, self.v))


def conformal_v4_online(T, t_mid, t_lo, t_hi, P, p_mid, p_lo, p_hi, cfg: Config, bands, tier):
    """Normalised, season-weighted, two-sided conformal with adaptive miscoverage control.

    Each side is calibrated separately (asymmetric CQR): the lower score
    (lo - y) / w and the upper score (y - hi) / w, with w = max(half-width, 0.25 * residual SD),
    each target (1 - coverage) / 2 misses. v3's symmetric margin left 18-21% of t+30 cells
    above q90 and 3-9% below q10 — the network under-predicts monsoon rises, and one shared
    margin cannot fix a one-sided miss.
    The pool P (cal block + earlier folds' test cells, season-weighted) gives each
    (district, band, side) a weighted score distribution, pulled toward the district's tier
    with pseudo-count 100.
    ACI: cells are processed in anchor order; a cell's outcome is fed back only once its
    target date has passed the current anchor (delayed feedback, as in deployment), and
    alpha <- clip(alpha + gamma * (alpha* - miss), aci_alpha_min, aci_alpha_max) per
    (district, band, side).
    aci_feedback='cell' (v4.1) feeds back every cell, i.e. ~10 strongly correlated outcomes per
    anchor and band for the TFT (all 30 steps are scored) but 1 for the Ensemble (eval steps only).
    'anchor' feeds back ONE outcome per (anchor, district, band): the mean miss of its cells, due
    once the last of them is observed.
    """
    import heapq
    a_star = (1.0 - cfg.target_coverage) / 2.0
    y_p = P["y"].values
    pw = _norm_width(p_mid, p_lo, p_hi, P["sd"].values)
    scores = {"lo": (p_lo - y_p) / pw, "hi": (y_p - p_hi) / pw}
    pb = bands[P["step"].values]
    pdist, pwt = P["district"].values, P["w"].values
    ptier = np.array([tier.get(d, "B") for d in pdist])
    cache = {}

    def side_q(side, dist, bb, level):
        key = (side, dist, bb)
        if key not in cache:
            sc = scores[side]
            ok = np.isfinite(sc) & (pb == bb)
            m_d, m_t = ok & (pdist == dist), ok & (ptier == tier.get(dist, "B"))
            cache[key] = (_WQ(sc[m_d], pwt[m_d]),
                          _WQ(sc[m_t], pwt[m_t]) if m_t.sum() >= 30 else _WQ(sc[ok], pwt[ok]))
        qd_, qb = cache[key]
        base = qb.q(level)
        return (qd_.n * qd_.q(level) + 100.0 * base) / (qd_.n + 100.0) if qd_.n > 0 else base

    tw = _norm_width(t_mid, t_lo, t_hi, T["sd"].values)
    lo_c, hi_c = t_lo.copy(), t_hi.copy()
    tb = bands[T["step"].values]
    order = np.lexsort((T["step"].values, T["anchor_date"].values))
    anchors, tdates = T["anchor_date"].values, T["target_date"].values
    tdist, y = T["district"].values, T["y"].values
    alpha, pending = {}, []                      # pending: (target_date, district, band, miss_lo, miss_hi)
    i = 0
    while i < len(order):
        a = anchors[order[i]]
        j = i
        while j < len(order) and anchors[order[j]] == a:
            j += 1
        while pending and pending[0][0] <= a:
            _, d_, b_, ml, mh = heapq.heappop(pending)
            for side, miss in (("lo", ml), ("hi", mh)):
                al = alpha.get((d_, b_, side), a_star)
                alpha[(d_, b_, side)] = float(np.clip(al + cfg.aci_gamma * (a_star - miss),
                                                      cfg.aci_alpha_min, cfg.aci_alpha_max))
        groups = {}                            # 'anchor' feedback: (district, band) -> [last date, misses]
        for k in order[i:j]:
            if not np.isfinite(t_mid[k]):
                continue                       # no forecast at this step (e.g. ensemble off eval steps)
            d_, b_ = tdist[k], int(tb[k])
            m_lo = side_q("lo", d_, b_, 1.0 - alpha.get((d_, b_, "lo"), a_star))
            m_hi = side_q("hi", d_, b_, 1.0 - alpha.get((d_, b_, "hi"), a_star))
            lo_c[k] = min(t_lo[k] - m_lo * tw[k], t_mid[k])
            hi_c[k] = max(t_hi[k] + m_hi * tw[k], t_mid[k])
            miss = (float(y[k] < lo_c[k]), float(y[k] > hi_c[k]))
            if cfg.aci_feedback == "anchor":
                g = groups.setdefault((d_, b_), [tdates[k], [], []])
                g[0] = max(g[0], tdates[k])
                g[1].append(miss[0])
                g[2].append(miss[1])
            else:
                heapq.heappush(pending, (tdates[k], d_, b_) + miss)
        for (d_, b_), (td, ml, mh) in groups.items():
            heapq.heappush(pending, (td, d_, b_, float(np.mean(ml)), float(np.mean(mh))))
        i = j
    return lo_c, hi_c


def frame_rows(F, mid, lo, hi, model_name, fold):
    """Scored rows (eval steps only) in the same schema as rows_daily."""
    m = np.isin(F["step"].values, EVAL_STEPS) & np.isfinite(mid)
    G = F.loc[m]
    a = G["anchor"].values
    return pd.DataFrame({
        "model": model_name, "fold": fold, "district": G["district"].values,
        "anchor_date": pd.to_datetime(G["anchor_date"].values), "target_date": pd.to_datetime(G["target_date"].values),
        "horizon": G["step"].map(HNAME).values, "step": G["step"].values, "actual": a + G["y"].values,
        "q10": a + lo[m], "q50": a + mid[m], "q90": a + hi[m], "persistence": a, "snaive": a + G["sn"].values,
    })


# ─── Rows + metrics ───────────────────────────────────────────────────────────
def rows_daily(ds, qd, model_name, fold):
    """One row per (window, eval horizon) whose target day is observed and in the split."""
    recs = []
    for s in EVAL_STEPS:
        m = ds.M[:, s] > 0
        if not m.any():
            continue
        a = ds.anchor[m]
        ad = ds.anchor_date[m]
        recs.append(pd.DataFrame({
            "model": model_name, "fold": fold, "district": ds.district[m],
            "anchor_date": ad, "target_date": ad + pd.Timedelta(days=s + 1),
            "horizon": HNAME[s], "step": s, "actual": ds.actual[m, s],
            "q10": a + qd[m, s, 0], "q50": a + qd[m, s, 1], "q90": a + qd[m, s, 2],
            "persistence": a, "snaive": a + ds.sn[m, s],
        }))
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def rows_weekly(ds, qd, model_name, fold, win=7, min_obs=4):
    """7-day-mean level over the week ending at the horizon; all 7 days must be in-split."""
    recs = []
    for s in EVAL_STEPS:
        sl = slice(max(0, s - win + 1), s + 1)
        keep = ds.insplit[:, sl].all(1) & (ds.M[:, sl].sum(1) >= min_obs)
        if not keep.any():
            continue
        act = np.where(ds.M[keep][:, sl] > 0, ds.actual[keep][:, sl], np.nan)
        a = ds.anchor[keep]
        q = qd[keep][:, sl, :].mean(axis=1)
        sn = ds.sn[keep][:, sl].mean(axis=1)
        ad = ds.anchor_date[keep]
        recs.append(pd.DataFrame({
            "model": model_name, "fold": fold, "district": ds.district[keep],
            "anchor_date": ad, "target_date": ad + pd.Timedelta(days=s + 1),
            "horizon": HNAME[s], "step": s, "actual": np.nanmean(act, axis=1),
            "q10": a + q[:, 0], "q50": a + q[:, 1], "q90": a + q[:, 2],
            "persistence": a, "snaive": a + sn,
        }))
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def pinball(a, q10, q50, q90):
    tot = 0.0
    for q, p in [(0.1, q10), (0.5, q50), (0.9, q90)]:
        d = a - p
        tot += np.mean(np.maximum(q * d, (q - 1) * d))
    return tot / 3.0


def metrics(pred, by=None):
    """rMAE vs persistence (= v3 'MASE') and vs seasonal-naive, plus the usual set."""
    keys = ["horizon"] if by is None else [by, "horizon"]
    rows = []
    for k, g in pred.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        a, p = g["actual"].values, g["q50"].values
        nv, sn = g["persistence"].values, g["snaive"].values
        mae, mae_p, mae_s = np.mean(np.abs(a - p)), np.mean(np.abs(a - nv)), np.mean(np.abs(a - sn))
        mse, mse_p = np.mean((a - p) ** 2), np.mean((a - nv) ** 2)
        if "district" in g.columns and g["district"].nunique() > 1:
            clim = g.groupby("district")["actual"].transform("mean").values
        else:
            clim = a.mean()
        r = {
            "MAE (m)": mae, "RMSE (m)": np.sqrt(mse), "Persist MAE": mae_p, "SN MAE": mae_s,
            "rMAE_P": mae / mae_p if mae_p > 1e-9 else np.nan,
            "rMAE_SN": mae / mae_s if mae_s > 1e-9 else np.nan,
            "Skill %": (1 - mse / mse_p) * 100 if mse_p > 1e-12 else np.nan,
            "MAPE (%)": np.mean(np.abs((a - p) / np.maximum(np.abs(a), 0.5))) * 100,
            "NSE": 1 - mse / max(np.mean((a - clim) ** 2), 1e-12),
            "Pinball": pinball(a, g["q10"].values, p, g["q90"].values),
            "Cov (%)": ((a >= g["q10"].values) & (a <= g["q90"].values)).mean() * 100,
            "Below q10 (%)": (a < g["q10"].values).mean() * 100,
            "Above q90 (%)": (a > g["q90"].values).mean() * 100,
            "Width (m)": (g["q90"].values - g["q10"].values).mean(),
            # Winkler interval score for the 80% interval (lower = sharper AND better calibrated)
            "Interval score (m)": float(np.mean((g["q90"].values - g["q10"].values)
                                                + 10.0 * np.maximum(g["q10"].values - a, 0)
                                                + 10.0 * np.maximum(a - g["q90"].values, 0))),
            "n": len(g),
        }
        if by is None:
            r["Horizon"] = k[0]
        else:
            r[by], r["Horizon"] = k[0], k[1]
        rows.append(r)
    out = pd.DataFrame(rows)
    out["_o"] = out["Horizon"].map({h: i for i, h in enumerate(HORDER)})
    out = out.sort_values(([by] if by else []) + ["_o"]).drop(columns="_o").reset_index(drop=True)
    return out.round(4)


def headline_subset(d):
    return d[~d["district"].isin(HEADLINE_EXCLUDE)]


def district_gap(per_dist, col="rMAE_P", k=5):
    """worst-k mean minus best-k mean of per-district rMAE, per horizon."""
    out = {}
    for h, g in per_dist.groupby("Horizon"):
        v = g[col].dropna().sort_values().values
        if len(v) >= 2 * k:
            out[h] = {"best5": v[:k].mean(), "worst5": v[-k:].mean(), "gap": v[-k:].mean() - v[:k].mean(),
                      "n_lt1": int((v < 1).sum()), "n": len(v), "median": float(np.median(v))}
    return out


def dm_test(a, p, ref, lag):
    """Diebold-Mariano on absolute error with a Newey-West HAC variance. Negative => model better."""
    d = np.abs(a - p) - np.abs(a - ref)
    n = len(d)
    if n < 30:
        return np.nan, np.nan
    dc = d - d.mean()
    var = np.mean(dc * dc)
    for l in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - l / (lag + 1)) * np.mean(dc[l:] * dc[:-l])
    stat = d.mean() / np.sqrt(max(var, 1e-12) / n)
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(stat) / math.sqrt(2))))
    return float(stat), float(pval)


def cluster_bootstrap(g, ref_col, n_boot=2000, seed=0):
    """Resample DISTRICTS; pooled rMAE = sum|e_model| / sum|e_ref|. One-sided p for H0: rMAE >= 1."""
    rng = np.random.default_rng(seed)
    agg = g.assign(em=(g.actual - g.q50).abs(), er=(g.actual - g[ref_col]).abs()) \
        .groupby("district")[["em", "er"]].sum()
    em, er = agg["em"].values, agg["er"].values
    k = len(em)
    idx = rng.integers(0, k, size=(n_boot, k))
    draws = em[idx].sum(1) / er[idx].sum(1)
    point = em.sum() / er.sum()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = (1 + (draws >= 1.0).sum()) / (n_boot + 1)
    return point, lo, hi, p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def significance(daily, model_name, refs=("persistence", "snaive")):
    rows = []
    for h, lag in [("t+7", 7), ("t+14", 14), ("t+30", 30)]:
        g = headline_subset(daily[(daily.model == model_name) & (daily.horizon == h)]) \
            .sort_values(["district", "target_date"])
        if not len(g):
            continue
        for ref in refs:
            pt, lo, hi, pb = cluster_bootstrap(g, ref)
            s, pd_ = dm_test(g["actual"].values, g["q50"].values, g[ref].values, lag)
            rows.append({"horizon": h, "vs": ref, "pooled rMAE": round(pt, 4),
                         "95% CI": f"[{lo:.3f}, {hi:.3f}]", "p_boot": pb, "DM stat": round(s, 3), "p_DM": pd_})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_boot_holm"] = holm(out["p_boot"].values)
        out["p_DM_holm"] = holm(out["p_DM"].values)
    return out


# ─── Fold driver ──────────────────────────────────────────────────────────────
def fold_dates(fold, cfg: Config):
    from dateutil.relativedelta import relativedelta
    test_start = pd.Timestamp(fold["test_start"])
    train_end = test_start - pd.Timedelta(days=1)
    val_start = train_end - relativedelta(months=cfg.val_months) + pd.Timedelta(days=1)
    fit_end = val_start - pd.Timedelta(days=1)
    es_end = val_start + pd.Timedelta(days=int((train_end - val_start).days * cfg.val_es_frac))
    return dict(test_start=test_start, test_end=pd.Timestamp(fold["test_end"]), train_end=train_end,
                val_start=val_start, fit_end=fit_end, es_end=es_end, cal_start=es_end + pd.Timedelta(days=1))


def _assert_no_leak(ds, cutoff, name):
    if len(ds) == 0:
        return
    td = ds.target_dates()
    used = ds.M > 0
    if used.any():
        mx = td[used].max()
        assert mx <= np.datetime64(cutoff), f"{name}: target date {mx} beyond {cutoff}"


def _fold_season(fd):
    months = pd.date_range(fd["test_start"], fd["test_end"], freq="D").month
    return pd.Series(months).map(SEASON_OF_MONTH).mode().iloc[0]


class DSLite:
    """What post-processing needs from a GWLDataset once the networks have predicted
    (no input tensors), so stage-1 results can be cached compactly."""

    FIELDS = ("district", "anchor_date", "anchor", "actual", "M", "insplit", "sd_mat", "base", "sn", "H")

    def __init__(self, ds):
        for f in self.FIELDS:
            setattr(self, f, getattr(ds, f))

    def target_dates(self):
        return self.anchor_date.values[:, None] + (np.arange(1, self.H + 1) * np.timedelta64(1, "D"))[None, :]

    def __len__(self):
        return len(self.anchor)


def _sn_at(df, cutoff, cfg: Config):
    """SN table fitted on data up to `cutoff`, plus leave-own-year-out tables (snaive mode only)
    for training windows, so a training target's own year never feeds its SN base."""
    tab = fit_snaive_table(df, cutoff, cfg)
    if cfg.target_mode != "snaive":
        return tab, None
    years = sorted(df.loc[df.date <= pd.Timestamp(cutoff), "date"].dt.year.unique())
    return tab, {y: fit_snaive_table(df, cutoff, cfg, exclude_year=y) for y in years}


def fold_inputs(cfg: Config, df, meta, fold):
    """Everything a fold's datasets are built from; identical for the network and tabular stages."""
    fd = fold_dates(fold, cfg)
    sn_fit, sn_loyo = _sn_at(df, fd["fit_end"], cfg)
    sn_test = fit_snaive_table(df, fd["train_end"], cfg)
    dfn = normalize_for_fold(df, fd["fit_end"], cfg, meta)
    sd_enc, sd_h = compute_delta_scales(df, fd["fit_end"], cfg,
                                        sn_tables=sn_fit if cfg.target_mode == "snaive" else None)

    def mk(s0, s1_, stride, tables, by_year=None, gate=True, keep=True):
        return GWLDataset(dfn, cfg, meta, sd_enc, sd_h, split_start=s0, split_end=s1_, stride=stride,
                          sn_tables=tables, sn_tables_by_year=by_year, dec_gate=gate, keep_tensors=keep)
    return dict(fd=fd, sn_fit=sn_fit, sn_loyo=sn_loyo, sn_test=sn_test, dfn=dfn, mk=mk)


def train_fold(cfg: Config, df, meta, fold, device, verbose=True):
    """Stage 1a: datasets, seed-ensemble networks (pass 1, and pass 2 when cfg.refit), predictions."""
    t0 = time.time()
    I = fold_inputs(cfg, df, meta, fold)
    fd, mk, name = I["fd"], I["mk"], fold["name"]
    train_ds = mk(None, fd["fit_end"], cfg.train_stride, I["sn_fit"], I["sn_loyo"])
    es_ds = mk(fd["val_start"], fd["es_end"], cfg.val_stride, I["sn_fit"])
    cal_ds = mk(fd["cal_start"], fd["train_end"], cfg.val_stride, I["sn_fit"], gate=False)
    test_ds = mk(fd["test_start"], fd["test_end"], cfg.test_stride, I["sn_test"], gate=False)
    _assert_no_leak(train_ds, fd["fit_end"], "train")
    _assert_no_leak(es_ds, fd["es_end"], "es")
    _assert_no_leak(cal_ds, fd["train_end"], "cal")
    if "gwl_frac_real" in df.columns and cfg.loss_weighting in ("frac_real", "both"):
        fr = I["dfn"].set_index(["district", "date"])["gwl_frac_real"]
        td = train_ds.target_dates()
        frac = np.array([[fr.get((d, pd.Timestamp(t)), 1.0) for t in row] for d, row in zip(train_ds.district, td)])
        apply_loss_weighting(train_ds, cfg, frac)
    else:
        apply_loss_weighting(train_ds, cfg)
    refit_ds = None
    if cfg.refit:
        # pass-2 training set: every window with targets up to the end of the early-stopping block
        refit_ds = mk(None, fd["es_end"], cfg.train_stride, I["sn_fit"], I["sn_loyo"])
        _assert_no_leak(refit_ds, fd["es_end"], "refit")
        apply_loss_weighting(refit_ds, cfg)

    win = {"fold": name, "train": len(train_ds), "es": len(es_ds), "cal": len(cal_ds), "test": len(test_ds),
           "refit": len(refit_ds) if refit_ds is not None else 0}
    if verbose:
        print(f"[{name}] fit<= {fd['fit_end'].date()} | es {fd['val_start'].date()}..{fd['es_end'].date()} | "
              f"cal {fd['cal_start'].date()}..{fd['train_end'].date()} | test {fd['test_start'].date()}.."
              f"{fd['test_end'].date()} | windows {win}", flush=True)

    H = cfg.decoder_len
    acc = {k: np.zeros((len(d), H, 3)) for k, d in
           (("cal", cal_ds), ("test", test_ds), ("cal2", cal_ds), ("test2", test_ds))}
    curves, curves_p2, best_epochs, refit_epochs = [], [], [], []
    state_p1 = state_p2 = None
    for seed in cfg.seeds:
        model, tr_h, va_h, lrs = train_one(cfg, meta, train_ds, es_ds, seed, device, f"{name}|s{seed}", verbose)
        curves.append((tr_h, va_h))
        best_ep = int(np.argmin(va_h)) + 1
        best_epochs.append(best_ep)
        acc["cal"] += predict_scaled(cal_ds, model, cfg, device)
        acc["test"] += predict_scaled(test_ds, model, cfg, device)
        state_p1 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if cfg.refit:
            n_ep = max(cfg.refit_min_epochs, best_ep)
            trace = (lrs + [lrs[-1]] * n_ep)[:n_ep]
            m2, tr2 = train_fixed(cfg, meta, refit_ds, seed, device, trace)
            curves_p2.append(tr2)
            refit_epochs.append(n_ep)
            acc["cal2"] += predict_scaled(cal_ds, m2, cfg, device)
            acc["test2"] += predict_scaled(test_ds, m2, cfg, device)
            state_p2 = {k: v.detach().cpu().clone() for k, v in m2.state_dict().items()}
            del m2
        del model
        gc.collect()
    ns = len(cfg.seeds)
    s1 = dict(
        fd=fd, win=win, curves=curves, curves_p2=curves_p2, best_epochs=best_epochs, refit_epochs=refit_epochs,
        model_state_p1=state_p1, model_state_p2=state_p2,
        cal_qd=to_delta_m(cal_ds, (acc["cal"] / ns).astype(np.float32)),
        test_qd=to_delta_m(test_ds, (acc["test"] / ns).astype(np.float32)),
        cal_qd_p2=to_delta_m(cal_ds, (acc["cal2"] / ns).astype(np.float32)) if cfg.refit else None,
        test_qd_p2=to_delta_m(test_ds, (acc["test2"] / ns).astype(np.float32)) if cfg.refit else None,
        train=DSLite(train_ds), cal=DSLite(cal_ds), test=DSLite(test_ds),
        train_seconds=time.time() - t0,
    )
    if verbose:
        extra = f" | refit epochs {refit_epochs}" if refit_epochs else ""
        print(f"[{name}] networks trained in {s1['train_seconds']:.0f}s | best epochs {best_epochs}{extra}",
              flush=True)
    return s1


def tab_fold(cfg: Config, df, meta, fold, s1, verbose=True):
    """Stage 1b: LightGBM + ridge, cached separately from the networks.

    tab_fresh: the model whose predictions feed the stacking weights (cal block) is trained on targets up
    to es_end; the model that forecasts the test period is refitted on everything up to the day before
    it (train_end). Neither ever sees its own evaluation block.
    """
    t0 = time.time()
    I = fold_inputs(cfg, df, meta, fold)
    fd, mk = I["fd"], I["mk"]
    cal_t = mk(fd["cal_start"], fd["train_end"], cfg.val_stride, I["sn_fit"], gate=False, keep=False)
    test_t = mk(fd["test_start"], fd["test_end"], cfg.test_stride, I["sn_test"], gate=False, keep=False)
    for a, b, nm in ((cal_t, s1["cal"], "cal"), (test_t, s1["test"], "test")):
        assert len(a) == len(b) and (a.district == b.district).all() and (a.anchor_date == b.anchor_date).all(), \
            f"tabular {nm} windows differ from the network's"
    if cfg.tab_fresh:
        tr_cal = mk(None, fd["es_end"], cfg.tab_stride, *_sn_at(df, fd["es_end"], cfg), keep=False)
        tr_test = mk(None, fd["train_end"], cfg.tab_stride, *_sn_at(df, fd["train_end"], cfg), keep=False)
        _assert_no_leak(tr_cal, fd["es_end"], "tabular cal model")
        _assert_no_leak(tr_test, fd["train_end"], "tabular test model")
        gbm_cal = fit_predict_tabular(tr_cal, cal_t, "gbm", cfg)
        gbm_test = fit_predict_tabular(tr_test, test_t, "gbm", cfg)
    else:
        tr_cal = tr_test = mk(None, fd["fit_end"], cfg.tab_stride, I["sn_fit"], I["sn_loyo"], keep=False)
        _assert_no_leak(tr_cal, fd["fit_end"], "tabular model")
        gbm_cal, gbm_test = fit_predict_tabular(tr_cal, [cal_t, test_t], "gbm", cfg)
    out = dict(gbm_cal=gbm_cal, gbm_test=gbm_test, ridge_test=fit_predict_tabular(tr_test, test_t, "ridge", cfg),
               n_cal_train=len(tr_cal), n_test_train=len(tr_test), seconds=time.time() - t0)
    if verbose:
        print(f"[{fold['name']}] tabular models in {out['seconds']:.0f}s "
              f"(train windows {out['n_cal_train']} / {out['n_test_train']})", flush=True)
    return out


def _pick_pass(cfg: Config, s1):
    use2 = cfg.use_refit and s1.get("cal_qd_p2") is not None
    if use2:
        return s1["cal_qd_p2"], s1["test_qd_p2"], s1["model_state_p2"], True
    return s1["cal_qd"], s1["test_qd"], s1.get("model_state_p1", s1.get("model_state")), False


def postprocess_fold(cfg: Config, meta, fold, s1, tab, pool=None, verbose=True):
    """Stage 2 (cheap, never cached): shrinkage, stacking, conformal, scored rows."""
    fd, name = s1["fd"], fold["name"]
    cal_ds, test_ds, train_ds = s1["cal"], s1["test"], s1["train"]
    cal_qd, test_qd, _, _ = _pick_pass(cfg, s1)
    tab = tab or {"gbm_cal": None, "gbm_test": None, "ridge_test": None}
    bands = _bands(cfg)
    C = window_frame(cal_ds, cal_qd, tab["gbm_cal"])
    T = window_frame(test_ds, test_qd, tab["gbm_test"])
    season = _fold_season(fd)
    use_pool = cfg.calib_mode == "v4" or cfg.combine == "stack"
    if use_pool:
        # pool = cal block + earlier folds' out-of-sample test cells, season-weighted
        P = pd.concat([C] + ([pool] if pool is not None and len(pool) else []), ignore_index=True)
        P["w"] = np.where(P["season"] == season, 1.0, cfg.season_weight_other)
    else:
        P = C.assign(w=1.0)
    lam = fit_lambda_frame(P if cfg.calib_mode == "v4" else C.assign(w=1.0), cfg, bands)
    t_mid, t_lo, t_hi = apply_lambda_frame(T, lam, bands)

    def calibrate(F_mid, F_lo, F_hi, combo):
        """combo(frame) -> (mid, lo, hi): the same point-forecast combination on another frame."""
        if cfg.calib_mode == "v4":
            p_mid, p_lo, p_hi = combo(P)
            ok = np.isfinite(p_mid)
            return conformal_v4_online(T, F_mid, F_lo, F_hi, P.loc[ok], p_mid[ok], p_lo[ok], p_hi[ok],
                                       cfg, bands, meta["tier"])
        c_mid, c_lo, c_hi = combo(C)
        ok = np.isfinite(c_mid)
        margins = conformal_v3(C.loc[ok], c_mid[ok], c_lo[ok], c_hi[ok], cfg, bands)
        return apply_conformal_v3(T, F_lo, F_hi, margins, bands)

    lo_c, hi_c = calibrate(t_mid, t_lo, t_hi, lambda F: apply_lambda_frame(F, lam, bands))
    parts = [frame_rows(T, T["tft_mid"].values, T["tft_lo"].values, T["tft_hi"].values, "TFT (raw)", name),
             frame_rows(T, t_mid, t_lo, t_hi, "TFT", name),
             frame_rows(T, t_mid, lo_c, hi_c, "TFT (calibrated)", name)]
    stack, guarded = {}, []
    if cfg.combine == "stack" and cfg.with_tabular:
        stack = fit_stack(P, cfg, guarded)
        e_mid, e_lo, e_hi = apply_stack(T, stack)
        elo_c, ehi_c = calibrate(e_mid, e_lo, e_hi, lambda F: apply_stack(F, stack))
        parts += [frame_rows(T, e_mid, e_lo, e_hi, "Ensemble", name),
                  frame_rows(T, e_mid, elo_c, ehi_c, "Ensemble (calibrated)", name)]
    daily = pd.concat(parts, ignore_index=True)

    # weekly-mean target for the shrunk TFT (array form)
    Lmat = np.ones(test_qd.shape[:2])
    for dist in np.unique(test_ds.district):
        rows_ = test_ds.district == dist
        for bb in np.unique(bands):
            Lmat[np.ix_(rows_, bands == bb)] = lam.get((dist, int(bb)), lam["_global"].get(int(bb), 1.0))
    test_s = test_qd.copy()
    cmat = test_ds.sn if lam.get("_center") == "sn" else 0.0
    mid_ = cmat + Lmat * (test_qd[:, :, 1] - cmat)
    test_s[:, :, 0] = mid_ - (test_qd[:, :, 1] - test_qd[:, :, 0])
    test_s[:, :, 2] = mid_ + (test_qd[:, :, 2] - test_qd[:, :, 1])
    test_s[:, :, 1] = mid_
    weekly = rows_weekly(test_ds, test_s, "TFT", name)

    base = [rows_daily(test_ds, baseline_snaive(test_ds, train_ds), "Seasonal-naive", name)]
    if tab["gbm_test"] is not None:
        base.append(rows_daily(test_ds, tab["ridge_test"], "Ridge", name))
        base.append(rows_daily(test_ds, tab["gbm_test"], "LightGBM", name))
    base = pd.concat([b for b in base if len(b)], ignore_index=True)

    # every model scored on the identical key set; every target inside the fold
    kcols = ["district", "target_date", "horizon"]
    ref_keys = set(map(tuple, daily.loc[daily.model == "TFT", kcols].astype(str).values))
    for mname, g in pd.concat([daily, base]).groupby("model"):
        assert set(map(tuple, g[kcols].astype(str).values)) == ref_keys, f"key mismatch: {mname}"
    assert ((daily.target_date >= fd["test_start"]) & (daily.target_date <= fd["test_end"])).all()
    assert ((daily.target_date - daily.anchor_date).dt.days == daily.step + 1).all()

    keys = rows_daily(test_ds, test_qd, "keys", name)[["fold", "district", "anchor_date", "target_date",
                                                        "horizon", "actual", "persistence", "snaive"]]
    if verbose:
        m = metrics(headline_subset(daily[daily.model == "TFT"])).set_index("Horizon")
        msg = f"[{name}] TFT rMAE_P {m['rMAE_P'].round(3).to_dict()} | rMAE_SN {m['rMAE_SN'].round(3).to_dict()}"
        if stack:
            e = metrics(headline_subset(daily[daily.model == "Ensemble"])).set_index("Horizon")
            msg += f" | Ensemble rMAE_SN {e['rMAE_SN'].round(3).to_dict()}"
        print(msg, flush=True)
    return dict(daily=daily, weekly=weekly, base=base, keys=keys, lam=lam, stack=stack, pool=T,
                guarded=pd.DataFrame(guarded).assign(fold=name) if guarded else pd.DataFrame())


# ── cache keys: three independent stages ─────────────────────────────────────
# post-processing settings: changing these never retrains anything
POST_FIELDS = ("calib_mode", "combine", "aci_gamma", "stack_prior_n", "season_weight_other", "lambda_center",
               "use_refit", "lambda_prior_n", "conf_min_n", "band_size", "target_coverage",
               "aci_feedback", "aci_alpha_min", "aci_alpha_max", "stack_guard", "stack_guard_min_n",
               "interval_choice")
# tabular-model settings: changing these never retrains the networks
TAB_FIELDS = ("with_tabular", "tab_features", "tab_stride", "tab_fresh", "gbm_n_estimators", "gbm_lr",
              "gbm_min_child", "gbm_leaves")
# network-only settings: changing these never refits the tabular models
TFT_FIELDS = ("d_model", "d_select", "grn_hidden_factor", "n_heads", "lstm_layers", "dropout", "cat_embed_dim",
              "weight_decay", "batch_size", "eval_batch", "max_epochs", "lr", "warmup_epochs", "patience",
              "grad_clip", "lr_patience", "lr_factor", "min_lr", "sample_weight_cap", "seeds", "loss_weighting",
              "refit", "refit_min_epochs", "epoch_frac", "train_stride")


def _stage_key(cfg: Config, exclude, tag) -> str:
    d = {k: v for k, v in asdict(cfg).items() if k not in Config._NON_RESULT and k not in exclude}
    d["data_md5"] = file_md5(cfg.data_path)
    d["stage"] = f"{tag}-{PIPELINE_VERSION}"
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def train_key(cfg: Config) -> str:
    return _stage_key(cfg, POST_FIELDS + TAB_FIELDS, "net")


def tab_key(cfg: Config) -> str:
    return _stage_key(cfg, POST_FIELDS + TFT_FIELDS, "tab")


def run(cfg: Config, verbose=True):
    """Run the configured folds: networks (cached), tabular models (cached separately),
    post-processing (recomputed every run)."""
    device = get_device()
    df, meta = load_data(cfg, verbose)
    out_dir = Path(cfg.out_root) / cfg.tag
    cache_dir = Path(cfg.out_root) / "_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    nkey, tkey, ckey = train_key(cfg), tab_key(cfg), cfg.key()
    if verbose:
        print(f"Device: {device} | net key {nkey} | tab key {tkey} | config key {ckey} | tag {cfg.tag} | "
              f"mode {cfg.mode} | seeds {cfg.seeds}", flush=True)

    parts = {"daily": [], "weekly": [], "base": [], "keys": []}
    curves, curves_p2, lams, stacks, wins, states, epochs, guards = {}, {}, {}, {}, [], {}, [], []
    t0 = time.time()
    req = [f for f in RUN_FOLDS[cfg.mode] if not cfg.only_folds or f["name"] in cfg.only_folds]
    # v4 calibration / stacking carry earlier folds' out-of-sample errors forward, so every
    # earlier fold must exist (trained or cached) even when only later folds are reported
    need_pool = cfg.calib_mode == "v4" or cfg.combine == "stack"
    seq = FOLDS[:max(FOLDS.index(f) for f in req) + 1] if need_pool else req
    pool = None
    for fold in seq:
        c1 = cache_dir / f"net_{nkey}_{fold['name']}.pkl"
        if c1.exists():
            s1 = pd.read_pickle(c1)
            if verbose:
                print(f"[{fold['name']}] networks cached ({c1.name})", flush=True)
        else:
            s1 = train_fold(cfg, df, meta, fold, device, verbose=verbose)
            pd.to_pickle(s1, c1)
        tab = None
        if cfg.with_tabular:
            c2 = cache_dir / f"tab_{tkey}_{fold['name']}.pkl"
            if c2.exists():
                tab = pd.read_pickle(c2)
            else:
                tab = tab_fold(cfg, df, meta, fold, s1, verbose=verbose)
                pd.to_pickle(tab, c2)
        blob = postprocess_fold(cfg, meta, fold, s1, tab, pool=pool, verbose=verbose)
        if need_pool:
            pool = pd.concat([p for p in (pool, blob["pool"]) if p is not None], ignore_index=True)
        if fold not in req:
            continue
        for k in parts:
            parts[k].append(blob[k])
        nm = fold["name"]
        curves[nm], curves_p2[nm], lams[nm], stacks[nm] = s1["curves"], s1.get("curves_p2", []), blob["lam"], \
            blob["stack"]
        wins.append(s1["win"])
        guards.append(blob["guarded"])
        _, _, state, used2 = _pick_pass(cfg, s1)
        states[nm] = state
        epochs.append({"fold": nm, "best_epochs": s1.get("best_epochs"), "refit_epochs": s1.get("refit_epochs"),
                       "refit_used": used2})
        torch.save(state, out_dir / f"model_{nm}_lastseed.pt")

    res = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    res["all"] = pd.concat([res["daily"], res["base"]], ignore_index=True)
    res.update(curves=curves, curves_p2=curves_p2, lams=lams, stacks=stacks, wins=pd.DataFrame(wins),
               guarded=pd.concat(guards, ignore_index=True) if guards else pd.DataFrame(),
               epochs=pd.DataFrame(epochs), model_states=states, meta=meta, cfg=cfg, ckey=ckey,
               out_dir=out_dir, minutes=(time.time() - t0) / 60)
    return res


def subset_folds(res, folds, tag):
    """A copy of `res` restricted to some folds (e.g. the locked test folds 5-6), with its own tag."""
    r = dict(res)
    for k in ("daily", "weekly", "base", "keys", "all"):
        r[k] = res[k][res[k]["fold"].isin(folds)].reset_index(drop=True)
    cfg = copy.deepcopy(res["cfg"])
    cfg.tag, cfg.mode = tag, "final"
    r["cfg"] = cfg
    r["wins"] = res["wins"][res["wins"]["fold"].isin(folds)]
    g = res.get("guarded", pd.DataFrame())
    r["guarded"] = g[g["fold"].isin(folds)] if len(g) else g
    r["out_dir"] = Path(cfg.out_root) / tag
    r["out_dir"].mkdir(parents=True, exist_ok=True)
    return r


# ─── Reporting ────────────────────────────────────────────────────────────────
LADDER = ["Seasonal-naive", "Ridge", "LightGBM", "TFT (raw)", "TFT", "TFT (calibrated)",
          "Ensemble", "Ensemble (calibrated)"]


def interval_source(cfg: Config, model, horizon):
    """'raw' or 'calibrated': which interval is reported for this model and horizon (v4.2)."""
    return (cfg.interval_choice or {}).get(model, {}).get(horizon, "calibrated")


def reported_rows(D, model, cfg: Config):
    """Rows of `model` whose intervals are the REPORTED ones: per horizon either '<model>' (the network's
    own spreads) or '<model> (calibrated)' (conformal), as cfg.interval_choice says. q50 is the same in both."""
    parts = []
    for h in HORDER:
        src = model if interval_source(cfg, model, h) == "raw" else f"{model} (calibrated)"
        parts.append(D[(D.model == src) & (D.horizon == h)])
    return pd.concat(parts, ignore_index=True)


def _cov_tables(sub):
    cov = metrics(sub).set_index("Horizon")
    cov_d = metrics(sub, by="district")
    in_band = cov_d.groupby("Horizon")["Cov (%)"].apply(lambda v: float(((v >= 65) & (v <= 90)).mean() * 100))
    cov["districts in 65-90% (%)"] = in_band
    return cov, cov_d


def _summary(DAILY, model, tier, cfg: Config):
    sub = headline_subset(DAILY[DAILY.model == model])
    if not len(sub):
        return None
    head = metrics(sub).set_index("Horizon")
    per_dist = metrics(sub, by="district")
    per_dist["tier"] = per_dist["district"].map(tier)
    rsub = headline_subset(reported_rows(DAILY, model, cfg))
    cov, cov_d = _cov_tables(rsub)
    cov_cal, cov_cal_d = _cov_tables(headline_subset(DAILY[DAILY.model == f"{model} (calibrated)"]))
    return dict(head=head, per_fold=metrics(sub, by="fold"), per_dist=per_dist, gap=district_gap(per_dist),
                sig=significance(DAILY, model), cov=cov, cov_d=cov_d, cov_fold=metrics(rsub, by="fold"),
                cov_cal=cov_cal, cov_cal_d=cov_cal_d,
                interval={h: interval_source(cfg, model, h) for h in HORDER})


def report(res, verbose=True, save_predictions=True):
    cfg, out = res["cfg"], res["out_dir"]
    ALL, DAILY, WEEKLY = res["all"], res["daily"], res["weekly"]
    tier = res["meta"]["tier"]
    if save_predictions:
        for k in ("daily", "base", "keys"):
            res[k].to_csv(out / f"predictions_{k}.csv", index=False)
        WEEKLY.to_csv(out / "predictions_weekly.csv", index=False)

    ladder = []
    for mname in LADDER:
        sub = headline_subset(ALL[ALL.model == mname])
        if len(sub):
            ladder.append(metrics(sub).assign(model=mname))
    ladder = pd.concat(ladder, ignore_index=True)
    ladder.to_csv(out / "baseline_ladder.csv", index=False)
    S = {"TFT": _summary(DAILY, "TFT", tier, cfg),
         "Ensemble": _summary(DAILY, "Ensemble", tier, cfg)}
    S = {k: v for k, v in S.items() if v is not None}
    for name, sm in S.items():
        pre = "" if name == "TFT" else "ensemble_"
        sm["head"].to_csv(out / f"{pre}metrics_headline.csv")
        sm["per_fold"].to_csv(out / f"{pre}metrics_per_fold.csv", index=False)
        sm["per_dist"].to_csv(out / f"{pre}metrics_per_district.csv", index=False)
        sm["sig"].to_csv(out / f"{pre}significance.csv", index=False)
        # reported intervals (per cfg.interval_choice), and the conformal ones as diagnostics
        sm["cov"].assign(interval=pd.Series(sm["interval"])).to_csv(out / f"{pre}coverage_reported.csv")
        sm["cov_d"].to_csv(out / f"{pre}coverage_reported_by_district.csv", index=False)
        sm["cov_cal"].to_csv(out / f"{pre}coverage_calibrated.csv")
        sm["cov_cal_d"].to_csv(out / f"{pre}coverage_calibrated_by_district.csv", index=False)
        pd.DataFrame(sm["gap"]).T.to_csv(out / f"{pre}district_gap.csv")
    if len(res.get("guarded", [])):
        res["guarded"].to_csv(out / "stack_guard.csv", index=False)
    wk = metrics(headline_subset(WEEKLY)).set_index("Horizon") if len(WEEKLY) else None
    res["wins"].to_csv(out / "windows.csv", index=False)
    if "epochs" in res and len(res["epochs"]):
        res["epochs"].to_csv(out / "best_epochs.csv", index=False)
    stk = [{"fold": f, "district": k[0], "horizon": HNAME.get(k[1], k[1]),
            **dict(zip(["no-change", "seasonal-naive", "TFT", "LightGBM"], np.round(v, 3)))}
           for f, st in res.get("stacks", {}).items() for k, v in st.items()]
    if stk:
        pd.DataFrame(stk).to_csv(out / "stack_weights.csv", index=False)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "config_key": res["ckey"]}, f, indent=2, default=str)

    # one ablation row per run
    row = {"tag": cfg.tag, "mode": cfg.mode, "key": res["ckey"], "minutes": round(res["minutes"], 1),
           "target_mode": cfg.target_mode, "calib_mode": cfg.calib_mode, "combine": cfg.combine,
           "features": "+".join(cfg.feature_groups), "enc": cfg.encoder_len, "d_model": cfg.d_model,
           "dropout": cfg.dropout, "seeds": len(cfg.seeds), "loss_w": cfg.loss_weighting,
           "aci_gamma": cfg.aci_gamma, "stack_prior_n": cfg.stack_prior_n, "refit": cfg.refit,
           "use_refit": cfg.use_refit, "epoch_frac": cfg.epoch_frac, "train_stride": cfg.train_stride,
           "tab_features": cfg.tab_features, "tab_stride": cfg.tab_stride, "tab_fresh": cfg.tab_fresh,
           "lambda_center": cfg.lambda_center, "aci_feedback": cfg.aci_feedback,
           "aci_alpha": f"{cfg.aci_alpha_min}-{cfg.aci_alpha_max}", "stack_guard": cfg.stack_guard,
           "interval_choice": json.dumps(cfg.interval_choice, sort_keys=True),
           "data": Path(cfg.data_path).name, "data_md5": file_md5(cfg.data_path)[:8]}
    for name, sm in S.items():
        pre = "" if name == "TFT" else "ens "
        for h in HORDER:
            if h in sm["head"].index:
                row[f"{pre}rMAE_P {h}"] = sm["head"].loc[h, "rMAE_P"]
                row[f"{pre}rMAE_SN {h}"] = sm["head"].loc[h, "rMAE_SN"]
                row[f"{pre}MAE {h}"] = sm["head"].loc[h, "MAE (m)"]
                row[f"{pre}cov {h}"] = sm["cov"].loc[h, "Cov (%)"] if h in sm["cov"].index else np.nan
                row[f"{pre}gap {h}"] = sm["gap"].get(h, {}).get("gap", np.nan)
                row[f"{pre}median rMAE_P {h}"] = sm["gap"].get(h, {}).get("median", np.nan)
                row[f"{pre}n<1 {h}"] = sm["gap"].get(h, {}).get("n_lt1", np.nan)
    ab_path = Path(cfg.out_root) / "ablations.csv"
    ab = pd.read_csv(ab_path) if ab_path.exists() else pd.DataFrame()
    pd.concat([ab, pd.DataFrame([row])], ignore_index=True).to_csv(ab_path, index=False)

    if verbose:
        pd.set_option("display.width", 220)
        nd = len(set(DAILY.district) - set(HEADLINE_EXCLUDE))
        order = [m for m in LADDER if m in set(ladder.model)]
        print("\n" + "=" * 100)
        print(f"{cfg.tag} — {cfg.mode} folds, {nd} headline districts, daily target")
        print("=" * 100)
        print("BASELINE LADDER — pooled rMAE vs persistence")
        print(ladder.pivot(index="model", columns="Horizon", values="rMAE_P")[HORDER].reindex(order).round(3).to_string())
        print("BASELINE LADDER — pooled rMAE vs seasonal-naive")
        print(ladder.pivot(index="model", columns="Horizon", values="rMAE_SN")[HORDER].reindex(order).round(3).to_string())
        for name, sm in S.items():
            print(f"\n----- {name} -----")
            print(sm["head"][["MAE (m)", "RMSE (m)", "Persist MAE", "SN MAE", "rMAE_P", "rMAE_SN", "Skill %",
                              "MAPE (%)", "NSE", "n"]].to_string())
            print("per fold rMAE_P:")
            print(sm["per_fold"].pivot(index="fold", columns="Horizon", values="rMAE_P")[HORDER].round(3).to_string())
            print("district evenness (per-district rMAE_P):")
            print(pd.DataFrame(sm["gap"]).T.round(3).to_string())
            tiers = sm["per_dist"].groupby(["tier", "Horizon"])["rMAE_P"].agg(
                ["median", "count", lambda s: int((s < 1).sum())])
            tiers.columns = ["median rMAE_P", "n", "n<1"]
            print(tiers.round(3).unstack("Horizon").to_string())
            print("significance (district-cluster bootstrap + DM, Holm-adjusted):")
            print(sm["sig"].to_string(index=False))
            print(f"coverage of the reported intervals {sm['interval']}:")
            print(sm["cov"][["Cov (%)", "Below q10 (%)", "Above q90 (%)", "Width (m)", "Interval score (m)",
                             "districts in 65-90% (%)"]].to_string())
        if wk is not None:
            print("\nWEEKLY-MEAN TARGET (TFT)")
            print(wk[["MAE (m)", "rMAE_P", "rMAE_SN", "n"]].to_string())
        print(f"\nOutputs -> {out}   | ablation row appended -> {ab_path}")
    return S


# ══════════════════════════════════════════════════════════════════════════════
# Simple (v1-style) exports — per-district tables and plots for one model
# ══════════════════════════════════════════════════════════════════════════════
LOCKED_FOLDS = ("fold5_postmonsoon2025", "fold6_dry2026")
SEASON_OF_FOLD = {"fold1_monsoon2024": "monsoon", "fold4_monsoon2025": "monsoon",
                  "fold2_postmonsoon2024": "post-monsoon", "fold5_postmonsoon2025": "post-monsoon",
                  "fold3_dry2025": "dry", "fold6_dry2026": "dry"}
# districts the old (v1) results had that the v4 dataset cannot support
NO_DATA = {
    "Beed": "no usable data: its only well reads a flat dry-well floor most of the year; 66% of days "
            "missing after cleaning, so it is dropped from the master (>= 60% missing)",
    "Latur": "no usable data: tier D (1 well, 41% of days observed) - excluded from training",
}
# chart styling (reference palette: first three categorical slots, ink for text)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
TIER_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#9a9893"}
C_FORECAST, C_BAND, C_NOCHANGE = "#2a78d6", "#b7d3f6", "#eb6834"


def simple_metrics(g):
    """The v1 columns (RMSE, MAE, MAPE with v1's formula, coverage, interval width, n) plus the
    comparisons v1 lacked: MASE (error / no-change error) and error / seasonal-naive error."""
    a, p = g["actual"].values, g["q50"].values
    lo, hi = g["q10"].values, g["q90"].values
    e = np.abs(a - p)
    mae_p = np.mean(np.abs(a - g["persistence"].values))
    mae_s = np.mean(np.abs(a - g["snaive"].values))
    return {
        "RMSE (m)": float(np.sqrt(np.mean((a - p) ** 2))),
        "MAE (m)": float(e.mean()),
        "MAPE (%)": float(np.mean(e / (np.abs(a) + 1e-8)) * 100),          # v1 formula
        "MAPE floored (%)": float(np.mean(e / np.maximum(np.abs(a), 0.5)) * 100),
        "MASE (vs no-change)": float(e.mean() / mae_p) if mae_p > 1e-9 else np.nan,
        "vs seasonal-naive": float(e.mean() / mae_s) if mae_s > 1e-9 else np.nan,
        "Coverage (%)": float(((a >= lo) & (a <= hi)).mean() * 100),
        "Below q10 (%)": float((a < lo).mean() * 100),
        "Above q90 (%)": float((a > hi).mean() * 100),
        "Interval (m)": float((hi - lo).mean()),
        "n": int(len(g)),
    }


def _metric_table(rows, by):
    out = []
    for k, g in rows.groupby(by, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        out.append({**dict(zip(by, k)), **simple_metrics(g)})
    t = pd.DataFrame(out)
    if "Horizon" in t.columns:
        t["_o"] = t["Horizon"].map({h: i for i, h in enumerate(HORDER)})
        t = t.sort_values([c for c in by if c != "Horizon"] + ["_o"]).drop(columns="_o")
    return t.reset_index(drop=True)


def _new_fig(w, h):
    from matplotlib.figure import Figure       # no pyplot: never touches the notebook's backend
    fig = Figure(figsize=(w, h), facecolor="#fcfcfb")
    return fig


def _style(ax, grid_axis="y"):
    ax.set_facecolor("#fcfcfb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _bar_by_district(path, t, col, title, xlabel, ref=None, tier=None):
    t = t.sort_values(col)
    fig = _new_fig(9, 0.32 * len(t) + 1.6)
    ax = fig.add_subplot(111)
    _style(ax, "x")
    y = np.arange(len(t))
    cols = [TIER_COLOR.get(tier.get(d, "D"), TIER_COLOR["D"]) for d in t["district"]]
    ax.barh(y, t[col].values, height=0.7, color=cols, edgecolor="#fcfcfb", linewidth=2)
    if ref is not None:
        ax.axvline(ref, color=INK, lw=1, ls="--")
    for yy, v in zip(y, t[col].values):
        ax.text(v, yy, f" {v:.2f}" if v < 10 else f" {v:.1f}", va="center", fontsize=7, color=INK2)
    lab = [f"{d}{'  (1 well)' if d in HEADLINE_EXCLUDE else ''}" for d in t["district"]]
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    from matplotlib.patches import Patch
    present = [k for k in ("A", "B", "C") if k in {tier.get(d) for d in t["district"]}]
    ax.legend([Patch(color=TIER_COLOR[k]) for k in present], [f"data tier {k}" for k in present],
              loc="upper right", fontsize=8, frameon=False)   # bars are sorted ascending: top right is empty
    missing = [d for d in NO_DATA if d not in set(t["district"])]
    if missing:
        fig.text(0.01, 0.005, "Not shown (no usable data): " + ", ".join(missing), fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def _daily(s, col):
    """Reindex one district/horizon series to daily so data gaps break the line."""
    if not len(s):
        return pd.Series(dtype=float)
    x = s.set_index("target_date")[col]
    x = x[~x.index.duplicated()]
    return x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="D"))


def export_simple(res, model, out_dir, locked=LOCKED_FOLDS):
    """v1-style result files for one model ('Ensemble' or 'TFT'), all modelled districts.
    Intervals are the REPORTED ones (cfg.interval_choice, per horizon: the network's own spreads or
    the conformal '<model> (calibrated)' rows); q50 is the same either way."""
    out_dir = Path(out_dir)
    (out_dir / "quantile").mkdir(parents=True, exist_ok=True)
    pre = model.lower()
    D = res["daily"]
    rows = reported_rows(D, model, res["cfg"]).copy()
    if not len(rows):
        print(f"export_simple: no rows for {model}")
        return None
    raw = D[D.model == model].set_index(["district", "target_date", "horizon"])["q50"]
    chk = rows.set_index(["district", "target_date", "horizon"])["q50"]
    assert np.allclose(raw.reindex(chk.index).values, chk.values), "calibration changed q50"
    rows["Horizon"] = rows["horizon"]
    tier = res["meta"]["tier"]
    dists = sorted(rows.district.unique())

    # predictions
    pred = rows.rename(columns={"target_date": "date", "anchor_date": "issue_date",
                                "persistence": "no_change", "snaive": "seasonal_naive"})
    pred[["district", "date", "horizon", "actual", "q10", "q50", "q90", "fold", "issue_date", "no_change",
          "seasonal_naive"]].to_csv(out_dir / f"{pre}_predictions_all_folds.csv", index=False)

    # headline: 21 headline districts and all modelled districts, all folds and locked folds
    head = []
    for scope, r in (("21 headline districts", headline_subset(rows)), (f"all {len(dists)} modelled", rows)):
        head.append(_metric_table(r, ["Horizon"]).assign(**{"District set": scope}))
    head = pd.concat(head)[["District set"] + [c for c in head[0].columns if c != "District set"]]
    head.to_csv(out_dir / f"{pre}_metrics_headline.csv", index=False)
    lk = rows[rows.fold.isin(locked)]
    if len(lk):
        lh = pd.concat([_metric_table(headline_subset(lk), ["Horizon"]).assign(**{"District set": "21 headline districts"}),
                        _metric_table(lk, ["Horizon"]).assign(**{"District set": f"all {len(dists)} modelled"})])
        lh[["District set"] + [c for c in lh.columns if c != "District set"]].to_csv(
            out_dir / f"{pre}_metrics_headline_locked.csv", index=False)

    # per district (+ no-data rows)
    pdist = _metric_table(rows, ["district", "Horizon"])
    pdist.insert(1, "tier", pdist["district"].map(tier))
    pdist.insert(2, "in_headline", ~pdist["district"].isin(HEADLINE_EXCLUDE))
    pdist["note"] = np.where(pdist["district"].isin(HEADLINE_EXCLUDE), "single well - reported separately", "")
    nd = [{"district": d, "tier": "-", "in_headline": False, "Horizon": h, "n": 0, "note": note}
          for d, note in NO_DATA.items() if d not in dists for h in HORDER]
    pdist = pd.concat([pdist, pd.DataFrame(nd)], ignore_index=True)
    pdist.to_csv(out_dir / f"{pre}_metrics_per_district.csv", index=False)

    # per season: each fold, and folds grouped by season (headline districts)
    hs = headline_subset(rows).copy()
    per_fold = _metric_table(hs, ["fold", "Horizon"]).assign(level="fold")
    hs["season"] = hs["fold"].map(SEASON_OF_FOLD)
    per_season = _metric_table(hs, ["season", "Horizon"]).assign(level="season").rename(columns={"season": "fold"})
    pd.concat([per_fold, per_season]).rename(columns={"fold": "season"}).to_csv(
        out_dir / f"{pre}_metrics_per_season.csv", index=False)

    # pivots
    md = pdist[pdist["n"] > 0]
    for col, fn in (("MASE (vs no-change)", "mase"), ("MAPE (%)", "mape"), ("Coverage (%)", "coverage")):
        pv = md.pivot(index="district", columns="Horizon", values=col)[HORDER].round(3)
        pv.insert(0, "tier", pv.index.map(tier))
        for d in NO_DATA:
            if d not in pv.index:
                pv.loc[d] = ["-"] + [np.nan] * len(HORDER)
        pv.to_csv(out_dir / f"{pre}_{fn}_by_district.csv")

    # bar charts per horizon
    for h in HORDER:
        t = md[md.Horizon == h]
        hn = h.replace("+", "")
        _bar_by_district(out_dir / f"{pre}_per_district_mape_{hn}.png", t, "MAPE (%)",
                         f"{model} - MAPE per district @ {h} (all test folds)", "MAPE (%)", tier=tier)
        _bar_by_district(out_dir / f"{pre}_per_district_mase_{hn}.png", t, "MASE (vs no-change)",
                         f"{model} - MASE per district @ {h}  (< 1 = better than 'no change')",
                         "error / no-change error", ref=1.0, tier=tier)

    # coverage heatmap (diverging around the 80% target: red = too narrow, blue = too wide)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("cov", ["#d03b3b", "#f0efec", "#2a78d6"])
    cv = md.pivot(index="district", columns="Horizon", values="Coverage (%)")[HORDER]
    names = list(cv.index) + [d for d in NO_DATA if d not in cv.index]
    vals = np.vstack([cv.values, np.full((len(names) - len(cv), 3), np.nan)])
    fig = _new_fig(5.2, 0.3 * len(names) + 1.6)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=TwoSlopeNorm(vmin=40, vcenter=80, vmax=100),
                   aspect="auto")
    for i in range(len(names)):
        for j in range(3):
            txt = f"{vals[i, j]:.0f}" if np.isfinite(vals[i, j]) else "no data"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(HORDER, fontsize=8, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8, color=INK)
    ax.set_title(f"{model} - 80% interval coverage (%) per district\ntarget 80; red = too narrow, blue = too wide",
                 fontsize=9, color=INK, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.savefig(out_dir / f"{pre}_coverage_heatmap.png", dpi=130)

    # forecast grids and one quantile plot per district
    fold_starts = [pd.Timestamp(f["test_start"]) for f in FOLDS if f["name"] in set(rows.fold)]
    stats = pdist.set_index(["district", "Horizon"])
    for h in HORDER:
        ncol = 6
        nrow = int(np.ceil(len(dists) / ncol))
        fig = _new_fig(4.0 * ncol, 2.6 * nrow + 0.6)
        for k, d in enumerate(dists):
            ax = fig.add_subplot(nrow, ncol, k + 1)
            _style(ax)
            s = rows[(rows.district == d) & (rows.horizon == h)].sort_values("target_date")
            ax.fill_between(_daily(s, "q10").index, _daily(s, "q10").values, _daily(s, "q90").values,
                            color=C_BAND, lw=0)
            ax.plot(_daily(s, "q50"), color=C_FORECAST, lw=1.2)
            ax.plot(_daily(s, "actual"), color=INK, lw=0.9)
            st = stats.loc[(d, h)]
            ax.set_title(f"{d} ({tier.get(d)})  MASE {st['MASE (vs no-change)']:.2f}  cov {st['Coverage (%)']:.0f}%",
                         fontsize=8, color=INK, loc="left")
            ax.tick_params(axis="x", labelrotation=30, labelsize=6)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        fig.legend([Line2D([], [], color=INK), Line2D([], [], color=C_FORECAST), Patch(color=C_BAND)],
                   ["actual", f"{model} forecast (q50)", "80% interval"], loc="upper right", ncol=3,
                   fontsize=9, frameon=False)
        fig.suptitle(f"{model} - {h} ahead, every district, all test folds", x=0.01, ha="left", fontsize=12,
                     color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / f"{pre}_forecast_grid_{h.replace('+', '')}.png", dpi=110)

    for d in dists:
        fig = _new_fig(13, 8.5)
        for k, h in enumerate(HORDER):
            ax = fig.add_subplot(3, 1, k + 1)
            _style(ax)
            s = rows[(rows.district == d) & (rows.horizon == h)].sort_values("target_date")
            ax.fill_between(_daily(s, "q10").index, _daily(s, "q10").values, _daily(s, "q90").values,
                            color=C_BAND, lw=0, label="80% interval")
            ax.plot(_daily(s, "q50"), color=C_FORECAST, lw=1.4, label=f"{model} forecast (q50)")
            ax.plot(_daily(s, "persistence"), color=C_NOCHANGE, lw=1, ls=":", label="no-change")
            ax.plot(_daily(s, "actual"), color=INK, lw=1.1, label="actual")
            for fs in fold_starts:
                ax.axvline(fs, color=GRID, lw=1.2, zorder=0)
            st = stats.loc[(d, h)]
            ax.set_title(f"{h}:  MAE {st['MAE (m)']:.2f} m   MAPE {st['MAPE (%)']:.1f}%   MASE "
                         f"{st['MASE (vs no-change)']:.2f}   vs seasonal {st['vs seasonal-naive']:.2f}   "
                         f"coverage {st['Coverage (%)']:.0f}%", fontsize=9, color=INK, loc="left")
            ax.set_ylabel("GWL (m)", fontsize=8, color=INK2)
            if k == 0:
                ax.legend(fontsize=8, ncol=4, loc="upper left", frameon=False)
        fig.suptitle(f"{d} (data tier {tier.get(d)}{', single well' if d in HEADLINE_EXCLUDE else ''}) - "
                     f"{model}, all test folds (grey lines = fold starts)", x=0.01, ha="left", fontsize=11,
                     color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / "quantile" / f"quantile_{d.lower().replace(' ', '_')}.png", dpi=110)
    print(f"export_simple({model}): {len(dists)} districts -> {out_dir}")
    return pdist


def vsn_importance(res, fold_name=None, max_batches=40):
    """Variable-selection weights of the saved last-seed network of one fold (default: the last fold
    run, i.e. fold 6 in a full run) on that fold's test set."""
    cfg = res["cfg"]
    fold_name = fold_name or list(res["model_states"])[-1]
    device = get_device()
    df, meta = load_data(cfg, verbose=False)
    fold = next(f for f in FOLDS if f["name"] == fold_name)
    I = fold_inputs(cfg, df, meta, fold)
    fd = I["fd"]
    ds = I["mk"](fd["test_start"], fd["test_end"], 5, I["sn_test"], gate=False)
    model = build_model(cfg, meta, device)
    model.load_state_dict(res["model_states"][fold_name])
    model.eval()
    acc = {"past": 0, "future": 0, "static": 0}
    nb = 0
    with torch.no_grad():
        for i, b in enumerate(DataLoader(ds, batch_size=cfg.eval_batch, shuffle=False)):
            if i >= max_batches:
                break
            sc, sx, px, fx = [t.to(device) for t in b[:4]]
            _, vsn, _ = model(sc, sx, px, fx)
            for k in acc:
                w = vsn[k].detach().cpu().numpy()
                acc[k] = acc[k] + w.reshape(-1, w.shape[-1]).mean(0)
            nb += 1
    labels = {"past": meta["past"], "future": meta["future"],
              "static": ["district + aquifer (embedding)"] + meta["static_cont"]}
    return {k: pd.Series(acc[k] / max(nb, 1), index=labels[k][:len(acc[k])]).sort_values(ascending=False)
            for k in acc}


def export_shared(res, out_dir, summaries=None):
    """Plots and tables shared by both models: training curves, VSN importance, TFT vs Ensemble summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # training curves
    names = [f["name"] for f in FOLDS if f["name"] in res["curves"]]
    fig = _new_fig(15, 7)
    for k, nm in enumerate(names):
        ax = fig.add_subplot(2, 3, k + 1)
        _style(ax)
        for i, (tr, va) in enumerate(res["curves"][nm]):
            ax.plot(range(1, len(tr) + 1), tr, color=C_FORECAST, lw=0.9, alpha=0.8, label="train" if i == 0 else None)
            ax.plot(range(1, len(va) + 1), va, color=C_NOCHANGE, lw=0.9, alpha=0.8,
                    label="early-stopping block" if i == 0 else None)
            b = int(np.argmin(va))
            ax.plot([b + 1], [va[b]], "o", ms=4, color=C_NOCHANGE)
        for i, tr2 in enumerate(res.get("curves_p2", {}).get(nm, [])):
            ax.plot(range(1, len(tr2) + 1), tr2, color=C_FORECAST, lw=0.9, ls=":",
                    label="refit (train + ES block)" if i == 0 else None)
        ax.set_title(nm, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("epoch", fontsize=8, color=INK2)
        if k == 0:
            ax.set_ylabel("masked pinball loss", fontsize=8, color=INK2)
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("TFT training curves (one line per seed; dots = best epoch)", x=0.01, ha="left", fontsize=11,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "tft_training_curves.png", dpi=120)

    # VSN importance
    try:
        imp = vsn_importance(res)
        fig = _new_fig(18, 9)
        for k, (grp, title) in enumerate((("past", "Past inputs"), ("future", "Known-future inputs"),
                                          ("static", "Static inputs"))):
            ax = fig.add_subplot(1, 3, k + 1)
            _style(ax, "x")
            sr = imp[grp].head(30)
            ax.barh(range(len(sr)), sr.values, color=C_FORECAST, height=0.7)
            ax.set_yticks(range(len(sr)))
            ax.set_yticklabels(sr.index, fontsize=7, color=INK)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=10, color=INK, loc="left")
            ax.set_xlabel("mean selection weight", fontsize=8, color=INK2)
        fig.suptitle("TFT variable-selection weights (last fold's network, its test windows) - what the model attends "
                     "to, not causal importance", x=0.01, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / "tft_vsn_importance.png", dpi=120)
        pd.concat({k: v for k, v in imp.items()}).rename("weight").to_csv(out_dir / "tft_vsn_importance.csv")
    except Exception as ex:                               # never let a plot stop the export
        print("VSN importance skipped:", ex)

    # TFT vs Ensemble summary (headline districts)
    D = res["daily"]
    rows = []
    for model in ("TFT", "Ensemble"):
        r = reported_rows(D, model, res["cfg"])
        if not len(r):
            continue
        for scope, rr in (("all folds", r), ("locked folds 5-6", r[r.fold.isin(LOCKED_FOLDS)])):
            for h in HORDER:
                g = headline_subset(rr[rr.horizon == h])
                if len(g):
                    rows.append({"model": model, "test set": scope, "Horizon": h,
                                 "interval": interval_source(res["cfg"], model, h), **simple_metrics(g)})
    pd.DataFrame(rows).to_csv(out_dir / "summary_tft_vs_ensemble.csv", index=False)
    print(f"export_shared -> {out_dir}")


def compare_v1(v1_csv, simple_dir, label="v4.2"):
    """Old v1 per-district metrics beside this run (TFT and Ensemble). NOT like-for-like - see caveat column."""
    simple_dir = Path(simple_dir)
    v1 = pd.read_csv(v1_csv)
    keep = ["district", "Horizon", "RMSE (m)", "MAE (m)", "MAPE (%)", "Coverage (%)", "n"]
    out = v1[[c for c in keep if c in v1.columns]].rename(columns={c: f"v1 {c}" for c in keep[2:]})
    for model in ("tft", "ensemble"):
        f = simple_dir / model / f"{model}_metrics_per_district.csv"
        if f.exists():
            m = pd.read_csv(f)[keep + ["MASE (vs no-change)"]]
            m = m.rename(columns={c: f"{label} {model} {c}" for c in keep[2:] + ["MASE (vs no-change)"]})
            out = out.merge(m, on=["district", "Horizon"], how="outer")
    out["caveat"] = ("v1 = old 11.6M-parameter model on the OLD groundwater series (before the target rebuild), "
                     "absolute-level target, different scoring windows; v1 has no no-change comparison. "
                     "Directional comparison only, not like-for-like.")
    out.to_csv(simple_dir / f"comparison_v1_vs_{label.replace('.', '_')}_per_district.csv", index=False)
    return out


def results_to_quote(S_locked, S_all):
    """One table with the numbers to quote, for both models and both test sets."""
    rows = []
    for label, S in (("locked folds 5-6", S_locked), ("all 6 folds", S_all)):
        for name, sm in S.items():
            sig = sm["sig"].set_index(["horizon", "vs"])
            for h in HORDER:
                hd, cv, gp = sm["head"].loc[h], sm["cov"].loc[h], sm["gap"].get(h, {})
                rows.append({
                    "test set": label, "model": name, "horizon": h,
                    "MAE (m)": hd["MAE (m)"], "RMSE (m)": hd["RMSE (m)"], "MAPE (%)": hd["MAPE (%)"],
                    "rMAE vs persistence": hd["rMAE_P"], "rMAE vs seasonal": hd["rMAE_SN"],
                    "p vs persistence": sig.loc[(h, "persistence"), "p_boot_holm"],
                    "p vs seasonal": sig.loc[(h, "snaive"), "p_boot_holm"],
                    "interval": sm.get("interval", {}).get(h, "calibrated"),
                    "coverage 80% (%)": cv["Cov (%)"], "below q10 (%)": cv["Below q10 (%)"],
                    "above q90 (%)": cv["Above q90 (%)"], "width (m)": cv["Width (m)"],
                    "interval score (m)": cv["Interval score (m)"],
                    "districts beating persistence": f"{gp.get('n_lt1', np.nan):.0f}/{gp.get('n', np.nan):.0f}",
                    "worst5 - best5 gap": gp.get("gap", np.nan)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# v4.2 selection rules (fixed before the run; applied to the development folds only)
# ══════════════════════════════════════════════════════════════════════════════
DEV_FOLDS = tuple(f["name"] for f in RUN_FOLDS["dev"])


def _fold_filter(D, folds):
    return D if folds is None else D[D.fold.isin(folds)]


def guard_decision(res_off, res_on, folds=DEV_FOLDS, max_rise=0.02):
    """Rule G: adopt stack_guard if the mean Ensemble rMAE vs seasonal-naive (headline districts,
    t+7/t+14/t+30) does not get worse AND no headline district's Ensemble MASE (vs no-change) rises
    by more than `max_rise` at any horizon. Returns (adopt, mean_off, mean_on, per-district table)."""
    def per(res):
        sub = headline_subset(_fold_filter(res["daily"], folds).query("model == 'Ensemble'"))
        return (float(metrics(sub)["rMAE_SN"].mean()),
                metrics(sub, by="district").set_index(["district", "Horizon"])["rMAE_P"])
    m_off, d_off = per(res_off)
    m_on, d_on = per(res_on)
    tab = pd.DataFrame({"MASE guard off": d_off, "MASE guard on": d_on})
    tab["change"] = (tab["MASE guard on"] - tab["MASE guard off"]).round(4)
    adopt = bool(m_on <= m_off and tab["change"].max() <= max_rise)
    return adopt, m_off, m_on, tab.reset_index()


def interval_table(res, folds=DEV_FOLDS):
    """Per model x horizon x interval ('raw' = the network's own spreads, 'calibrated' = conformal),
    headline districts, given folds: coverage, tails, width and interval score."""
    D = _fold_filter(res["daily"], folds)
    cols = ["Cov (%)", "Below q10 (%)", "Above q90 (%)", "Width (m)", "Interval score (m)", "n"]
    out = []
    for model in ("TFT", "Ensemble"):
        for src, name in (("raw", model), ("calibrated", f"{model} (calibrated)")):
            sub = headline_subset(D[D.model == name])
            if len(sub):
                m = metrics(sub)
                out.append(m[["Horizon"] + cols].rename(columns={"Horizon": "horizon"})
                           .assign(model=model, interval=src))
    t = pd.concat(out, ignore_index=True)
    return t[["model", "horizon", "interval"] + cols]


def calibration_score(table):
    """Rule C score of one conformal variant: mean over model x horizon of
    (calibrated interval score / raw interval score). Below 1 = calibration helps."""
    t = table.pivot_table(index=["model", "horizon"], columns="interval", values="Interval score (m)")
    return float((t["calibrated"] / t["raw"]).mean())


def pick_interval_choice(table, lo=75.0, hi=85.0, tie=0.01):
    """Rule D, per model x horizon: if exactly one of raw / calibrated has coverage in [lo, hi], take it;
    otherwise take the lower interval score, preferring raw unless calibrated is lower by more than `tie`
    (relative). Returns (interval_choice dict for Config, table of the decisions)."""
    choice, why = {}, []
    for (model, h), g in table.groupby(["model", "horizon"], sort=False):
        g = g.set_index("interval")
        r, c = g.loc["raw"], g.loc["calibrated"]
        in_r, in_c = lo <= r["Cov (%)"] <= hi, lo <= c["Cov (%)"] <= hi
        if in_r != in_c:
            pick, reason = ("raw" if in_r else "calibrated"), f"only one in {lo:.0f}-{hi:.0f}% coverage"
        else:
            better = c["Interval score (m)"] < r["Interval score (m)"] * (1 - tie)
            pick = "calibrated" if better else "raw"
            reason = ("both" if in_r else "neither") + f" in {lo:.0f}-{hi:.0f}%: lower interval score" + \
                     ("" if better else " (or within 1%: simpler)")
        choice.setdefault(model, {})[h] = pick
        why.append({"model": model, "horizon": h, "raw cov": r["Cov (%)"], "cal cov": c["Cov (%)"],
                    "raw score": r["Interval score (m)"], "cal score": c["Interval score (m)"],
                    "pick": pick, "reason": reason})
    return choice, pd.DataFrame(why)
