"""
Phase 2.2 prerequisite: tune the seasonal-naive (SN) baseline on its OWN dev-fold
skill, before any network sees it (so the TFT residual target is not tuned on TFT results).

SN forecast at anchor t for horizon h:  y(t) + damping * kernel-mean of past h-day changes
around day-of-year(t)  (circular Gaussian, bandwidth bw days, shrunk to 0 by pseudo-count k).
Fitted on data before each dev fold's test period; scored on observed (t, t+h) pairs whose
target lies in the test period; headline districts only.

    python tune_snaive.py [--data ../../data/master/master_dataset_v4target.csv]
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import tft_v4 as v4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def evaluate(df, cfg, folds):
    rows = []
    for f in folds:
        fd = v4.fold_dates(f, cfg)
        tabs = v4.fit_snaive_table(df, fd["train_end"], cfg)
        for dist, g in df.groupby("district"):
            if dist in v4.HEADLINE_EXCLUDE:
                continue
            g = g.sort_values("date")
            y, dates, doy = g[v4.TARGET].values, g["date"].values, g["day_of_year"].values
            for h in (7, 14, 30):
                tgt_in = (dates[h:] >= np.datetime64(fd["test_start"])) & (dates[h:] <= np.datetime64(fd["test_end"]))
                ok = tgt_in & np.isfinite(y[h:]) & np.isfinite(y[:-h])
                if not ok.any():
                    continue
                a = y[:-h][ok]
                sn = a + tabs[dist][doy[:-h][ok] - 1, h - 1]
                act = y[h:][ok]
                rows.append({"fold": f["name"], "h": h, "err_sn": np.abs(act - sn).sum(),
                             "err_p": np.abs(act - a).sum(), "n": int(ok.sum())})
    r = pd.DataFrame(rows).groupby("h")[["err_sn", "err_p"]].sum()
    return (r["err_sn"] / r["err_p"]).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(v4.DEFAULT_DATA))
    a = ap.parse_args()
    cfg = v4.Config(data_path=a.data)
    df, _ = v4.load_data(cfg, verbose=False)
    folds = v4.RUN_FOLDS["dev"]
    res = []
    for bw, k, damp in itertools.product([7, 15, 30, 45], [0, 20, 50, 100], [0.5, 0.75, 1.0]):
        cfg.sn_bandwidth, cfg.sn_prior_k, cfg.sn_damping = bw, k, damp
        r = evaluate(df, cfg, folds)
        res.append({"bw": bw, "k": k, "damping": damp, **{f"rMAE_P h{h}": round(v, 4) for h, v in r.items()},
                    "mean": round(np.mean(list(r.values())), 4)})
        print(res[-1], flush=True)
    R = pd.DataFrame(res).sort_values("mean")
    print("\nBest 10 (pooled rMAE vs persistence, dev folds 1-4, headline districts):")
    print(R.head(10).to_string(index=False))
    out = Path(v4.DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "snaive_tuning.csv", index=False)


if __name__ == "__main__":
    main()
