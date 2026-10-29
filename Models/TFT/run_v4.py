"""
Command-line runner for the v4 pipeline (local CPU or Colab GPU).

Examples
--------
  # quick smoke test: one fold, one seed, 3 epochs, every v4.1 path, simple exports
  python run_v4.py --tag smoke --mode dev --folds fold1_monsoon2024 --seeds 42 --max-epochs 3 \
      --data master_dataset_v4target.csv --target-mode snaive --calib-mode v4 --combine stack \
      --tab-features v2 --tab-stride 1 --tab-fresh --refit --export-simple

  # the v4.0 reference set-up on the dev folds
  python run_v4.py --tag snaive --mode dev --target-mode snaive --calib-mode v4 --combine stack

Every run writes outputs/v4/<tag>/ and appends one row to outputs/v4/ablations.csv.
Networks, tabular models and post-processing are cached separately (see tft_v4.run), so changing
only post-processing or only the tabular settings never retrains the networks.
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tft_v4 as v4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="dev", choices=["dev", "final", "all"])
    ap.add_argument("--data", default=None, help="master_dataset csv path")
    ap.add_argument("--folds", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--target-mode", choices=["persistence", "snaive"], default=None)
    ap.add_argument("--calib-mode", choices=["v3", "v4"], default=None)
    ap.add_argument("--combine", choices=["lambda", "stack", "none"], default=None)
    ap.add_argument("--features", nargs="*", default=None, help="feature groups: gwl rain met era5 sentinel")
    ap.add_argument("--encoder-len", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--loss-weighting", choices=["none", "eval_steps", "frac_real", "both"], default=None)
    ap.add_argument("--sn-bandwidth", type=float, default=None)
    ap.add_argument("--sn-prior-k", type=float, default=None)
    ap.add_argument("--sn-damping", type=float, default=None)
    ap.add_argument("--aci-gamma", type=float, default=None)
    ap.add_argument("--stack-prior-n", type=float, default=None)
    # v4.1
    ap.add_argument("--lambda-center", choices=["auto", "zero", "sn"], default=None)
    ap.add_argument("--refit", action="store_true", help="pass-2 retrain on train + early-stopping block")
    ap.add_argument("--no-use-refit", action="store_true", help="score pass 1 even if pass 2 was trained")
    ap.add_argument("--train-stride", type=int, default=None)
    ap.add_argument("--epoch-frac", type=float, default=None)
    ap.add_argument("--tab-features", choices=["v1", "v2"], default=None)
    ap.add_argument("--tab-stride", type=int, default=None)
    ap.add_argument("--tab-fresh", action="store_true")
    ap.add_argument("--gbm-n-estimators", type=int, default=None)
    ap.add_argument("--gbm-lr", type=float, default=None)
    ap.add_argument("--gbm-min-child", type=int, default=None)
    ap.add_argument("--export-simple", action="store_true", help="write the v1-style simple outputs")
    ap.add_argument("--out-root", default=None, help="default Models/TFT/outputs/v4")
    ap.add_argument("--no-tabular", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cfg = v4.Config(tag=a.tag, mode=a.mode, only_folds=a.folds)
    overrides = {
        "out_root": a.out_root, "aci_gamma": a.aci_gamma, "stack_prior_n": a.stack_prior_n,
        "data_path": a.data, "seeds": a.seeds, "max_epochs": a.max_epochs, "target_mode": a.target_mode,
        "calib_mode": a.calib_mode, "combine": a.combine, "feature_groups": a.features,
        "encoder_len": a.encoder_len, "d_model": a.d_model, "dropout": a.dropout,
        "loss_weighting": a.loss_weighting, "sn_bandwidth": a.sn_bandwidth,
        "sn_prior_k": a.sn_prior_k, "sn_damping": a.sn_damping,
        "lambda_center": a.lambda_center, "train_stride": a.train_stride, "epoch_frac": a.epoch_frac,
        "tab_features": a.tab_features, "tab_stride": a.tab_stride, "gbm_n_estimators": a.gbm_n_estimators,
        "gbm_lr": a.gbm_lr, "gbm_min_child": a.gbm_min_child,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if a.refit:
        cfg.refit = True
    if a.no_use_refit:
        cfg.use_refit = False
    if a.tab_fresh:
        cfg.tab_fresh = True
    if a.no_tabular:
        cfg.with_tabular = False

    res = v4.run(cfg, verbose=not a.quiet)
    v4.report(res, verbose=True)
    if a.export_simple:
        simple = res["out_dir"] / "simple"
        for model in ("Ensemble", "TFT"):
            v4.export_simple(res, model, simple / model.lower())
        v4.export_shared(res, simple)
    print(f"\nTotal: {res['minutes']:.1f} min")


if __name__ == "__main__":
    main()
