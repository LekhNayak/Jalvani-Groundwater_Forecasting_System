# JalVaani v4.2 run summary

- pipeline 4.6, data master_dataset_v4target.csv (md5 e035dc26)
- networks / LightGBM: the v4.1 final set-up {'target_mode': 'snaive', 'calib_mode': 'v4', 'combine': 'stack', 'train_stride': 1, 'epoch_frac': 0.14285714285714285, 'use_refit': False, 'tab_features': 'v2', 'tab_stride': 1, 'tab_fresh': True, 'gbm_n_estimators': 400, 'gbm_lr': 0.03, 'gbm_min_child': 100, 'stack_prior_n': 500, 'aci_gamma': 0.01}
- seeds: [42, 1337, 2026, 7, 99, 123, 555, 777, 2024, 31337]
- total run time: 31 min
- all three rules used dev folds 1-4 only; folds 5-6 are the locked test

## Rule G (Pune guard): adopted = False
dev mean Ensemble vs seasonal: off 0.9501, on 0.9686; largest district rise +0.126

## Rule C (conformal variant): V4_static = {'aci_feedback': 'cell', 'aci_gamma': 0.0, 'aci_alpha_min': 0.01, 'aci_alpha_max': 0.3}
```
          variant aci_feedback  aci_gamma  aci_alpha_min  aci_alpha_max  score (cal / raw)  TFT cov t+7  TFT cov t+14  TFT cov t+30  Ensemble cov t+7  Ensemble cov t+14  Ensemble cov t+30
          V0_v4.1         cell       0.01           0.01            0.3              1.131       72.436        67.893        63.624            79.338             77.648             77.237
        V1_anchor       anchor       0.01           0.01            0.3              1.115       76.343        76.686        72.684            79.338             77.648             77.237
    V2_anchor_sym       anchor       0.01           0.00            0.2              1.129       76.861        77.264        73.229            79.608             78.327             77.600
V3_anchor_sym_g03       anchor       0.03           0.00            0.2              1.166       76.523        75.928        71.424            79.406             75.611             74.625
        V4_static         cell       0.00           0.01            0.3              1.097       71.906        73.506        70.902            76.151             76.154             76.124
```

## Rule D (reported interval per model and horizon)
```
   model horizon  raw cov  cal cov  raw score  cal score       pick                                                          reason
     TFT     t+7   79.608   71.906      2.818      2.929        raw                                     only one in 75-85% coverage
     TFT    t+14   76.584   73.506      3.978      4.537        raw                                     only one in 75-85% coverage
     TFT    t+30   70.879   70.902      5.717      6.838        raw neither in 75-85%: lower interval score (or within 1%: simpler)
Ensemble     t+7   80.498   76.151      2.769      2.767        raw    both in 75-85%: lower interval score (or within 1%: simpler)
Ensemble    t+14   78.406   76.154      3.831      4.044        raw    both in 75-85%: lower interval score (or within 1%: simpler)
Ensemble    t+30   73.706   76.124      5.441      6.260 calibrated                                     only one in 75-85% coverage
```

## v4.1 -> v4.2
```
        test set    model horizon  v4.1 MAE (m)  v4.1 vs no-change  v4.1 vs seasonal  v4.1 coverage  v4.1 interval score  same-net v4.1-post MAE (m)  same-net v4.1-post vs no-change  same-net v4.1-post vs seasonal  same-net v4.1-post coverage  same-net v4.1-post interval score same-net v4.1-post interval  v4.2 MAE (m)  v4.2 vs no-change  v4.2 vs seasonal  v4.2 coverage  v4.2 interval score v4.2 interval  v4.2 - same-net: vs no-change  v4.2 - same-net: interval score
locked folds 5-6 Ensemble     t+7         0.314              0.839             0.938           81.1                1.676                       0.314                            0.839                           0.938                       81.087                              1.676                  calibrated         0.314              0.839             0.938         81.628                1.731           raw                            0.0                            0.055
locked folds 5-6 Ensemble    t+14         0.406              0.763             0.922           80.1                2.057                       0.406                            0.763                           0.922                       80.138                              2.057                  calibrated         0.406              0.763             0.922         80.557                2.122           raw                            0.0                            0.065
locked folds 5-6 Ensemble    t+30         0.550              0.616             0.877           80.0                3.062                       0.550                            0.616                           0.877                       79.964                              3.062                  calibrated         0.550              0.616             0.877         79.053                3.109    calibrated                            0.0                            0.047
locked folds 5-6      TFT     t+7         0.326              0.872             0.976           74.2                2.010                       0.326                            0.872                           0.976                       74.152                              2.010                  calibrated         0.326              0.872             0.976         80.054                1.778           raw                            0.0                           -0.232
locked folds 5-6      TFT    t+14         0.428              0.803             0.971           70.0                2.490                       0.428                            0.803                           0.971                       70.010                              2.490                  calibrated         0.428              0.803             0.971         78.462                2.209           raw                            0.0                           -0.281
locked folds 5-6      TFT    t+30         0.602              0.675             0.960           65.7                3.557                       0.602                            0.675                           0.960                       65.710                              3.557                  calibrated         0.602              0.675             0.960         77.254                2.888           raw                            0.0                           -0.669
     all 6 folds Ensemble     t+7         0.429              0.903             0.967           79.9                2.403                       0.429                            0.903                           0.967                       79.887                              2.403                  calibrated         0.429              0.903             0.967         80.853                2.443           raw                            0.0                            0.040
     all 6 folds Ensemble    t+14         0.595              0.830             0.949           78.4                3.606                       0.595                            0.830                           0.949                       78.432                              3.606                  calibrated         0.595              0.830             0.949         79.083                3.293           raw                            0.0                           -0.312
     all 6 folds Ensemble    t+30         0.860              0.712             0.909           78.1                5.387                       0.860                            0.712                           0.909                       78.144                              5.387                  calibrated         0.860              0.712             0.909         77.098                5.211    calibrated                            0.0                           -0.176
     all 6 folds      TFT     t+7         0.444              0.935             1.000           73.0                2.740                       0.444                            0.935                           1.000                       72.974                              2.740                  calibrated         0.444              0.935             1.000         79.748                2.491           raw                            0.0                           -0.249
     all 6 folds      TFT    t+14         0.629              0.876             1.002           68.6                3.984                       0.629                            0.876                           1.002                       68.559                              3.984                  calibrated         0.629              0.876             1.002         77.175                3.421           raw                            0.0                           -0.563
     all 6 folds      TFT    t+30         0.924              0.764             0.976           64.3                5.767                       0.924                            0.764                           0.976                       64.318                              5.767                  calibrated         0.924              0.764             0.976         73.000                4.775           raw                            0.0                           -0.991
```

## Pune (Ensemble, all folds)
```
               run Horizon  rMAE_P  rMAE_SN  MAE (m)
same-net v4.1-post     t+7   1.000    1.022    0.213
same-net v4.1-post    t+14   1.050    1.133    0.334
same-net v4.1-post    t+30   1.128    1.301    0.572
              v4.2     t+7   1.000    1.022    0.213
              v4.2    t+14   1.050    1.133    0.334
              v4.2    t+30   1.128    1.301    0.572
```