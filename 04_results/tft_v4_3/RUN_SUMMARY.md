# JalVaani v4.3 run summary

- pipeline 4.7, data master_dataset_v4target.csv (md5 e035dc26)
- networks / LightGBM: the v4.1 final set-up {'target_mode': 'snaive', 'calib_mode': 'v4', 'combine': 'stack', 'train_stride': 1, 'epoch_frac': 0.14285714285714285, 'use_refit': False, 'tab_features': 'v2', 'tab_stride': 1, 'tab_fresh': True, 'gbm_n_estimators': 400, 'gbm_lr': 0.03, 'gbm_min_child': 100, 'stack_prior_n': 500}
- seeds: [42, 1337, 2026, 7, 99, 123, 555, 777, 2024, 31337]
- total run time: 39 min
- rules G2 and D2 used dev folds 1-4 only; the width cap (3) was fixed in advance
- v4.3 was designed after seeing v4.2 results incl. folds 5-6: its locked numbers are a refinement check

## Rule G2 (LightGBM-only guard): adopted = True
dev mean Ensemble vs seasonal: off 0.9501, on 0.9486; largest district rise +0.000

## Rule D2 (reported interval per model and horizon)
```
   model horizon  raw cov  raw score  raw districts 65-90  calibrated cov  calibrated score  calibrated districts 65-90       pick                                                            reason
     TFT     t+7   79.608      2.818               80.952          71.906             2.892                      90.476        raw                                   the only one in 75-85% coverage
     TFT    t+14   76.584      3.978               66.667          73.370             4.415                      90.476        raw                                   the only one in 75-85% coverage
     TFT    t+30   70.879      5.717               76.190          70.811             6.559                      80.952        raw                           neither in 75-85%: lower interval score
Ensemble     t+7   80.486      2.769               80.952          76.208             2.766                      95.238 calibrated               both in 75-85% coverage; more even across districts
Ensemble    t+14   78.633      3.822               76.190          76.381             4.035                      85.714        raw both in 75-85% coverage; the other's interval score is > 5% worse
Ensemble    t+30   73.728      5.440               76.190          76.453             5.987                      85.714 calibrated                                   the only one in 75-85% coverage
```

## v4.2 -> v4.3
```
        test set    model horizon  v4.2 MAE (m)  v4.2 vs no-change  v4.2 vs seasonal  v4.2 coverage  v4.2 interval score  same-net v4.2-post MAE (m)  same-net v4.2-post vs no-change  same-net v4.2-post vs seasonal  same-net v4.2-post coverage  same-net v4.2-post interval score same-net v4.2-post interval  v4.3 MAE (m)  v4.3 vs no-change  v4.3 vs seasonal  v4.3 coverage  v4.3 interval score v4.3 interval  v4.3 - same-net: vs no-change  v4.3 - same-net: interval score
locked folds 5-6      TFT     t+7         0.326              0.872             0.976         80.054                1.778                       0.326                            0.872                           0.976                       80.054                              1.778                         raw         0.326              0.872             0.976         80.054                1.778           raw                          0.000                            0.000
locked folds 5-6      TFT    t+14         0.428              0.803             0.971         78.462                2.209                       0.428                            0.803                           0.971                       78.462                              2.209                         raw         0.428              0.803             0.971         78.462                2.209           raw                          0.000                            0.000
locked folds 5-6      TFT    t+30         0.602              0.675             0.960         77.254                2.888                       0.602                            0.675                           0.960                       77.254                              2.888                         raw         0.602              0.675             0.960         77.254                2.888           raw                          0.000                            0.000
locked folds 5-6 Ensemble     t+7         0.314              0.839             0.938         81.628                1.731                       0.314                            0.839                           0.938                       81.628                              1.731                         raw         0.314              0.841             0.940         80.079                1.681    calibrated                          0.001                           -0.050
locked folds 5-6 Ensemble    t+14         0.406              0.763             0.922         80.557                2.122                       0.406                            0.763                           0.922                       80.557                              2.122                         raw         0.408              0.766             0.926         80.310                2.123           raw                          0.003                            0.001
locked folds 5-6 Ensemble    t+30         0.550              0.616             0.877         79.053                3.109                       0.550                            0.616                           0.877                       79.053                              3.109                  calibrated         0.554              0.620             0.882         78.210                3.086    calibrated                          0.004                           -0.023
     all 6 folds      TFT     t+7         0.444              0.935             1.000         79.748                2.491                       0.444                            0.935                           1.000                       79.748                              2.491                         raw         0.444              0.935             1.000         79.748                2.491           raw                          0.000                            0.000
     all 6 folds      TFT    t+14         0.629              0.876             1.002         77.175                3.421                       0.629                            0.876                           1.002                       77.175                              3.421                         raw         0.629              0.876             1.002         77.175                3.421           raw                          0.000                            0.000
     all 6 folds      TFT    t+30         0.924              0.764             0.976         73.000                4.775                       0.924                            0.764                           0.976                       73.000                              4.775                         raw         0.924              0.764             0.976         73.000                4.775           raw                          0.000                            0.000
     all 6 folds Ensemble     t+7         0.429              0.903             0.967         80.853                2.443                       0.429                            0.903                           0.967                       80.853                              2.443                         raw         0.429              0.903             0.967         77.423                2.425    calibrated                          0.000                           -0.018
     all 6 folds Ensemble    t+14         0.595              0.830             0.949         79.083                3.293                       0.595                            0.830                           0.949                       79.083                              3.293                         raw         0.595              0.829             0.948         79.161                3.287           raw                         -0.001                           -0.006
     all 6 folds Ensemble    t+30         0.860              0.712             0.909         77.098                5.211                       0.860                            0.712                           0.909                       77.098                              5.211                  calibrated         0.860              0.712             0.909         77.038                5.022    calibrated                         -0.000                           -0.190
```

## Districts with coverage 65-90% (% of 21)
```
run               same-net v4.2-post, all folds  same-net v4.2-post, locked  v4.3, all folds  v4.3, locked
model    horizon                                                                                          
Ensemble t+14                              81.0                        52.0             81.0          57.0
         t+30                              86.0                        57.0             86.0          52.0
         t+7                               86.0                        57.0            100.0          76.0
TFT      t+14                              76.0                        48.0             76.0          48.0
         t+30                              81.0                        57.0             81.0          57.0
         t+7                               86.0                        57.0             86.0          57.0
```

## Reported intervals wider than 20 m
```
   model  same-net v4.2-post  v4.3
Ensemble                 332   105
     TFT                   0     0
```

## Pune (Ensemble, all folds)
```
               run Horizon  rMAE_P  rMAE_SN  MAE (m)
same-net v4.2-post     t+7   1.000    1.022    0.213
same-net v4.2-post    t+14   1.050    1.133    0.334
same-net v4.2-post    t+30   1.128    1.301    0.572
              v4.3     t+7   1.012    1.034    0.216
              v4.3    t+14   1.004    1.083    0.319
              v4.3    t+30   1.121    1.293    0.569
```

## Latest forecast: issued 2026-06-04, 23 districts (dashboard_export/)