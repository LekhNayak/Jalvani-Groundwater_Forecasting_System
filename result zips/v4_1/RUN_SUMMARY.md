# JalVaani v4.1 run summary

- pipeline 4.5, data master_dataset_v4target.csv (md5 e035dc26)
- chosen set-up: **s1_tab2** = {'target_mode': 'snaive', 'calib_mode': 'v4', 'combine': 'stack', 'refit': True, 'train_stride': 1, 'epoch_frac': 0.14285714285714285, 'tab_features': 'v2', 'tab_stride': 1, 'tab_fresh': True, 'gbm_n_estimators': 400, 'gbm_lr': 0.03, 'gbm_min_child': 100, 'use_refit': False}
- stacking strength stack_prior_n = 500, interval adaptivity aci_gamma = 0.01
- final seeds: [42, 1337, 2026, 7, 99, 123, 555, 777, 2024, 31337]
- total run time: 54 min
- selection used dev folds 1-4 only; folds 5-6 are the locked test (viewed once before, in v4.0)

## Screening (dev folds 1-4)
```
           run  mean vs seasonal  vs no-change t+7  vs no-change t+14  vs no-change t+30
           ref             0.963             0.922              0.860              0.769
          tab2             0.949             0.924              0.848              0.744
         refit             0.969             0.920              0.858              0.786
    tab2_refit             0.946             0.920              0.845              0.743
       s1_tab2             0.939             0.907              0.841              0.739
 s1_tab2_refit             0.954             0.932              0.850              0.746
d48_tab2_refit             0.961             0.936              0.865              0.746
```

## v4.0 -> v4.1
```
        test set    model horizon  v4.0 MAE (m)  v4.0 vs no-change  v4.0 vs seasonal  v4.0 coverage  v4.1 MAE (m)  v4.1 vs no-change  v4.1 vs seasonal  v4.1 coverage  change vs no-change
locked folds 5-6 Ensemble     t+7         0.323              0.864             0.967           80.6         0.314              0.839             0.938         81.087               -0.025
locked folds 5-6 Ensemble    t+14         0.419              0.788             0.952           81.5         0.406              0.763             0.922         80.138               -0.025
locked folds 5-6 Ensemble    t+30         0.593              0.664             0.945           81.2         0.550              0.616             0.877         79.964               -0.048
locked folds 5-6      TFT     t+7         0.340              0.910             1.017           74.0         0.326              0.872             0.976         74.152               -0.037
locked folds 5-6      TFT    t+14         0.444              0.834             1.008           70.1         0.428              0.803             0.971         70.010               -0.030
locked folds 5-6      TFT    t+30         0.622              0.697             0.992           66.8         0.602              0.675             0.960         65.710               -0.023
     all 6 folds Ensemble     t+7         0.433              0.911             0.975           80.8         0.429              0.903             0.967         79.887               -0.008
     all 6 folds Ensemble    t+14         0.610              0.850             0.972           80.3         0.595              0.830             0.949         78.432               -0.020
     all 6 folds Ensemble    t+30         0.907              0.750             0.958           77.7         0.860              0.712             0.909         78.144               -0.038
     all 6 folds      TFT     t+7         0.445              0.937             1.003           74.1         0.444              0.935             1.000         72.974               -0.003
     all 6 folds      TFT    t+14         0.629              0.876             1.002           69.4         0.629              0.876             1.002         68.559               -0.000
     all 6 folds      TFT    t+30         0.934              0.772             0.987           65.0         0.924              0.764             0.976         64.318               -0.008
```