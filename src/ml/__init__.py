"""ML package: 1X2 classification & goal regression for Serie A.

Modules
-------
- ``data``:     load matches_clean.csv, derive season, build targets/features.
- ``splits``:   time-based split + expanding-window walk-forward.
- ``models``:   baselines + XGBoost/LightGBM/CatBoost classifiers & regressors.
- ``evaluate``: metrics (log loss, ROC-AUC, accuracy, MAE, RMSE) + reports.
- ``plots``:    confusion matrix, feature importance, calibration curve.
- ``train``:    CLI orchestrating the full training run.
- ``predict``:  CLI for inference with the saved pipeline.
"""
