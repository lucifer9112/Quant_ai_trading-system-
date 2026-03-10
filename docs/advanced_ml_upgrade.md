# Quant Trading Upgrade Guide

## Architecture changes

- Added `validation/time_series_cv.py` for panel-aware walk-forward and time-series cross-validation.
- Added `feature_engineering/panel_feature_expander.py` to expand the existing research feature pipeline with leakage-safe microstructure, cross-asset, and macro context features.
- Extended `feature_engineering/feature_analyzer.py` with correlation pruning and SHAP-based importance support.
- Rebuilt `ml_models/stacking_ensemble.py` and `ml_models/ensemble_builder.py` around time-series out-of-fold stacking.
- Added `ml_models/prediction_confidence.py` for probability-derived confidence, margin, and entropy diagnostics.
- Added `risk_management/confidence_position_sizer.py` and wired it into portfolio allocation and backtesting.
- Added `monitoring/model_drift.py` for PSI-based drift baselines and live drift checks.
- Added `explainability/shap_explainer.py` plus `dashboards/explainability_dashboard.py` for model explainability.

## New modules

- `validation/time_series_cv.py`
- `feature_engineering/panel_feature_expander.py`
- `explainability/shap_explainer.py`
- `ml_models/prediction_confidence.py`
- `risk_management/confidence_position_sizer.py`
- `monitoring/model_drift.py`
- `dashboards/explainability_dashboard.py`

## Integration points

- `PanelDatasetBuilder.build_panel(...)` now supports macro and benchmark inputs and expands panel features by default.
- `PanelDatasetBuilder.build_training_frame(...)` now supports metadata retention, correlation pruning, and feature caps.
- `AutoGluonTrainer.walk_forward_validate(...)` adds walk-forward evaluation without changing the existing `train(...)` contract.
- `AutoGluonPredictor.predict(...)` now appends `prediction_confidence`, `prediction_margin`, and `prediction_entropy` when probability output is available.
- `PortfolioAllocator` and `AdvancedBacktester` now scale exposure by model confidence.
- `apps/batch_train/train_tabular.py` exposes walk-forward validation, pruning, drift baseline export, and explainability reporting.

## Example usage

```bash
python apps/batch_train/train_tabular.py \
  --config config.yaml \
  --walk-forward \
  --prune-correlated \
  --max-features 120 \
  --save-drift-baseline \
  --save-shap-report \
  --benchmark-path data/benchmarks/nifty50.csv \
  --macro-path data/macro/macro_features.csv
```

## Explainability dashboard

```python
from dashboards.explainability_dashboard import ExplainabilityDashboard

dashboard = ExplainabilityDashboard()
dashboard.run(model, feature_frame)
```

## Drift monitoring

```python
from monitoring.model_drift import ModelDriftDetector

detector = ModelDriftDetector().fit(reference_features)
report = detector.detect(live_features)
print(report["metrics"].head())
```
