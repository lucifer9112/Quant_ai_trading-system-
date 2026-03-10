"""SHAP explainability helpers with graceful fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ShapExplainabilityReport:
    """Container for SHAP outputs used by reports and dashboards."""

    feature_importance: pd.DataFrame
    shap_values: pd.DataFrame
    base_values: Optional[np.ndarray] = None


class ShapExplainer:
    """Generate global and local SHAP explanations."""

    def __init__(
        self,
        sample_size: int = 500,
        background_size: int = 200,
        random_state: int = 42,
    ):
        self.sample_size = sample_size
        self.background_size = background_size
        self.random_state = random_state

    def explain(self, model, X: pd.DataFrame) -> ShapExplainabilityReport:
        feature_frame = self._to_frame(X)
        sampled = self._sample(feature_frame)

        try:
            import shap
        except ImportError:
            return self._fallback_report(model, sampled)

        background = sampled.head(min(self.background_size, len(sampled)))
        explainer = shap.Explainer(model, background)
        explanation = explainer(sampled)
        values = np.asarray(explanation.values)

        if values.ndim == 3:
            local_values = values.mean(axis=2)
            mean_abs = np.abs(values).mean(axis=(0, 2))
        else:
            local_values = values
            mean_abs = np.abs(values).mean(axis=0)

        shap_frame = pd.DataFrame(local_values, columns=sampled.columns, index=sampled.index)
        importance = pd.DataFrame(
            {
                "feature": sampled.columns,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

        base_values = getattr(explanation, "base_values", None)
        if base_values is not None:
            base_values = np.asarray(base_values)

        return ShapExplainabilityReport(
            feature_importance=importance,
            shap_values=shap_frame,
            base_values=base_values,
        )

    def save_report(self, report: ShapExplainabilityReport, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report.feature_importance.to_csv(output_path / "shap_feature_importance.csv", index=False)
        report.shap_values.to_csv(output_path / "shap_values_sample.csv", index=True)
        return output_path

    def _to_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    def _sample(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(X) <= self.sample_size:
            return X.copy()
        return X.sample(self.sample_size, random_state=self.random_state)

    def _fallback_report(self, model, X: pd.DataFrame) -> ShapExplainabilityReport:
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_, dtype=float))
            if values.ndim > 1:
                values = values.mean(axis=0)
        elif hasattr(model, "feature_importance"):
            try:
                importance_frame = model.feature_importance(X)
                if isinstance(importance_frame, pd.DataFrame):
                    if "importance" in importance_frame.columns:
                        importance_series = importance_frame.set_index("feature")["importance"]
                    else:
                        importance_series = importance_frame.iloc[:, 0]
                    values = importance_series.reindex(X.columns).fillna(0.0).to_numpy(dtype=float)
                else:
                    values = np.asarray(importance_frame, dtype=float)
            except Exception:
                values = np.zeros(len(X.columns), dtype=float)
        else:
            values = np.zeros(len(X.columns), dtype=float)

        importance = pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": values,
            }
        ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

        return ShapExplainabilityReport(
            feature_importance=importance,
            shap_values=pd.DataFrame(0.0, index=X.index, columns=X.columns),
            base_values=None,
        )
