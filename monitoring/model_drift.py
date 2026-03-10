"""Model drift detection using feature and prediction distribution shifts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json

import numpy as np
import pandas as pd


@dataclass
class DriftMetric:
    """Single drift metric summary."""

    feature: str
    psi: float
    mean_shift: float
    std_ratio: float
    drift_flag: bool


class ModelDriftDetector:
    """Track training distributions and compare them against live samples."""

    def __init__(self, bins: int = 10, psi_threshold: float = 0.25):
        self.bins = bins
        self.psi_threshold = psi_threshold
        self.reference_profile_: Dict[str, Dict[str, list | float]] = {}

    def fit(self, reference_df: pd.DataFrame) -> "ModelDriftDetector":
        numeric = reference_df.select_dtypes(include=[np.number])
        profile = {}
        for column in numeric.columns:
            series = numeric[column].dropna()
            if series.empty:
                continue
            quantiles = np.unique(np.quantile(series, np.linspace(0.0, 1.0, self.bins + 1)))
            if len(quantiles) < 2:
                continue
            profile[column] = {
                "breaks": quantiles.tolist(),
                "hist": self._histogram(series, quantiles).tolist(),
                "mean": float(series.mean()),
                "std": float(series.std() or 0.0),
            }
        self.reference_profile_ = profile
        return self

    def detect(self, current_df: pd.DataFrame) -> Dict[str, object]:
        if not self.reference_profile_:
            raise ValueError("Drift detector must be fitted before detect()")

        numeric = current_df.select_dtypes(include=[np.number])
        metrics = []
        for column, profile in self.reference_profile_.items():
            if column not in numeric.columns:
                continue
            series = numeric[column].dropna()
            if series.empty:
                continue

            breaks = np.asarray(profile["breaks"], dtype=float)
            reference_hist = np.asarray(profile["hist"], dtype=float)
            current_hist = self._histogram(series, breaks)
            psi = self._psi(reference_hist, current_hist)
            mean_shift = float(series.mean() - float(profile["mean"]))
            reference_std = float(profile["std"]) or 1e-9
            std_ratio = float((series.std() or 0.0) / reference_std)
            metrics.append(
                DriftMetric(
                    feature=column,
                    psi=float(psi),
                    mean_shift=mean_shift,
                    std_ratio=std_ratio,
                    drift_flag=psi >= self.psi_threshold,
                )
            )

        drift_frame = pd.DataFrame([metric.__dict__ for metric in metrics])
        drift_frame = drift_frame.sort_values("psi", ascending=False, ignore_index=True) if not drift_frame.empty else drift_frame

        return {
            "metrics": drift_frame,
            "drift_detected": bool((drift_frame["drift_flag"]).any()) if not drift_frame.empty else False,
            "max_psi": float(drift_frame["psi"].max()) if not drift_frame.empty else 0.0,
        }

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.reference_profile_, indent=2), encoding="utf-8")
        return output_path

    def load(self, path: str | Path) -> "ModelDriftDetector":
        self.reference_profile_ = json.loads(Path(path).read_text(encoding="utf-8"))
        return self

    def _histogram(self, series: pd.Series, breaks: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(series, bins=breaks)
        counts = counts.astype(float)
        total = counts.sum()
        if total == 0:
            return np.full_like(counts, 1.0 / len(counts))
        return counts / total

    def _psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        expected = np.clip(expected, 1e-9, 1.0)
        actual = np.clip(actual, 1e-9, 1.0)
        return float(np.sum((actual - expected) * np.log(actual / expected)))
