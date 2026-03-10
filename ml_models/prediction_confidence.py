"""Prediction confidence scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class PredictionConfidenceResult:
    """Container for confidence diagnostics."""

    diagnostics: pd.DataFrame
    probabilities: np.ndarray
    classes: list


class PredictionConfidenceScorer:
    """Turn model probability outputs into confidence diagnostics."""

    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon

    def score(
        self,
        probabilities,
        *,
        predicted_labels: Optional[Iterable] = None,
        classes: Optional[Iterable] = None,
    ) -> PredictionConfidenceResult:
        probability_array, class_labels = self._normalize_probabilities(
            probabilities,
            classes=classes,
        )
        top_indices = probability_array.argmax(axis=1)
        confidence = probability_array.max(axis=1)

        sorted_probabilities = np.sort(probability_array, axis=1)
        if probability_array.shape[1] > 1:
            margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        else:
            margin = confidence.copy()

        entropy = -np.sum(
            probability_array * np.log(np.clip(probability_array, self.epsilon, 1.0)),
            axis=1,
        )
        entropy_norm = entropy / np.log(max(probability_array.shape[1], 2))
        predicted = list(predicted_labels) if predicted_labels is not None else [class_labels[idx] for idx in top_indices]

        diagnostics = pd.DataFrame(
            {
                "predicted_class": predicted,
                "prediction_confidence": confidence,
                "prediction_margin": margin,
                "prediction_entropy": entropy_norm,
                "prediction_uncertainty": 1.0 - confidence,
            }
        )

        return PredictionConfidenceResult(
            diagnostics=diagnostics,
            probabilities=probability_array,
            classes=class_labels,
        )

    def merge(self, df: pd.DataFrame, probabilities, *, predicted_labels: Optional[Iterable] = None) -> pd.DataFrame:
        result = df.copy()
        confidence = self.score(probabilities, predicted_labels=predicted_labels)
        for column in confidence.diagnostics.columns:
            result[column] = confidence.diagnostics[column].values
        return result

    def _normalize_probabilities(self, probabilities, *, classes: Optional[Iterable] = None):
        if isinstance(probabilities, pd.DataFrame):
            class_labels = list(probabilities.columns)
            probability_array = probabilities.to_numpy(dtype=float)
        else:
            probability_array = np.asarray(probabilities, dtype=float)
            if probability_array.ndim == 1:
                probability_array = np.column_stack([1.0 - probability_array, probability_array])
            class_labels = list(classes) if classes is not None else list(range(probability_array.shape[1]))

        row_sums = probability_array.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        probability_array = np.clip(probability_array / row_sums, self.epsilon, 1.0)
        probability_array = probability_array / probability_array.sum(axis=1, keepdims=True)
        return probability_array, class_labels
