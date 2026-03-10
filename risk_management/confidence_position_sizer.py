"""Confidence-aware portfolio sizing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ConfidenceSizingDiagnostics:
    """Diagnostics for confidence-based sizing."""

    confidence: float
    entropy: float
    multiplier: float


class ConfidencePositionSizer:
    """Scale portfolio weights using calibrated model confidence."""

    def __init__(
        self,
        confidence_floor: float = 0.50,
        min_multiplier: float = 0.35,
        max_multiplier: float = 1.50,
    ):
        self.confidence_floor = confidence_floor
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def confidence_multiplier(
        self,
        confidence: float,
        *,
        entropy: Optional[float] = None,
    ) -> float:
        bounded_confidence = float(np.clip(confidence, 0.0, 1.0))
        normalized = max(0.0, bounded_confidence - self.confidence_floor)
        denominator = max(1e-9, 1.0 - self.confidence_floor)
        multiplier = self.min_multiplier + (
            (self.max_multiplier - self.min_multiplier) * (normalized / denominator)
        )

        if entropy is not None and not np.isnan(entropy):
            multiplier *= float(np.clip(1.0 - (entropy * 0.5), 0.5, 1.0))

        return float(np.clip(multiplier, self.min_multiplier, self.max_multiplier))

    def scale_weights(
        self,
        weights: Dict[str, float],
        confidences: Dict[str, float],
        entropies: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        adjusted = {}
        for symbol, weight in weights.items():
            entropy = None if entropies is None else entropies.get(symbol)
            multiplier = self.confidence_multiplier(confidences.get(symbol, 1.0), entropy=entropy)
            adjusted[symbol] = weight * multiplier
        return adjusted
