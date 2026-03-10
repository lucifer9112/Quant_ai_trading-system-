"""Probability calibration for improved prediction confidence."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrate raw model outputs to well-calibrated probabilities.
    
    Key issue: Raw predictions != true probabilities. Calibration ensures:
    - When model says 70%, actual success rate ≈ 70%
    - Prevents overconfident predictions causing wrong trade sizes
    
    Methods:
    - Isotonic regression (non-parametric, flexible)
    - Platt scaling (parametric, simpler)
    
    Expected benefit: Better risk management, improved position sizing
    """

    def __init__(self, method: str = 'isotonic'):
        """Initialize probability calibrator.
        
        Parameters
        ----------
        method : {'isotonic', 'platt'}
            Calibration method
        """
        if method not in ['isotonic', 'platt']:
            raise ValueError(f"Unknown calibration method: {method}")
        
        self.method = method
        self.calibrator = None
        self.calibrated = False
        
        logger.info(f"ProbabilityCalibrator initialized with method={method}")

    def fit(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """Fit calibrator on validation data.
        
        Parameters
        ----------
        predictions : array, shape (n_samples,)
            Raw model predictions (typically in range [0, 1] for classification)
        actuals : array, shape (n_samples,)
            Ground truth binary outcomes (0 or 1)
            
        Returns
        -------
        self
        """
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:  # platt
            self.calibrator = LogisticRegression(max_iter=1000)
        
        # Ensure predictions are in [0, 1]
        predictions_bounded = np.clip(predictions, 0, 1)
        
        if self.method == 'isotonic':
            self.calibrator.fit(predictions_bounded, actuals)
        else:  # platt
            # Need 2D array for LogisticRegression
            self.calibrator.fit(predictions_bounded.reshape(-1, 1), actuals)
        
        self.calibrated = True
        logger.info(f"Calibrator fitted on {len(predictions)} samples using {self.method} method")
        
        return self

    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Transform raw predictions to calibrated probabilities.
        
        Parameters
        ----------
        predictions : array, shape (n_samples,)
            Raw model predictions
            
        Returns
        -------
        calibrated : array, shape (n_samples,)
            Calibrated probabilities
        """
        if not self.calibrated:
            raise ValueError("Calibrator must be fitted before calibration")
        
        predictions_bounded = np.clip(predictions, 0, 1)
        
        if self.method == 'isotonic':
            calibrated = self.calibrator.predict(predictions_bounded)
        else:  # platt
            calibrated = self.calibrator.predict_proba(predictions_bounded.reshape(-1, 1))[:, 1]
        
        return np.clip(calibrated, 0, 1)

    def evaluate_calibration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate calibration quality using multiple metrics.
        
        Parameters
        ----------
        predictions : array
            Raw predictions
        actuals : array
            Ground truth
            
        Returns
        -------
        metrics : dict
            Calibration metrics (ECE, MCE, Brier score)
        """
        calibrated = self.calibrate(predictions)
        
        # Expected Calibration Error (ECE)
        ece = self._expected_calibration_error(calibrated, actuals)
        
        # Maximum Calibration Error (MCE)
        mce = self._max_calibration_error(calibrated, actuals)
        
        # Brier Score
        brier = np.mean((calibrated - actuals) ** 2)
        
        # Log-loss
        epsilon = 1e-15
        log_loss = -np.mean(
            actuals * np.log(np.clip(calibrated, epsilon, 1)) +
            (1 - actuals) * np.log(np.clip(1 - calibrated, epsilon, 1))
        )
        
        metrics = {
            'ECE': ece,
            'MCE': mce,
            'Brier': brier,
            'LogLoss': log_loss,
        }
        
        logger.info(f"Calibration metrics: ECE={ece:.4f}, MCE={mce:.4f}, Brier={brier:.4f}")
        
        return metrics

    def _expected_calibration_error(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Calculate Expected Calibration Error.
        
        Bins predictions and checks if average prediction ≈ actual success rate.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_acc = actuals[mask].mean()
            bin_conf = predictions[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(predictions)
        
        return ece

    def _max_calibration_error(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Calculate Maximum Calibration Error.
        
        Maximum absolute difference between predicted and actual success rates.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        mce = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_acc = actuals[mask].mean()
            bin_conf = predictions[mask].mean()
            mce = max(mce, np.abs(bin_acc - bin_conf))
        
        return mce

    def get_confidence_based_sizing(
        self,
        calibrated_probs: np.ndarray,
        base_position_size: float = 1.0,
        min_size: float = 0.1,
        max_size: float = 3.0,
    ) -> np.ndarray:
        """Generate position sizes based on calibrated confidence.
        
        Parameters
        ----------
        calibrated_probs : array
            Calibrated probabilities
        base_position_size : float
            Base size when probability = 0.5
        min_size : float
            Minimum position size
        max_size : float
            Maximum position size
            
        Returns
        -------
        sizes : array, shape like calibrated_probs
            Position sizes scaled by confidence
        """
        # Confidence: how far away from 0.5 (neutral)
        confidence = 2 * np.abs(calibrated_probs - 0.5)  # Range [0, 1]
        
        # Position size scales with confidence
        sizes = base_position_size * (1.0 + confidence)
        sizes = np.clip(sizes, min_size, max_size)
        
        return sizes

    def summary(self) -> str:
        """Get calibrator summary.
        
        Returns
        -------
        summary : str
        """
        lines = [
            "ProbabilityCalibrator Summary",
            "=" * 50,
            f"Method: {self.method}",
            f"Status: {'Calibrated' if self.calibrated else 'Not calibrated'}",
        ]
        
        return "\n".join(lines)
