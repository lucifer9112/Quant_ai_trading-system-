"""
Probability Calibration - Ensure Predictions Match Reality

When a model says 60% confidence, actual accuracy should be ~60%.
Poor calibration leads to over-confident bets and worse risk management.

Techniques:
- Platt scaling
- Isotonic regression
- Temperature scaling
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Calibrate predicted probabilities to match observed frequencies."""
    
    def __init__(self, method: str = "isotonic"):  # isotonic, platt, temperature
        """
        Args:
            method: Calibration method to use
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,  # Predicted probabilities (0-1)
        actuals: np.ndarray,      # Actual binary outcomes (0 or 1)
    ):
        """
        Fit calibrator on validation set.
        
        Args:
            predictions: Array of predicted probabilities
            actuals: Array of actual binary outcomes
        """
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(predictions, actuals)
        
        elif self.method == "platt":
            # Platt scaling: fit logistic regression
            self.calibrator = LogisticRegression()
            log_odds = np.log(predictions / (1 - predictions + 1e-6) + 1e-6)
            self.calibrator.fit(log_odds.reshape(-1, 1), actuals)
        
        elif self.method == "temperature":
            # Temperature scaling: optimize single temperature parameter
            self.calibrator = self._fit_temperature(predictions, actuals)
        
        self.is_fitted = True
    
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calibrate raw predictions.
        
        Args:
            predictions: Raw predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            return predictions
        
        if self.method == "isotonic":
            return self.calibrator.predict(predictions)
        
        elif self.method == "platt":
            log_odds = np.log(predictions / (1 - predictions + 1e-6) + 1e-6)
            return self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]
        
        elif self.method == "temperature":
            temperature = self.calibrator
            return self._apply_temperature(predictions, temperature)
        
        return predictions
    
    def _fit_temperature(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Fit temperature parameter."""
        # Grid search for best temperature
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in np.linspace(0.5, 2.0, 20):
            calibrated = self._apply_temperature(predictions, temp)
            loss = self._nll_loss(calibrated, actuals)
            
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        
        return best_temp
    
    def _apply_temperature(
        self,
        predictions: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Apply temperature scaling."""
        # T < 1: sharpen probabilities
        # T > 1: soften probabilities
        log_odds = np.log(predictions / (1 - predictions + 1e-6) + 1e-6)
        return 1.0 / (1.0 + np.exp(-log_odds / temperature))
    
    def _nll_loss(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Negative log-likelihood loss."""
        eps = 1e-6
        predictions = np.clip(predictions, eps, 1 - eps)
        return -np.mean(actuals * np.log(predictions) + 
                       (1 - actuals) * np.log(1 - predictions))
    
    def evaluate_calibration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict:
        """
        Evaluate calibration quality.
        
        Args:
            predictions: Predicted probabilities
            actuals: Actual outcomes
            
        Returns:
            Calibration metrics
        """
        # Bin predictions
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        
        calibration_curve = []
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) == 0:
                continue
            
            predicted_prob = np.mean(predictions[mask])
            actual_freq = np.mean(actuals[mask])
            
            calibration_curve.append({
                'bin': bin_idx,
                'predicted': predicted_prob,
                'actual': actual_freq,
                'error': abs(predicted_prob - actual_freq),
                'count': np.sum(mask),
            })
        
        # Overall metrics
        mean_calibration_error = np.mean([c['error'] for c in calibration_curve])
        
        # Brier score
        brier = np.mean((predictions - actuals) ** 2)
        
        # NLL
        eps = 1e-6
        predictions_clipped = np.clip(predictions, eps, 1 - eps)
        nll = -np.mean(actuals * np.log(predictions_clipped) + 
                      (1 - actuals) * np.log(1 - predictions_clipped))
        
        return {
            'mean_calibration_error': mean_calibration_error,
            'brier_score': brier,
            'nll': nll,
            'calibration_curve': pd.DataFrame(calibration_curve),
        }


class ConfidenceCalibrator:
    """Calibrate continuous confidence scores (0-1)."""
    
    def __init__(self):
        self.calibrator = None
        self.is_fitted = False
    
    def fit(
        self,
        confidences: np.ndarray,  # Model confidence scores
        errors: np.ndarray,       # Prediction errors (absolute)
    ):
        """
        Fit confidence calibration.
        
        Args:
            confidences: Model confidence scores (0-1)
            errors: Absolute prediction errors
        """
        # Simple: fit relationship between confidence and error
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        
        # Higher confidence should predict lower error
        targets = 1.0 - (errors / (np.max(errors) + 1e-6))
        
        self.calibrator.fit(confidences, targets)
        self.is_fitted = True
    
    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """Calibrate confidence scores."""
        if not self.is_fitted:
            return confidences
        
        return self.calibrator.predict(confidences)
    
    def get_expected_error(self, confidence: float) -> float:
        """Estimate expected error given confidence."""
        if not self.is_fitted:
            return 1.0 - confidence
        
        calibrated = self.calibrator.predict([[confidence]])[0]
        return 1.0 - calibrated


class ThresholdOptimizer:
    """
    Optimize decision thresholds based on cost function.
    
    Example: If false positives (wrong buys) cost more than false negatives,
    set threshold higher.
    """
    
    def __init__(
        self,
        cost_false_positive: float = 1.0,
        cost_false_negative: float = 1.0,
    ):
        """
        Args:
            cost_false_positive: Cost of incorrect BUY signal
            cost_false_negative: Cost of missed BUY signal
        """
        self.cost_fp = cost_false_positive
        self.cost_fn = cost_false_negative
        self.optimal_threshold = 0.5
    
    def optimize(
        self,
        predictions: np.ndarray,  # Predicted probabilities
        actuals: np.ndarray,      # Actual outcomes
    ) -> float:
        """
        Find optimal decision threshold.
        
        Args:
            predictions: Predicted probabilities
            actuals: Actual outcomes (binary)
            
        Returns:
            Optimal threshold
        """
        thresholds = np.linspace(0, 1, 101)
        best_cost = float('inf')
        best_threshold = 0.5
        
        for thresh in thresholds:
            predicted_pos = (predictions >= thresh).astype(int)
            
            # Cost matrix
            tp = np.sum((predicted_pos == 1) & (actuals == 1))
            fp = np.sum((predicted_pos == 1) & (actuals == 0))
            fn = np.sum((predicted_pos == 0) & (actuals == 1))
            
            cost = (self.cost_fp * fp) + (self.cost_fn * fn)
            
            if cost < best_cost:
                best_cost = cost
                best_threshold = thresh
        
        self.optimal_threshold = best_threshold
        return best_threshold
    
    def classify(self, predictions: np.ndarray) -> np.ndarray:
        """Classify using optimal threshold."""
        return (predictions >= self.optimal_threshold).astype(int)


class CalibrationPlotter:
    """Generate calibration plots."""
    
    @staticmethod
    def reliability_diagram(
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Generate data for reliability diagram.
        Perfect calibration = points on diagonal.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        
        diagram = []
        
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) == 0:
                continue
            
            mean_pred = np.mean(predictions[mask])
            mean_actual = np.mean(actuals[mask])
            count = np.sum(mask)
            
            diagram.append({
                'predicted_prob': mean_pred,
                'actual_freq': mean_actual,
                'count': count,
                'cal': abs(mean_pred - mean_actual),
            })
        
        return pd.DataFrame(diagram)
