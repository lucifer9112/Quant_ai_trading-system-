"""
Ensemble ML Model - Combine Multiple Predictions

Combines:
1. Cross-asset model predictions
2. Sector model predictions
3. Technical signals
4. Sentiment features
5. Individual autogluon predictions

Significantly improves accuracy and robustness.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


@dataclass
class EnsemblePrediction:
    """Ensemble prediction with components"""
    symbol: str
    prediction: float
    confidence: float
    component_predictions: Dict[str, float]
    component_weights: Dict[str, float]
    prediction_std: float  # Uncertainty estimate


class EnsemblePredictor:
    """
    Combine multiple ML models into single ensemble prediction.
    
    Ensemble methods:
    - Weighted average (by recent R²)
    - Stacking (meta-learner)
    - Boosting (sequential improvement)
    """
    
    def __init__(
        self,
        ensemble_method: str = "weighted_avg",  # weighted_avg, stacking, boosting
        update_frequency: int = 10,  # Recalculate weights every N days
    ):
        """
        Args:
            ensemble_method: How to combine predictions
            update_frequency: How often to update model weights
        """
        self.ensemble_method = ensemble_method
        self.update_frequency = update_frequency
        self.component_weights = {
            'cross_asset': 0.30,
            'sector': 0.25,
            'technical': 0.20,
            'sentiment': 0.15,
            'autogluon': 0.10,
        }
        self.component_scores = {}  # Track recent performance
        self.meta_learner = None
        self.days_since_update = 0
    
    def combine_predictions(
        self,
        cross_asset_pred: Optional[float] = None,
        sector_pred: Optional[float] = None,
        technical_pred: Optional[float] = None,
        sentiment_pred: Optional[float] = None,
        autogluon_pred: Optional[float] = None,
        symbol: str = "",
    ) -> EnsemblePrediction:
        """
        Combine predictions from multiple models.
        
        Args:
            cross_asset_pred: Prediction from cross-asset model
            sector_pred: Prediction from sector model
            technical_pred: Prediction from technical signals
            sentiment_pred: Prediction from sentiment analysis
            autogluon_pred: Prediction from AutoGluon
            symbol: Stock symbol
            
        Returns:
            EnsemblePrediction with combined result
        """
        components = {
            'cross_asset': cross_asset_pred,
            'sector': sector_pred,
            'technical': technical_pred,
            'sentiment': sentiment_pred,
            'autogluon': autogluon_pred,
        }
        
        # Remove None values
        valid_components = {k: v for k, v in components.items() if v is not None}
        
        if not valid_components:
            return EnsemblePrediction(
                symbol=symbol,
                prediction=0.0,
                confidence=0.0,
                component_predictions={},
                component_weights={},
                prediction_std=0.0,
            )
        
        # Combine based on method
        if self.ensemble_method == "weighted_avg":
            pred, conf, std = self._weighted_average(valid_components)
        elif self.ensemble_method == "stacking":
            pred, conf, std = self._stacking(valid_components)
        elif self.ensemble_method == "boosting":
            pred, conf, std = self._boosting(valid_components)
        else:
            pred, conf, std = self._weighted_average(valid_components)
        
        # Get active weights
        active_weights = {
            k: (self.component_weights[k] / sum(self.component_weights[c] 
                for c in valid_components.keys()))
            for k in valid_components.keys()
        }
        
        return EnsemblePrediction(
            symbol=symbol,
            prediction=pred,
            confidence=conf,
            component_predictions=valid_components,
            component_weights=active_weights,
            prediction_std=std,
        )
    
    def _weighted_average(
        self,
        predictions: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """Weighted average ensemble."""
        total_weight = sum(
            self.component_weights[k] for k in predictions.keys()
        )
        
        if total_weight == 0:
            return 0.0, 0.0, 0.0
        
        # Weighted prediction
        pred = sum(
            self.component_weights[k] * v
            for k, v in predictions.items()
        ) / total_weight
        
        # Confidence: agreement between models
        preds = np.array(list(predictions.values()))
        agreement = 1.0 - (np.std(preds) / (np.mean(np.abs(preds)) + 1e-6))
        confidence = np.clip(agreement, 0.0, 1.0)
        
        # Std: predicted uncertainty
        std = np.std(preds)
        
        return float(pred), float(confidence), float(std)
    
    def _stacking(
        self,
        predictions: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """Stacking - train meta-learner on predictions."""
        preds = np.array(list(predictions.values())).reshape(1, -1)
        
        if self.meta_learner is None:
            # Initialize simple meta-learner
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0)
            # Default weights until trained
            return self._weighted_average(predictions)
        
        # Use meta-learner
        pred = self.meta_learner.predict(preds)[0]
        
        # Confidence from prediction magnitude
        confidence = min(1.0, abs(pred) / 0.05)
        
        # Std from component variance
        std = np.std(np.array(list(predictions.values())))
        
        return float(pred), float(confidence), float(std)
    
    def _boosting(
        self,
        predictions: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """Boosting - emphasize disagreement."""
        preds_array = np.array(list(predictions.values()))
        
        # Find consensus
        consensus = np.median(preds_array)
        
        # Boost outliers
        adjusted = []
        for k, v in predictions.items():
            error = abs(v - consensus)
            boost = 1.0 + (error / (np.std(preds_array) + 1e-6))
            adjusted.append(v * boost)
        
        pred = np.mean(adjusted)
        
        # Confidence from unanimity
        disagreement = np.std(preds_array)
        confidence = 1.0 / (1.0 + disagreement)
        
        std = disagreement
        
        return float(pred), float(confidence), float(std)
    
    def update_weights_from_performance(
        self,
        actual_returns: Dict[str, float],
        predictions: Dict[str, Dict[str, float]],
    ):
        """
        Update component weights based on recent performance.
        
        Args:
            actual_returns: Actual symbol returns
            predictions: Dict mapping symbol to {component: prediction}
        """
        self.days_since_update += 1
        
        if self.days_since_update < self.update_frequency:
            return
        
        # Calculate R² for each component
        for component in self.component_weights.keys():
            component_preds = [
                pred.get(component)
                for pred in predictions.values()
            ]
            component_preds = [p for p in component_preds if p is not None]
            
            if not component_preds:
                continue
            
            actual = list(actual_returns.values())
            
            # Calculate R²
            ss_res = sum((actual[i] - component_preds[i]) ** 2 
                         for i in range(len(actual)))
            ss_tot = sum((a - np.mean(actual)) ** 2 for a in actual)
            
            r2 = 1.0 - (ss_res / (ss_tot + 1e-6))
            self.component_scores[component] = max(0.0, r2)
        
        # Normalize scores to weights
        if self.component_scores:
            total = sum(self.component_scores.values())
            if total > 0:
                self.component_weights = {
                    k: v / total for k, v in self.component_scores.items()
                }
        
        self.days_since_update = 0
    
    def get_ensemble_metrics(self) -> Dict:
        """Get ensemble performance metrics."""
        return {
            'weights': self.component_weights,
            'scores': self.component_scores,
            'method': self.ensemble_method,
            'prediction_std': self.prediction_std if hasattr(self, 'prediction_std') else None,
        }


class VotingEnsemble:
    """
    Simple voting ensemble for classification (long/hold/short).
    """
    
    def __init__(self, voting_type: str = "soft"):  # soft or hard
        """
        Args:
            voting_type: "soft" = averaged probabilities, "hard" = majority vote
        """
        self.voting_type = voting_type
    
    def vote(
        self,
        signals: Dict[str, str],  # component_name -> "BUY"/"HOLD"/"SELL"
    ) -> str:
        """
        Generate ensemble signal from component votes.
        
        Args:
            signals: Dict mapping components to signals
            
        Returns:
            Final signal: "BUY", "HOLD", or "SELL"
        """
        signal_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        
        values = [signal_map.get(v, 0) for v in signals.values()]
        
        if not values:
            return "HOLD"
        
        avg = np.mean(values)
        
        if avg > 0.3:
            return "BUY"
        elif avg < -0.3:
            return "SELL"
        else:
            return "HOLD"
    
    def weighted_vote(
        self,
        signals: Dict[str, str],  # component -> signal
        weights: Dict[str, float],  # component -> weight
    ) -> str:
        """Weighted voting."""
        signal_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for component, signal in signals.items():
            weight = weights.get(component, 1.0)
            value = signal_map.get(signal, 0)
            
            weighted_sum += weight * value
            total_weight += weight
        
        if total_weight == 0:
            return "HOLD"
        
        avg = weighted_sum / total_weight
        
        if avg > 0.3:
            return "BUY"
        elif avg < -0.3:
            return "SELL"
        else:
            return "HOLD"
