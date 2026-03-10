"""Dynamic weighting of ensemble predictions based on market regime."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DynamicWeighting:
    """Adapt ensemble weights based on market regime.
    
    Key insight: Different models perform better in different regimes
    - XGBoost better in trending markets
    - LightGBM better in mean-reverting markets
    - Random Forest robust in complex regimes
    
    Approach:
    1. Identify current regime (volatility, trend, correlation)
    2. Use regime-specific weights
    3. Continuously update weights based on recent OOS performance
    
    Expected improvement: 1-3% accuracy boost via regime adaptation
    """

    def __init__(self, window_size: int = 63):
        """Initialize dynamic weighting system.
        
        Parameters
        ----------
        window_size : int
            Days to use for recent performance evaluation
        """
        self.window_size = window_size
        self.base_weights = {}
        self.regime_weights = {}
        self.performance_history = {}
        self.current_regime = None
        
        logger.info(f"DynamicWeighting initialized with window_size={window_size}")

    def identify_regime(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        correlation_matrix: np.ndarray = None,
    ) -> str:
        """Identify current market regime.
        
        Parameters
        ----------
        returns : array, shape (n_days,)
            Recent returns
        volatility : array, shape (n_days,)
            Recent volatility measures
        correlation_matrix : array, optional
            Pairwise correlations
            
        Returns
        -------
        regime : str
            One of: 'low_vol_trending', 'high_vol_trending', 'mean_reverting', 'choppy'
        """
        # Volatility percentile
        vol_pctl = np.percentile(volatility[-self.window_size:], q=50)
        current_vol = volatility[-1]
        
        # Trend strength (autocorrelation)
        if len(returns) > 10:
            trend_strength = np.abs(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        else:
            trend_strength = 0.5
        
        # Determine regime
        if current_vol < vol_pctl and trend_strength > 0.3:
            regime = 'low_vol_trending'
        elif current_vol >= vol_pctl and trend_strength > 0.3:
            regime = 'high_vol_trending'
        elif trend_strength < 0.1:
            regime = 'mean_reverting'
        else:
            regime = 'choppy'
        
        self.current_regime = regime
        logger.debug(f"Regime identified: {regime} (vol_ratio: {current_vol/vol_pctl:.2f}, trend: {trend_strength:.2f})")
        
        return regime

    def get_regime_weights(self) -> Dict[str, float]:
        """Get weights for current regime.
        
        Returns
        -------
        weights : dict
            Model name -> weight (sum to 1.0)
        """
        if self.current_regime is None:
            # Return equal weights if regime not yet identified
            n_models = len(self.base_weights) if self.base_weights else 5
            return {f'model_{i}': 1.0 / n_models for i in range(n_models)}
        
        if self.current_regime == 'low_vol_trending':
            # XGBoost and LightGBM excel in trending markets
            return {
                'xgb': 0.35,
                'lgb': 0.30,
                'cat': 0.15,
                'rf': 0.15,
                'gb': 0.05,
            }
        elif self.current_regime == 'high_vol_trending':
            # More robust models for volatile trending
            return {
                'xgb': 0.25,
                'lgb': 0.25,
                'cat': 0.20,
                'rf': 0.20,
                'gb': 0.10,
            }
        elif self.current_regime == 'mean_reverting':
            # Random Forest better for mean reversion
            return {
                'xgb': 0.20,
                'lgb': 0.20,
                'cat': 0.20,
                'rf': 0.30,
                'gb': 0.10,
            }
        else:  # choppy
            # Equal weights in uncertain regimes
            return {
                'xgb': 0.20,
                'lgb': 0.20,
                'cat': 0.20,
                'rf': 0.20,
                'gb': 0.20,
            }

    def update_weights_from_performance(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> Dict[str, float]:
        """Update weights based on recent out-of-sample performance.
        
        Parameters
        ----------
        predictions : dict
            Model name -> predictions array
        actuals : array
            Actual values
            
        Returns
        -------
        weights : dict
            Updated weights
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        recent_actuals = actuals[-self.window_size:]
        
        # Calculate performance for each model
        performance = {}
        for model_name, preds in predictions.items():
            recent_preds = preds[-self.window_size:]
            
            mse = mean_squared_error(recent_actuals, recent_preds)
            mae = mean_absolute_error(recent_actuals, recent_preds)
            
            # Lower error = better performance = higher weight
            performance[model_name] = 1.0 / (1.0 + mse)  # Softmax-like
        
        # Normalize to sum to 1.0
        total_perf = sum(performance.values())
        weights = {k: v / total_perf for k, v in performance.items()}
        
        self.performance_history[len(self.performance_history)] = {
            'regime': self.current_regime,
            'weights': weights,
            'performance': performance,
        }
        
        logger.info(f"Weights updated based on {self.window_size}-day performance")
        logger.debug(f"New weights: {weights}")
        
        return weights

    def blend_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Blend predictions from multiple models using weights.
        
        Parameters
        ----------
        predictions : dict
            Model name -> predictions array
        weights : dict, optional
            Model name -> weight. If None, uses regime weights.
            
        Returns
        -------
        blended : array
            Weighted average predictions
        """
        if weights is None:
            weights = self.get_regime_weights()
        
        # Ensure all models in predictions are in weights
        blended = np.zeros_like(next(iter(predictions.values())))
        
        for model_name, preds in predictions.items():
            weight = weights.get(model_name, 0.0)
            blended += weight * preds
        
        return blended

    def adaptive_weighted_prediction(
        self,
        base_predictions: Dict[str, np.ndarray],
        recent_actuals: np.ndarray,
        returns: np.ndarray,
        volatility: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate predictions with adaptive weighting.
        
        Combines regime identification + performance-based weighting.
        
        Parameters
        ----------
        base_predictions : dict
            Model name -> predictions
        recent_actuals : array
            Recent actual values for performance calculation
        returns : array
            Returns for regime identification
        volatility : array
            Volatility for regime identification
            
        Returns
        -------
        predictions : array
            Adaptively weighted predictions
        weights : dict
            Final weights used
        """
        # Identify current regime
        self.identify_regime(returns, volatility)
        
        # Get regime-based weights
        regime_weights = self.get_regime_weights()
        
        # Update with recent performance
        performance_weights = self.update_weights_from_performance(
            base_predictions, recent_actuals
        )
        
        # Blend regime and performance weights (70% regime, 30% performance)
        final_weights = {}
        for model_name in base_predictions.keys():
            regime_w = regime_weights.get(model_name, 0.2)
            perf_w = performance_weights.get(model_name, 0.2)
            final_weights[model_name] = 0.7 * regime_w + 0.3 * perf_w
        
        # Normalize
        total = sum(final_weights.values())
        final_weights = {k: v / total for k, v in final_weights.items()}
        
        # Generate blended predictions
        predictions = self.blend_predictions(base_predictions, final_weights)
        
        logger.info(f"Adaptive weighted predictions generated. Regime: {self.current_regime}")
        
        return predictions, final_weights

    def get_weight_history(self) -> pd.DataFrame:
        """Get historical weight evolution.
        
        Returns
        -------
        history : DataFrame
            Columns: timestamp, regime, model, weight, performance
        """
        records = []
        for ts, entry in self.performance_history.items():
            regime = entry['regime']
            for model_name, weight in entry['weights'].items():
                performance = entry['performance'].get(model_name, np.nan)
                records.append({
                    'timestamp': ts,
                    'regime': regime,
                    'model': model_name,
                    'weight': weight,
                    'performance': performance,
                })
        
        return pd.DataFrame(records)

    def summary(self) -> str:
        """Get summary of dynamic weighting state.
        
        Returns
        -------
        summary : str
        """
        lines = [
            "Dynamic Weighting Summary",
            "=" * 50,
            f"Current Regime: {self.current_regime or 'Not identified'}",
            f"Window Size: {self.window_size} days",
            f"Weight Updates: {len(self.performance_history)}",
        ]
        
        if self.current_regime:
            weights = self.get_regime_weights()
            lines.append(f"\nRegime Weights ({self.current_regime}):")
            for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {model:10s}: {weight:.1%}")
        
        return "\n".join(lines)
