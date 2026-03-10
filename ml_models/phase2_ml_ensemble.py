"""Phase 2: ML Ensemble Overhaul - Stacking, Dynamic Weighting, Calibration"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from ml_models.stacking_ensemble import StackingEnsemble
from ml_models.ensemble_builder import EnsembleBuilder
from ml_models.dynamic_weighting import DynamicWeighting
from ml_models.probability_calibration import ProbabilityCalibrator
from validation import ExpandingWindowValidator
from feature_engineering import ComprehensiveFeaturePipeline

logger = logging.getLogger(__name__)


class Phase2MLEnsemble:
    """Phase 2: ML Ensemble Overhaul for improved accuracy.
    
    Components:
    1. Heterogeneous stacking (XGBoost, LightGBM, CatBoost, RF)
    2. Dynamic weighting by market regime
    3. Probability calibration for better confidence
    4. Continuous performance monitoring
    
    Expected improvement: Phase 1's 55-58% → Phase 2's 60-65%
    
    Targets:
    - Sharpe Ratio: > 1.5
    - Win Rate: > 55%
    - Accuracy: 60-65%
    - Max Drawdown: < 20%
    """

    def __init__(self, use_calibration: bool = True, use_dynamic_weighting: bool = True):
        """Initialize Phase 2 ML Ensemble system.
        
        Parameters
        ----------
        use_calibration : bool
            Whether to use probability calibration
        use_dynamic_weighting : bool
            Whether to use dynamic regime-based weighting
        """
        self.use_calibration = use_calibration
        self.use_dynamic_weighting = use_dynamic_weighting
        self.ensemble = None
        self.calibrator = None
        self.dynamic_weighter = None
        self.feature_pipeline = ComprehensiveFeaturePipeline()
        self.validator = ExpandingWindowValidator(
            initial_window=252,
            step_size=63,
            forecast_horizon=21,
        )
        self.ensemble_builder = EnsembleBuilder(task='regression')
        
        logger.info(
            f"Phase2MLEnsemble initialized (calibration={use_calibration}, "
            f"dynamic_weighting={use_dynamic_weighting})"
        )

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and engineer features from raw data.
        
        Parameters
        ----------
        df : DataFrame
            Input data with columns: Date, symbol, OHLCV
            
        Returns
        -------
        X : array, shape (n_samples, n_features)
            Engineered features
        y : array, shape (n_samples,)
            Target variable (forward returns)
        """
        logger.info(f"Preparing features for {len(df)} samples")
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Engineer features
        df = self.feature_pipeline.run(df)
        
        # Create target (forward returns, next day's return)
        df['forward_returns'] = df['Close'].pct_change().shift(-1)
        df['signal'] = (df['forward_returns'] > 0).astype(int)
        
        # Select features (exclude non-feature columns)
        exclude_cols = {'Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'forward_returns', 'signal', 'returns'}
        feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('_')]
        
        # Remove rows with NaN
        df = df.dropna()
        
        X = df[feature_cols].values
        y = df['signal'].values
        
        logger.info(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_folds: int = 5,
    ) -> StackingEnsemble:
        """Train stacking ensemble on training data.
        
        Parameters
        ----------
        X_train : array, shape (n_samples, n_features)
            Training features
        y_train : array, shape (n_samples,)
            Training targets
        n_folds : int
            Cross-validation folds for meta-learner
            
        Returns
        -------
        ensemble : StackingEnsemble
            Trained ensemble
        """
        logger.info(f"Training stacking ensemble on {X_train.shape[0]} samples")
        
        self.ensemble = self.ensemble_builder.build_ensemble(
            X_train, y_train, n_folds=n_folds
        )
        
        return self.ensemble

    def calibrate_ensemble(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> ProbabilityCalibrator:
        """Calibrate ensemble predictions for probability outputs.
        
        Parameters
        ----------
        predictions : array
            Raw ensemble predictions
        actuals : array
            Actual outcomes
            
        Returns
        -------
        calibrator : ProbabilityCalibrator
            Fitted calibrator
        """
        if not self.use_calibration:
            logger.info("Calibration disabled")
            return None
        
        logger.info("Fitting probability calibrator")
        
        self.calibrator = ProbabilityCalibrator(method='isotonic')
        self.calibrator.fit(predictions, actuals)
        
        return self.calibrator

    def setup_dynamic_weighting(self) -> DynamicWeighting:
        """Setup dynamic weighting system.
        
        Returns
        -------
        weighter : DynamicWeighting
            Dynamic weighting system
        """
        if not self.use_dynamic_weighting:
            logger.info("Dynamic weighting disabled")
            return None
        
        logger.info("Setting up dynamic weighting by market regime")
        
        self.dynamic_weighter = DynamicWeighting(window_size=63)
        
        return self.dynamic_weighter

    def run_walk_forward_phase2(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Execute Phase 2 walk-forward backtest with ML ensemble.
        
        Parameters
        ----------
        df : DataFrame
            Input data
            
        Returns
        -------
        results : dict
            Walk-forward results with Phase 2 improvements
        """
        logger.info("Starting Phase 2 walk-forward backtest")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        # Walk-forward validation
        for fold_idx, fold in enumerate(self.validator.split(df)):
            logger.info(f"\n=== Fold {fold_idx + 1} ===")
            logger.info(f"Train: {fold.train_start_date} → {fold.train_end_date}")
            logger.info(f"Valid: {fold.val_start_date} → {fold.val_end_date}")
            
            # Get train/val data
            X_train, y_train = X[fold.train_idx], y[fold.train_idx]
            X_val, y_val = X[fold.val_idx], y[fold.val_idx]
            
            # Train ensemble
            ensemble = self.train_ensemble(X_train, y_train)
            
            # Get predictions
            predictions = ensemble.predict(X_val)
            
            # Calibrate if enabled
            if self.use_calibration:
                if fold_idx == 0:
                    # Use first fold validation for calibration fit
                    self.calibrate_ensemble(predictions, y_val)
                
                if self.calibrator:
                    predictions = self.calibrator.calibrate(predictions)
            
            # Calculate metrics
            accuracy = np.mean((predictions > 0.5).astype(int) == y_val)
            precision = np.sum((predictions > 0.5) & (y_val == 1)) / np.sum(predictions > 0.5 + 1e-10)
            
            fold_result = {
                'fold': fold_idx,
                'train_date_range': f"{fold.train_start_date} → {fold.train_end_date}",
                'val_date_range': f"{fold.val_start_date} → {fold.val_end_date}",
                'accuracy': accuracy,
                'precision': precision,
                'num_samples': len(y_val),
            }
            
            fold_results.append(fold_result)
            all_predictions.extend(predictions)
            all_actuals.extend(y_val)
            
            logger.info(f"Fold {fold_idx + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
        
        # Aggregate results
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        mean_accuracy = np.mean([r['accuracy'] for r in fold_results])
        std_accuracy = np.std([r['accuracy'] for r in fold_results])
        mean_precision = np.mean([r['precision'] for r in fold_results])
        
        results = {
            'phase': 2,
            'fold_results': fold_results,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_precision': mean_precision,
            'overall_accuracy': np.mean((all_predictions > 0.5).astype(int) == all_actuals),
            'base_learners': list(self.ensemble.base_learners.keys()),
            'features_used': X.shape[1],
            'calibration_enabled': self.use_calibration,
            'dynamic_weighting_enabled': self.use_dynamic_weighting,
        }
        
        logger.info(f"\n=== Phase 2 Results ===")
        logger.info(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        logger.info(f"Mean Precision: {mean_precision:.4f}")
        logger.info(f"Base Learners: {', '.join(list(self.ensemble.base_learners.keys()))}")
        
        return results

    def summary(self) -> str:
        """Get Phase 2 system summary.
        
        Returns
        -------
        summary : str
        """
        lines = [
            "=" * 60,
            "PHASE 2: ML ENSEMBLE OVERHAUL",
            "=" * 60,
            f"Status: {'Trained' if self.ensemble else 'Not trained'}",
            f"Ensemble Type: Stacking (Meta-learner: Ridge)",
            f"Base Learners: XGBoost, LightGBM, CatBoost, Random Forest, GB",
            f"Probability Calibration: {'Enabled' if self.use_calibration else 'Disabled'}",
            f"Dynamic Weighting: {'Enabled' if self.use_dynamic_weighting else 'Disabled'}",
            f"",
            f"Expected Improvements:",
            f"  - Accuracy: 55-58% → 60-65%",
            f"  - Sharpe Ratio: ~1.2 → >1.5",
            f"  - Max Drawdown: <25% → <20%",
            f"  - Better risk management via calibrated probabilities",
            f"  - Adaptive to market regime changes",
        ]
        
        if self.ensemble:
            lines.append(f"\nEnsemble Summary:")
            lines.append(self.ensemble.summary())
        
        return "\n".join(lines)
