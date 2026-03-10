"""Ensemble builder for constructing heterogeneous model stacks."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# optional ML libraries - guard imports so missing packages don't break module import
try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - may not be installed in some environments
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None
import warnings

from .stacking_ensemble import StackingEnsemble

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """Build and configure heterogeneous stacking ensembles.
    
    Combines multiple model types:
    - XGBoost (gradient boosting)
    - LightGBM (light gradient boosting)
    - CatBoost (categorical features)
    - Random Forest (ensemble)
    - Gradient Boosting (sklearn)
    
    Expected behavior:
    - Better generalization through model diversity
    - Lower variance via ensemble averaging
    - Target: 60-65% accuracy from Phase 1's 55-58%
    """

    def __init__(self, task: str = 'regression', random_state: int = 42):
        """Initialize ensemble builder.
        
        Parameters
        ----------
        task : {'regression', 'classification'}
            Task type for base learner configuration
        random_state : int
            Random seed for reproducibility
        """
        self.task = task
        self.random_state = random_state
        self.base_learners = None
        self.ensemble = None
        
        logger.info(f"EnsembleBuilder initialized for {task} task")

    def build_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_config: Optional[Dict] = None,
        n_folds: int = 5,
    ) -> StackingEnsemble:
        """Build and train stacking ensemble.
        
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training features
        y_train : array-like, shape (n_samples,)
            Training targets
        model_config : dict, optional
            Configuration for base learners. If None, uses defaults.
        n_folds : int
            Cross-validation folds for meta-learner training
            
        Returns
        -------
        ensemble : StackingEnsemble
            Trained stacking ensemble
        """
        # Create base learners
        if model_config is None:
            base_learners = self._create_default_base_learners()
        else:
            base_learners = self._create_base_learners_from_config(model_config)
        
        # Create and train ensemble
        self.ensemble = StackingEnsemble(
            base_learners=base_learners,
            random_state=self.random_state,
            n_folds=n_folds,
        )
        
        self.ensemble.fit(X_train, y_train)
        logger.info(f"Ensemble trained on {X_train.shape[0]} samples")
        
        return self.ensemble

    def evaluate_ensemble(
        self,
        ensemble: StackingEnsemble,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate ensemble performance on test set.
        
        Parameters
        ----------
        ensemble : StackingEnsemble
            Trained ensemble
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
            
        Returns
        -------
        metrics : dict
            Performance metrics: mae, rmse, r2_score, accuracy (for classification)
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
        
        predictions = ensemble.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2_score': r2_score(y_test, predictions),
        }
        
        # For classification tasks
        if self.task == 'classification':
            pred_binary = (predictions > 0.5).astype(int)
            metrics['accuracy'] = accuracy_score(y_test, pred_binary)
        
        logger.info(f"Ensemble evaluation - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2_score']:.4f}")
        
        return metrics

    def compare_base_learners(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Compare individual base learner performance.
        
        Parameters
        ----------
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
            
        Returns
        -------
        comparison : DataFrame
            Performance comparison of base learners
        """
        if self.ensemble is None:
            raise ValueError("Ensemble must be built first")
        
        from sklearn.metrics import mean_squared_error, r2_score
        
        results = []
        base_preds = self.ensemble.get_base_predictions(X_test)
        
        for col in base_preds.columns:
            pred = base_preds[col].values
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            mae = np.mean(np.abs(y_test - pred))
            
            results.append({
                'Model': col.upper(),
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
            })
        
        # Add ensemble results
        ensemble_pred = self.ensemble.predict(X_test)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = np.mean(np.abs(y_test - ensemble_pred))
        
        results.append({
            'Model': 'ENSEMBLE',
            'RMSE': ensemble_rmse,
            'MAE': ensemble_mae,
            'R2': ensemble_r2,
        })
        
        df = pd.DataFrame(results).sort_values('R2', ascending=False)
        logger.info(f"\nBase learner comparison:\n{df.to_string()}")
        
        return df

    def get_optimal_config(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Find optimal hyperparameter configuration via grid search.
        
        Parameters
        ----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
            
        Returns
        -------
        config : dict
            Optimal configuration for base learners
        """
        # This would involve hyperparameter tuning
        # For now, return defaults
        logger.info("Optimal configuration search not yet implemented")
        return self._create_default_base_learners()

    def _create_default_base_learners(self) -> Dict:
        """Create default heterogeneous base learners.
        
        Returns
        -------
        base_learners : dict
            Name -> model pairs with reasonable defaults
        """
        learners = {}
        # add optional base learners only if the corresponding library was imported
        if XGBRegressor is not None:
            learners['xgb'] = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0,
            )
        else:
            logger.warning("XGBoost not available; skipping xgb base learner")

        if LGBMRegressor is not None:
            learners['lgb'] = LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            logger.warning("LightGBM not available; skipping lgb base learner")

        if CatBoostRegressor is not None:
            learners['cat'] = CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_state=self.random_state,
                verbose=False,
            )
        else:
            logger.warning("CatBoost not available; skipping cat base learner")

        learners['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        learners['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=self.random_state,
        )
        
        logger.info(f"Created {len(learners)} default base learners: {', '.join(learners.keys())}")
        return learners

    def _create_base_learners_from_config(self, config: Dict) -> Dict:
        """Create base learners from configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Configuration for each learner
            
        Returns
        -------
        base_learners : dict
            Configured learners
        """
        learners = {}
        
        if 'xgb' in config:
            learners['xgb'] = XGBRegressor(**config['xgb'], random_state=self.random_state)
        
        if 'lgb' in config:
            learners['lgb'] = LGBMRegressor(**config['lgb'], random_state=self.random_state)
        
        if 'cat' in config:
            learners['cat'] = CatBoostRegressor(**config['cat'], random_state=self.random_state)
        
        if 'rf' in config:
            learners['rf'] = RandomForestRegressor(**config['rf'], random_state=self.random_state)
        
        if 'gb' in config:
            learners['gb'] = GradientBoostingRegressor(**config['gb'], random_state=self.random_state)
        
        logger.info(f"Created {len(learners)} configured base learners")
        return learners
