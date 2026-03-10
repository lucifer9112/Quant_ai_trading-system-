"""Stacking ensemble framework for heterogeneous base learners."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
import logging
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Stack heterogeneous models for improved predictions.
    
    Architecture:
    - Level 0: Base learners (XGBoost, LightGBM, Random Forest, etc.)
    - Level 1: Meta-learner (Ridge regression) learns optimal weighting
    
    Expected improved accuracy: 50% → 60-65%
    """

    def __init__(
        self,
        base_learners: Dict[str, Any],
        meta_learner=None,
        random_state: int = 42,
        n_folds: int = 5,
    ):
        """Initialize stacking ensemble.
        
        Parameters
        ----------
        base_learners : dict
            Name -> model pairs: {'xgb': XGBRegressor(), 'lgb': LGBMRegressor()}
        meta_learner : model, optional
            Meta-learner for combining predictions. Default: Ridge(alpha=1.0)
        random_state : int
            Random seed for reproducibility
        n_folds : int
            Cross-validation folds for training meta-learner
        """
        self.base_learners = base_learners
        self.meta_learner = meta_learner or Ridge(alpha=1.0)
        self.random_state = random_state
        self.n_folds = n_folds
        self.scaler = StandardScaler()
        self.trained = False
        self.training_date = None
        
        logger.info(f"StackingEnsemble initialized with {len(base_learners)} base learners")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """Train stacking ensemble.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Target variable (returns or binary signals)
            
        Returns
        -------
        self
        """
        logger.info(f"Training stacking ensemble on {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train all base learners
        for name, learner in self.base_learners.items():
            logger.info(f"Training base learner: {name}")
            learner.fit(X, y)
        
        # Generate meta-features via cross-validation
        logger.info("Generating meta-features for level 1 learner")
        meta_features = self._generate_meta_features(X, y)
        
        # Train meta-learner on meta-features
        logger.info("Training meta-learner (Ridge regression)")
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        self.meta_learner.fit(meta_features_scaled, y)
        
        self.trained = True
        self.training_date = datetime.now()
        
        logger.info("StackingEnsemble training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using stacking ensemble.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Features to predict
            
        Returns
        -------
        predictions : array, shape (n_samples,)
            Predicted values
        """
        if not self.trained:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all base learners
        meta_features = np.zeros((X.shape[0], len(self.base_learners)))
        for i, (name, learner) in enumerate(self.base_learners.items()):
            meta_features[:, i] = learner.predict(X)
        
        # Scale and predict with meta-learner
        meta_features_scaled = self.scaler.transform(meta_features)
        predictions = self.meta_learner.predict(meta_features_scaled)
        
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions (for classification).
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Features to predict
            
        Returns
        -------
        proba : array, shape (n_samples, 2)
            Probability of class 0 and 1
        """
        if not self.trained:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = self.predict(X)
        # Convert regression output to probabilities
        proba_positive = 1.0 / (1.0 + np.exp(-predictions))  # Sigmoid
        proba_negative = 1.0 - proba_positive
        
        return np.column_stack([proba_negative, proba_positive])

    def get_base_predictions(self, X: np.ndarray) -> pd.DataFrame:
        """Get predictions from each base learner.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Features to predict
            
        Returns
        -------
        predictions : DataFrame
            Column per base learner with predictions
        """
        base_preds = {}
        for name, learner in self.base_learners.items():
            base_preds[name] = learner.predict(X)
        
        return pd.DataFrame(base_preds)

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from base learners.
        
        Returns
        -------
        importance : dict
            Structure: {'xgb': {'feature_0': 0.1, ...}, 'lgb': {...}, ...}
        """
        importance = {}
        
        for name, learner in self.base_learners.items():
            if hasattr(learner, 'feature_importances_'):
                importance[name] = dict(enumerate(learner.feature_importances_))
            elif hasattr(learner, 'coef_'):
                importance[name] = dict(enumerate(np.abs(learner.coef_)))
            else:
                importance[name] = {'unknown': 0}
        
        return importance

    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate meta-features via cross-validation predictions.
        
        This ensures the meta-learner doesn't overfit to in-sample predictions.
        """
        meta_features = np.zeros((X.shape[0], len(self.base_learners)))
        
        for i, (name, learner) in enumerate(self.base_learners.items()):
            # Use cross-validation predictions for meta-features
            meta_features[:, i] = cross_val_predict(
                learner, X, y, cv=self.n_folds, method='predict'
            )
        
        return meta_features

    def save(self, filepath: str) -> None:
        """Save ensemble to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save ensemble (.pkl)
        """
        if not self.trained:
            logger.warning("Saving untrained ensemble")
        
        state = {
            'base_learners': self.base_learners,
            'meta_learner': self.meta_learner,
            'scaler': self.scaler,
            'trained': self.trained,
            'training_date': self.training_date,
            'random_state': self.random_state,
            'n_folds': self.n_folds,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Ensemble saved to {filepath}")

    def load(self, filepath: str) -> "StackingEnsemble":
        """Load ensemble from disk.
        
        Parameters
        ----------
        filepath : str
            Path to load ensemble (.pkl)
            
        Returns
        -------
        self
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.base_learners = state['base_learners']
        self.meta_learner = state['meta_learner']
        self.scaler = state['scaler']
        self.trained = state['trained']
        self.training_date = state['training_date']
        self.random_state = state['random_state']
        self.n_folds = state['n_folds']
        
        logger.info(f"Ensemble loaded from {filepath}")
        return self

    def summary(self) -> str:
        """Get human-readable summary of ensemble.
        
        Returns
        -------
        summary : str
            Summary statistics
        """
        lines = [
            f"StackingEnsemble Summary",
            f"{'=' * 50}",
            f"Status: {'Trained' if self.trained else 'Not trained'}",
            f"Training Date: {self.training_date}",
            f"Base Learners: {len(self.base_learners)}",
        ]
        
        for name in self.base_learners.keys():
            lines.append(f"  - {name}")
        
        lines.extend([
            f"Meta-learner: Ridge",
            f"Cross-validation Folds: {self.n_folds}",
            f"Feature Scaling: StandardScaler",
        ])
        
        return "\n".join(lines)
