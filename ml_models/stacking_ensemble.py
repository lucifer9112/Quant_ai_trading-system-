"""Time-series-safe stacking ensemble framework."""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Stack heterogeneous models using time-series out-of-fold predictions."""

    def __init__(
        self,
        base_learners: Dict[str, Any],
        meta_learner=None,
        random_state: int = 42,
        n_folds: int = 5,
        task: str = "regression",
        use_probabilities: Optional[bool] = None,
    ):
        self.base_learners = base_learners
        self.meta_learner = meta_learner or (
            LogisticRegression(max_iter=1000) if task == "classification" else Ridge(alpha=1.0)
        )
        self.random_state = random_state
        self.n_folds = n_folds
        self.task = task
        self.use_probabilities = task == "classification" if use_probabilities is None else use_probabilities
        self.scaler = StandardScaler()
        self.trained = False
        self.training_date = None
        self.fitted_base_learners: Dict[str, Any] = {}
        self.meta_feature_names_: List[str] = []

        logger.info("StackingEnsemble initialized with %d base learners", len(base_learners))

    def fit(self, X: np.ndarray, y: np.ndarray, splitter=None) -> "StackingEnsemble":
        logger.info("Training stacking ensemble on %d samples", X.shape[0])
        X_array = np.asarray(X)
        y_array = np.asarray(y)

        meta_features = self._generate_meta_features(X_array, y_array, splitter=splitter)
        valid_rows = ~np.isnan(meta_features).any(axis=1)
        if not valid_rows.any():
            raise ValueError("Unable to build out-of-fold meta-features for stacking ensemble")

        meta_features_scaled = self.scaler.fit_transform(meta_features[valid_rows])
        self.meta_learner.fit(meta_features_scaled, y_array[valid_rows])

        self.fitted_base_learners = {}
        for name, learner in self.base_learners.items():
            fitted = clone(learner)
            fitted.fit(X_array, y_array)
            self.fitted_base_learners[name] = fitted

        self.trained = True
        self.training_date = datetime.now()
        logger.info("StackingEnsemble training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise ValueError("Ensemble must be fitted before prediction")

        meta_features = self._compose_meta_features(np.asarray(X))
        meta_features_scaled = self.scaler.transform(meta_features)
        predictions = self.meta_learner.predict(meta_features_scaled)
        return np.asarray(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise ValueError("Ensemble must be fitted before prediction")

        meta_features = self._compose_meta_features(np.asarray(X))
        meta_features_scaled = self.scaler.transform(meta_features)

        if hasattr(self.meta_learner, "predict_proba"):
            return np.asarray(self.meta_learner.predict_proba(meta_features_scaled))

        predictions = np.asarray(self.meta_learner.predict(meta_features_scaled))
        if predictions.ndim == 1:
            proba_positive = 1.0 / (1.0 + np.exp(-predictions))
            return np.column_stack([1.0 - proba_positive, proba_positive])

        shifted = predictions - predictions.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / exp_values.sum(axis=1, keepdims=True)

    def get_base_predictions(self, X: np.ndarray) -> pd.DataFrame:
        meta_features = self._compose_meta_features(np.asarray(X))
        return pd.DataFrame(meta_features, columns=self.meta_feature_names_)

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        importance = {}
        for name, learner in self.fitted_base_learners.items():
            if hasattr(learner, "feature_importances_"):
                importance[name] = dict(enumerate(np.asarray(learner.feature_importances_, dtype=float)))
            elif hasattr(learner, "coef_"):
                coefficients = np.asarray(learner.coef_, dtype=float)
                if coefficients.ndim > 1:
                    coefficients = np.abs(coefficients).mean(axis=0)
                importance[name] = dict(enumerate(np.abs(coefficients)))
            else:
                importance[name] = {"unknown": 0.0}
        return importance

    def save(self, filepath: str) -> None:
        state = {
            "base_learners": self.base_learners,
            "meta_learner": self.meta_learner,
            "scaler": self.scaler,
            "trained": self.trained,
            "training_date": self.training_date,
            "random_state": self.random_state,
            "n_folds": self.n_folds,
            "task": self.task,
            "use_probabilities": self.use_probabilities,
            "fitted_base_learners": self.fitted_base_learners,
            "meta_feature_names_": self.meta_feature_names_,
        }
        with open(filepath, "wb") as handle:
            pickle.dump(state, handle)
        logger.info("Ensemble saved to %s", filepath)

    def load(self, filepath: str) -> "StackingEnsemble":
        with open(filepath, "rb") as handle:
            state = pickle.load(handle)

        self.base_learners = state["base_learners"]
        self.meta_learner = state["meta_learner"]
        self.scaler = state["scaler"]
        self.trained = state["trained"]
        self.training_date = state["training_date"]
        self.random_state = state["random_state"]
        self.n_folds = state["n_folds"]
        self.task = state["task"]
        self.use_probabilities = state["use_probabilities"]
        self.fitted_base_learners = state["fitted_base_learners"]
        self.meta_feature_names_ = state["meta_feature_names_"]
        logger.info("Ensemble loaded from %s", filepath)
        return self

    def summary(self) -> str:
        lines = [
            "StackingEnsemble Summary",
            "=" * 50,
            f"Status: {'Trained' if self.trained else 'Not trained'}",
            f"Task: {self.task}",
            f"Training Date: {self.training_date}",
            f"Base Learners: {len(self.base_learners)}",
        ]
        for name in self.base_learners.keys():
            lines.append(f"  - {name}")
        lines.extend(
            [
                f"Meta-learner: {self.meta_learner.__class__.__name__}",
                f"Cross-validation Folds: {self.n_folds}",
                f"Feature Scaling: StandardScaler",
            ]
        )
        return "\n".join(lines)

    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray, splitter=None) -> np.ndarray:
        splits = self._resolve_splits(X, y, splitter=splitter)
        meta_blocks = {}

        for name, learner in self.base_learners.items():
            meta_blocks[name] = None

        feature_names: List[str] = []
        for train_idx, val_idx in splits:
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            for name, learner in self.base_learners.items():
                fold_model = clone(learner)
                fold_model.fit(X_train, y_train)
                prediction_matrix, columns = self._prediction_matrix(fold_model, X_val, prefix=name)

                if meta_blocks[name] is None:
                    meta_blocks[name] = np.full((len(X), prediction_matrix.shape[1]), np.nan)
                    feature_names.extend(columns)

                meta_blocks[name][val_idx, :] = prediction_matrix

        self.meta_feature_names_ = feature_names
        if not meta_blocks:
            return np.empty((len(X), 0))

        return np.concatenate([meta_blocks[name] for name in self.base_learners], axis=1)

    def _compose_meta_features(self, X: np.ndarray) -> np.ndarray:
        matrices = []
        feature_names: List[str] = []
        for name, learner in self.fitted_base_learners.items():
            prediction_matrix, columns = self._prediction_matrix(learner, X, prefix=name)
            matrices.append(prediction_matrix)
            feature_names.extend(columns)
        if feature_names:
            self.meta_feature_names_ = feature_names
        return np.concatenate(matrices, axis=1)

    def _prediction_matrix(self, learner, X: np.ndarray, *, prefix: str):
        if self.task == "classification" and self.use_probabilities and hasattr(learner, "predict_proba"):
            predictions = np.asarray(learner.predict_proba(X), dtype=float)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            if predictions.shape[1] == 2:
                return predictions[:, [1]], [f"{prefix}_proba_positive"]
            columns = [f"{prefix}_class_{index}" for index in range(predictions.shape[1])]
            return predictions, columns

        predictions = np.asarray(learner.predict(X), dtype=float)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        columns = [f"{prefix}_prediction_{index}" for index in range(predictions.shape[1])]
        return predictions, columns

    def _resolve_splits(self, X: np.ndarray, y: np.ndarray, splitter=None):
        if splitter is None:
            n_splits = min(self.n_folds, max(2, len(X) - 1))
            return list(TimeSeriesSplit(n_splits=n_splits).split(X, y))

        if isinstance(splitter, list):
            return [(fold.train_idx, fold.val_idx) for fold in splitter]

        if hasattr(splitter, "split"):
            try:
                return list(splitter.split(X, y))
            except TypeError:
                return list(splitter.split(X))

        raise ValueError("Unsupported splitter supplied to StackingEnsemble.fit")
