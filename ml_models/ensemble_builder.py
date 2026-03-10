"""Ensemble builder for heterogeneous model stacks."""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None

from .stacking_ensemble import StackingEnsemble

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """Build and train heterogeneous stacking ensembles."""

    def __init__(self, task: str = "regression", random_state: int = 42):
        self.task = task
        self.random_state = random_state
        self.ensemble = None
        logger.info("EnsembleBuilder initialized for %s task", task)

    def build_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_config: Optional[Dict] = None,
        n_folds: int = 5,
        splitter=None,
        meta_learner=None,
    ) -> StackingEnsemble:
        if model_config is None:
            base_learners = self._create_default_base_learners()
        else:
            base_learners = self._create_base_learners_from_config(model_config)

        if not base_learners:
            raise ValueError("No base learners available for ensemble construction")

        default_meta = meta_learner or (
            LogisticRegression(max_iter=1000) if self.task == "classification" else Ridge(alpha=1.0)
        )
        self.ensemble = StackingEnsemble(
            base_learners=base_learners,
            meta_learner=default_meta,
            random_state=self.random_state,
            n_folds=n_folds,
            task=self.task,
        )
        self.ensemble.fit(X_train, y_train, splitter=splitter)
        logger.info("Ensemble trained on %d samples", X_train.shape[0])
        return self.ensemble

    def evaluate_ensemble(
        self,
        ensemble: StackingEnsemble,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            log_loss,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
        )

        predictions = ensemble.predict(X_test)
        metrics = {}

        if self.task == "classification":
            pred_binary = np.asarray(predictions)
            metrics["accuracy"] = float(accuracy_score(y_test, pred_binary))
            metrics["precision"] = float(
                precision_score(y_test, pred_binary, average="weighted", zero_division=0)
            )
            if hasattr(ensemble.meta_learner, "predict_proba"):
                metrics["log_loss"] = float(log_loss(y_test, ensemble.predict_proba(X_test)))
        else:
            metrics["mae"] = float(mean_absolute_error(y_test, predictions))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, predictions)))
            metrics["r2_score"] = float(r2_score(y_test, predictions))

        return metrics

    def compare_base_learners(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        if self.ensemble is None:
            raise ValueError("Ensemble must be built first")

        from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, r2_score

        results = []
        base_preds = self.ensemble.get_base_predictions(X_test)

        for col in base_preds.columns:
            pred = base_preds[col].values
            if self.task == "classification":
                discrete = np.rint(pred).astype(int)
                results.append(
                    {
                        "Model": col.upper(),
                        "Accuracy": accuracy_score(y_test, discrete),
                        "Precision": precision_score(y_test, discrete, average="weighted", zero_division=0),
                    }
                )
            else:
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)
                mae = np.mean(np.abs(y_test - pred))
                results.append({"Model": col.upper(), "RMSE": rmse, "MAE": mae, "R2": r2})

        return pd.DataFrame(results)

    def _create_default_base_learners(self) -> Dict:
        return (
            self._create_default_classification_learners()
            if self.task == "classification"
            else self._create_default_regression_learners()
        )

    def _create_default_regression_learners(self) -> Dict:
        learners = {}
        if XGBRegressor is not None:
            learners["xgb"] = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0,
            )
        if LGBMRegressor is not None:
            learners["lgb"] = LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
        if CatBoostRegressor is not None:
            learners["cat"] = CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_state=self.random_state,
                verbose=False,
            )
        learners["rf"] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        learners["gb"] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            random_state=self.random_state,
        )
        return learners

    def _create_default_classification_learners(self) -> Dict:
        learners = {}
        if XGBClassifier is not None:
            learners["xgb"] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0,
                eval_metric="logloss",
            )
        if LGBMClassifier is not None:
            learners["lgb"] = LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
        if CatBoostClassifier is not None:
            learners["cat"] = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_state=self.random_state,
                verbose=False,
            )
        learners["rf"] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        learners["gb"] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            random_state=self.random_state,
        )
        return learners

    def _create_base_learners_from_config(self, config: Dict) -> Dict:
        learners = {}
        if self.task == "classification":
            if "xgb" in config and XGBClassifier is not None:
                learners["xgb"] = XGBClassifier(**config["xgb"], random_state=self.random_state)
            if "lgb" in config and LGBMClassifier is not None:
                learners["lgb"] = LGBMClassifier(**config["lgb"], random_state=self.random_state)
            if "cat" in config and CatBoostClassifier is not None:
                learners["cat"] = CatBoostClassifier(**config["cat"], random_state=self.random_state)
            if "rf" in config:
                learners["rf"] = RandomForestClassifier(**config["rf"], random_state=self.random_state)
            if "gb" in config:
                learners["gb"] = GradientBoostingClassifier(**config["gb"], random_state=self.random_state)
            return learners

        if "xgb" in config and XGBRegressor is not None:
            learners["xgb"] = XGBRegressor(**config["xgb"], random_state=self.random_state)
        if "lgb" in config and LGBMRegressor is not None:
            learners["lgb"] = LGBMRegressor(**config["lgb"], random_state=self.random_state)
        if "cat" in config and CatBoostRegressor is not None:
            learners["cat"] = CatBoostRegressor(**config["cat"], random_state=self.random_state)
        if "rf" in config:
            learners["rf"] = RandomForestRegressor(**config["rf"], random_state=self.random_state)
        if "gb" in config:
            learners["gb"] = GradientBoostingRegressor(**config["gb"], random_state=self.random_state)
        return learners
