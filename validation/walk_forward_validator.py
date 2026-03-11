"""Walk-forward validation wrappers built on shared time-series CV utilities."""

from __future__ import annotations

import logging
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from .time_series_cv import PurgedWalkForwardSplitter, TimeSeriesFold, TimeSeriesSplitConfig

logger = logging.getLogger(__name__)

ValidationFold = TimeSeriesFold


class WalkForwardValidator:
    """Backward-compatible walk-forward validator."""

    def __init__(
        self,
        min_train_days: int = 252,
        val_days: int = 60,
        step_days: int = 20,
        mode: str = "expanding",
        *,
        initial_window: int | None = None,
        forecast_horizon: int | None = None,
        step_size: int | None = None,
        gap_days: int = 0,
        embargo_days: int = 0,
        max_train_days: int | None = None,
    ):
        self.min_train_days = initial_window or min_train_days
        self.val_days = forecast_horizon or val_days
        self.step_days = step_size or step_days
        self.mode = mode
        self.gap_days = gap_days
        self.embargo_days = embargo_days
        self.max_train_days = max_train_days
        self.splitter = PurgedWalkForwardSplitter(
            TimeSeriesSplitConfig(
                min_train_size=self.min_train_days,
                test_size=self.val_days,
                step_size=self.step_days,
                gap_size=self.gap_days,
                embargo_size=self.embargo_days,
                mode=self.mode,
                max_train_size=self.max_train_days,
            )
        )

    def split(self, df: pd.DataFrame, date_col: str = "Date") -> List[ValidationFold]:
        folds = self.splitter.split(df, date_col=date_col)
        logger.info("Generated %d walk-forward folds", len(folds))
        return folds

    def get_train_val_data(
        self,
        df: pd.DataFrame,
        fold: ValidationFold,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset = df.copy()
        dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
        dataset = dataset.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        train_df = dataset.iloc[fold.train_idx].copy()
        val_df = dataset.iloc[fold.val_idx].copy()
        return train_df, val_df

    def cross_validate(
        self,
        df: pd.DataFrame,
        model_fn: Callable,
        eval_fn: Callable,
        date_col: str = "Date",
    ) -> dict:
        folds = self.split(df, date_col=date_col)
        fold_scores = []
        fold_results = []

        for fold in folds:
            logger.info("Running %s", fold)
            train_df, val_df = self.get_train_val_data(df, fold)
            model = model_fn(train_df)
            score = eval_fn(model, val_df)
            fold_scores.append(float(score))
            fold_results.append(
                {
                    "fold": fold.fold_idx,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "val_start": fold.val_start,
                    "val_end": fold.val_end,
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                    "score": float(score),
                }
            )

        scores = np.asarray(fold_scores, dtype=float)
        if scores.size == 0:
            return {
                "folds": [],
                "scores": [],
                "mean_score": np.nan,
                "std_score": np.nan,
                "min_score": np.nan,
                "max_score": np.nan,
            }

        results = {
            "folds": fold_results,
            "scores": scores.tolist(),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
        }
        logger.info(
            "Walk-forward results: mean=%.4f std=%.4f range=[%.4f, %.4f]",
            results["mean_score"],
            results["std_score"],
            results["min_score"],
            results["max_score"],
        )
        return results


class ExpandingWindowValidator(WalkForwardValidator):
    """Expanding window only."""

    def __init__(self, min_train_days: int = 252, val_days: int = 60, step_days: int = 20, **kwargs):
        super().__init__(
            min_train_days=min_train_days,
            val_days=val_days,
            step_days=step_days,
            mode="expanding",
            **kwargs,
        )


class RollingWindowValidator(WalkForwardValidator):
    """Rolling window with a fixed-size training window."""

    def __init__(self, train_days: int = 252, val_days: int = 60, step_days: int = 20, **kwargs):
        super().__init__(
            min_train_days=train_days,
            val_days=val_days,
            step_days=step_days,
            mode="rolling",
            max_train_days=train_days,
            **kwargs,
        )
