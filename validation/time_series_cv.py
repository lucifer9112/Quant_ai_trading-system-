"""Time-series cross-validation utilities for panel and single-asset data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplitConfig:
    """Configuration for time-series cross-validation."""

    min_train_size: int = 252
    test_size: int = 60
    step_size: int = 20
    gap_size: int = 0
    embargo_size: int = 0
    mode: str = "expanding"
    max_train_size: Optional[int] = None


@dataclass
class TimeSeriesFold:
    """Single train-validation fold with resolved row indices."""

    fold_idx: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    gap_size: int = 0
    embargo_size: int = 0

    @property
    def train_start_date(self) -> str:
        return self.train_start

    @property
    def train_end_date(self) -> str:
        return self.train_end

    @property
    def val_start_date(self) -> str:
        return self.val_start

    @property
    def val_end_date(self) -> str:
        return self.val_end

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_idx}: "
            f"Train [{self.train_start}:{self.train_end}] -> "
            f"Val [{self.val_start}:{self.val_end}]"
        )


class PurgedWalkForwardSplitter:
    """Generate walk-forward folds using unique timestamps.

    Splits are built on unique dates and then expanded back to row indices.
    This keeps multi-asset panel rows from the same date in the same fold.
    """

    def __init__(self, config: Optional[TimeSeriesSplitConfig] = None):
        self.config = config or TimeSeriesSplitConfig()

        if self.config.mode not in {"expanding", "rolling", "anchored"}:
            raise ValueError(
                "mode must be one of {'expanding', 'rolling', 'anchored'}"
            )
        if self.config.min_train_size < 1:
            raise ValueError("min_train_size must be >= 1")
        if self.config.test_size < 1:
            raise ValueError("test_size must be >= 1")
        if self.config.step_size < 1:
            raise ValueError("step_size must be >= 1")
        if self.config.gap_size < 0 or self.config.embargo_size < 0:
            raise ValueError("gap_size and embargo_size must be >= 0")

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "Date",
    ) -> List[TimeSeriesFold]:
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in dataframe")

        ordered = df.copy()
        ordered[date_col] = pd.to_datetime(ordered[date_col], errors="coerce")
        ordered = ordered.dropna(subset=[date_col]).sort_values(date_col).reset_index()

        unique_dates = ordered[date_col].drop_duplicates().to_numpy()
        if len(unique_dates) == 0:
            return []

        folds: List[TimeSeriesFold] = []
        fold_idx = 0
        validation_start = self.config.min_train_size + self.config.gap_size
        max_validation_start = len(unique_dates) - self.config.test_size

        while validation_start <= max_validation_start:
            train_end = validation_start - self.config.gap_size
            if train_end <= 0:
                break

            if self.config.mode in {"expanding", "anchored"}:
                train_start = 0
            else:
                rolling_window = self.config.max_train_size or self.config.min_train_size
                train_start = max(0, train_end - rolling_window)

            train_dates = unique_dates[train_start:train_end]
            validation_end = validation_start + self.config.test_size
            validation_dates = unique_dates[validation_start:validation_end]

            train_mask = ordered[date_col].isin(train_dates)
            val_mask = ordered[date_col].isin(validation_dates)

            train_idx = ordered.loc[train_mask, "index"].to_numpy(dtype=int)
            val_idx = ordered.loc[val_mask, "index"].to_numpy(dtype=int)

            if len(train_idx) == 0 or len(val_idx) == 0:
                break

            folds.append(
                TimeSeriesFold(
                    fold_idx=fold_idx,
                    train_start=str(pd.Timestamp(train_dates[0]).date()),
                    train_end=str(pd.Timestamp(train_dates[-1]).date()),
                    val_start=str(pd.Timestamp(validation_dates[0]).date()),
                    val_end=str(pd.Timestamp(validation_dates[-1]).date()),
                    train_idx=np.sort(train_idx),
                    val_idx=np.sort(val_idx),
                    gap_size=self.config.gap_size,
                    embargo_size=self.config.embargo_size,
                )
            )

            fold_idx += 1
            validation_start += self.config.step_size
            if self.config.embargo_size:
                validation_start += self.config.embargo_size

        return folds

    def get_n_splits(self, df: pd.DataFrame, date_col: str = "Date") -> int:
        return len(self.split(df, date_col=date_col))


class TimeSeriesCrossValidator:
    """Run model training and evaluation over walk-forward folds."""

    def __init__(self, splitter: Optional[PurgedWalkForwardSplitter] = None):
        self.splitter = splitter or PurgedWalkForwardSplitter()

    def evaluate(
        self,
        df: pd.DataFrame,
        train_fn: Callable[[pd.DataFrame], Any],
        predict_fn: Callable[[Any, pd.DataFrame], Sequence[float]],
        metric_fns: Dict[str, Callable[[Sequence[float], Sequence[float]], float]],
        *,
        date_col: str = "Date",
        target_col: str,
    ) -> Dict[str, Any]:
        folds = self.splitter.split(df, date_col=date_col)
        fold_results: List[Dict[str, Any]] = []

        for fold in folds:
            train_df = df.iloc[fold.train_idx].copy()
            val_df = df.iloc[fold.val_idx].copy()
            model = train_fn(train_df)
            predictions = np.asarray(predict_fn(model, val_df))
            actuals = np.asarray(val_df[target_col])

            metrics = {
                name: float(metric(actuals, predictions))
                for name, metric in metric_fns.items()
            }
            fold_results.append(
                {
                    "fold": fold.fold_idx,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "val_start": fold.val_start,
                    "val_end": fold.val_end,
                    "train_size": int(len(train_df)),
                    "val_size": int(len(val_df)),
                    **metrics,
                }
            )

        summary: Dict[str, Any] = {"folds": fold_results}
        for metric_name in metric_fns:
            values = [fold[metric_name] for fold in fold_results]
            summary[f"{metric_name}_mean"] = float(np.mean(values)) if values else np.nan
            summary[f"{metric_name}_std"] = float(np.std(values)) if values else np.nan

        return summary
