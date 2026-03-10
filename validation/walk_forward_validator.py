"""
Walk-Forward Validation Framework

Implements time-series cross-validation with proper time boundaries:
- Expanding window (train on [0:t], validate on [t:t+k])
- Rolling window (train on [t-n:t], validate on [t:t+k])
- Anchored windows (fixed train start, expanding test)

Prevents look-ahead bias and measures true generalization.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationFold:
    """Single train-validation fold with time boundaries"""
    fold_idx: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    
    def __repr__(self):
        return (
            f"Fold {self.fold_idx}: "
            f"Train [{self.train_start}:{self.train_end}] "
            f"→ Val [{self.val_start}:{self.val_end}]"
        )


class WalkForwardValidator:
    """
    Time-series aware cross-validation.
    
    Ensures:
    - No training data leaks into validation
    - Proper temporal ordering
    - Realistic evaluation of model generalization
    """
    
    def __init__(
        self,
        min_train_days: int = 252,  # ~1 year of trading days
        val_days: int = 60,  # ~3 months
        step_days: int = 20,  # Roll forward 1 month at a time
        mode: str = "expanding",  # expanding, rolling
    ):
        """
        Args:
            min_train_days: Minimum training set size (trading days)
            val_days: Validation set size (trading days)
            step_days: Days to advance window for next fold
            mode: 'expanding' (train grows) or 'rolling' (fixed window size)
        """
        self.min_train_days = min_train_days
        self.val_days = val_days
        self.step_days = step_days
        self.mode = mode
        
    def split(self, df: pd.DataFrame, date_col: str = "Date") -> List[ValidationFold]:
        """
        Generate walk-forward folds from time-indexed dataframe.
        
        Args:
            df: DataFrame with Date column
            date_col: Name of date column
            
        Returns:
            List of ValidationFold objects with train/val indices
        """
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in dataframe")
        
        # Ensure dates are sorted
        df = df.sort_values(date_col).reset_index(drop=True)
        dates = pd.to_datetime(df[date_col])
        
        folds = []
        train_start_idx = 0
        val_start_idx = self.min_train_days
        
        fold_idx = 0
        while val_start_idx + self.val_days <= len(df):
            val_end_idx = val_start_idx + self.val_days
            
            # Define fold boundaries
            train_indices = np.arange(train_start_idx, val_start_idx)
            val_indices = np.arange(val_start_idx, val_end_idx)
            
            fold = ValidationFold(
                fold_idx=fold_idx,
                train_start=str(dates.iloc[train_start_idx].date()),
                train_end=str(dates.iloc[val_start_idx - 1].date()),
                val_start=str(dates.iloc[val_start_idx].date()),
                val_end=str(dates.iloc[val_end_idx - 1].date()),
                train_idx=train_indices,
                val_idx=val_indices,
            )
            
            folds.append(fold)
            logger.info(f"Created {fold}")
            
            # Advance to next fold
            if self.mode == "expanding":
                # Train window grows
                val_start_idx += self.step_days
            else:  # rolling
                # Fixed-size train window
                train_start_idx += self.step_days
                val_start_idx += self.step_days
            
            fold_idx += 1
        
        logger.info(f"Generated {len(folds)} walk-forward folds")
        return folds
    
    def get_train_val_data(
        self,
        df: pd.DataFrame,
        fold: ValidationFold,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract train and validation data for a fold.
        
        Args:
            df: Full dataframe
            fold: ValidationFold object
            
        Returns:
            (train_df, val_df) with no temporal overlap
        """
        train_df = df.iloc[fold.train_idx].copy()
        val_df = df.iloc[fold.val_idx].copy()
        
        return train_df, val_df
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        model_fn: Callable,
        eval_fn: Callable,
        date_col: str = "Date",
    ) -> dict:
        """
        Run walk-forward cross-validation.
        
        Args:
            df: Full dataframe
            model_fn: Function that trains model: model = model_fn(train_df)
            eval_fn: Function that evaluates: score = eval_fn(model, val_df)
            date_col: Date column name
            
        Returns:
            Dictionary with fold-wise and aggregate metrics
        """
        folds = self.split(df, date_col)
        
        fold_scores = []
        fold_results = []
        
        for fold in folds:
            logger.info(f"Running {fold}")
            
            train_df, val_df = self.get_train_val_data(df, fold)
            
            # Train model
            model = model_fn(train_df)
            
            # Evaluate on validation set
            score = eval_fn(model, val_df)
            fold_scores.append(score)
            
            fold_results.append({
                "fold": fold.fold_idx,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "val_start": fold.val_start,
                "val_end": fold.val_end,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "score": score,
            })
        
        # Aggregate metrics
        fold_scores = np.array(fold_scores)
        
        results = {
            "folds": fold_results,
            "scores": fold_scores.tolist(),
            "mean_score": float(fold_scores.mean()),
            "std_score": float(fold_scores.std()),
            "min_score": float(fold_scores.min()),
            "max_score": float(fold_scores.max()),
        }
        
        logger.info(
            f"Walk-Forward Results: "
            f"Mean Score={results['mean_score']:.4f} ± {results['std_score']:.4f}, "
            f"Range=[{results['min_score']:.4f}, {results['max_score']:.4f}]"
        )
        
        return results


class ExpandingWindowValidator(WalkForwardValidator):
    """Expanding window only."""
    
    def __init__(self, min_train_days: int = 252, val_days: int = 60, step_days: int = 20):
        super().__init__(
            min_train_days=min_train_days,
            val_days=val_days,
            step_days=step_days,
            mode="expanding",
        )


class RollingWindowValidator(WalkForwardValidator):
    """Rolling window (fixed train size, advancing)."""
    
    def __init__(self, train_days: int = 252, val_days: int = 60, step_days: int = 20):
        super().__init__(
            min_train_days=train_days,
            val_days=val_days,
            step_days=step_days,
            mode="rolling",
        )
