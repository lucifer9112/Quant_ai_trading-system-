from pathlib import Path

import numpy as np
import pandas as pd

from validation.walk_forward_validator import WalkForwardValidator


class AutoGluonTrainer:

    def __init__(
        self,
        label="target_return",
        model_path="models/autogluon",
        problem_type="multiclass",
        eval_metric="accuracy"
    ):

        self.label = label
        self.model_path = model_path
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self._tabular_predictor = None

    def _predictor_class(self):

        if self._tabular_predictor is None:
            try:
                from autogluon.tabular import TabularPredictor
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "autogluon is not installed. Install it with `pip install autogluon.tabular`."
                ) from exc

            self._tabular_predictor = TabularPredictor

        return self._tabular_predictor

    def train(self, df, time_limit=600, presets="medium_quality_faster_train"):

        if self.label not in df.columns:
            raise ValueError(f"Training dataframe must include label column '{self.label}'.")

        model_dir = Path(self.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        predictor = self._predictor_class()(
            label=self.label,
            path=self.model_path,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric
        )

        predictor.fit(
            train_data=df,
            time_limit=time_limit,
            presets=presets
        )

        return predictor

    def walk_forward_validate(
        self,
        df,
        *,
        date_col="Date",
        time_limit=300,
        presets="medium_quality_faster_train",
        validator=None,
        excluded_columns=None,
    ):
        if date_col not in df.columns:
            raise ValueError(f"Training dataframe must include date column '{date_col}'.")
        if self.label not in df.columns:
            raise ValueError(f"Training dataframe must include label column '{self.label}'.")

        validator = validator or WalkForwardValidator()
        folds = validator.split(df, date_col=date_col)
        excluded = set(excluded_columns or [])
        excluded.add(date_col)

        fold_results = []
        for fold in folds:
            train_df = df.iloc[fold.train_idx].copy()
            val_df = df.iloc[fold.val_idx].copy()

            train_frame = train_df.drop(columns=list(excluded & set(train_df.columns)), errors="ignore")
            val_frame = val_df.drop(columns=list(excluded & set(val_df.columns)), errors="ignore")

            predictor = self._predictor_class()(
                label=self.label,
                path=str(Path(self.model_path) / f"walk_forward_fold_{fold.fold_idx}"),
                problem_type=self.problem_type,
                eval_metric=self.eval_metric,
            )
            predictor.fit(
                train_data=train_frame,
                time_limit=time_limit,
                presets=presets,
            )
            predictions = np.asarray(predictor.predict(val_frame.drop(columns=[self.label], errors="ignore")))
            actuals = np.asarray(val_frame[self.label])
            accuracy = float((predictions == actuals).mean())
            fold_results.append(
                {
                    "fold": fold.fold_idx,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "val_start": fold.val_start,
                    "val_end": fold.val_end,
                    "accuracy": accuracy,
                    "val_size": int(len(val_frame)),
                }
            )

        accuracies = [fold["accuracy"] for fold in fold_results]
        return {
            "folds": fold_results,
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else np.nan,
            "std_accuracy": float(np.std(accuracies)) if accuracies else np.nan,
        }
