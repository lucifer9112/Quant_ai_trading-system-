from pathlib import Path


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
                    "autogluon is not installed. Install it with `pip install autogluon`."
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
