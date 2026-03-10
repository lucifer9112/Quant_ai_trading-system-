from pathlib import Path

from ml_models.prediction_confidence import PredictionConfidenceScorer


class AutoGluonPredictor:

    def __init__(self, model_path="models/autogluon", prediction_column="ml_prediction"):

        self.model_path = model_path
        self.prediction_column = prediction_column
        self._predictor = None
        self.confidence_scorer = PredictionConfidenceScorer()

    def _load_predictor(self):

        if self._predictor is not None:
            return self._predictor

        predictor_artifact = Path(self.model_path) / "predictor.pkl"

        if not predictor_artifact.exists():
            raise FileNotFoundError(
                f"AutoGluon model not found at '{predictor_artifact}'. "
                "Train and save a model first, or update model.autogluon_path in config.yaml."
            )

        try:
            from autogluon.tabular import TabularPredictor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "autogluon is not installed. Install it with `pip install autogluon.tabular`."
            ) from exc

        self._predictor = TabularPredictor.load(self.model_path)

        return self._predictor

    def predict(self, df):

        predictor = self._load_predictor()

        predictions = predictor.predict(df)

        result = df.copy()
        result[self.prediction_column] = predictions

        if hasattr(predictor, "predict_proba"):
            try:
                probabilities = predictor.predict_proba(df)
                result = self.confidence_scorer.merge(
                    result,
                    probabilities,
                    predicted_labels=predictions,
                )
            except Exception:
                pass

        return result
