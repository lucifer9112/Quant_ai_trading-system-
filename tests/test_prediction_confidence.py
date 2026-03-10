import numpy as np

from ml_models.prediction_confidence import PredictionConfidenceScorer


def test_prediction_confidence_scores_probability_outputs():

    scorer = PredictionConfidenceScorer()
    probabilities = np.array([
        [0.1, 0.9],
        [0.45, 0.55],
    ])

    result = scorer.score(probabilities)

    assert {"prediction_confidence", "prediction_margin", "prediction_entropy"}.issubset(
        result.diagnostics.columns
    )
    assert result.diagnostics["prediction_confidence"].iloc[0] > result.diagnostics["prediction_confidence"].iloc[1]
    assert result.diagnostics["prediction_entropy"].iloc[0] < result.diagnostics["prediction_entropy"].iloc[1]
