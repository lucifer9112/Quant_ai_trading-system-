import numpy as np
import pandas as pd

from monitoring.model_drift import ModelDriftDetector


def test_model_drift_detector_flags_shifted_feature_distribution():

    rng = np.random.default_rng(42)
    reference = pd.DataFrame({
        "feature_a": rng.normal(0, 1, size=200),
        "feature_b": rng.normal(0, 1, size=200),
    })
    current = pd.DataFrame({
        "feature_a": rng.normal(2, 1, size=200),
        "feature_b": rng.normal(0, 1, size=200),
    })

    detector = ModelDriftDetector(psi_threshold=0.1).fit(reference)
    report = detector.detect(current)

    assert report["drift_detected"] is True
    assert not report["metrics"].empty
    assert report["metrics"].iloc[0]["psi"] > 0
