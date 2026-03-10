"""Machine learning models module for ensemble methods."""

__all__ = [
    "StackingEnsemble",
    "EnsembleBuilder",
    "DynamicWeighting",
    "ProbabilityCalibrator",
    "Phase2MLEnsemble",
]


def __getattr__(name):

    if name == "StackingEnsemble":
        from .stacking_ensemble import StackingEnsemble
        return StackingEnsemble
    if name == "EnsembleBuilder":
        from .ensemble_builder import EnsembleBuilder
        return EnsembleBuilder
    if name == "DynamicWeighting":
        from .dynamic_weighting import DynamicWeighting
        return DynamicWeighting
    if name == "ProbabilityCalibrator":
        from .probability_calibration import ProbabilityCalibrator
        return ProbabilityCalibrator
    if name == "Phase2MLEnsemble":
        from .phase2_ml_ensemble import Phase2MLEnsemble
        return Phase2MLEnsemble

    raise AttributeError(f"module 'ml_models' has no attribute '{name}'")
