import importlib
import sys
import types

import pandas as pd

from feature_engineering.feature_analyzer import FeatureAnalyzer


def test_feature_pipeline_imports_and_executes_in_order(monkeypatch):

    step_map = [
        (
            "feature_engineering.indicators.trend_indicators",
            "TrendIndicators",
            "add",
            "trend"
        ),
        (
            "feature_engineering.indicators.momentum_indicators",
            "MomentumIndicators",
            "add",
            "momentum"
        ),
        (
            "feature_engineering.indicators.volatility_indicators",
            "VolatilityIndicators",
            "add",
            "volatility"
        ),
        (
            "feature_engineering.indicators.volume_indicators",
            "VolumeIndicators",
            "add",
            "volume"
        ),
        (
            "feature_engineering.price_action.support_resistance",
            "SupportResistance",
            "add",
            "support_resistance"
        ),
        (
            "feature_engineering.price_action.breakout_detector",
            "BreakoutDetector",
            "detect",
            "breakout"
        ),
        (
            "feature_engineering.price_action.candlestick_patterns",
            "CandlePatterns",
            "detect",
            "candles"
        ),
        (
            "feature_engineering.regime_detection.trend_classifier",
            "TrendClassifier",
            "classify",
            "trend_regime"
        ),
        (
            "feature_engineering.regime_detection.volatility_regime",
            "VolatilityRegime",
            "classify",
            "volatility_regime"
        ),
        (
            "feature_engineering.regime_detection.momentum_score",
            "MomentumScore",
            "compute",
            "momentum_score"
        ),
    ]

    for module_name, class_name, method_name, marker in step_map:
        module = types.ModuleType(module_name)

        class Stub:
            pass

        def method(self, data, marker=marker):
            data.append(marker)
            return data

        setattr(Stub, method_name, method)
        setattr(module, class_name, Stub)

        monkeypatch.setitem(sys.modules, module_name, module)

    monkeypatch.delitem(sys.modules, "feature_engineering.feature_pipeline", raising=False)

    feature_pipeline_module = importlib.import_module("feature_engineering.feature_pipeline")
    pipeline = feature_pipeline_module.FeaturePipeline()

    data = []
    result = pipeline.run(data)

    assert result == [
        "trend",
        "momentum",
        "volatility",
        "volume",
        "support_resistance",
        "breakout",
        "candles",
        "trend_regime",
        "volatility_regime",
        "momentum_score",
    ]


def test_feature_analyzer_low_variance_filter_is_scale_safe():

    frame = pd.DataFrame({
        "tiny_signal": [0.10, 0.11, 0.09, 0.12],
        "huge_scale_signal": [1000.0, 1100.0, 900.0, 1200.0],
        "constant_feature": [1.0, 1.0, 1.0, 1.0],
    })

    analyzer = FeatureAnalyzer(variance_threshold=0.01)
    low_variance = analyzer._find_low_variance_features(frame)

    assert "constant_feature" in low_variance
    assert "tiny_signal" not in low_variance
    assert "huge_scale_signal" not in low_variance
