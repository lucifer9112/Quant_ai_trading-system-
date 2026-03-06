from .indicators.trend_indicators import TrendIndicators
from .indicators.momentum_indicators import MomentumIndicators
from .indicators.volatility_indicators import VolatilityIndicators
from .indicators.volume_indicators import VolumeIndicators

from .price_action.support_resistance import SupportResistance
from .price_action.breakout_detector import BreakoutDetector
from .price_action.candlestick_patterns import CandlePatterns

from .regime_detection.trend_classifier import TrendClassifier
from .regime_detection.volatility_regime import VolatilityRegime
from .regime_detection.momentum_score import MomentumScore


class FeaturePipeline:

    def run(self, df):

        df = TrendIndicators().add(df)
        df = MomentumIndicators().add(df)
        df = VolatilityIndicators().add(df)
        df = VolumeIndicators().add(df)

        df = SupportResistance().add(df)
        df = BreakoutDetector().detect(df)
        df = CandlePatterns().detect(df)

        df = TrendClassifier().classify(df)
        df = VolatilityRegime().classify(df)
        df = MomentumScore().compute(df)

        return df
