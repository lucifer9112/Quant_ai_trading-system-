from .strategies.trend_following import TrendFollowingStrategy
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.breakout_strategy import BreakoutStrategy
from .strategies.momentum_strategy import MomentumStrategy


class StrategyScoring:

    def _ensure_signal_columns(self, df):

        signal_generators = [
            ("trend_signal", TrendFollowingStrategy()),
            ("mean_reversion_signal", MeanReversionStrategy()),
            ("breakout_signal", BreakoutStrategy()),
            ("momentum_signal", MomentumStrategy()),
        ]

        for signal_column, generator in signal_generators:

            if signal_column in df.columns:
                continue

            df = generator.generate_signals(df)

        return df

    def compute_score(self, df):

        df = self._ensure_signal_columns(df)

        scores = []

        for i in range(len(df)):

            score = (
                df["trend_signal"].iloc[i] * 0.4 +
                df["mean_reversion_signal"].iloc[i] * 0.2 +
                df["breakout_signal"].iloc[i] * 0.2 +
                df["momentum_signal"].iloc[i] * 0.2
            )

            scores.append(score)

        df["strategy_score"] = scores

        return df
