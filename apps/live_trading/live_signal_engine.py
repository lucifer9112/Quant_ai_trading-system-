import pandas as pd

from decision_engine.signal_generator import SignalGenerator
from features.store.online_feature_state import OnlineFeatureState


class LiveSignalEngine:

    def __init__(self, feature_state=None, signal_generator=None):

        self.feature_state = feature_state or OnlineFeatureState()
        self.signal_generator = signal_generator or SignalGenerator()

    def on_bar(self, bar):

        latest_features = self.feature_state.update(bar)

        if latest_features is None:
            return None

        latest_frame = pd.DataFrame([latest_features])
        signal = self.signal_generator.generate_latest(latest_frame)

        return {
            "symbol": bar["symbol"],
            "date": latest_features["Date"],
            "signal": signal,
            "strategy_score": latest_features.get("strategy_score", 0.0),
            "features": latest_features,
        }
