from collections import defaultdict, deque

import numpy as np
import pandas as pd


class OnlineFeatureState:

    def __init__(self, window_size=60):

        self.window_size = window_size
        self.buffers = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, bar):

        symbol = bar["symbol"]
        self.buffers[symbol].append(bar)

        frame = pd.DataFrame(self.buffers[symbol])
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.sort_values("Date").reset_index(drop=True)

        if frame.empty:
            return None

        frame["return_1d"] = frame["Close"].pct_change()
        frame["return_5d"] = frame["Close"].pct_change(5)
        frame["sma_20"] = frame["Close"].rolling(20).mean()
        frame["rolling_vol_20"] = frame["Close"].pct_change().rolling(20).std() * np.sqrt(252)
        frame["rolling_high_20"] = frame["High"].rolling(20).max()
        frame["rolling_low_20"] = frame["Low"].rolling(20).min()
        cumulative_volume = frame["Volume"].replace(0, np.nan).cumsum()
        typical_price = (frame["High"] + frame["Low"] + frame["Close"]) / 3.0
        frame["vwap_live"] = (typical_price * frame["Volume"]).cumsum() / cumulative_volume
        frame["vwap_deviation"] = (
            frame["Close"] - frame["vwap_live"]
        ) / frame["vwap_live"].replace(0, np.nan).abs()
        frame["gap_ratio"] = frame["Open"] / frame["Close"].shift(1) - 1.0
        frame["volume_zscore_20"] = (
            (frame["Volume"] - frame["Volume"].rolling(20).mean()) /
            frame["Volume"].rolling(20).std().replace(0, np.nan)
        )

        frame["trend_signal"] = np.where(frame["Close"] > frame["sma_20"], 1, -1)
        frame["mean_reversion_signal"] = np.where(frame["vwap_deviation"] < -0.01, 1, 0)
        frame["mean_reversion_signal"] = np.where(
            frame["vwap_deviation"] > 0.01,
            -1,
            frame["mean_reversion_signal"]
        )
        frame["breakout_signal"] = np.where(frame["Close"] > frame["rolling_high_20"].shift(1), 1, 0)
        frame["breakout_signal"] = np.where(
            frame["Close"] < frame["rolling_low_20"].shift(1),
            -1,
            frame["breakout_signal"]
        )
        frame["momentum_signal"] = np.where(frame["return_5d"] > 0, 1, 0)
        frame["momentum_signal"] = np.where(frame["return_5d"] < 0, -1, frame["momentum_signal"])
        frame["strategy_score"] = (
            frame["trend_signal"] * 0.4 +
            frame["mean_reversion_signal"] * 0.2 +
            frame["breakout_signal"] * 0.2 +
            frame["momentum_signal"] * 0.2
        )

        return frame.iloc[-1].to_dict()
