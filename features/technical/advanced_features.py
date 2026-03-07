import numpy as np


class AdvancedTechnicalFeatures:

    def add(self, df):

        df = df.copy()

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        typical_price = (high + low + close) / 3.0
        cumulative_volume = volume.replace(0, np.nan).cumsum()
        rolling_returns = close.pct_change()

        if "VWAP" not in df.columns:
            df["VWAP"] = (typical_price * volume).cumsum() / cumulative_volume

        true_range = np.maximum(
            high - low,
            np.maximum(
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            )
        )

        if "ATR" not in df.columns:
            df["ATR"] = true_range.rolling(14).mean()

        if {"BB_upper", "BB_lower", "BB_middle"}.issubset(df.columns):
            denominator = df["BB_middle"].replace(0, np.nan).abs()
            df["bollinger_band_width"] = (df["BB_upper"] - df["BB_lower"]) / denominator
        else:
            rolling_mean = close.rolling(20).mean()
            rolling_std = close.rolling(20).std()
            df["bollinger_band_width"] = (4 * rolling_std) / rolling_mean.replace(0, np.nan).abs()

        df["atr_to_close"] = df["ATR"] / close.replace(0, np.nan).abs()
        df["rolling_vol_5"] = rolling_returns.rolling(5).std() * np.sqrt(252)
        df["rolling_vol_20"] = rolling_returns.rolling(20).std() * np.sqrt(252)
        df["return_1d"] = rolling_returns
        df["return_3d"] = close.pct_change(3)
        df["return_5d"] = close.pct_change(5)
        df["return_10d"] = close.pct_change(10)
        df["vwap_deviation"] = (close - df["VWAP"]) / df["VWAP"].replace(0, np.nan).abs()

        return df
