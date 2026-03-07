import numpy as np


class MarketMicrostructureFeatures:

    def add(self, df):

        df = df.copy()

        range_ = (df["High"] - df["Low"]).replace(0, np.nan)
        close_return = df["Close"].pct_change().abs()
        dollar_volume = (df["Close"] * df["Volume"]).replace(0, np.nan)

        df["gap_ratio"] = df["Open"] / df["Close"].shift(1) - 1.0
        df["intraday_range_efficiency"] = (df["Close"] - df["Open"]).abs() / range_
        df["close_location_value"] = ((df["Close"] - df["Low"]) / range_) * 2 - 1
        df["volume_zscore_20"] = (
            (df["Volume"] - df["Volume"].rolling(20).mean()) /
            df["Volume"].rolling(20).std().replace(0, np.nan)
        )
        df["price_impact_proxy"] = close_return / dollar_volume
        df["overnight_reversal"] = -df["gap_ratio"].shift(1)

        return df
