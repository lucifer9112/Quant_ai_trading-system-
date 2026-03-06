import numpy as np


class TrendClassifier:

    def classify(self, df):

        df["Trend"] = np.where(
            (df["Close"] > df["SMA200"]) &
            (df["ADX"] > 25),
            "Bullish",
            "Sideways"
        )

        df["Trend"] = np.where(
            (df["Close"] < df["SMA200"]) &
            (df["ADX"] > 25),
            "Bearish",
            df["Trend"]
        )

        return df