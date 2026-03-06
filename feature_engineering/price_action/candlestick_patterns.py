import numpy as np


class CandlePatterns:

    def detect(self, df):

        body = abs(df["Close"] - df["Open"])
        range_ = df["High"] - df["Low"]

        df["Doji"] = np.where(
            body <= range_ * 0.1,
            1,
            0
        )

        df["Hammer"] = np.where(
            (df["Close"] > df["Open"]) &
            ((df["Low"] < df[["Open","Close"]].min(axis=1) - body)),
            1,
            0
        )

        df["Engulfing"] = np.where(
            (df["Close"] > df["Open"]) &
            (df["Close"].shift(1) < df["Open"].shift(1)),
            1,
            0
        )

        return df