import numpy as np


class BreakoutDetector:

    def detect(self, df):

        df["Breakout"] = np.where(
            df["Close"] > df["Resistance"].shift(1),
            1,
            0
        )

        return df