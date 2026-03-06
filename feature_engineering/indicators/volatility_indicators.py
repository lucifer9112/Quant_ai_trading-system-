from ta.volatility import BollingerBands, AverageTrueRange
import numpy as np


class VolatilityIndicators:

    def add(self, df):

        bb = BollingerBands(df["Close"])

        df["BB_upper"] = bb.bollinger_hband()
        df["BB_middle"] = bb.bollinger_mavg()
        df["BB_lower"] = bb.bollinger_lband()

        atr = AverageTrueRange(
            df["High"],
            df["Low"],
            df["Close"]
        )

        df["ATR"] = atr.average_true_range()

        df["Historical_Vol"] = (
            df["Close"]
            .pct_change()
            .rolling(20)
            .std()
            * np.sqrt(252)
        )

        return df