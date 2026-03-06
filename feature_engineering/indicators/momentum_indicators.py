from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import CCIIndicator


class MomentumIndicators:

    def add(self, df):

        df["RSI"] = RSIIndicator(df["Close"], 14).rsi()

        stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])

        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()

        df["CCI"] = CCIIndicator(
            df["High"],
            df["Low"],
            df["Close"]
        ).cci()

        df["ROC"] = ROCIndicator(df["Close"]).roc()

        return df