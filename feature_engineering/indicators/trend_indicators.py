import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator


class TrendIndicators:

    def add(self, df):

        df["SMA20"] = SMAIndicator(df["Close"], 20).sma_indicator()
        df["SMA50"] = SMAIndicator(df["Close"], 50).sma_indicator()
        df["SMA200"] = SMAIndicator(df["Close"], 200).sma_indicator()

        df["EMA9"] = EMAIndicator(df["Close"], 9).ema_indicator()
        df["EMA21"] = EMAIndicator(df["Close"], 21).ema_indicator()

        macd = MACD(df["Close"])

        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_diff"] = macd.macd_diff()

        adx = ADXIndicator(df["High"], df["Low"], df["Close"])

        df["ADX"] = adx.adx()

        return df