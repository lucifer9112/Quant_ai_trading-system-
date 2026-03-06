class TrendFollowingStrategy:

    def generate_signals(self, df):

        signals = []

        for i in range(len(df)):

            if df["Close"].iloc[i] > df["SMA50"].iloc[i] and df["ADX"].iloc[i] > 25:
                signals.append(1)  # Buy

            elif df["Close"].iloc[i] < df["SMA50"].iloc[i] and df["ADX"].iloc[i] > 25:
                signals.append(-1)  # Sell

            else:
                signals.append(0)

        df["trend_signal"] = signals

        return df
