class MeanReversionStrategy:

    def generate_signals(self, df):

        signals = []

        for i in range(len(df)):

            if df["Close"].iloc[i] < df["BB_lower"].iloc[i]:
                signals.append(1)

            elif df["Close"].iloc[i] > df["BB_upper"].iloc[i]:
                signals.append(-1)

            else:
                signals.append(0)

        df["mean_reversion_signal"] = signals

        return df
