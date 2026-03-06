class BreakoutStrategy:

    def generate_signals(self, df):

        signals = []

        for i in range(len(df)):

            if df["Close"].iloc[i] > df["Resistance"].iloc[i]:
                signals.append(1)

            elif df["Close"].iloc[i] < df["Support"].iloc[i]:
                signals.append(-1)

            else:
                signals.append(0)

        df["breakout_signal"] = signals

        return df
