class MomentumStrategy:

    def generate_signals(self, df):

        signals = []

        for i in range(len(df)):

            if df["RSI"].iloc[i] < 30 and df["ROC"].iloc[i] > 0:
                signals.append(1)

            elif df["RSI"].iloc[i] > 70 and df["ROC"].iloc[i] < 0:
                signals.append(-1)

            else:
                signals.append(0)

        df["momentum_signal"] = signals

        return df
