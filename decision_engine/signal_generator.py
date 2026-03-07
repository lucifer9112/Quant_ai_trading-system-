from .decision_model import DecisionModel


class SignalGenerator:

    def __init__(self):

        self.model = DecisionModel()

    def generate(self, df):

        signals = []

        for _, row in df.iterrows():

            signal = self.model.decide(row)

            signals.append(signal)

        df["final_signal"] = signals

        return df

    def generate_latest(self, df):

        row = df.iloc[-1] if hasattr(df, "iloc") else df

        return self.model.decide(row)
