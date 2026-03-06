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