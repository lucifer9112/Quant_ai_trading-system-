import pandas as pd


class HistoricalLoader:

    def __init__(self, path):
        self.path = path

    def load(self):

        df = pd.read_csv(self.path)

        df["Date"] = pd.to_datetime(df["Date"])

        df = df.sort_values("Date")

        df = df.dropna()

        return df


if __name__ == "__main__":

    loader = HistoricalLoader("data/historical.csv")

    data = loader.load()

    print(data.head())