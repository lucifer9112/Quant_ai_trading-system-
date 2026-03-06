class SupportResistance:

    def add(self, df, window=20):

        df["Support"] = df["Low"].rolling(window).min()
        df["Resistance"] = df["High"].rolling(window).max()

        return df