import matplotlib.pyplot as plt


class IndicatorPlots:

    def plot_price_sma(self, df):

        plt.figure(figsize=(12,6))

        plt.plot(df["Date"], df["Close"], label="Close")

        if "SMA50" in df.columns:
            plt.plot(df["Date"], df["SMA50"], label="SMA50")

        if "SMA200" in df.columns:
            plt.plot(df["Date"], df["SMA200"], label="SMA200")

        plt.legend()
        plt.title("Price with Moving Averages")

        plt.show()

    def plot_rsi(self, df):

        if "RSI" not in df.columns:
            return

        plt.figure(figsize=(12,4))

        plt.plot(df["Date"], df["RSI"])

        plt.axhline(70)
        plt.axhline(30)

        plt.title("RSI Indicator")

        plt.show()