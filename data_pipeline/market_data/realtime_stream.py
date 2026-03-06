import time
import yfinance as yf


class RealtimeStreamer:

    def __init__(self, symbol):
        self.symbol = symbol + ".NS"

    def stream(self, interval=10):

        print("Starting realtime stream...")

        while True:

            data = yf.Ticker(self.symbol)

            price = data.history(period="1d", interval="1m")

            latest = price.iloc[-1]

            print(
                f"Price: {latest['Close']} Volume: {latest['Volume']}"
            )

            time.sleep(interval)


if __name__ == "__main__":

    streamer = RealtimeStreamer("RELIANCE")

    streamer.stream()