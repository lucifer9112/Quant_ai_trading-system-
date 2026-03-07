import time


class RealtimeStreamer:

    def __init__(self, symbol):
        self.symbol = symbol + ".NS"

    @staticmethod
    def _ticker(symbol):

        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "yfinance is not installed. Install it with `pip install yfinance`."
            ) from exc

        return yf.Ticker(symbol)

    def get_latest_bar(self, history_period="1d", bar_interval="1m"):

        ticker = self._ticker(self.symbol)
        frame = ticker.history(period=history_period, interval=bar_interval)

        if frame.empty:
            raise ValueError(f"No realtime data available for {self.symbol}.")

        latest = frame.iloc[-1]
        timestamp = frame.index[-1]

        return {
            "Date": timestamp,
            "symbol": self.symbol.replace(".NS", ""),
            "Open": float(latest["Open"]),
            "High": float(latest["High"]),
            "Low": float(latest["Low"]),
            "Close": float(latest["Close"]),
            "Volume": float(latest["Volume"]),
        }

    def stream_quotes(self, interval=10, limit=None):

        emitted = 0

        while limit is None or emitted < limit:
            yield self.get_latest_bar()
            emitted += 1
            time.sleep(interval)

    def stream(self, interval=10):

        print("Starting realtime stream...")

        for latest in self.stream_quotes(interval=interval):
            print(
                f"Price: {latest['Close']} Volume: {latest['Volume']}"
            )


if __name__ == "__main__":

    streamer = RealtimeStreamer("RELIANCE")

    streamer.stream()
