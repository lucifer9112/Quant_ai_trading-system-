import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path


class NSEDownloader:

    def __init__(self, symbol):
        self.symbol = symbol + ".NS"
        self._configure_yfinance_cache()

    @staticmethod
    def _configure_yfinance_cache():

        cache_dir = Path(".cache") / "yfinance"
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir.resolve()))

    def download(self, start="2015-01-01", end=None):

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        df = yf.download(
            self.symbol,
            start=start,
            end=end,
            progress=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            # yfinance can return a (field, ticker) MultiIndex even for one symbol.
            df.columns = df.columns.get_level_values(0)
            df.columns.name = None

        if df.empty:
            raise ValueError(
                f"No market data downloaded for {self.symbol} "
                f"between {start} and {end}."
            )

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Downloaded data is missing required columns: {missing_columns}"
            )

        df.reset_index(inplace=True)

        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        if "Date" not in df.columns:
            raise ValueError("Downloaded data does not include a Date column.")

        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


if __name__ == "__main__":

    downloader = NSEDownloader("RELIANCE")

    df = downloader.download()

    print(df.head())
