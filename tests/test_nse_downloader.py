from data_pipeline.market_data.nse_downloader import NSEDownloader


class FakeYFinance:

    def __init__(self):

        self.download_calls = []
        self.cache_location = None

    def set_tz_cache_location(self, path):

        self.cache_location = path

    def download(self, symbol, start, end, progress, auto_adjust):

        self.download_calls.append(
            {
                "symbol": symbol,
                "start": start,
                "end": end,
                "progress": progress,
                "auto_adjust": auto_adjust,
            }
        )

        class FakeFrame:

            columns = ["Open", "High", "Low", "Close", "Volume"]
            empty = False

            def reset_index(self, inplace=True):

                self.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

            def __getitem__(self, columns):

                return columns

        return FakeFrame()


def test_nse_downloader_requests_explicit_non_adjusted_prices(monkeypatch):

    fake_yfinance = FakeYFinance()
    monkeypatch.setattr(NSEDownloader, "_yfinance_module", staticmethod(lambda: fake_yfinance))

    downloader = NSEDownloader("RELIANCE")
    downloader.download(start="2024-01-01", end="2024-01-31")

    assert fake_yfinance.download_calls == [
        {
            "symbol": "RELIANCE.NS",
            "start": "2024-01-01",
            "end": "2024-01-31",
            "progress": False,
            "auto_adjust": False,
        }
    ]
