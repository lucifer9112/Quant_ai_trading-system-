import pandas as pd


class MultiAssetMarketLoader:

    def __init__(self, downloader_factory=None):

        if downloader_factory is None:
            self.downloader_factory = self._default_downloader_factory
        else:
            self.downloader_factory = downloader_factory

    @staticmethod
    def _default_downloader_factory(symbol):

        from data_pipeline.market_data.nse_downloader import NSEDownloader

        return NSEDownloader(symbol)

    def load(self, universe, start=None, end=None):

        market_frames = []

        for asset in universe.assets:
            downloader = self.downloader_factory(asset.symbol)
            frame = downloader.download(start=start or universe.start_date, end=end or universe.end_date)
            frame = frame.copy()
            frame["symbol"] = asset.symbol
            frame["sector"] = asset.sector
            market_frames.append(frame)

        if not market_frames:
            raise ValueError("Universe contains no assets to download.")

        panel = pd.concat(market_frames, ignore_index=True)
        panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
        panel = panel.dropna(subset=["Date"])

        return panel.sort_values(["Date", "symbol"]).reset_index(drop=True)
