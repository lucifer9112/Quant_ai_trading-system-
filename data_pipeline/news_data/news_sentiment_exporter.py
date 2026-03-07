import re

import pandas as pd

from data_pipeline.news_data.news_cleaner import NewsCleaner
from data_pipeline.news_data.news_collector import NewsCollector
from feature_engineering.sentiment_features.news_sentiment import NewsSentiment


class NewsSentimentExporter:

    OUTPUT_COLUMNS = ["Date", "symbol", "sector", "sentiment", "source", "text", "link"]

    def __init__(self, collector=None, cleaner=None, scorer=None):

        self.collector = collector or NewsCollector()
        self.cleaner = cleaner or NewsCleaner()
        self.scorer = scorer or NewsSentiment()

    @staticmethod
    def _matches_asset(text, asset):

        pattern = re.escape(asset.symbol.lower())

        return re.search(pattern, text.lower()) is not None

    def build_records(self, universe):

        news_items = self.collector.collect()
        cleaned_items = self.cleaner.clean_news(news_items)
        records = []

        for item in cleaned_items:
            matched_assets = [
                asset
                for asset in universe.assets
                if self._matches_asset(item["title"], asset)
            ]

            for asset in matched_assets:
                records.append({
                    "Date": pd.to_datetime(item["published"], errors="coerce"),
                    "symbol": asset.symbol,
                    "sector": asset.sector,
                    "sentiment": self.scorer.score(item["title"]),
                    "source": "news",
                    "text": item["title"],
                    "link": item["link"],
                })

        frame = pd.DataFrame(records, columns=self.OUTPUT_COLUMNS)
        if not frame.empty:
            frame = frame.dropna(subset=["Date"]).sort_values(["Date", "symbol"]).reset_index(drop=True)

        return frame
