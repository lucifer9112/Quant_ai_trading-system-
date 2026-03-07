import logging

import pandas as pd

from data_pipeline.twitter_data.tweet_cleaner import TweetCleaner
from data_pipeline.twitter_data.twitter_collector import TwitterCollector
from feature_engineering.sentiment_features.twitter_sentiment import TwitterSentiment


logger = logging.getLogger("quant_system")


class TwitterSentimentExporter:

    OUTPUT_COLUMNS = ["Date", "symbol", "sector", "sentiment", "source", "text"]

    def __init__(self, collector=None, cleaner=None, scorer=None):

        self.collector = collector or TwitterCollector()
        self.cleaner = cleaner or TweetCleaner()
        self.scorer = scorer or TwitterSentiment()

    def build_records(self, universe, limit_per_symbol=50):

        records = []

        for asset in universe.assets:
            query = f"{asset.symbol} stock"
            try:
                tweets = self.collector.search(query, limit=limit_per_symbol)
            except RuntimeError as exc:
                logger.warning("Twitter sentiment collection unavailable: %s", exc)
                break
            except Exception as exc:
                logger.warning("Skipping Twitter sentiment for %s: %s", asset.symbol, exc)
                continue

            for tweet in tweets:
                cleaned_text = self.cleaner.clean(tweet["text"])
                records.append({
                    "Date": pd.to_datetime(tweet["date"], errors="coerce"),
                    "symbol": asset.symbol,
                    "sector": asset.sector,
                    "sentiment": self.scorer.score(cleaned_text),
                    "source": "twitter",
                    "text": cleaned_text,
                })

        frame = pd.DataFrame(records, columns=self.OUTPUT_COLUMNS)
        if not frame.empty:
            frame = frame.dropna(subset=["Date"]).sort_values(["Date", "symbol"]).reset_index(drop=True)

        return frame
