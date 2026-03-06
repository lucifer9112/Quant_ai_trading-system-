import pandas as pd


class SentimentAggregator:

    def aggregate(self, df):

        daily = df.groupby("Date").agg({
            "sentiment": ["mean", "sum", "count"]
        })

        daily.columns = [
            "news_sentiment_mean",
            "news_sentiment_sum",
            "news_volume"
        ]

        return daily.reset_index()