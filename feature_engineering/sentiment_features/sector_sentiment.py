import pandas as pd


class SectorSentiment:

    def aggregate(self, df):

        sector_sentiment = df.groupby("sector")["sentiment"].mean()

        return sector_sentiment