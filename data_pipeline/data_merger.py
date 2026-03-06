import pandas as pd


class DataMerger:

    def merge_market_news(self, market_df, news_df):

        market_df = market_df.copy()
        market_df["Date"] = pd.to_datetime(market_df["Date"], errors="coerce")
        market_df = market_df.dropna(subset=["Date"])

        if news_df is None or len(news_df) == 0:
            return market_df.sort_values("Date")

        news_df = news_df.copy()

        if "published" not in news_df.columns:
            return market_df.sort_values("Date")

        news_df["published"] = pd.to_datetime(news_df["published"], errors="coerce")
        news_df = news_df.dropna(subset=["published"])

        if news_df.empty:
            return market_df.sort_values("Date")

        merged = pd.merge_asof(
            market_df.sort_values("Date"),
            news_df.sort_values("published"),
            left_on="Date",
            right_on="published",
            direction="backward"
        )

        return merged

    def merge_twitter(self, df, tweets):

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

        if tweets is None or len(tweets) == 0:
            return df.sort_values("Date")

        tweets_df = pd.DataFrame(tweets)

        if "date" not in tweets_df.columns:
            return df.sort_values("Date")

        tweets_df["date"] = pd.to_datetime(tweets_df["date"], errors="coerce")
        tweets_df = tweets_df.dropna(subset=["date"])

        if tweets_df.empty:
            return df.sort_values("Date")

        merged = pd.merge_asof(
            df.sort_values("Date"),
            tweets_df.sort_values("date"),
            left_on="Date",
            right_on="date",
            direction="backward"
        )

        return merged
