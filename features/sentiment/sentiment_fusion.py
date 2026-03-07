import numpy as np
import pandas as pd


class SentimentFusionPipeline:

    def __init__(self, weights=None, smoothing_window=3):

        self.weights = weights or {
            "news": 0.45,
            "twitter": 0.35,
            "sector": 0.20,
        }
        self.smoothing_window = max(1, smoothing_window)

    @staticmethod
    def _normalize_dates(values):

        dates = pd.to_datetime(values, errors="coerce")

        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)

        return dates.dt.normalize()

    @staticmethod
    def _prepare_symbol_frame(df, prefix):

        if df is None or len(df) == 0:
            return None

        frame = df.copy()

        if "Date" not in frame.columns or "symbol" not in frame.columns or "sentiment" not in frame.columns:
            raise ValueError(
                f"{prefix} sentiment data must include 'Date', 'symbol', and 'sentiment' columns."
            )

        frame["Date"] = SentimentFusionPipeline._normalize_dates(frame["Date"])
        frame = frame.dropna(subset=["Date", "symbol"])

        aggregated = (
            frame.groupby(["Date", "symbol"], as_index=False)
            .agg(
                sentiment=("sentiment", "mean"),
                volume=("sentiment", "count"),
            )
            .rename(columns={
                "sentiment": f"{prefix}_sentiment",
                "volume": f"{prefix}_sentiment_volume",
            })
        )

        return aggregated

    @staticmethod
    def _prepare_sector_frame(df):

        if df is None or len(df) == 0:
            return None

        frame = df.copy()

        if "Date" not in frame.columns or "sector" not in frame.columns or "sentiment" not in frame.columns:
            raise ValueError(
                "sector sentiment data must include 'Date', 'sector', and 'sentiment' columns."
            )

        frame["Date"] = SentimentFusionPipeline._normalize_dates(frame["Date"])
        frame = frame.dropna(subset=["Date", "sector"])

        aggregated = (
            frame.groupby(["Date", "sector"], as_index=False)
            .agg(
                sentiment=("sentiment", "mean"),
                volume=("sentiment", "count"),
            )
            .rename(columns={
                "sentiment": "sector_sentiment",
                "volume": "sector_sentiment_volume",
            })
        )

        return aggregated

    def _smooth_symbol_series(self, panel, column):

        panel[column] = (
            panel.groupby("symbol")[column]
            .transform(lambda series: series.fillna(0.0).rolling(self.smoothing_window, min_periods=1).mean())
        )

    def _smooth_sector_series(self, panel, column):

        panel[column] = (
            panel.groupby("sector")[column]
            .transform(lambda series: series.fillna(0.0).rolling(self.smoothing_window, min_periods=1).mean())
        )

    def enrich(self, panel_df, news_df=None, twitter_df=None, sector_df=None):

        panel = panel_df.copy()
        panel["Date"] = self._normalize_dates(panel["Date"])

        if "sector" not in panel.columns:
            panel["sector"] = "UNKNOWN"

        news_features = self._prepare_symbol_frame(news_df, "news")
        twitter_features = self._prepare_symbol_frame(twitter_df, "twitter")
        sector_features = self._prepare_sector_frame(sector_df)

        if news_features is not None:
            panel = panel.merge(news_features, on=["Date", "symbol"], how="left")
        if twitter_features is not None:
            panel = panel.merge(twitter_features, on=["Date", "symbol"], how="left")
        if sector_features is not None:
            panel = panel.merge(sector_features, on=["Date", "sector"], how="left")

        defaults = {
            "news_sentiment": 0.0,
            "news_sentiment_volume": 0,
            "twitter_sentiment": 0.0,
            "twitter_sentiment_volume": 0,
            "sector_sentiment": 0.0,
            "sector_sentiment_volume": 0,
        }

        for column, default in defaults.items():
            if column not in panel.columns:
                panel[column] = default
            else:
                panel[column] = panel[column].fillna(default)

        self._smooth_symbol_series(panel, "news_sentiment")
        self._smooth_symbol_series(panel, "twitter_sentiment")
        self._smooth_sector_series(panel, "sector_sentiment")

        panel["sentiment_divergence"] = (panel["news_sentiment"] - panel["twitter_sentiment"]).abs()
        panel["sentiment_alignment"] = np.sign(panel["news_sentiment"] * panel["twitter_sentiment"])
        panel["sentiment_confidence"] = (
            np.log1p(panel["news_sentiment_volume"] + panel["twitter_sentiment_volume"]) /
            np.log(10)
        ).clip(lower=0.0, upper=1.0)
        panel["sentiment_composite"] = (
            panel["news_sentiment"] * self.weights["news"] +
            panel["twitter_sentiment"] * self.weights["twitter"] +
            panel["sector_sentiment"] * self.weights["sector"]
        )
        panel["sentiment_momentum"] = panel.groupby("symbol")["sentiment_composite"].diff().fillna(0.0)
        panel["sentiment_regime_interaction"] = panel["sentiment_composite"] * panel.get("trend_regime_code", 0)

        return panel
