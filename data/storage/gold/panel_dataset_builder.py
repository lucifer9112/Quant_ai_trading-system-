import numpy as np
import pandas as pd

from features.pipelines.research_feature_pipeline import ResearchFeaturePipeline
from features.sentiment.sentiment_fusion import SentimentFusionPipeline
from strategy_engine.strategy_scoring import StrategyScoring


class PanelDatasetBuilder:

    def __init__(self, feature_pipeline=None, strategy_scoring=None, sentiment_pipeline=None):

        self.feature_pipeline = feature_pipeline or ResearchFeaturePipeline()
        self.strategy_scoring = strategy_scoring or StrategyScoring()
        self.sentiment_pipeline = sentiment_pipeline or SentimentFusionPipeline()

    def build_panel(self, market_df, news_sentiment_df=None, twitter_sentiment_df=None, sector_sentiment_df=None):

        required_columns = {"Date", "Open", "High", "Low", "Close", "Volume", "symbol"}
        missing_columns = required_columns.difference(market_df.columns)

        if missing_columns:
            raise ValueError(f"Market dataframe is missing required columns: {sorted(missing_columns)}")

        panel_frames = []
        market_df = market_df.copy()

        if "sector" not in market_df.columns:
            market_df["sector"] = "UNKNOWN"

        for _, frame in market_df.groupby("symbol", sort=False):
            enriched = frame.sort_values("Date").reset_index(drop=True)
            enriched = self.feature_pipeline.run(enriched)
            enriched = self.strategy_scoring.compute_score(enriched)
            panel_frames.append(enriched)

        panel = pd.concat(panel_frames, ignore_index=True)
        panel = panel.sort_values(["Date", "symbol"]).reset_index(drop=True)

        sector_group = panel.groupby(["Date", "sector"])["return_1d"]
        panel["sector_return_mean_1d"] = sector_group.transform("mean")
        panel["sector_relative_return_1d"] = panel["return_1d"] - panel["sector_return_mean_1d"]
        panel["cross_sectional_momentum_rank"] = panel.groupby("Date")["return_5d"].rank(pct=True)
        panel["symbol_code"] = panel["symbol"].astype("category").cat.codes
        panel["sector_code"] = panel["sector"].fillna("UNKNOWN").astype("category").cat.codes
        panel = self.sentiment_pipeline.enrich(
            panel,
            news_df=news_sentiment_df,
            twitter_df=twitter_sentiment_df,
            sector_df=sector_sentiment_df,
        )

        return panel

    def build_training_frame(self, panel_df, horizon=1, threshold=0.002):

        if horizon < 1:
            raise ValueError("horizon must be >= 1.")

        dataset = panel_df.copy()
        dataset["future_close"] = dataset.groupby("symbol")["Close"].shift(-horizon)
        dataset["forward_return"] = dataset["future_close"] / dataset["Close"] - 1.0

        dataset["target_return"] = np.where(
            dataset["forward_return"] > threshold,
            1,
            np.where(dataset["forward_return"] < -threshold, -1, 0)
        )

        dataset = dataset.dropna(subset=["future_close"]).copy()

        numeric_columns = dataset.select_dtypes(include=["number", "bool"]).columns.tolist()
        excluded_columns = {"future_close", "forward_return", "target_return"}
        feature_columns = [column for column in numeric_columns if column not in excluded_columns]

        training_frame = dataset[feature_columns + ["target_return"]].dropna().copy()
        training_frame["target_return"] = training_frame["target_return"].astype(int)

        return training_frame
