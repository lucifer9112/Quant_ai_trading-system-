import numpy as np
import pandas as pd

from feature_engineering.feature_analyzer import FeatureAnalyzer
from feature_engineering.panel_feature_expander import PanelFeatureExpander
from features.pipelines.research_feature_pipeline import ResearchFeaturePipeline
from features.sentiment.sentiment_fusion import SentimentFusionPipeline
from strategy_engine.strategy_scoring import StrategyScoring


class PanelDatasetBuilder:

    def __init__(
        self,
        feature_pipeline=None,
        strategy_scoring=None,
        sentiment_pipeline=None,
        panel_feature_expander=None,
        feature_analyzer=None,
    ):

        self.feature_pipeline = feature_pipeline or ResearchFeaturePipeline()
        self.strategy_scoring = strategy_scoring or StrategyScoring()
        self.sentiment_pipeline = sentiment_pipeline or SentimentFusionPipeline()
        self.panel_feature_expander = panel_feature_expander or PanelFeatureExpander()
        self.feature_analyzer = feature_analyzer or FeatureAnalyzer()

    def build_panel(
        self,
        market_df,
        news_sentiment_df=None,
        twitter_sentiment_df=None,
        sector_sentiment_df=None,
        macro_df=None,
        benchmark_df=None,
        use_expanded_features=True,
    ):

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
        if use_expanded_features:
            panel = self.panel_feature_expander.transform(
                panel,
                macro_df=macro_df,
                benchmark_df=benchmark_df,
            )

        return panel

    def build_training_frame(
        self,
        panel_df,
        horizon=1,
        threshold=0.002,
        *,
        include_metadata=False,
        prune_correlated=False,
        max_features=None,
    ):

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

        if prune_correlated and feature_columns:
            prune_report = self.feature_analyzer.prune_correlated_features(
                training_frame[feature_columns],
                y=training_frame["target_return"],
            )
            selected = prune_report["selected_features"]
            if max_features is not None:
                selected = self.feature_analyzer.select_features(
                    training_frame[selected],
                    y=training_frame["target_return"],
                    n_features=max_features,
                )
            training_frame = training_frame[selected + ["target_return"]].copy()
            training_frame.attrs["feature_pruning_report"] = prune_report
        elif max_features is not None and feature_columns:
            selected = self.feature_analyzer.select_features(
                training_frame[feature_columns],
                y=training_frame["target_return"],
                n_features=max_features,
            )
            training_frame = training_frame[selected + ["target_return"]].copy()

        if include_metadata:
            metadata_columns = [column for column in ["Date", "symbol", "sector"] if column in dataset.columns]
            metadata_frame = dataset.loc[training_frame.index, metadata_columns].copy()
            training_frame = metadata_frame.join(training_frame)

        return training_frame
