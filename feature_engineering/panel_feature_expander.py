"""Leakage-safe panel feature expansion for cross-asset and macro context."""

from __future__ import annotations

from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd

from .microstructure_features import MicrostructureFeatures


class PanelFeatureExpander:
    """Expand a panel with market microstructure, cross-asset, and macro features."""

    def __init__(
        self,
        microstructure_window: int = 20,
        cross_asset_window: int = 20,
        feature_availability_lag: int = 1,
    ):
        self.microstructure_window = microstructure_window
        self.cross_asset_window = cross_asset_window
        self.feature_availability_lag = feature_availability_lag
        self.microstructure = MicrostructureFeatures(window=microstructure_window)

    def transform(
        self,
        panel_df: pd.DataFrame,
        *,
        macro_df: Optional[pd.DataFrame] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        dataset = panel_df.copy()
        dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
        dataset = dataset.dropna(subset=["Date"]).sort_values(["symbol", "Date"]).reset_index(drop=True)

        original_columns = set(dataset.columns)
        dataset = self._add_symbol_features(dataset)
        dataset = self._add_cross_asset_features(dataset)
        dataset = self._add_macro_features(dataset, macro_df=macro_df, benchmark_df=benchmark_df)

        new_columns = [column for column in dataset.columns if column not in original_columns]
        return self._lag_features(dataset, new_columns)

    def _add_symbol_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for _, frame in dataset.groupby("symbol", sort=False):
            symbol_frame = frame.sort_values("Date").reset_index(drop=True)
            symbol_frame = self.microstructure.add(symbol_frame)
            symbol_frame["dollar_volume"] = symbol_frame["Close"] * symbol_frame["Volume"]
            symbol_frame["intraday_return"] = (symbol_frame["Close"] / symbol_frame["Open"]) - 1.0
            symbol_frame["overnight_gap_return"] = (
                symbol_frame["Open"] / symbol_frame["Close"].shift(1)
            ) - 1.0
            symbol_frame["realized_vol_10"] = symbol_frame["Close"].pct_change().rolling(10).std()
            symbol_frame["realized_vol_21"] = symbol_frame["Close"].pct_change().rolling(21).std()
            symbol_frame["amihud_illiquidity_20"] = (
                symbol_frame["Close"].pct_change().abs() /
                symbol_frame["dollar_volume"].replace(0, np.nan)
            ).rolling(20).mean()
            intraday_range = (symbol_frame["High"] - symbol_frame["Low"]).replace(0, np.nan)
            symbol_frame["intraday_range_zscore_20"] = (
                (intraday_range - intraday_range.rolling(20).mean()) /
                intraday_range.rolling(20).std().replace(0, np.nan)
            )
            symbol_frame["volume_surge_5"] = (
                symbol_frame["Volume"] / symbol_frame["Volume"].rolling(5).mean().replace(0, np.nan)
            )
            frames.append(symbol_frame)

        return pd.concat(frames, ignore_index=True).sort_values(["Date", "symbol"]).reset_index(drop=True)

    def _add_cross_asset_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        frame = dataset.copy()
        if "return_1d" not in frame.columns:
            frame["return_1d"] = frame.groupby("symbol")["Close"].pct_change()
        if "return_5d" not in frame.columns:
            frame["return_5d"] = frame.groupby("symbol")["Close"].pct_change(5)

        frame["market_return_mean_1d"] = frame.groupby("Date")["return_1d"].transform("mean")
        frame["market_return_dispersion_1d"] = frame.groupby("Date")["return_1d"].transform("std")
        frame["market_breadth_positive"] = frame.groupby("Date")["return_1d"].transform(
            lambda series: float((series.fillna(0.0) > 0).mean())
        )
        frame["market_breadth_negative"] = frame.groupby("Date")["return_1d"].transform(
            lambda series: float((series.fillna(0.0) < 0).mean())
        )
        frame["cross_sectional_return_rank_1d"] = frame.groupby("Date")["return_1d"].rank(pct=True)
        frame["cross_sectional_return_rank_5d"] = frame.groupby("Date")["return_5d"].rank(pct=True)
        frame["cross_sectional_volume_rank"] = frame.groupby("Date")["Volume"].rank(pct=True)
        frame["cross_sectional_dollar_volume_rank"] = frame.groupby("Date")["dollar_volume"].rank(pct=True)
        frame["cross_sectional_zscore_return_1d"] = frame.groupby("Date")["return_1d"].transform(
            self._zscore_series
        )

        if "sector" in frame.columns:
            frame["sector_breadth_positive"] = frame.groupby(["Date", "sector"])["return_1d"].transform(
                lambda series: float((series.fillna(0.0) > 0).mean())
            )
            frame["sector_relative_momentum_5d"] = frame["return_5d"] - frame.groupby(
                ["Date", "sector"]
            )["return_5d"].transform("mean")

        market_series = (
            frame.groupby("Date")["return_1d"]
            .mean()
            .rename("market_return_mean_1d")
            .reset_index()
        )
        frame = frame.merge(market_series, on="Date", how="left", suffixes=("", "_dup"))
        frame = frame.drop(columns=["market_return_mean_1d_dup"], errors="ignore")
        frame["rolling_beta_to_market_20"] = (
            frame.groupby("symbol", group_keys=False)
            .apply(self._rolling_beta)
            .reset_index(level=0, drop=True)
        )
        frame["rolling_corr_to_market_20"] = (
            frame.groupby("symbol", group_keys=False)
            .apply(self._rolling_correlation)
            .reset_index(level=0, drop=True)
        )
        return frame

    def _add_macro_features(
        self,
        dataset: pd.DataFrame,
        *,
        macro_df: Optional[pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        frame = dataset.copy()

        daily_market = frame.groupby("Date").agg(
            macro_market_return_1d=("return_1d", "mean"),
            macro_market_dispersion_1d=("return_1d", "std"),
            macro_market_breadth_positive=("market_breadth_positive", "mean"),
        )
        daily_market["macro_market_return_5d"] = daily_market["macro_market_return_1d"].rolling(5).mean()
        daily_market["macro_market_vol_21"] = daily_market["macro_market_return_1d"].rolling(21).std()
        daily_market["macro_breadth_momentum_10"] = daily_market["macro_market_breadth_positive"].rolling(10).mean()
        frame = frame.merge(daily_market.reset_index(), on="Date", how="left")

        if benchmark_df is not None and {"Date", "Close"}.issubset(benchmark_df.columns):
            benchmark = benchmark_df.copy()
            benchmark["Date"] = pd.to_datetime(benchmark["Date"], errors="coerce")
            benchmark = benchmark.dropna(subset=["Date"]).sort_values("Date")
            benchmark["benchmark_return_1d"] = benchmark["Close"].pct_change()
            benchmark["benchmark_return_5d"] = benchmark["Close"].pct_change(5)
            benchmark["benchmark_vol_21"] = benchmark["benchmark_return_1d"].rolling(21).std()
            benchmark["benchmark_trend_20_60"] = (
                benchmark["Close"].rolling(20).mean() / benchmark["Close"].rolling(60).mean()
            ) - 1.0
            frame = frame.merge(
                benchmark[
                    [
                        "Date",
                        "benchmark_return_1d",
                        "benchmark_return_5d",
                        "benchmark_vol_21",
                        "benchmark_trend_20_60",
                    ]
                ],
                on="Date",
                how="left",
            )

        if macro_df is not None and "Date" in macro_df.columns:
            macro_frame = macro_df.copy()
            macro_frame["Date"] = pd.to_datetime(macro_frame["Date"], errors="coerce")
            macro_frame = macro_frame.dropna(subset=["Date"]).sort_values("Date")
            rename_map = {
                column: f"macro_external_{column}"
                for column in macro_frame.columns
                if column != "Date"
            }
            macro_frame = macro_frame.rename(columns=rename_map)
            frame = frame.merge(macro_frame, on="Date", how="left")

        return frame

    def _lag_features(self, dataset: pd.DataFrame, new_columns: Iterable[str]) -> pd.DataFrame:
        if self.feature_availability_lag <= 0:
            return dataset

        laggable_columns = [
            column
            for column in new_columns
            if column not in {"Date", "symbol", "sector"}
        ]
        if not laggable_columns:
            return dataset

        lagged = dataset.copy()
        lagged[laggable_columns] = lagged.groupby("symbol")[laggable_columns].shift(
            self.feature_availability_lag
        )
        return lagged

    @staticmethod
    def _zscore_series(series: pd.Series) -> pd.Series:
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    def _rolling_beta(self, frame: pd.DataFrame) -> pd.Series:
        asset = frame["return_1d"]
        market = frame["market_return_mean_1d"]
        covariance = asset.rolling(self.cross_asset_window).cov(market)
        variance = market.rolling(self.cross_asset_window).var()
        return covariance / variance.replace(0, np.nan)

    def _rolling_correlation(self, frame: pd.DataFrame) -> pd.Series:
        return frame["return_1d"].rolling(self.cross_asset_window).corr(frame["market_return_mean_1d"])
