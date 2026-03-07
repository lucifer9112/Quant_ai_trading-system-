import pandas as pd

from features.pipelines.research_feature_pipeline import ResearchFeaturePipeline


class StubBasePipeline:

    def run(self, df):

        frame = df.copy()
        frame["Momentum_Score"] = 50.0
        frame["Trend"] = "Bullish"
        frame["Volatility_Regime"] = "Medium"
        frame["ATR"] = 2.0
        frame["VWAP"] = frame["Close"] * 0.99
        frame["BB_upper"] = frame["Close"] * 1.02
        frame["BB_middle"] = frame["Close"]
        frame["BB_lower"] = frame["Close"] * 0.98

        return frame


def test_research_feature_pipeline_adds_advanced_columns():

    frame = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "Open": [100 + value for value in range(30)],
        "High": [101 + value for value in range(30)],
        "Low": [99 + value for value in range(30)],
        "Close": [100 + value for value in range(30)],
        "Volume": [1000 + value * 10 for value in range(30)],
    })

    pipeline = ResearchFeaturePipeline(base_pipeline=StubBasePipeline())
    result = pipeline.run(frame)

    expected_columns = {
        "bollinger_band_width",
        "rolling_vol_20",
        "return_5d",
        "gap_ratio",
        "volume_zscore_20",
        "regime_adjusted_momentum",
    }

    assert expected_columns.issubset(result.columns)
