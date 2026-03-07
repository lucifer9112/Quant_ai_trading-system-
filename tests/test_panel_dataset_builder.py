import pandas as pd

from data.storage.gold.panel_dataset_builder import PanelDatasetBuilder


class StubFeaturePipeline:

    def run(self, df):

        frame = df.copy()
        frame["return_1d"] = frame["Close"].pct_change().fillna(0.0)
        frame["return_5d"] = frame["Close"].pct_change().fillna(0.0)

        return frame


class StubStrategyScoring:

    def compute_score(self, df):

        frame = df.copy()
        frame["strategy_score"] = 0.1

        return frame


def test_panel_dataset_builder_builds_panel_and_training_frame():

    frame = pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03",
            "2024-01-01", "2024-01-02", "2024-01-03",
        ]),
        "Open": [100, 101, 102, 200, 201, 202],
        "High": [101, 102, 103, 201, 202, 203],
        "Low": [99, 100, 101, 199, 200, 201],
        "Close": [100, 102, 103, 200, 198, 202],
        "Volume": [1000, 1100, 1200, 1500, 1550, 1600],
        "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
        "sector": ["Tech", "Tech", "Tech", "Finance", "Finance", "Finance"],
    })

    builder = PanelDatasetBuilder(
        feature_pipeline=StubFeaturePipeline(),
        strategy_scoring=StubStrategyScoring(),
    )

    panel = builder.build_panel(frame)
    training = builder.build_training_frame(panel, horizon=1, threshold=0.0)

    assert {"sector_relative_return_1d", "cross_sectional_momentum_rank", "symbol_code", "sector_code"}.issubset(panel.columns)
    assert "target_return" in training.columns
    assert set(training["target_return"].unique()).issubset({-1, 0, 1})
