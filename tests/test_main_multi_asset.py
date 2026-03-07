from pathlib import Path

import pandas as pd

from main import QuantTradingSystem


def test_main_runs_multi_asset_path_with_backtest(monkeypatch, tmp_path):

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join([
            "symbol: RELIANCE",
            "pipeline:",
            "  mode: multi_asset",
            "backtesting:",
            "  enabled: true",
            "trading:",
            "  initial_capital: 100000",
            "portfolio:",
            "  max_position_weight: 0.5",
            "universe:",
            "  symbols: [AAA, BBB]",
            "  sectors:",
            "    AAA: Tech",
            "    BBB: Finance",
        ]),
        encoding="utf-8",
    )

    class StubLoader:
        def load(self, universe, start=None, end=None):
            return pd.DataFrame({
                "Date": pd.to_datetime([
                    "2024-01-01", "2024-01-02", "2024-01-03",
                    "2024-01-01", "2024-01-02", "2024-01-03",
                ]),
                "Open": [100, 101, 102, 200, 201, 202],
                "High": [101, 102, 103, 201, 202, 203],
                "Low": [99, 100, 101, 199, 200, 201],
                "Close": [100, 101, 103, 200, 198, 204],
                "Volume": [1000, 1050, 1100, 2000, 1950, 2100],
                "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
                "sector": ["Tech", "Tech", "Tech", "Finance", "Finance", "Finance"],
            })

    class StubPanelBuilder:
        def build_panel(self, market_panel, **kwargs):
            frame = market_panel.copy()
            frame["strategy_score"] = [0.8, 0.7, 0.6, -0.7, -0.6, -0.5]
            frame["rolling_vol_20"] = [0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
            frame["sentiment_composite"] = [0.3, 0.2, 0.1, -0.2, -0.1, -0.1]
            frame["sentiment_confidence"] = 0.8
            frame["trend_signal"] = [1, 1, 1, -1, -1, -1]
            frame["mean_reversion_signal"] = 0
            frame["breakout_signal"] = 0
            frame["momentum_signal"] = 0

            return frame

    class StubPredictor:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, frame):
            frame = frame.copy()
            frame["ml_prediction"] = [1, 1, 1, -1, -1, -1]
            return frame

    monkeypatch.setattr("main.MultiAssetMarketLoader", lambda: StubLoader())
    monkeypatch.setattr("main.PanelDatasetBuilder", lambda: StubPanelBuilder())
    monkeypatch.setattr("main.AutoGluonPredictor", StubPredictor)

    system = QuantTradingSystem(config_path=str(config_path))
    result = system.run()

    assert {"portfolio_weight", "portfolio_value", "drawdown", "transaction_cost"}.issubset(result.columns)
    assert set(result["symbol"]) == {"AAA", "BBB"}
