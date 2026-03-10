import pandas as pd

from decision_engine.portfolio_allocator import PortfolioAllocator


def test_portfolio_allocator_constructs_multi_asset_weights():

    frame = pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-01-02", "2024-01-02",
        ]),
        "symbol": ["AAA", "BBB", "AAA", "BBB"],
        "Close": [100.0, 120.0, 101.0, 118.0],
        "final_signal": ["BUY", "SELL", "BUY", "SELL"],
        "strategy_score": [0.8, -0.7, 0.6, -0.5],
        "ml_prediction": [1.0, -1.0, 0.5, -0.4],
        "sentiment_composite": [0.4, -0.3, 0.2, -0.2],
        "rolling_vol_20": [0.20, 0.30, 0.22, 0.28],
    })

    allocator = PortfolioAllocator(max_position_weight=0.6, max_gross_exposure=1.0)
    result = allocator.construct_portfolio(frame, capital=100000)

    assert {"portfolio_weight", "target_position_units", "portfolio_value", "gross_exposure"}.issubset(result.columns)
    gross_by_date = result.groupby("Date")["portfolio_weight"].apply(lambda series: series.abs().sum())
    assert (gross_by_date <= 1.000001).all()
    assert result.loc[result["symbol"] == "AAA", "portfolio_weight"].iloc[0] > 0
    assert result.loc[result["symbol"] == "BBB", "portfolio_weight"].iloc[0] < 0


def test_portfolio_allocator_uses_prediction_confidence_in_weighting():

    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "symbol": ["AAA", "BBB"],
        "Close": [100.0, 100.0],
        "final_signal": ["BUY", "BUY"],
        "strategy_score": [0.5, 0.5],
        "ml_prediction": [0.5, 0.5],
        "sentiment_composite": [0.0, 0.0],
        "prediction_confidence": [0.9, 0.55],
        "rolling_vol_20": [0.2, 0.2],
    })

    allocator = PortfolioAllocator(max_position_weight=1.0, max_gross_exposure=1.0)
    result = allocator.construct_portfolio(frame, capital=100000)

    weights = result.set_index("symbol")["portfolio_weight"]
    assert weights["AAA"] > weights["BBB"]
