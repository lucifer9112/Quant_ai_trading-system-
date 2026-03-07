import pandas as pd

from backtesting.engine.advanced_backtester import AdvancedBacktester


def test_advanced_backtester_returns_metrics_and_costs():

    frame = pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-01-02", "2024-01-02",
            "2024-01-03", "2024-01-03",
            "2024-01-04", "2024-01-04",
        ]),
        "symbol": ["AAA", "BBB"] * 4,
        "Close": [100, 200, 102, 202, 104, 198, 103, 205],
        "final_signal": ["BUY", "BUY", "BUY", "HOLD", "SELL", "BUY", "HOLD", "SELL"],
        "rolling_vol_20": [0.20, 0.25, 0.20, 0.25, 0.22, 0.23, 0.21, 0.24],
    })

    backtester = AdvancedBacktester(transaction_cost_bps=10, slippage_bps=5)
    result = backtester.backtest(frame)

    assert not result.equity_curve.empty
    assert result.equity_curve["transaction_cost"].sum() > 0
    assert {"total_return", "sharpe_ratio", "max_drawdown", "win_rate"}.issubset(result.metrics.keys())
