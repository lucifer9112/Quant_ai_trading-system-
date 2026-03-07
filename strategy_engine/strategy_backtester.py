from backtesting.engine.advanced_backtester import AdvancedBacktester


class StrategyBacktester:

    def __init__(self, **backtester_kwargs):

        self.engine = AdvancedBacktester(**backtester_kwargs)

    def backtest(self, df, signal_column="strategy_score"):

        result = self.engine.backtest(df, signal_column=signal_column)
        equity_curve = result.equity_curve.copy()
        equity_curve.attrs["metrics"] = result.metrics

        return equity_curve
