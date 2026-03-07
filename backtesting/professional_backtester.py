"""
Enhanced Backtester Integration Example

Shows how to integrate:
1. Advanced Risk Management (Kelly, Position Sizing)
2. Comprehensive Metrics Engine
3. Professional Backtesting

This is an enhanced wrapper around the existing AdvancedBacktester.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
from dataclasses import dataclass

from backtesting.engine.advanced_backtester import AdvancedBacktester, AdvancedBacktestResult
from risk_management import KellyCriterion, AdvancedPositionSizer, PortfolioRiskManager
from metrics_engine import MetricsAggregator, Trade
from utils.logger import get_logger


logger = get_logger()


class ProfessionalBacktester:
    """
    Enhanced backtester combining advanced risk management and comprehensive metrics.
    
    Usage:
        backtester = ProfessionalBacktester(config)
        result = backtester.run(df)
        report = result.metrics_report
        print(report.summary)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 3.0,
        max_drawdown_pct: float = 0.20,
        kelly_enabled: bool = True,
        risk_management_config: Optional[Dict] = None,
    ):
        """
        Args:
            initial_capital: Starting capital
            transaction_cost_bps: Transaction costs in basis points
            slippage_bps: Slippage in basis points
            max_drawdown_pct: Maximum acceptable drawdown
            kelly_enabled: Whether to use Kelly criterion for position sizing
            risk_management_config: Optional config dict for risk management
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.max_drawdown_pct = max_drawdown_pct
        self.kelly_enabled = kelly_enabled
        
        # Initialize risk management components
        self.kelly = KellyCriterion(
            max_kelly_fraction=0.25,
            safety_factor=2.0,
        )
        self.position_sizer = AdvancedPositionSizer(
            initial_capital=initial_capital,
            max_position_weight=0.25,
            max_leverage=2.0,
            volatility_target=0.15,
        )
        self.portfolio_risk = PortfolioRiskManager(
            initial_capital=initial_capital,
            max_drawdown_pct=max_drawdown_pct,
        )
        
        # Initialize metrics aggregator
        self.metrics_agg = MetricsAggregator(
            risk_free_rate=0.04,
            trading_days_per_year=252,
        )
        
        # Initialize base backtester
        self.backtester = AdvancedBacktester(
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            max_drawdown_pct=max_drawdown_pct,
        )
    
    def run(
        self,
        df: pd.DataFrame,
        signal_column: str = "final_signal",
    ) -> 'ProfessionalBacktestResult':
        """
        Run professional backtest with full metrics.
        
        Args:
            df: DataFrame with OHLCV data and signals
            signal_column: Column name for trading signals
            
        Returns:
            ProfessionalBacktestResult with metrics and analysis
        """
        logger.info("Starting professional backtest...")
        
        # Run base backtest
        backtest_result = self.backtester.backtest(df, signal_column=signal_column)
        
        # Extract equity curve
        equity_curve = backtest_result.equity_curve['Portfolio_Value'].tolist()
        
        # Calculate comprehensive metrics
        metrics_report = self.metrics_agg.calculate_all_metrics(
            equity_curve=equity_curve,
            trades=None,  # TODO: extract trades from backtest result
        )
        
        # Create result object
        result = ProfessionalBacktestResult(
            equity_curve=backtest_result.equity_curve,
            metrics_report=metrics_report,
            original_metrics=backtest_result.metrics,
        )
        
        logger.info(f"Backtest complete. Total return: {metrics_report.summary['total_return_pct']:.2f}%")
        logger.info(f"Sharpe ratio: {metrics_report.performance.sharpe_ratio:.3f}")
        logger.info(f"Max drawdown: {metrics_report.summary['max_drawdown_pct']:.2f}%")
        
        return result
    
    def run_with_kelly_sizing(
        self,
        df: pd.DataFrame,
        signal_column: str = "final_signal",
        win_rate: float = 0.55,
        avg_win: float = 1.02,
        avg_loss: float = 0.98,
    ) -> 'ProfessionalBacktestResult':
        """
        Run backtest with Kelly criterion position sizing.
        
        Args:
            df: DataFrame with data
            signal_column: Signal column name
            win_rate: Historical win rate (0-1)
            avg_win: Average profit on winning trades
            avg_loss: Average loss on losing trades
            
        Returns:
            Backtest result with Kelly-optimized sizing
        """
        logger.info("Running backtest with Kelly criterion position sizing...")
        
        # Calculate Kelly fraction
        kelly_fraction = self.kelly.from_win_loss(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )
        
        logger.info(f"Calculated Kelly fraction: {kelly_fraction:.2%}")
        
        # Adjust position sizing in dataframe
        df_kelly = df.copy()
        
        if 'portfolio_weight' in df_kelly.columns:
            df_kelly['portfolio_weight'] = df_kelly['portfolio_weight'] * kelly_fraction
        elif 'target_weight' in df_kelly.columns:
            df_kelly['target_weight'] = df_kelly['target_weight'] * kelly_fraction
        
        # Run backtest
        return self.run(df_kelly, signal_column=signal_column)


@dataclass
class ProfessionalBacktestResult:
    """Result from professional backtest with comprehensive metrics."""
    equity_curve: pd.DataFrame
    metrics_report: 'BacktestMetricsReport'
    original_metrics: Dict
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*60)
        print(f"{'BACKTEST SUMMARY':^60}")
        print("="*60)
        
        summary = self.metrics_report.summary
        print(f"Starting Capital:        ${summary['starting_capital']:>15,.2f}")
        print(f"Ending Capital:          ${summary['ending_capital']:>15,.2f}")
        print(f"Total Gain:              ${summary['total_gain']:>15,.2f}")
        print(f"Total Return:            {summary['total_return_pct']:>15.2f}%")
        print("-"*60)
        print(f"Sharpe Ratio:            {self.metrics_report.performance.sharpe_ratio:>15.3f}")
        print(f"Sortino Ratio:           {self.metrics_report.performance.sortino_ratio:>15.3f}")
        print(f"Calmar Ratio:            {self.metrics_report.performance.calmar_ratio:>15.3f}")
        print("-"*60)
        print(f"Max Drawdown:            {self.metrics_report.risk.max_drawdown:>15.2%}")
        print(f"Avg Drawdown:            {self.metrics_report.risk.avg_drawdown:>15.2%}")
        print(f"Recovery Factor:         {self.metrics_report.risk.recovery_factor:>15.3f}")
        print(f"VaR (95%):               {self.metrics_report.risk.var_95:>15.2%}")
        print("-"*60)
        print(f"Total Trades:            {self.metrics_report.trades.num_trades:>15}")
        print(f"Winning Trades:          {self.metrics_report.trades.num_wins:>15}")
        print(f"Losing Trades:           {self.metrics_report.trades.num_losses:>15}")
        print(f"Win Rate:                {self.metrics_report.trades.win_rate:>15.2%}")
        print(f"Profit Factor:           {self.metrics_report.trades.profit_factor:>15.3f}")
        print(f"Payoff Ratio:            {self.metrics_report.trades.payoff_ratio:>15.3f}")
        print("-"*60)
        print(f"Avg Win:                 ${self.metrics_report.trades.avg_win:>15.2f}")
        print(f"Avg Loss:                ${self.metrics_report.trades.avg_loss:>15.2f}")
        print(f"Expectancy per Trade:    ${self.metrics_report.trades.expectancy:>15.2f}")
        print("="*60 + "\n")
    
    def export_metrics_csv(self, filename: str):
        """Export metrics to CSV."""
        metrics_df = self.metrics_report.to_dataframe()
        metrics_df.to_csv(filename, index=False)
        logger.info(f"Metrics exported to {filename}")
    
    def export_html_report(self, filename: str, strategy_name: str = "Trading Strategy"):
        """Export HTML report."""
        html = self.metrics_report.to_dataframe().to_html()
        
        full_html = f"""
        <html>
        <head>
            <title>{strategy_name} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>{strategy_name}</h1>
            {html}
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(full_html)
        
        logger.info(f"HTML report exported to {filename}")


# Example usage
if __name__ == "__main__":
    # This would be integrated into main.py
    # Example:
    # backtester = ProfessionalBacktester(initial_capital=100000)
    # result = backtester.run(df)
    # result.print_summary()
    # result.export_html_report("backtest_report.html")
    pass
