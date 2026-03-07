"""
Metrics Aggregator - Unified Interface for All Backtesting Metrics

Combines:
- Performance Metrics (Sharpe, Sortino, etc.)
- Risk Metrics (VaR, CVaR, Drawdown)
- Trade Metrics (Win Rate, Profit Factor)
- Drawdown Analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from metrics_engine.performance_metrics import PerformanceAnalyzer, PerformanceMetrics
from metrics_engine.risk_metrics import RiskAnalyzer, RiskMetrics
from metrics_engine.trade_metrics import TradeAnalyzer, Trade, TradeMetrics
from metrics_engine.drawdown_analysis import DrawdownAnalyzer, DrawdownPeriod


@dataclass
class BacktestMetricsReport:
    """Complete backtesting metrics report"""
    performance: PerformanceMetrics
    risk: RiskMetrics
    trades: TradeMetrics
    summary: Dict  # High-level summary
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'performance': asdict(self.performance),
            'risk': asdict(self.risk),
            'trades': asdict(self.trades),
            'summary': self.summary,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for display."""
        rows = []
        
        # Performance metrics
        rows.append({'Category': 'Performance', 'Metric': 'Total Return', 'Value': f"{self.performance.total_return:.2%}"})
        rows.append({'Category': 'Performance', 'Metric': 'Annualized Return', 'Value': f"{self.performance.annualized_return:.2%}"})
        rows.append({'Category': 'Performance', 'Metric': 'Sharpe Ratio', 'Value': f"{self.performance.sharpe_ratio:.3f}"})
        rows.append({'Category': 'Performance', 'Metric': 'Sortino Ratio', 'Value': f"{self.performance.sortino_ratio:.3f}"})
        rows.append({'Category': 'Performance', 'Metric': 'Calmar Ratio', 'Value': f"{self.performance.calmar_ratio:.3f}"})
        
        # Risk metrics
        rows.append({'Category': 'Risk', 'Metric': 'Max Drawdown', 'Value': f"{self.risk.max_drawdown:.2%}"})
        rows.append({'Category': 'Risk', 'Metric': 'Avg Drawdown', 'Value': f"{self.risk.avg_drawdown:.2%}"})
        rows.append({'Category': 'Risk', 'Metric': 'VaR (95%)', 'Value': f"{self.risk.var_95:.2%}"})
        rows.append({'Category': 'Risk', 'Metric': 'CVaR (95%)', 'Value': f"{self.risk.cvar_95:.2%}"})
        rows.append({'Category': 'Risk', 'Metric': 'Recovery Factor', 'Value': f"{self.risk.recovery_factor:.3f}"})
        rows.append({'Category': 'Risk', 'Metric': 'Tail Ratio', 'Value': f"{self.risk.tail_ratio:.3f}"})
        
        # Trade metrics
        rows.append({'Category': 'Trading', 'Metric': 'Total Trades', 'Value': f"{self.trades.num_trades}"})
        rows.append({'Category': 'Trading', 'Metric': 'Win Rate', 'Value': f"{self.trades.win_rate:.2%}"})
        rows.append({'Category': 'Trading', 'Metric': 'Profit Factor', 'Value': f"{self.trades.profit_factor:.3f}"})
        rows.append({'Category': 'Trading', 'Metric': 'Payoff Ratio', 'Value': f"{self.trades.payoff_ratio:.3f}"})
        rows.append({'Category': 'Trading', 'Metric': 'Avg Win', 'Value': f"${self.trades.avg_win:.2f}"})
        rows.append({'Category': 'Trading', 'Metric': 'Avg Loss', 'Value': f"${self.trades.avg_loss:.2f}"})
        rows.append({'Category': 'Trading', 'Metric': 'Expectancy', 'Value': f"${self.trades.expectancy:.2f}"})
        
        return pd.DataFrame(rows)


class MetricsAggregator:
    """
    Unified interface for calculating all backtesting metrics.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.04,
        trading_days_per_year: int = 252,
        benchmark_returns: Optional[List[float]] = None,
    ):
        """
        Args:
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Trading days per year
            benchmark_returns: Optional benchmark returns
        """
        self.perf_analyzer = PerformanceAnalyzer(
            risk_free_rate=risk_free_rate,
            trading_days_per_year=trading_days_per_year,
            benchmark_returns=benchmark_returns,
        )
        self.risk_analyzer = RiskAnalyzer(
            trading_days_per_year=trading_days_per_year,
        )
        self.trade_analyzer = TradeAnalyzer()
        self.drawdown_analyzer = DrawdownAnalyzer()
    
    def calculate_all_metrics(
        self,
        equity_curve: List[float],
        trades: Optional[List[Trade]] = None,
    ) -> BacktestMetricsReport:
        """
        Calculate all metrics for a backtest.
        
        Args:
            equity_curve: List of portfolio values over time
            trades: Optional list of Trade objects
            
        Returns:
            BacktestMetricsReport with all metrics
        """
        # Performance metrics
        performance = self.perf_analyzer.calculate_metrics(equity_curve)
        
        # Risk metrics
        returns = self.perf_analyzer.calculate_returns(equity_curve)
        risk = self.risk_analyzer.calculate_metrics(equity_curve, returns)
        
        # Trade metrics
        if trades is None:
            trades = []
        trade = self.trade_analyzer.calculate_metrics(trades)
        
        # Summary
        summary = self._create_summary(performance, risk, trade, equity_curve)
        
        return BacktestMetricsReport(
            performance=performance,
            risk=risk,
            trades=trade,
            summary=summary,
        )
    
    def _create_summary(
        self,
        performance: PerformanceMetrics,
        risk: RiskMetrics,
        trades: TradeMetrics,
        equity_curve: List[float],
    ) -> Dict:
        """
        Create high-level summary metrics.
        
        Args:
            performance: Performance metrics
            risk: Risk metrics
            trades: Trade metrics
            equity_curve: Equity curve
            
        Returns:
            Summary dict
        """
        equity_array = np.array(equity_curve)
        start_val = equity_array[0]
        end_val = equity_array[-1]
        
        return {
            'starting_capital': start_val,
            'ending_capital': end_val,
            'total_gain': end_val - start_val,
            'total_return_pct': ((end_val / start_val) - 1.0) * 100,
            'sharpe_ratio': performance.sharpe_ratio,
            'sortino_ratio': performance.sortino_ratio,
            'calmar_ratio': performance.calmar_ratio,
            'max_drawdown_pct': risk.max_drawdown * 100,
            'recovery_factor': risk.recovery_factor,
            'win_rate_pct': trades.win_rate * 100 if trades.num_trades > 0 else 0,
            'profit_factor': trades.profit_factor,
            'total_trades': trades.num_trades,
        }
    
    def create_performance_comparison_table(
        self,
        backtest_results: List[BacktestMetricsReport],
        strategy_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create comparison table for multiple backtest results.
        
        Args:
            backtest_results: List of BacktestMetricsReport objects
            strategy_names: Optional names for strategies
            
        Returns:
            DataFrame with comparison
        """
        if strategy_names is None:
            strategy_names = [f"Strategy_{i}" for i in range(len(backtest_results))]
        
        rows = []
        
        for name, result in zip(strategy_names, backtest_results):
            rows.append({
                'Strategy': name,
                'Total Return': f"{result.performance.total_return:.2%}",
                'Annualized': f"{result.performance.annualized_return:.2%}",
                'Sharpe': f"{result.performance.sharpe_ratio:.3f}",
                'Sortino': f"{result.performance.sortino_ratio:.3f}",
                'Max DD': f"{result.risk.max_drawdown:.2%}",
                'Win Rate': f"{result.trades.win_rate:.2%}",
                'Profit Factor': f"{result.trades.profit_factor:.3f}",
                'Num Trades': result.trades.num_trades,
            })
        
        return pd.DataFrame(rows)
    
    def generate_html_report(
        self,
        report: BacktestMetricsReport,
        strategy_name: str = "Trading Strategy",
        output_file: Optional[str] = None,
    ) -> str:
        """
        Generate HTML report of metrics.
        
        Args:
            report: BacktestMetricsReport object
            strategy_name: Name of strategy
            output_file: Optional file to write HTML to
            
        Returns:
            HTML string
        """
        metrics_df = report.to_dataframe()
        metrics_html = metrics_df.to_html(index=False)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{strategy_name} - Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{strategy_name}</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Starting Capital:</strong> ${report.summary['starting_capital']:.2f}</p>
                <p><strong>Ending Capital:</strong> ${report.summary['ending_capital']:.2f}</p>
                <p><strong>Total Return:</strong> {report.summary['total_return_pct']:.2f}%</p>
                <p><strong>Sharpe Ratio:</strong> {report.summary['sharpe_ratio']:.3f}</p>
                <p><strong>Max Drawdown:</strong> {report.summary['max_drawdown_pct']:.2f}%</p>
                <p><strong>Win Rate:</strong> {report.summary['win_rate_pct']:.2f}%</p>
                <p><strong>Total Trades:</strong> {report.summary['total_trades']}</p>
            </div>
            <h2>Detailed Metrics</h2>
            {metrics_html}
        </body>
        </html>
        """
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html)
        
        return html
