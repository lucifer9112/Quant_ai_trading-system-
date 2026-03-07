"""
USAGE EXAMPLES - Improved Professional Quant Trading System

This file demonstrates how to use the new risk management and metrics modules.
"""

# =============================================================================
# EXAMPLE 1: Kelly Criterion Position Sizing
# =============================================================================

from risk_management.kelly_criterion import KellyCriterion

# Initialize Kelly calculator
kelly = KellyCriterion(
    max_kelly_fraction=0.25,  # Cap at 25% of capital
    safety_factor=2.0,         # Use half-Kelly for safety
    min_trades=20,             # Need 20+ trades for reliability
)

# Method 1: Calculate from historical trade returns
trade_returns = [0.02, -0.01, 0.03, -0.015, 0.025, 0.01, -0.02, 0.04]
kelly_result = kelly.from_trade_returns(trade_returns, start_capital=100000)

print(f"Win Rate: {kelly_result.win_rate:.2%}")
print(f"Payoff Ratio: {kelly_result.payoff_ratio:.3f}")
print(f"Kelly Fraction: {kelly_result.kelly_fraction:.2%}")
print(f"Safe Fraction (half-Kelly): {kelly_result.safe_fraction:.2%}")
print(f"Optimal Leverage: {kelly_result.optimal_leverage:.2f}x")


# Method 2: Calculate from win/loss statistics
kelly_fraction_safe = kelly.from_win_loss(
    win_rate=0.55,      # 55% win rate
    avg_win=1.02,       # Average 2% win
    avg_loss=0.98,      # Average 2% loss
    num_trades=100,
)
print(f"\nKelly fraction from stats: {kelly_fraction_safe:.2%}")


# Method 3: Calculate position size for a specific trade
shares = kelly.position_size(
    account_equity=100000,
    entry_price=150.0,
    stop_loss_price=140.0,
    win_rate=0.55,
    avg_win=1.02,
    avg_loss=0.98,
    num_trades=100,
)
print(f"Shares to buy based on Kelly: {shares:.2f}")


# =============================================================================
# EXAMPLE 2: Advanced Position Sizing
# =============================================================================

import numpy as np
from risk_management.position_sizer import AdvancedPositionSizer

sizer = AdvancedPositionSizer(
    initial_capital=100000,
    max_position_weight=0.25,
    max_leverage=2.0,
    volatility_target=0.15,  # Target 15% annual volatility
)

# Volatility-adjusted sizing
prices = np.array([100, 101, 99, 102, 101, 100, 103, 102, 101, 104])
size_multiplier = sizer.volatility_adjusted(
    prices=prices,
    signal_strength=0.8,  # Signal confidence
    lookback_periods=5,
)
print(f"Position size multiplier (vol-adjusted): {size_multiplier:.2%}")


# Risk-parity sizing across multiple assets
volatilities = [0.15, 0.20, 0.12, 0.18]  # Annual vols
signals = [0.7, -0.5, 0.9, 0.3]          # Signal strengths
rp_weights = sizer.risk_parity(volatilities, signals)
print(f"\nRisk-parity weights: {rp_weights}")


# =============================================================================
# EXAMPLE 3: Portfolio Risk Management
# =============================================================================

import pandas as pd
from risk_management.portfolio_risk import PortfolioRiskManager

risk_mgr = PortfolioRiskManager(
    initial_capital=100000,
    max_drawdown_pct=0.20,     # 20% max drawdown
    max_concentration=0.30,     # 30% max per position
    max_sector_concentration=0.50,
)

# Calculate drawdown metrics
equity_curve = np.array([100000, 102000, 101000, 99000, 98000, 105000, 110000])
max_dd, current_dd, duration = risk_mgr.calculate_drawdown(equity_curve)
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Current Drawdown: {current_dd:.2%}")
print(f"Duration: {duration} periods")


# Calculate VaR and CVaR
returns = [-0.02, 0.03, -0.01, 0.05, -0.03, 0.02, 0.04, -0.015]
var_95 = risk_mgr.value_at_risk(returns, confidence=0.95)
cvar_95 = risk_mgr.conditional_var(returns, confidence=0.95)
print(f"\nVaR (95%): {var_95:.2%}")
print(f"CVaR (95%): {cvar_95:.2%}")


# Check portfolio risk limits
position_weights = {'RELIANCE': 0.25, 'TCS': 0.20, 'INFY': 0.15, 'WIPRO': 0.10}
is_valid, violations = risk_mgr.check_portfolio_limits(position_weights)
print(f"Portfolio valid: {is_valid}")
if violations:
    for v in violations:
        print(f"  - {v}")


# =============================================================================
# EXAMPLE 4: Performance Metrics
# =============================================================================

from metrics_engine.performance_metrics import PerformanceAnalyzer

perf_analyzer = PerformanceAnalyzer(
    risk_free_rate=0.04,
    trading_days_per_year=252,
)

equity_curve = [100000, 102000, 103500, 102000, 105000, 108000, 110000, 109000, 115000]
metrics = perf_analyzer.calculate_metrics(equity_curve)

print(f"\nPerformance Metrics:")
print(f"  Total Return: {metrics.total_return:.2%}")
print(f"  Annualized Return: {metrics.annualized_return:.2%}")
print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
print(f"  Calmar Ratio: {metrics.calmar_ratio:.3f}")
print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")


# =============================================================================
# EXAMPLE 5: Risk Metrics (VaR, Drawdown, CVaR)
# =============================================================================

from metrics_engine.risk_metrics import RiskAnalyzer

risk_analyzer = RiskAnalyzer(trading_days_per_year=252)

returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns
equity_curve = [100000]
for ret in returns:
    equity_curve.append(equity_curve[-1] * (1 + ret))

risk_metrics = risk_analyzer.calculate_metrics(equity_curve)

print(f"\nRisk Metrics:")
print(f"  Max Drawdown: {risk_metrics.max_drawdown:.2%}")
print(f"  Avg Drawdown: {risk_metrics.avg_drawdown:.2%}")
print(f"  VaR (95%): {risk_metrics.var_95:.2%}")
print(f"  CVaR (95%): {risk_metrics.cvar_95:.2%}")
print(f"  Recovery Factor: {risk_metrics.recovery_factor:.3f}")
print(f"  Tail Ratio: {risk_metrics.tail_ratio:.3f}")
print(f"  Skewness: {risk_metrics.skewness:.3f}")
print(f"  Excess Kurtosis: {risk_metrics.kurtosis:.3f}")


# =============================================================================
# EXAMPLE 6: Trade Metrics - Win Rate, Profit Factor, etc.
# =============================================================================

from metrics_engine.trade_metrics import TradeAnalyzer, Trade
from datetime import datetime, timedelta

trade_analyzer = TradeAnalyzer()

# Create sample trades
trades = [
    Trade(
        entry_date=datetime(2024, 1, 1),
        exit_date=datetime(2024, 1, 5),
        symbol='RELIANCE',
        entry_price=2500,
        exit_price=2550,
        quantity=100,
        trade_type='LONG',
        pnl=5000,
        pnl_pct=0.02,
        duration_days=5,
        reason='TAKE_PROFIT'
    ),
    Trade(
        entry_date=datetime(2024, 1, 2),
        exit_date=datetime(2024, 1, 3),
        symbol='TCS',
        entry_price=3500,
        exit_price=3420,
        quantity=50,
        trade_type='LONG',
        pnl=-4000,
        pnl_pct=-0.0228,
        duration_days=1,
        reason='STOP_LOSS'
    ),
]

trade_metrics = trade_analyzer.calculate_metrics(trades)

print(f"\nTrade Metrics:")
print(f"  Total Trades: {trade_metrics.num_trades}")
print(f"  Winning Trades: {trade_metrics.num_wins}")
print(f"  Losing Trades: {trade_metrics.num_losses}")
print(f"  Win Rate: {trade_metrics.win_rate:.2%}")
print(f"  Profit Factor: {trade_metrics.profit_factor:.3f}")
print(f"  Payoff Ratio: {trade_metrics.payoff_ratio:.3f}")
print(f"  Avg Win: ${trade_metrics.avg_win:.2f}")
print(f"  Avg Loss: ${trade_metrics.avg_loss:.2f}")
print(f"  Expectancy: ${trade_metrics.expectancy:.2f}")


# =============================================================================
# EXAMPLE 7: Drawdown Analysis
# =============================================================================

from metrics_engine.drawdown_analysis import DrawdownAnalyzer

dd_analyzer = DrawdownAnalyzer()

equity_curve = [100000, 95000, 92000, 88000, 90000, 98000, 102000, 100000, 99000, 105000]
analysis = dd_analyzer.analyze_equity_curve(equity_curve)

print(f"\nDrawdown Analysis:")
print(f"  Max Drawdown: {analysis['max_drawdown']:.2%}")
print(f"  Avg Drawdown: {analysis['avg_drawdown']:.2%}")
print(f"  Num Drawdown Periods: {analysis['num_drawdown_periods']}")

for i, period in enumerate(analysis['periods']):
    print(f"\n  Period {i+1}:")
    print(f"    Depth: {period.depth:.2%}")
    print(f"    Duration: {period.duration} periods")
    if period.recovery_duration:
        print(f"    Recovery Time: {period.recovery_duration} periods")


# =============================================================================
# EXAMPLE 8: Comprehensive Metrics Aggregation
# =============================================================================

from metrics_engine.metrics_aggregator import MetricsAggregator

agg = MetricsAggregator(risk_free_rate=0.04, trading_days_per_year=252)

# Generate sample equity curve
np.random.seed(42)
daily_returns = np.random.normal(0.0005, 0.015, 252)
equity_curve = [100000]
for ret in daily_returns:
    equity_curve.append(equity_curve[-1] * (1 + ret))

# Calculate all metrics at once
report = agg.calculate_all_metrics(equity_curve, trades=None)

print(f"\nComprehensive Backtest Report:")
print(f"  Starting Capital: ${report.summary['starting_capital']:.2f}")
print(f"  Ending Capital: ${report.summary['ending_capital']:.2f}")
print(f"  Total Return: {report.summary['total_return_pct']:.2f}%")
print(f"  Sharpe Ratio: {report.summary['sharpe_ratio']:.3f}")
print(f"  Sortino Ratio: {report.summary['sortino_ratio']:.3f}")
print(f"  Max Drawdown: {report.summary['max_drawdown_pct']:.2f}%")

# Display as table
metrics_table = report.to_dataframe()
print("\nMetrics Table:")
print(metrics_table.to_string())

# Export HTML report
html = report.to_dataframe().to_html()
# Can save to file
# with open('report.html', 'w') as f:
#     f.write(html)


# =============================================================================
# EXAMPLE 9: Integration with Professional Backtester
# =============================================================================

from backtesting.professional_backtester import ProfessionalBacktester

# Initialize professional backtester
pb = ProfessionalBacktester(
    initial_capital=100000,
    transaction_cost_bps=5.0,
    slippage_bps=3.0,
    max_drawdown_pct=0.20,
    kelly_enabled=True,
)

# Run a backtest (assuming you have the data)
# result = pb.run(df)
# result.print_summary()
# result.export_html_report('backtest_report.html')


# =============================================================================
# EXAMPLE 10: Using Kelly Criterion in Live Trading Context
# =============================================================================

# Track historical performance
historical_trades = [
    {'pnl': 500, 'return': 0.02},
    {'pnl': -300, 'return': -0.03},
    {'pnl': 800, 'return': 0.025},
    # ... more trades
]

# Estimate win statistics
wins = [t for t in historical_trades if t['pnl'] > 0]
losses = [t for t in historical_trades if t['pnl'] < 0]

win_rate = len(wins) / len(historical_trades)
avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses)) if losses else 0

# Calculate kelly for next trade
kelly_calc = KellyCriterion(safety_factor=2.0)
kelly_frac = kelly_calc.from_win_loss(win_rate, avg_win, avg_loss)

print(f"\nNext trade Kelly sizing:")
print(f"  Win Rate: {win_rate:.2%}")
print(f"  Avg Win: ${avg_win:.2f}")
print(f"  Avg Loss: ${avg_loss:.2f}")
print(f"  Kelly Fraction (safe): {kelly_frac:.2%}")

# Size position accordingly
account_equity = 100000
position_value = account_equity * kelly_frac
print(f"  Position Size: ${position_value:.2f}")

