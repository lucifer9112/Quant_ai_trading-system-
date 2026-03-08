"""
Phase 3: Advanced Visualization - Usage Examples

Demonstrates all visualization modules and report generation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from visualization import (
    EquityCurveVisualizer,
    TradeSignalVisualizer,
    PerformanceDashboard,
    IndicatorVisualizer,
    ReportGenerator,
)


# ============================================================================
# Example 1: Equity Curve Visualization
# ============================================================================

def example_equity_curves():
    """Generate equity curve visualizations."""
    print("=" * 60)
    print("EXAMPLE 1: Equity Curve Visualization")
    print("=" * 60)
    
    # Generate synthetic equity curve
    n_days = 252
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Portfolio equity curve (random walk with drift)
    returns = np.random.normal(0.0005, 0.015, n_days)
    equity_curve = 100 * np.cumprod(1 + returns)
    
    # Benchmark equity curve
    benchmark_returns = np.random.normal(0.0003, 0.012, n_days)
    benchmark = 100 * np.cumprod(1 + benchmark_returns)
    
    # Create visualizations
    viz = EquityCurveVisualizer()
    
    # 1. Basic equity curve
    print("\n1. Creating equity curve chart...")
    fig1 = viz.plot_equity_curve(equity_curve, dates, benchmark,
                                  title="Portfolio vs Benchmark",
                                  save_path="visualization_reports/equity_vs_benchmark.png")
    
    # 2. Equity curve with drawdown bands
    print("2. Creating equity curve with drawdown bands...")
    fig2 = viz.plot_equity_with_drawdown(equity_curve, dates,
                                         title="Equity Curve with Drawdown Analysis",
                                         save_path="visualization_reports/equity_with_dd.png")
    
    # 3. Underwater plot
    print("3. Creating underwater plot...")
    fig3 = viz.plot_underwater(equity_curve, dates,
                               title="Underwater Plot - Drawdown Severity",
                               save_path="visualization_reports/underwater.png")
    
    # 4. Cumulative returns
    print("4. Creating cumulative returns chart...")
    fig4 = viz.plot_cumulative_returns(equity_curve, dates, benchmark,
                                       title="Cumulative Returns Comparison",
                                       save_path="visualization_reports/cumulative_returns.png")
    
    # 5. Rolling Sharpe ratio
    print("5. Creating rolling Sharpe ratio...")
    fig5 = viz.plot_rolling_metrics(equity_curve, dates, window=60,
                                    title="Rolling Sharpe Ratio (60-Day Window)",
                                    save_path="visualization_reports/rolling_sharpe.png")
    
    print("\n✓ All equity curve visualizations created!")
    return fig1, fig2, fig3, fig4, fig5


# ============================================================================
# Example 2: Trade Signal Visualization
# ============================================================================

def example_trade_signals():
    """Generate trade signal visualizations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Trade Signal Visualization")
    print("=" * 60)
    
    # Generate synthetic price and signals
    n_days = 100
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    prices = 100 + np.cumsum(np.random.normal(0, 1, n_days))
    
    # Synthetic trade signals
    buy_signals = [(10, prices[10]), (30, prices[30]), (50, prices[50]), (70, prices[70])]
    sell_signals = [(20, prices[20]), (40, prices[40]), (60, prices[60]), (80, prices[80])]
    
    # Stop losses and take profits
    stop_losses = [(11, prices[10] * 0.98), (31, prices[30] * 0.97)]
    take_profits = [(11, prices[10] * 1.02), (31, prices[30] * 1.03)]
    
    # Position P&L
    position_pnl = 100 + np.cumsum(np.random.normal(0.1, 2, n_days))
    position_sizes = np.random.randint(10, 100, n_days)
    
    viz = TradeSignalVisualizer()
    
    # 1. Price with signals
    print("\n1. Creating price chart with trading signals...")
    fig1 = viz.plot_price_with_signals(
        dates, prices, buy_signals, sell_signals, stop_losses, take_profits,
        title="Price Chart with Trading Signals",
        save_path="visualization_reports/price_with_signals.png"
    )
    
    # 2. Position P&L
    print("2. Creating position P&L chart...")
    fig2 = viz.plot_position_pnl(
        dates, position_pnl, position_sizes,
        title="Position P&L with Position Sizing",
        save_path="visualization_reports/position_pnl.png"
    )
    
    # 3. Trade analysis
    wins = [100, 250, 150, 75, 300]
    losses = [-50, -100, -75, -120]
    print("3. Creating trade analysis chart...")
    fig3 = viz.plot_trade_analysis(
        wins, losses,
        title="Trade P&L Distribution Analysis",
        save_path="visualization_reports/trade_analysis.png"
    )
    
    # 4. Holding periods
    holding_periods = [5, 10, 7, 12, 8, 15, 6, 9, 11, 14]
    pnls = [50, 150, 75, 200, 100, 250, 25, 125, 175, 300]
    print("4. Creating holding period analysis...")
    fig4 = viz.plot_holding_periods(
        holding_periods, pnls,
        title="Holding Period vs Trade P&L",
        save_path="visualization_reports/holding_periods.png"
    )
    
    print("\n✓ All trade signal visualizations created!")
    return fig1, fig2, fig3, fig4


# ============================================================================
# Example 3: Performance Dashboard
# ============================================================================

def example_performance_dashboard():
    """Generate performance dashboard visualizations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Performance Dashboard")
    print("=" * 60)
    
    # Metrics
    metrics = {
        'Sharpe Ratio': 1.85,
        'Sortino Ratio': 2.45,
        'Calmar Ratio': 1.65,
        'Max Drawdown': -15.2,
        'Win Rate': 58.5,
        'Profit Factor': 2.35,
        'Annual Return': 18.5,
        'Annual Volatility': 10.2,
        'Expectancy': 125.50,
    }
    
    dashboard = PerformanceDashboard()
    
    # 1. Metrics grid
    print("\n1. Creating metrics grid...")
    fig1 = dashboard.plot_metrics_grid(
        metrics,
        title="Performance Metrics Summary",
        save_path="visualization_reports/metrics_grid.png"
    )
    
    # 2. Sector attribution
    print("2. Creating sector attribution pie chart...")
    sectors = {
        'Technology': 8.5,
        'Financials': 4.2,
        'Healthcare': 3.1,
        'Energy': -1.5,
        'Consumer': 2.7,
    }
    fig2 = dashboard.plot_attribution_pie(
        sectors,
        title="Return Attribution by Sector",
        save_path="visualization_reports/sector_attribution.png"
    )
    
    # 3. Monthly returns heatmap
    print("3. Creating monthly returns heatmap...")
    monthly_returns = {
        'Jan': 2.5, 'Feb': 1.8, 'Mar': 3.2, 'Apr': -0.5, 'May': 2.1,
        'Jun': 1.5, 'Jul': 2.8, 'Aug': -1.2, 'Sep': 3.5, 'Oct': 2.1,
        'Nov': 1.9, 'Dec': 2.6
    }
    fig3 = dashboard.plot_monthly_returns_heatmap(
        monthly_returns,
        title="Monthly Returns Performance",
        save_path="visualization_reports/monthly_returns.png"
    )
    
    # 4. Risk metrics timeline
    n_periods = 100
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_periods)]
    volatility = 10 + 2 * np.sin(np.arange(n_periods) / 10) + np.random.normal(0, 1, n_periods)
    max_dd = -15 + 5 * np.sin(np.arange(n_periods) / 15) + np.random.normal(0, 1, n_periods)
    var_95 = 2 + 0.5 * np.sin(np.arange(n_periods) / 10) + np.random.normal(0, 0.2, n_periods)
    
    print("4. Creating risk metrics timeline...")
    fig4 = dashboard.plot_risk_metrics_timeline(
        dates, volatility, max_dd, var_95,
        title="Risk Metrics Over Time",
        save_path="visualization_reports/risk_timeline.png"
    )
    
    # 5. Risk vs Return scatter
    print("5. Creating risk-return scatter...")
    strategies = {
        'Strategy A': (15.5, 9.2),
        'Strategy B': (18.2, 11.5),
        'Strategy C': (12.1, 7.8),
        'Benchmark': (10.5, 8.5),
    }
    fig5 = dashboard.plot_performance_vs_risk(
        strategies,
        title="Risk-Return Profile",
        save_path="visualization_reports/risk_return_scatter.png"
    )
    
    # 6. Rolling Sharpe ratio
    print("6. Creating rolling Sharpe ratio...")
    sharpe_ratios = 1.5 + 0.5 * np.sin(np.arange(n_periods) / 20) + np.random.normal(0, 0.3, n_periods)
    fig6 = dashboard.plot_rolling_sharpe(
        dates, sharpe_ratios,
        title="Rolling Sharpe Ratio (20-Day)",
        save_path="visualization_reports/rolling_sharpe_dashboard.png"
    )
    
    print("\n✓ All performance dashboard visualizations created!")
    return fig1, fig2, fig3, fig4, fig5, fig6


# ============================================================================
# Example 4: Technical Indicators
# ============================================================================

def example_indicators():
    """Generate technical indicator visualizations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Technical Indicators")
    print("=" * 60)
    
    # Generate price and indicators
    n_days = 100
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Price
    prices = 100 + np.cumsum(np.random.normal(0, 1, n_days))
    
    # Moving averages
    sma_20 = pd.Series(prices).rolling(20).mean().fillna(prices[0]).tolist()
    sma_50 = pd.Series(prices).rolling(50).mean().fillna(prices[0]).tolist()
    sma_200 = pd.Series(prices).rolling(200).mean().fillna(prices[0]).tolist()
    
    # Bollinger Bands
    sma = pd.Series(prices).rolling(20).mean()
    std = pd.Series(prices).rolling(20).std()
    upper_band = (sma + 2 * std).fillna(prices[0]).tolist()
    lower_band = (sma - 2 * std).fillna(prices[0]).tolist()
    
    # MACD
    ema_12 = pd.Series(prices).ewm(span=12).mean().tolist()
    ema_26 = pd.Series(prices).ewm(span=26).mean().tolist()
    macd_line = [e12 - e26 for e12, e26 in zip(ema_12, ema_26)]
    signal_line = pd.Series(macd_line).ewm(span=9).mean().tolist()
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    
    # RSI
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).fillna(50).tolist()
    
    # Volume
    volumes = np.random.randint(1000000, 5000000, n_days)
    
    viz = IndicatorVisualizer()
    
    # 1. Price with moving averages
    print("\n1. Creating price chart with moving averages...")
    fig1 = viz.plot_price_with_ma(
        dates, prices, sma_20, sma_50, sma_200,
        title="Price with SMA (20, 50, 200)",
        save_path="visualization_reports/ma_chart.png"
    )
    
    # 2. Bollinger Bands
    print("2. Creating Bollinger Bands chart...")
    fig2 = viz.plot_bollinger_bands(
        dates, prices, upper_band, lower_band, sma_20,
        title="Price with Bollinger Bands (20, 2)",
        save_path="visualization_reports/bollinger_bands.png"
    )
    
    # 3. MACD
    print("3. Creating MACD chart...")
    fig3 = viz.plot_macd(
        dates, macd_line, signal_line, histogram,
        title="MACD Indicator",
        save_path="visualization_reports/macd.png"
    )
    
    # 4. RSI
    print("4. Creating RSI chart...")
    fig4 = viz.plot_rsi(
        dates, rsi,
        title="Relative Strength Index (RSI 14)",
        save_path="visualization_reports/rsi.png"
    )
    
    # 5. Volume profile
    print("5. Creating volume profile...")
    fig5 = viz.plot_volume_profile(
        dates, volumes, prices,
        title="Trading Volume",
        save_path="visualization_reports/volume.png"
    )
    
    # 6. Multiple indicators dashboard
    print("6. Creating multi-indicator dashboard...")
    indicators_dict = {
        'RSI': rsi,
        'MACD': macd_line,
        'Volume (millions)': [v/1e6 for v in volumes],
    }
    fig6 = viz.plot_multiple_indicators(
        dates, prices, indicators_dict,
        title="Multi-Indicator Dashboard",
        save_path="visualization_reports/multi_indicators.png"
    )
    
    print("\n✓ All technical indicator visualizations created!")
    return fig1, fig2, fig3, fig4, fig5, fig6


# ============================================================================
# Example 5: Report Generation
# ============================================================================

def example_report_generation():
    """Generate comprehensive PDF reports."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Report Generation")
    print("=" * 60)
    
    # Generate all visualizations
    figs_equity = example_equity_curves()
    figs_signals = example_trade_signals()
    figs_dashboard = example_performance_dashboard()
    figs_indicators = example_indicators()
    
    all_figures = figs_equity + figs_signals + figs_dashboard + figs_indicators
    
    # Report generator
    gen = ReportGenerator('visualization_reports/')
    
    # Summary metrics
    summary = {
        'Total Return': '18.5%',
        'Annual Volatility': '10.2%',
        'Sharpe Ratio': 1.85,
        'Max Drawdown': '-15.2%',
        'Win Rate': '58.5%',
    }
    
    # 1. Full strategy report
    print("\n1. Generating comprehensive strategy report...")
    report_path = gen.generate_strategy_report(
        all_figures,
        title="Quantitative Trading Strategy - Comprehensive Analysis",
        summary=summary,
        filename="strategy_report.pdf"
    )
    print(f"   ✓ Report saved: {report_path}")
    
    # 2. Trade analysis report
    print("2. Generating trade analysis report...")
    trade_stats = {
        'total_trades': 45,
        'winning_trades': 26,
        'losing_trades': 19,
        'win_rate': 57.8,
        'avg_win': 245.50,
        'avg_loss': -125.75,
        'profit_factor': 2.45,
        'expectancy': 127.35,
        'max_win_streak': 5,
        'max_loss_streak': 3,
        'longest_trade': 25,
        'shortest_trade': 1,
    }
    trade_report = gen.generate_trade_analysis_report(
        [figs_signals[2], figs_signals[3]],
        trade_stats,
        filename="trade_analysis_report.pdf"
    )
    print(f"   ✓ Report saved: {trade_report}")
    
    # 3. Risk analysis report
    print("3. Generating risk analysis report...")
    risk_metrics = {
        'max_drawdown': -15.2,
        'avg_drawdown': -4.5,
        'max_dd_duration': 45,
        'var_95': -2.15,
        'cvar_95': -3.45,
        'annual_volatility': 10.2,
        'skewness': -0.35,
        'kurtosis': 2.15,
        'recovery_factor': 1.22,
        'tail_ratio': 0.95,
    }
    risk_report = gen.generate_risk_report(
        [figs_equity[2]],
        risk_metrics,
        filename="risk_analysis_report.pdf"
    )
    print(f"   ✓ Report saved: {risk_report}")
    
    # 4. Summary table report
    print("4. Generating summary table...")
    table_data = {
        'Month': ['January', 'February', 'March', 'April', 'May'],
        'Return (%)': [2.5, 1.8, 3.2, -0.5, 2.1],
        'Sharpe': [1.85, 1.92, 2.15, 0.85, 1.75],
        'Max DD (%)': [-8.5, -6.2, -5.1, -12.1, -7.8],
    }
    table_report = gen.generate_summary_table(
        table_data,
        filename="monthly_summary_table.pdf"
    )
    print(f"   ✓ Report saved: {table_report}")
    
    print("\n✓ All reports generated successfully!")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("PHASE 3: ADVANCED VISUALIZATION - COMPREHENSIVE EXAMPLES")
    print("=" * 60)
    
    # Create output directory
    import os
    os.makedirs("visualization_reports", exist_ok=True)
    
    # Run all examples
    example_equity_curves()
    example_trade_signals()
    example_performance_dashboard()
    example_indicators()
    example_report_generation()
    
    print("\n")
    print("=" * 60)
    print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - visualization_reports/*.png (individual charts)")
    print("  - visualization_reports/*.pdf (comprehensive reports)")
    print("\nNext Steps:")
    print("  1. Integrate visualizations into main.py pipeline")
    print("  2. Phase 4: Implement Regime Detection & Adaptive Allocation")
    print("  3. Phase 5: Enhanced Professional Backtester")
