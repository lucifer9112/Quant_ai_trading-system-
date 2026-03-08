"""
Visualization Module - Professional trading visualization and reporting

Modules:
- equity_curves: Equity curve, drawdown bands, underwater plots
- trade_signals: Buy/sell signals, position P&L, trade analysis
- performance_dashboard: Metrics grid, attribution, monthly heatmaps
- indicators: Technical indicators (MA, MACD, RSI, Bollinger Bands, Volume)
- report_generator: PDF report generation

Usage:
    from visualization import EquityCurveVisualizer, TradeSignalVisualizer
    
    # Plot equity curve with drawdown
    viz = EquityCurveVisualizer()
    fig = viz.plot_equity_with_drawdown(equity_curve, dates)
    
    # Plot price with indicators
    ind_viz = IndicatorVisualizer()
    fig = ind_viz.plot_price_with_ma(dates, prices, sma_20, sma_50, sma_200)
    
    # Generate PDF report
    from visualization import ReportGenerator
    gen = ReportGenerator('reports/')
    pdf_path = gen.generate_strategy_report(figures, title="My Strategy")
"""

from .equity_curves import EquityCurveVisualizer
from .trade_signals import TradeSignalVisualizer
from .performance_dashboard import PerformanceDashboard
from .indicators import IndicatorVisualizer
from .report_generator import ReportGenerator

__all__ = [
    'EquityCurveVisualizer',
    'TradeSignalVisualizer',
    'PerformanceDashboard',
    'IndicatorVisualizer',
    'ReportGenerator',
]

__version__ = '1.0.0'
