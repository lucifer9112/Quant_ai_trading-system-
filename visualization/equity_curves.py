"""
Equity Curve Visualization - Professional performance charts

Creates publication-quality figures showing:
- Equity curve with drawdown bands
- Cumulative returns
- Rolling statistics
- Underwater plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional, Tuple
import seaborn as sns


class EquityCurveVisualizer:
    """Generate professional equity curve charts."""
    
    def __init__(self, figsize: Tuple = (14, 8), style: str = "seaborn"):
        """
        Args:
            figsize: Figure size
            style: matplotlib style
        """
        self.figsize = figsize
        self.style = style
        sns.set_style(style)
    
    def plot_equity_curve(
        self,
        equity_curve: List[float],
        dates: Optional[List] = None,
        benchmark: Optional[List[float]] = None,
        title: str = "Equity Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot equity curve with statistics.
        
        Args:
            equity_curve: List of portfolio values
            dates: Optional dates
            benchmark: Optional benchmark equity curve
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure
        """
        if dates is None:
            dates = range(len(equity_curve))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Main equity curve
        ax.plot(dates, equity_curve, linewidth=2.5, label='Portfolio', 
                color='#2E86AB', zorder=3)
        
        # Benchmark
        if benchmark is not None:
            ax.plot(dates, benchmark, linewidth=2, label='Benchmark', 
                   color='#A23B72', linestyle='--', alpha=0.7, zorder=2)
        
        # Fill between start and equity curve
        start_value = equity_curve[0]
        ax.fill_between(dates, start_value, equity_curve, alpha=0.1, 
                        color='#2E86AB', zorder=1)
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_equity_with_drawdown(
        self,
        equity_curve: List[float],
        dates: Optional[List] = None,
        title: str = "Equity Curve with Drawdown Bands",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot equity curve with drawdown bands.
        
        Args:
            equity_curve: List of portfolio values
            dates: Optional dates
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure
        """
        if dates is None:
            dates = range(len(equity_curve))
        
        equity_array = np.array(equity_curve)
        
        # Calculate running maximum and drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Top plot: Equity curve with drawdown bands
        ax1.plot(dates, equity_curve, linewidth=2.5, color='#2E86AB', 
                label='Portfolio', zorder=3)
        ax1.fill_between(dates, equity_array[0], equity_curve, alpha=0.1, 
                        color='#2E86AB', zorder=1)
        ax1.plot(dates, running_max, linewidth=1.5, color='#F18F01', 
                linestyle='--', alpha=0.7, label='Peak', zorder=2)
        
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=11)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Bottom plot: Drawdown
        colors = ['#2E86AB' if dd < -0.10 else '#F18F01' if dd < -0.05 else '#06A77D' 
                 for dd in drawdown]
        ax2.bar(dates, drawdown * 100, color=colors, alpha=0.7, width=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, zorder=1)
        ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_underwater(
        self,
        equity_curve: List[float],
        dates: Optional[List] = None,
        title: str = "Underwater Plot (Drawdown Over Time)",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Underwater plot showing drawdown periods as shaded regions.
        
        Args:
            equity_curve: List of portfolio values
            dates: Optional dates
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib figure
        """
        if dates is None:
            dates = range(len(equity_curve))
        
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = ((equity_array - running_max) / running_max) * 100
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Fill underwater areas
        ax.fill_between(dates, drawdown, 0, where=(drawdown < 0), 
                       color='#E63946', alpha=0.6, label='Drawdown')
        ax.plot(dates, drawdown, linewidth=2, color='#1D3557', zorder=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cumulative_returns(
        self,
        equity_curve: List[float],
        dates: Optional[List] = None,
        benchmark: Optional[List[float]] = None,
        title: str = "Cumulative Returns",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot cumulative returns (%)."""
        if dates is None:
            dates = range(len(equity_curve))
        
        equity_array = np.array(equity_curve)
        cum_returns = ((equity_array / equity_array[0]) - 1) * 100
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(dates, cum_returns, linewidth=2.5, label='Portfolio', 
               color='#2E86AB', zorder=2)
        
        if benchmark is not None:
            benchmark_array = np.array(benchmark)
            benchmark_returns = ((benchmark_array / benchmark_array[0]) - 1) * 100
            ax.plot(dates, benchmark_returns, linewidth=2, label='Benchmark', 
                   color='#A23B72', linestyle='--', alpha=0.7, zorder=1)
        
        ax.fill_between(dates, 0, cum_returns, alpha=0.1, color='#2E86AB')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_rolling_metrics(
        self,
        equity_curve: List[float],
        dates: Optional[List] = None,
        window: int = 60,
        title: str = "Rolling Sharpe Ratio",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot rolling Sharpe ratio over time."""
        if dates is None:
            dates = range(len(equity_curve))
        
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        rolling_sharpe = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            annual_return = np.mean(window_returns) * 252
            annual_vol = np.std(window_returns) * np.sqrt(252)
            sharpe = annual_return / (annual_vol + 1e-6) - 0.04
            rolling_sharpe.append(sharpe)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(dates[window:], rolling_sharpe, linewidth=2.5, color='#2E86AB')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.fill_between(dates[window:], rolling_sharpe, 0, 
                       where=(np.array(rolling_sharpe) >= 0),
                       color='#06A77D', alpha=0.3, label='Positive')
        ax.fill_between(dates[window:], rolling_sharpe, 0, 
                       where=(np.array(rolling_sharpe) < 0),
                       color='#E63946', alpha=0.3, label='Negative')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
