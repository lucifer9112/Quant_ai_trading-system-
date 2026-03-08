"""
Performance Dashboard Visualization - Multi-panel performance analysis

Creates comprehensive dashboards with:
- Performance metrics grid
- Attribution breakdown
- Risk analysis
- Monthly/yearly heat maps
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
import seaborn as sns


class PerformanceDashboard:
    """Create professional performance dashboards."""
    
    def __init__(self, figsize: Tuple = (16, 10), style: str = "whitegrid"):
        self.figsize = figsize
        allowed = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
        self.style = style if style in allowed else "whitegrid"
        sns.set_style(self.style)
    
    def plot_metrics_grid(
        self,
        metrics: Dict[str, float],
        title: str = "Performance Metrics",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Display key metrics in grid format.
        
        Args:
            metrics: Dict mapping metric names to values
            title: Plot title
            save_path: Optional path to save
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Create grid of subplots
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            ax.axis('off')
            
            # Color code based on type
            if 'Sharpe' in metric_name or 'Return' in metric_name:
                color = '#06A77D' if metric_value > 0 else '#E63946'
            elif 'Drawdown' in metric_name or 'Loss' in metric_name:
                color = '#E63946' if metric_value < 0 else '#06A77D'
            else:
                color = '#2E86AB'
            
            # Format value appropriately
            if isinstance(metric_value, float):
                if abs(metric_value) < 10:
                    value_str = f"{metric_value:.3f}"
                elif 'Return' in metric_name or 'Drawdown' in metric_name or '%' in metric_name:
                    value_str = f"{metric_value:.2f}%"
                else:
                    value_str = f"{metric_value:.2f}"
            else:
                value_str = str(metric_value)
            
            # Draw box
            rect = Rectangle((0.05, 0.3), 0.9, 0.6, fill=True, facecolor=color, 
                           alpha=0.1, edgecolor=color, linewidth=2, transform=ax.transAxes)
            ax.add_patch(rect)
            
            # Text
            ax.text(0.5, 0.65, metric_name, ha='center', va='center',
                   fontsize=11, fontweight='bold', transform=ax.transAxes)
            ax.text(0.5, 0.35, value_str, ha='center', va='center',
                   fontsize=14, fontweight='bold', color=color, 
                   family='monospace', transform=ax.transAxes)
        
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attribution_pie(
        self,
        sectors: Dict[str, float],
        title: str = "Return Attribution by Sector",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot sector contribution to returns."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#06A77D' if v > 0 else '#E63946' for v in sectors.values()]
        
        wedges, texts, autotexts = ax.pie(
            [abs(v) for v in sectors.values()],
            labels=sectors.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_monthly_returns_heatmap(
        self,
        returns_by_month: Dict[str, float],
        title: str = "Monthly Returns Heatmap",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Heatmap of monthly returns.
        
        Args:
            returns_by_month: Dict mapping month names to returns (%)
            title: Plot title
            save_path: Optional path to save
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        months = list(returns_by_month.keys())
        returns = list(returns_by_month.values())
        
        # Create color map
        colors = []
        for r in returns:
            if r > 5:
                colors.append('#06A77D')
            elif r > 0:
                colors.append('#A8DADC')
            elif r > -5:
                colors.append('#F1ACAC')
            else:
                colors.append('#E63946')
        
        bars = ax.bar(range(len(months)), returns, color=colors, 
                     edgecolor='black', linewidth=1)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels
        for i, (bar, ret) in enumerate(zip(bars, returns)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontweight='bold', fontsize=10)
        
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha='right')
        ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_risk_metrics_timeline(
        self,
        dates: List,
        volatility: List[float],
        max_dd: List[float],
        var_95: List[float],
        title: str = "Risk Metrics Timeline",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot risk metrics over time."""
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, sharex=True)
        
        # Volatility
        axes[0].plot(dates, volatility, linewidth=2, color='#2E86AB')
        axes[0].fill_between(dates, volatility, alpha=0.3, color='#2E86AB')
        axes[0].set_ylabel('Volatility (Annual %)', fontsize=11, fontweight='bold')
        axes[0].set_title(title, fontsize=16, fontweight='bold', pad=20)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Max Drawdown
        axes[1].plot(dates, max_dd, linewidth=2, color='#E63946')
        axes[1].fill_between(dates, max_dd, alpha=0.3, color='#E63946')
        axes[1].set_ylabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # Value at Risk
        axes[2].plot(dates, var_95, linewidth=2, color='#F18F01')
        axes[2].fill_between(dates, var_95, alpha=0.3, color='#F18F01')
        axes[2].set_ylabel('95% VaR (%)', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Time', fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_performance_vs_risk(
        self,
        strategies: Dict[str, Tuple[float, float]],  # {name: (return, volatility)}
        title: str = "Risk-Return Scatter",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot risk vs return for multiple strategies."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        returns = []
        volatilities = []
        labels = []
        
        for strategy, (ret, vol) in strategies.items():
            returns.append(ret)
            volatilities.append(vol)
            labels.append(strategy)
        
        colors = ['#2E86AB', '#06A77D', '#F18F01', '#E63946', '#A23B72']
        scatter = ax.scatter(volatilities, returns, s=300, c=colors[:len(strategies)],
                            alpha=0.6, edgecolors='black', linewidth=2)
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (volatilities[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Volatility (Annual %)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (% Annual)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_rolling_sharpe(
        self,
        dates: List,
        sharpe_ratios: List[float],
        title: str = "Rolling Sharpe Ratio",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot rolling Sharpe ratio with bands."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sharpe_array = np.array(sharpe_ratios)
        
        ax.plot(dates, sharpe_array, linewidth=2.5, color='#2E86AB', label='Sharpe Ratio')
        ax.fill_between(dates, sharpe_array, 0, where=(sharpe_array >= 0),
                       color='#06A77D', alpha=0.3, label='Good')
        ax.fill_between(dates, sharpe_array, 0, where=(sharpe_array < 0),
                       color='#E63946', alpha=0.3, label='Poor')
        
        # Add bands
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=2, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
