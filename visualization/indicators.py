"""
Technical Indicators Visualization - Display technical analysis on price charts

Plots:
- Price with moving averages
- MACD
- RSI
- Bollinger Bands
- Volume profile
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
import seaborn as sns


class IndicatorVisualizer:
    """Visualize technical indicators on price charts."""
    
    def __init__(self, figsize: Tuple = (16, 10), style: str = "whitegrid"):
        self.figsize = figsize
        allowed = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
        self.style = style if style in allowed else "whitegrid"
        sns.set_style(self.style)
    
    def plot_price_with_ma(
        self,
        dates: List,
        prices: List[float],
        sma_20: Optional[List[float]] = None,
        sma_50: Optional[List[float]] = None,
        sma_200: Optional[List[float]] = None,
        ema_12: Optional[List[float]] = None,
        title: str = "Price with Moving Averages",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot price with multiple moving averages."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(dates, prices, linewidth=2.5, color='#1D3557', label='Price', zorder=3)
        ax.fill_between(dates, prices, alpha=0.1, color='#1D3557')
        
        if sma_20 is not None:
            ax.plot(dates, sma_20, linewidth=1.5, color='#F18F01', 
                   label='SMA 20', alpha=0.8, linestyle='-')
        
        if sma_50 is not None:
            ax.plot(dates, sma_50, linewidth=1.5, color='#2E86AB',
                   label='SMA 50', alpha=0.8, linestyle='-')
        
        if sma_200 is not None:
            ax.plot(dates, sma_200, linewidth=1.5, color='#E63946',
                   label='SMA 200', alpha=0.8, linestyle='-')
        
        if ema_12 is not None:
            ax.plot(dates, ema_12, linewidth=2, color='#06A77D',
                   label='EMA 12', alpha=0.8, linestyle='--')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_bollinger_bands(
        self,
        dates: List,
        prices: List[float],
        upper_band: List[float],
        lower_band: List[float],
        middle_band: List[float],
        title: str = "Bollinger Bands",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot price with Bollinger Bands."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Price
        ax.plot(dates, prices, linewidth=2.5, color='#1D3557', label='Price', zorder=3)
        
        # Bollinger Bands
        ax.plot(dates, upper_band, linewidth=1, color='#E63946', 
               linestyle='--', alpha=0.7, label='Upper Band')
        ax.plot(dates, middle_band, linewidth=1.5, color='#2E86AB',
               linestyle='-', alpha=0.8, label='Middle Band')
        ax.plot(dates, lower_band, linewidth=1, color='#06A77D',
               linestyle='--', alpha=0.7, label='Lower Band')
        
        # Fill between bands
        ax.fill_between(dates, lower_band, upper_band, alpha=0.1, color='#2E86AB')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_macd(
        self,
        dates: List,
        macd_line: List[float],
        signal_line: List[float],
        histogram: List[float],
        title: str = "MACD",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot MACD indicator."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # MACD and Signal lines
        ax.plot(dates, macd_line, linewidth=2, color='#2E86AB', label='MACD')
        ax.plot(dates, signal_line, linewidth=2, color='#F18F01', label='Signal')
        
        # Histogram
        colors = ['#06A77D' if h >= 0 else '#E63946' for h in histogram]
        ax.bar(dates, histogram, color=colors, alpha=0.3, width=1, label='Histogram')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('MACD Value', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_rsi(
        self,
        dates: List,
        rsi_values: List[float],
        title: str = "Relative Strength Index (RSI)",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot RSI indicator with overbought/oversold levels."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        rsi_array = np.array(rsi_values)
        
        # RSI line
        ax.plot(dates, rsi_array, linewidth=2.5, color='#2E86AB', label='RSI')
        
        # Color fill based on levels
        ax.fill_between(dates, rsi_array, 70, where=(rsi_array >= 70),
                       color='#E63946', alpha=0.3, label='Overbought (>70)')
        ax.fill_between(dates, 30, rsi_array, where=(rsi_array <= 30),
                       color='#06A77D', alpha=0.3, label='Oversold (<30)')
        ax.fill_between(dates, rsi_array, 30, where=((rsi_array > 30) & (rsi_array < 70)),
                       color='#F18F01', alpha=0.2, label='Neutral')
        
        # Key levels
        ax.axhline(y=70, color='#E63946', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=50, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axhline(y=30, color='#06A77D', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_ylim([0, 100])
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('RSI', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='lower left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_volume_profile(
        self,
        dates: List,
        volumes: List[float],
        prices: Optional[List[float]] = None,
        title: str = "Volume Profile",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot volume bars."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = ['#06A77D' if (i == 0 or prices[i] >= prices[i-1]) else '#E63946'
                 for i in range(len(volumes))] if prices else ['#2E86AB'] * len(volumes)
        
        ax.bar(dates, volumes, color=colors, alpha=0.6, edgecolor='black', linewidth=0.3)
        
        # Moving average of volume
        if len(volumes) > 20:
            ma_20 = pd.Series(volumes).rolling(20).mean()
            ax.plot(dates, ma_20, linewidth=2, color='#F18F01', label='MA(20)', zorder=3)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volume (Shares)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_indicators(
        self,
        dates: List,
        prices: List[float],
        indicators: Dict[str, List[float]],
        title: str = "Multi-Indicator Dashboard",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create dashboard with price and multiple indicators."""
        n_indicators = len(indicators)
        fig, axes = plt.subplots(n_indicators + 1, 1, figsize=self.figsize,
                                 gridspec_kw={'height_ratios': [2] + [1] * n_indicators},
                                 sharex=True)
        
        if n_indicators == 0:
            axes = [axes]
        
        # Price on first axis
        axes[0].plot(dates, prices, linewidth=2.5, color='#1D3557', zorder=3)
        axes[0].fill_between(dates, prices, alpha=0.1, color='#1D3557')
        axes[0].set_title(title, fontsize=16, fontweight='bold', pad=20)
        axes[0].set_ylabel('Price ($)', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Indicators on subsequent axes
        colors = ['#2E86AB', '#F18F01', '#E63946', '#06A77D', '#A23B72']
        
        for idx, (indicator_name, indicator_values) in enumerate(indicators.items()):
            ax = axes[idx + 1]
            
            indicator_array = np.array(indicator_values)
            
            # Plot line
            ax.plot(dates, indicator_array, linewidth=1.5, 
                   color=colors[idx % len(colors)], label=indicator_name)
            
            # Zero line if values cross it
            if np.min(indicator_array) < 0 < np.max(indicator_array):
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            ax.set_ylabel(indicator_name, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=9, loc='upper left')
        
        axes[-1].set_xlabel('Time', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
