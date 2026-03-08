"""
Trade Signal Visualization - Overlay buy/sell signals on price charts

Displays:
- Price candlesticks/OHLC
- Buy/sell entry signals
- Take profit/stop loss levels
- Position sizing information
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import seaborn as sns


class TradeSignalVisualizer:
    """Visualize trade signals on price charts."""
    
    def __init__(self, figsize: Tuple = (16, 8), style: str = "whitegrid"):
        self.figsize = figsize
        allowed = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
        self.style = style if style in allowed else "whitegrid"
        sns.set_style(self.style)
    
    def plot_price_with_signals(
        self,
        dates: List,
        prices: List[float],
        buy_signals: Optional[List[Tuple[int, float]]] = None,
        sell_signals: Optional[List[Tuple[int, float]]] = None,
        stop_losses: Optional[List[Tuple[int, float]]] = None,
        take_profits: Optional[List[Tuple[int, float]]] = None,
        title: str = "Trade Signals",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot price with trading signals.
        
        Args:
            dates: List of dates
            prices: List of prices
            buy_signals: List of (index, price) tuples
            sell_signals: List of (index, price) tuples
            stop_losses: List of (index, price) tuples
            take_profits: List of (index, price) tuples
            title: Plot title
            save_path: Optional path to save
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot price line
        ax.plot(dates, prices, linewidth=2, color='#1D3557', label='Price', zorder=2)
        ax.fill_between(dates, prices, alpha=0.1, color='#457B9D')
        
        # Buy signals
        if buy_signals:
            buy_dates = [dates[idx] for idx, _ in buy_signals]
            buy_prices = [price for _, price in buy_signals]
            ax.scatter(buy_dates, buy_prices, color='#06A77D', s=200, 
                      marker='^', label='Buy Signal', zorder=5, edgecolors='black', linewidth=1.5)
        
        # Sell signals
        if sell_signals:
            sell_dates = [dates[idx] for idx, _ in sell_signals]
            sell_prices = [price for _, price in sell_signals]
            ax.scatter(sell_dates, sell_prices, color='#E63946', s=200, 
                      marker='v', label='Sell Signal', zorder=5, edgecolors='black', linewidth=1.5)
        
        # Stop losses (horizontal lines)
        if stop_losses:
            for idx, price in stop_losses:
                ax.axhline(y=price, color='#E63946', linestyle='--', alpha=0.5, linewidth=1)
        
        # Take profits (horizontal lines)
        if take_profits:
            for idx, price in take_profits:
                ax.axhline(y=price, color='#06A77D', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_position_pnl(
        self,
        dates: List,
        position_pnl: List[float],
        position_sizes: Optional[List[int]] = None,
        title: str = "Position P&L Over Time",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot position P&L with position sizing."""
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        # P&L line
        ax1.plot(dates, position_pnl, linewidth=2.5, color='#2E86AB', 
                label='P&L', zorder=2)
        
        # Fill profit/loss areas
        pnl_array = np.array(position_pnl)
        ax1.fill_between(dates, pnl_array, 0, where=(pnl_array >= 0),
                        color='#06A77D', alpha=0.3, label='Profit')
        ax1.fill_between(dates, pnl_array, 0, where=(pnl_array < 0),
                        color='#E63946', alpha=0.3, label='Loss')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('P&L ($)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=11)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Position sizing on secondary axis
        if position_sizes:
            ax2 = ax1.twinx()
            ax2.bar(dates, position_sizes, alpha=0.2, color='#F18F01', 
                   label='Position Size', zorder=1, width=1)
            ax2.set_ylabel('Position Size (Units)', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trade_analysis(
        self,
        wins: List[float],
        losses: List[float],
        title: str = "Trade Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot winning vs losing trades."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Win/Loss distribution
        all_trades = wins + losses
        colors = ['#06A77D'] * len(wins) + ['#E63946'] * len(losses)
        x_pos = range(len(all_trades))
        
        ax1.bar(x_pos, all_trades, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_title('Individual Trades P&L', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
        ax1.set_ylabel('P&L ($)', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Summary statistics
        win_sum = sum(wins)
        loss_sum = sum(losses)
        win_count = len(wins)
        loss_count = len(losses)
        
        stats = {
            'Wins': win_count,
            'Losses': loss_count,
            'Total Profit': win_sum,
            'Total Loss': loss_sum,
            'Net P&L': win_sum + loss_sum
        }
        
        ax2.axis('off')
        y_pos = 0.95
        for key, value in stats.items():
            if isinstance(value, float):
                text = f"{key}: ${value:,.2f}"
                color = '#06A77D' if value >= 0 else '#E63946'
            else:
                text = f"{key}: {value}"
                color = '#2E86AB'
            
            ax2.text(0.1, y_pos, text, fontsize=13, fontweight='bold', color=color,
                    family='monospace', transform=ax2.transAxes)
            y_pos -= 0.15
        
        win_rate = (win_count / (win_count + loss_count) * 100) if (win_count + loss_count) > 0 else 0
        ax2.text(0.1, y_pos - 0.1, f"Win Rate: {win_rate:.1f}%", fontsize=13,
                fontweight='bold', color='#2E86AB', transform=ax2.transAxes)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_holding_periods(
        self,
        holding_periods: List[int],
        pnls: List[float],
        title: str = "Holding Period Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Scatter plot of holding period vs P&L."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = ['#06A77D' if pnl >= 0 else '#E63946' for pnl in pnls]
        
        ax.scatter(holding_periods, pnls, c=colors, s=100, alpha=0.6, 
                  edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add trend line
        if len(holding_periods) > 1:
            z = np.polyfit(holding_periods, pnls, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(min(holding_periods), max(holding_periods), 100)
            ax.plot(x_trend, p(x_trend), "b--", alpha=0.5, linewidth=2, label='Trend')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Holding Period (Days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trade P&L ($)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        if len(holding_periods) > 1:
            ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
