"""
Trade Metrics - Individual Trade Performance Analysis

Calculates:
- Win Rate
- Profit Factor
- Payoff Ratio (avg win / avg loss)
- Consecutive wins/losses
- Trade duration
- Trade P&L distribution
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Trade:
    """Single trade record"""
    entry_date: any  # Date/timestamp
    exit_date: any
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    trade_type: str  # "LONG" or "SHORT"
    pnl: float  # Profit/loss in dollars
    pnl_pct: float  # P&L as percentage
    duration_days: int
    reason: str  # "STOP_LOSS", "TAKE_PROFIT", "SIGNAL", etc.


@dataclass
class TradeMetrics:
    """Trade performance metrics"""
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    payoff_ratio: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    longest_winning_streak: int
    longest_losing_streak: int
    avg_consecutive_wins: float
    avg_consecutive_losses: float
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    expectancy: float  # Expected value per trade
    
    def __repr__(self) -> str:
        return (
            f"TradeMetrics(\n"
            f"  num_trades={self.num_trades}\n"
            f"  win_rate={self.win_rate:.2%}\n"
            f"  profit_factor={self.profit_factor:.3f}\n"
            f"  payoff_ratio={self.payoff_ratio:.3f}\n"
            f"  avg_win=${self.avg_win:.2f}\n"
            f"  avg_loss=${self.avg_loss:.2f}\n"
            f"  expectancy=${self.expectancy:.2f}\n"
            f")"
        )


class TradeAnalyzer:
    """
    Analyzes individual trade performance.
    """
    
    def __init__(self):
        pass
    
    def calculate_metrics(
        self,
        trades: List[Trade],
    ) -> TradeMetrics:
        """
        Calculate trade performance metrics.
        
        Args:
            trades: List of Trade objects
            
        Returns:
            TradeMetrics object
        """
        if not trades:
            return self._empty_metrics()
        
        # Separate wins and losses
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        breakeven_trades = [t for t in trades if t.pnl == 0]
        
        num_trades = len(trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        
        # Win rate
        win_rate = num_wins / num_trades if num_trades > 0 else 0.0
        
        # Profits and losses
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        
        # Profit Factor = Gross Profit / Gross Loss
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Average win/loss
        avg_win = gross_profit / num_wins if num_wins > 0 else 0.0
        avg_loss = gross_loss / num_losses if num_losses > 0 else 0.0
        
        # Payoff Ratio = Avg Win / Avg Loss
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Largest win/loss
        largest_win = max((t.pnl for t in winning_trades), default=0.0)
        largest_loss = abs(min((t.pnl for t in losing_trades), default=0.0))
        
        # Consecutive wins/losses
        win_streak = self._calculate_streaks(trades)
        longest_winning_streak = max(win_streak.values(), default=0) if win_streak else 0
        
        loss_streak = self._calculate_streaks(trades, is_loss=True)
        longest_losing_streak = max(loss_streak.values(), default=0) if loss_streak else 0
        
        avg_consecutive_wins = np.mean(list(win_streak.values())) if win_streak else 0.0
        avg_consecutive_losses = np.mean(list(loss_streak.values())) if loss_streak else 0.0
        
        # Duration
        durations = [t.duration_days for t in trades if t.duration_days > 0]
        avg_duration = np.mean(durations) if durations else 0.0
        
        # Best and worst
        best_trade = max((t.pnl for t in trades), default=0.0)
        worst_trade = min((t.pnl for t in trades), default=0.0)
        
        # Expectancy (expected value per trade)
        expectancy = self._calculate_expectancy(win_rate, avg_win, avg_loss)
        
        return TradeMetrics(
            num_trades=num_trades,
            num_wins=num_wins,
            num_losses=num_losses,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            longest_winning_streak=longest_winning_streak,
            longest_losing_streak=longest_losing_streak,
            avg_consecutive_wins=avg_consecutive_wins,
            avg_consecutive_losses=avg_consecutive_losses,
            avg_trade_duration=avg_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            expectancy=expectancy,
        )
    
    def _empty_metrics(self) -> TradeMetrics:
        """Return empty metrics."""
        return TradeMetrics(
            num_trades=0,
            num_wins=0,
            num_losses=0,
            win_rate=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=0.0,
            payoff_ratio=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            longest_winning_streak=0,
            longest_losing_streak=0,
            avg_consecutive_wins=0.0,
            avg_consecutive_losses=0.0,
            avg_trade_duration=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            expectancy=0.0,
        )
    
    def _calculate_streaks(
        self,
        trades: List[Trade],
        is_loss: bool = False,
    ) -> Dict[int, int]:
        """
        Calculate winning or losing streaks.
        
        Args:
            trades: List of trades
            is_loss: If True, count losing streaks; if False, winning streaks
            
        Returns:
            Dict mapping streak_index to streak_length
        """
        streaks = {}
        current_streak = 0
        streak_index = 0
        
        for trade in trades:
            is_winning = trade.pnl > 0
            
            if (is_winning and not is_loss) or (not is_winning and is_loss):
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks[streak_index] = current_streak
                    streak_index += 1
                    current_streak = 0
        
        # Record final streak
        if current_streak > 0:
            streaks[streak_index] = current_streak
        
        return streaks
    
    def _calculate_expectancy(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate expectancy (expected profit per trade).
        
        Formula: Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
        """
        if win_rate == 0:
            return -avg_loss
        
        loss_rate = 1.0 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        return expectancy
    
    def pnl_distribution(
        self,
        trades: List[Trade],
        bins: int = 20,
    ) -> pd.DataFrame:
        """
        Calculate P&L distribution for histogram.
        
        Args:
            trades: List of trades
            bins: Number of histogram bins
            
        Returns:
            DataFrame with P&L distribution
        """
        if not trades:
            return pd.DataFrame()
        
        pnls = [t.pnl for t in trades]
        
        counts, edges = np.histogram(pnls, bins=bins)
        
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        df = pd.DataFrame({
            'PnL': bin_centers,
            'Count': counts,
            'Frequency': counts / len(trades),
        })
        
        return df
    
    def monthly_trade_summary(
        self,
        trades: List[Trade],
    ) -> pd.DataFrame:
        """
        Summarize trades by month.
        
        Args:
            trades: List of trades
            
        Returns:
            DataFrame with monthly summary
        """
        if not trades:
            return pd.DataFrame()
        
        # Convert to DataFrame for easier grouping
        trade_data = []
        for t in trades:
            trade_data.append({
                'entry_date': t.entry_date,
                'pnl': t.pnl,
                'is_win': t.pnl > 0,
            })
        
        df = pd.DataFrame(trade_data)
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['YearMonth'] = df['entry_date'].dt.to_period('M')
        
        monthly = df.groupby('YearMonth').agg({
            'pnl': ['count', 'sum', 'mean', 'min', 'max'],
            'is_win': 'sum',
        })
        
        monthly.columns = ['NumTrades', 'TotalPnL', 'AvgPnL', 'MinPnL', 'MaxPnL', 'Wins']
        monthly['WinRate'] = monthly['Wins'] / monthly['NumTrades']
        
        return monthly.reset_index()
    
    def consecutive_stats(
        self,
        trades: List[Trade],
    ) -> Dict:
        """
        Calculate statistics about consecutive wins/losses.
        
        Args:
            trades: List of trades
            
        Returns:
            Dict with consecutive stats
        """
        win_streaks = self._calculate_streaks(trades, is_loss=False)
        loss_streaks = self._calculate_streaks(trades, is_loss=True)
        
        return {
            'longest_win_streak': max(win_streaks.values(), default=0),
            'longest_loss_streak': max(loss_streaks.values(), default=0),
            'avg_win_streak': np.mean(list(win_streaks.values())) if win_streaks else 0.0,
            'avg_loss_streak': np.mean(list(loss_streaks.values())) if loss_streaks else 0.0,
            'num_win_streaks': len(win_streaks),
            'num_loss_streaks': len(loss_streaks),
        }
