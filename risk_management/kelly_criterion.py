"""
Kelly Criterion - Optimal Position Sizing Based on Historical Statistics

The Kelly criterion provides the optimal fraction of capital to allocate to a position
based on the historical win rate and payoff ratio. This maximizes long-term wealth growth.

Formula: f = (bp - q) / b
where:
- f = fraction of capital to wager
- b = odds (win amount / loss amount)
- p = probability of winning
- q = probability of losing (1 - p)

Reference: "A Random Walk Down Wall Street" by Burton Malkiel
"""

import math
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class KellyMetrics:
    """Results from Kelly criterion calculation"""
    win_rate: float
    payoff_ratio: float
    kelly_fraction: float
    safe_fraction: float  # Kelly fraction / 2 or /4 for safety
    optimal_leverage: float
    returns: List[float]
    start_capital: float
    end_capital: float
    
    def __repr__(self) -> str:
        return (
            f"KellyMetrics(\n"
            f"  win_rate={self.win_rate:.2%}\n"
            f"  payoff_ratio={self.payoff_ratio:.3f}\n"
            f"  kelly_fraction={self.kelly_fraction:.2%}\n"
            f"  safe_fraction={self.safe_fraction:.2%}\n"
            f"  optimal_leverage={self.optimal_leverage:.2f}x\n"
            f"  total_return={((self.end_capital / self.start_capital) - 1):.2%}\n"
            f")"
        )


class KellyCriterion:
    """
    Implements Kelly Criterion for optimal position sizing.
    
    Can be used in three modes:
    1. Post-trade analysis: Calculate Kelly fraction from historical trade returns
    2. Pre-trade sizing: Generate position size based on win/loss estimates
    3. Risk-adjusted: Apply fractional Kelly (1/2, 1/4) for safety
    """
    
    def __init__(
        self,
        max_kelly_fraction: float = 0.25,
        safety_factor: float = 2.0,  # Use Kelly/2 by default (full Kelly / safety_factor)
        min_win_rate: float = 0.4,
        min_trades: int = 20,
    ):
        """
        Args:
            max_kelly_fraction: Cap Kelly fraction at this value (safety limit)
            safety_factor: Divide Kelly by this (2 = half Kelly, 4 = quarter Kelly)
            min_win_rate: Minimum win rate required (below this, kelly = 0)
            min_trades: Minimum trades to calculate Kelly (otherwise flat position)
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.safety_factor = safety_factor
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades
    
    def from_trade_returns(
        self,
        returns: List[float],
        start_capital: float = 100000,
    ) -> KellyMetrics:
        """
        Calculate Kelly fraction from historical trade returns.
        
        Args:
            returns: List of trade returns (as decimals, e.g., 0.02 for +2%)
            start_capital: Initial capital for growth calculation
            
        Returns:
            KellyMetrics with win rate, payoff ratio, and Kelly fraction
        """
        if not returns or len(returns) < self.min_trades:
            return KellyMetrics(
                win_rate=0.0,
                payoff_ratio=0.0,
                kelly_fraction=0.0,
                safe_fraction=0.0,
                optimal_leverage=1.0,
                returns=returns,
                start_capital=start_capital,
                end_capital=start_capital,
            )
        
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        win_rate = len(wins) / len(returns) if returns else 0
        
        # Avoid division by zero
        if not losses or not wins:
            if win_rate >= self.min_win_rate:
                kelly = self.max_kelly_fraction
            else:
                kelly = 0.0
        else:
            avg_win = sum(wins) / len(wins)
            avg_loss = abs(sum(losses) / len(losses))
            
            # Payoff ratio: average win / average loss
            if avg_loss > 0:
                payoff_ratio = avg_win / avg_loss
            else:
                payoff_ratio = 1.0
            
            kelly = self._calculate_kelly_fraction(win_rate, payoff_ratio)
        
        # Apply safety factor
        safe_fraction = kelly / self.safety_factor
        
        # Simulate capital growth using sequential compounding
        end_capital = self._simulate_growth(start_capital, returns, safe_fraction)
        
        return KellyMetrics(
            win_rate=win_rate,
            payoff_ratio=avg_win / avg_loss if not losses == 0 else 1.0,
            kelly_fraction=kelly,
            safe_fraction=safe_fraction,
            optimal_leverage=1.0 / safe_fraction if safe_fraction > 0 else 1.0,
            returns=returns,
            start_capital=start_capital,
            end_capital=end_capital,
        )
    
    def from_win_loss(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        num_trades: int = 100,
    ) -> float:
        """
        Calculate Kelly fraction from win rate and average win/loss amounts.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average profit on winning trades
            avg_loss: Average loss on losing trades (positive amount)
            num_trades: Number of trades (used for validation)
            
        Returns:
            Kelly fraction (safe fraction after applying safety factor)
        """
        if num_trades < self.min_trades or win_rate < self.min_win_rate:
            return 0.0
        
        if avg_loss <= 0:
            return self.max_kelly_fraction
        
        payoff_ratio = avg_win / avg_loss
        kelly = self._calculate_kelly_fraction(win_rate, payoff_ratio)
        
        return kelly / self.safety_factor
    
    def _calculate_kelly_fraction(self, win_rate: float, payoff_ratio: float) -> float:
        """
        Calculate Kelly fraction using the formula: f = (b*p - q) / b
        where b = payoff_ratio, p = win_rate, q = 1 - win_rate
        """
        if win_rate <= 0 or payoff_ratio <= 0:
            return 0.0
        
        loss_rate = 1.0 - win_rate
        
        # Kelly formula: f = (p*b - (1-p)) / b
        numerator = (win_rate * payoff_ratio) - loss_rate
        denominator = payoff_ratio
        
        if denominator == 0:
            return 0.0
        
        kelly = numerator / denominator
        
        # Ensure kelly is positive and capped
        kelly = max(0.0, min(kelly, self.max_kelly_fraction))
        
        return kelly
    
    def _simulate_growth(
        self,
        start_capital: float,
        returns: List[float],
        fraction: float,
    ) -> float:
        """Simulate capital growth using sequential compounding with Kelly fraction."""
        capital = start_capital
        for ret in returns:
            # Apply the Kelly fraction to the return
            adjusted_return = ret * fraction
            capital *= (1.0 + adjusted_return)
        
        return capital
    
    def position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        num_trades: int = 100,
    ) -> float:
        """
        Calculate position size based on Kelly criterion and risk parameters.
        
        Args:
            account_equity: Current account equity
            entry_price: Price to enter position
            stop_loss_price: Price to exit if losing
            win_rate: Historical win rate
            avg_win: Average profit per winning trade
            avg_loss: Average loss per losing trade (positive)
            num_trades: Number of historical trades
            
        Returns:
            Number of shares/contracts to buy
        """
        kelly_fraction = self.from_win_loss(win_rate, avg_win, avg_loss, num_trades)
        
        if kelly_fraction <= 0:
            return 0.0
        
        # Risk amount is Kelly fraction of account equity
        risk_amount = account_equity * kelly_fraction
        
        # Position size is risk amount divided by entry price
        shares = risk_amount / entry_price if entry_price > 0 else 0.0
        
        return shares
    
    def leverage_from_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate optimal leverage based on Kelly fraction.
        Leverage = 1 / kelly_fraction
        
        Example: Kelly = 0.25 → Leverage = 4x
        """
        kelly = self._calculate_kelly_fraction(win_rate, avg_win / avg_loss)
        safe_kelly = kelly / self.safety_factor
        
        if safe_kelly <= 0:
            return 1.0
        
        return 1.0 / safe_kelly
