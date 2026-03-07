"""
Portfolio-Level Risk Management

Handles:
- Maximum drawdown constraints
- Value at Risk (VaR) calculations
- Conditional VaR (CVaR)
- Portfolio concentration limits
- Sector/industry concentration
- Correlation-based risk
- Dynamic stop-loss/take-profit
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PortfolioRiskMetrics:
    """Portfolio risk measurement results"""
    max_drawdown: float
    current_drawdown: float
    drawdown_duration: int
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var_95: float
    conditional_var_99: float
    concentration_ratio: float
    concentration_risk: float
    sector_concentration: Dict[str, float]
    portfolio_volatility: float
    portfolio_correlation: float


class PortfolioRiskManager:
    """
    Manages portfolio-level risk constraints and monitoring.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        max_drawdown_pct: float = 0.20,  # 20% maximum drawdown
        var_confidence: float = 0.95,     # 95% VaR
        max_concentration: float = 0.30,  # 30% in any single stock
        max_sector_concentration: float = 0.50,  # 50% in any sector
    ):
        """
        Args:
            initial_capital: Starting portfolio value
            max_drawdown_pct: Maximum acceptable drawdown
            var_confidence: Confidence level for VaR (0.95 or 0.99)
            max_concentration: Maximum weight for single position
            max_sector_concentration: Maximum weight for any sector
        """
        self.initial_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.var_confidence = var_confidence
        self.max_concentration = max_concentration
        self.max_sector_concentration = max_sector_concentration
        self.peak_capital = initial_capital
    
    def calculate_drawdown(
        self,
        equity_curve: np.ndarray,
    ) -> Tuple[float, float, int]:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: Array of portfolio values over time
            
        Returns:
            Tuple of (max_drawdown, current_drawdown, duration_in_periods)
        """
        if len(equity_curve) < 2:
            return 0.0, 0.0, 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_dd = np.min(drawdown)
        
        # Current drawdown
        current_dd = drawdown[-1]
        
        # Duration: count periods in current drawdown
        duration = 0
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown[i] == 0:
                break
            duration += 1
        
        return max_dd, current_dd, duration
    
    def value_at_risk(
        self,
        returns: List[float],
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Historical returns
            confidence: Confidence level (0.95 = 95%)
            method: "historical" or "parametric"
            
        Returns:
            VaR as a negative value (e.g., -0.05 for 5% loss)
        """
        if not returns or len(returns) < 20:
            return 0.0
        
        returns_array = np.array(returns)
        
        if method == "historical":
            # Historical quantile
            quantile = 1.0 - confidence
            var = np.quantile(returns_array, quantile)
        else:
            # Parametric (assume normal distribution)
            mean = np.mean(returns_array)
            std = np.std(returns_array)
            from scipy import stats
            z_score = stats.norm.ppf(1.0 - confidence)
            var = mean + (z_score * std)
        
        return var
    
    def conditional_var(
        self,
        returns: List[float],
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Conditional VaR (CVaR) = Expected Shortfall.
        Average of all returns worse than VaR level.
        
        Args:
            returns: Historical returns
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            CVaR as a negative value
        """
        if not returns or len(returns) < 20:
            return 0.0
        
        var = self.value_at_risk(returns, confidence, method="historical")
        
        # Get all returns worse than VaR
        returns_array = np.array(returns)
        tail_returns = returns_array[returns_array <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return np.mean(tail_returns)
    
    def calculate_concentration(
        self,
        weights: Dict[str, float],
    ) -> float:
        """
        Calculate portfolio concentration using Herfindahl index.
        Higher values = more concentrated
        
        Formula: sum(weight_i^2)
        - Equal weight = 1/n
        - Fully concentrated = 1.0
        
        Args:
            weights: Position weights
            
        Returns:
            Concentration ratio (0-1)
        """
        if not weights:
            return 0.0
        
        weights_array = np.array(list(weights.values()))
        herfindahl = np.sum(weights_array ** 2)
        
        # Normalize to 0-1 scale
        n = len(weights)
        min_concentration = 1.0 / n
        normalized = (herfindahl - min_concentration) / (1.0 - min_concentration)
        
        return np.clip(normalized, 0.0, 1.0)
    
    def calculate_sector_concentration(
        self,
        position_weights: Dict[str, float],
        symbol_to_sector: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Calculate weights by sector.
        
        Args:
            position_weights: Weight per symbol
            symbol_to_sector: Mapping of symbol to sector
            
        Returns:
            Dict of sector -> total weight
        """
        sector_weights = {}
        
        for symbol, weight in position_weights.items():
            sector = symbol_to_sector.get(symbol, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
        
        return sector_weights
    
    def check_portfolio_limits(
        self,
        position_weights: Dict[str, float],
        symbol_to_sector: Dict[str, str] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Check if portfolio violates risk limits.
        
        Args:
            position_weights: Current position weights
            symbol_to_sector: Optional sector mapping
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check individual position concentration
        for symbol, weight in position_weights.items():
            if abs(weight) > self.max_concentration:
                violations.append(
                    f"Position {symbol} ({abs(weight):.2%}) exceeds "
                    f"max concentration ({self.max_concentration:.2%})"
                )
        
        # Check sector concentration
        if symbol_to_sector:
            sector_weights = self.calculate_sector_concentration(
                position_weights, symbol_to_sector
            )
            for sector, weight in sector_weights.items():
                if abs(weight) > self.max_sector_concentration:
                    violations.append(
                        f"Sector {sector} ({abs(weight):.2%}) exceeds "
                        f"max sector concentration ({self.max_sector_concentration:.2%})"
                    )
        
        return len(violations) == 0, violations
    
    def should_cut_positions(
        self,
        current_capital: float,
        drawdown_pct: float,
    ) -> bool:
        """
        Determine if positions should be cut due to drawdown.
        
        Args:
            current_capital: Current portfolio value
            drawdown_pct: Current drawdown as percentage
            
        Returns:
            True if drawdown exceeds limits
        """
        return drawdown_pct <= -self.max_drawdown_pct
    
    def calculate_risk_adjusted_weights(
        self,
        proposed_weights: Dict[str, float],
        correlations: Dict[Tuple[str, str], float],
        target_var: float = 0.15,
    ) -> Dict[str, float]:
        """
        Adjust weights to manage correlation risk.
        
        Args:
            proposed_weights: Initial position weights
            correlations: Pairwise correlations between positions
            target_var: Target portfolio variance (annual)
            
        Returns:
            Adjusted weights
        """
        if not proposed_weights or not correlations:
            return proposed_weights
        
        symbols = list(proposed_weights.keys())
        n = len(symbols)
        
        # Build correlation matrix
        corr_matrix = np.eye(n)
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i != j:
                    key = (sym1, sym2) if (sym1, sym2) in correlations else (sym2, sym1)
                    corr_matrix[i, j] = correlations.get(key, 0.5)
        
        # Calculate portfolio variance
        weights = np.array([proposed_weights[s] for s in symbols])
        
        # Assume unit variances (volatilities already in position sizing)
        portfolio_variance = weights @ corr_matrix @ weights.T
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Scale to target
        if portfolio_vol > 0:
            scale = target_var / portfolio_vol
            scale = np.clip(scale, 0.5, 2.0)
            weights = weights * scale
        
        # Return adjusted weights
        adjusted = {sym: float(w) for sym, w in zip(symbols, weights)}
        
        return adjusted
    
    def dynamic_stop_loss(
        self,
        entry_price: float,
        volatility: float,
        atr_value: float = None,
        volatility_multiplier: float = 2.0,
    ) -> float:
        """
        Calculate stop-loss based on volatility.
        
        Args:
            entry_price: Entry price
            volatility: Stock volatility (daily %)
            atr_value: Average True Range value
            volatility_multiplier: Multiplier for volatility
            
        Returns:
            Stop-loss price
        """
        if entry_price <= 0:
            return 0.0
        
        if atr_value:
            # ATR-based stop loss
            stop_loss = entry_price - (atr_value * volatility_multiplier)
        else:
            # Volatility-based stop loss
            stop_loss = entry_price * (1.0 - (volatility * volatility_multiplier))
        
        return max(0.0, stop_loss)
    
    def dynamic_take_profit(
        self,
        entry_price: float,
        volatility: float,
        win_loss_ratio: float = 1.5,
    ) -> float:
        """
        Calculate take-profit based on risk/reward ratio.
        
        Args:
            entry_price: Entry price
            volatility: Stock volatility
            win_loss_ratio: Target win/loss ratio
            
        Returns:
            Take-profit price
        """
        if entry_price <= 0:
            return 0.0
        
        # Risk (using 2x volatility)
        risk = entry_price * volatility * 2.0
        
        # Reward = Risk * ratio
        reward = risk * win_loss_ratio
        
        # Take profit = entry + reward
        take_profit = entry_price + reward
        
        return take_profit
