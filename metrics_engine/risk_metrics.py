"""
Risk Metrics - Value at Risk, Expected Shortfall, and Risk Analysis

Calculates:
- Value at Risk (VaR) - 95%, 99%
- Conditional VaR / Expected Shortfall (CVaR/ES)
- Maximum Drawdown
- Drawdown Duration
- Recovery Factor
- Tail Ratio
- Skewness and Kurtosis
- Concentration metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """Risk metrics dataclass"""
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    recovery_factor: float
    tail_ratio: float
    skewness: float
    kurtosis: float
    best_trade: float
    worst_trade: float
    
    def __repr__(self) -> str:
        return (
            f"RiskMetrics(\n"
            f"  max_drawdown={self.max_drawdown:.2%}\n"
            f"  avg_drawdown={self.avg_drawdown:.2%}\n"
            f"  var_95={self.var_95:.2%}\n"
            f"  cvar_95={self.cvar_95:.2%}\n"
            f"  recovery_factor={self.recovery_factor:.3f}\n"
            f"  tail_ratio={self.tail_ratio:.3f}\n"
            f")"
        )


class RiskAnalyzer:
    """
    Analyzes risk metrics for portfolio returns.
    """
    
    def __init__(
        self,
        trading_days_per_year: int = 252,
    ):
        """
        Args:
            trading_days_per_year: Days per year for annualization
        """
        self.trading_days_per_year = trading_days_per_year
    
    def calculate_metrics(
        self,
        equity_curve: List[float],
        returns: Optional[List[float]] = None,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            equity_curve: List of portfolio values
            returns: List of returns (calculated if not provided)
            
        Returns:
            RiskMetrics object
        """
        equity_array = np.array(equity_curve)
        
        if returns is None:
            returns = self._calculate_returns(equity_array)
        
        returns_array = np.array(returns)
        
        # Drawdown metrics
        max_dd, avg_dd, duration = self._calculate_drawdowns(equity_array)
        
        # VaR and CVaR
        var_95 = self._value_at_risk(returns_array, 0.95)
        var_99 = self._value_at_risk(returns_array, 0.99)
        cvar_95 = self._conditional_var(returns_array, 0.95)
        cvar_99 = self._conditional_var(returns_array, 0.99)
        
        # Recovery factor
        total_return = (equity_array[-1] / equity_array[0]) - 1.0
        recovery = self._recovery_factor(total_return, max_dd)
        
        # Tail ratio
        tail_ratio = self._tail_ratio(returns_array)
        
        # Distribution properties
        skewness = self._skewness(returns_array)
        kurtosis = self._kurtosis(returns_array)
        
        # Best and worst trades
        best_trade = np.max(returns_array) if len(returns_array) > 0 else 0.0
        worst_trade = np.min(returns_array) if len(returns_array) > 0 else 0.0
        
        return RiskMetrics(
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            drawdown_duration=duration,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            recovery_factor=recovery,
            tail_ratio=tail_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            best_trade=best_trade,
            worst_trade=worst_trade,
        )
    
    def _calculate_returns(self, equity_curve: np.ndarray) -> List[float]:
        """Calculate returns from equity curve."""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        return returns.tolist()
    
    def _calculate_drawdowns(
        self,
        equity_curve: np.ndarray,
    ) -> Tuple[float, float, int]:
        """
        Calculate max drawdown, average drawdown, and duration.
        
        Returns:
            Tuple of (max_drawdown, avg_drawdown, duration_periods)
        """
        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Drawdown at each point
        drawdown = (equity_curve - running_max) / running_max
        
        # Max drawdown
        max_dd = np.min(drawdown)
        
        # Average drawdown (only negative periods)
        dd_negative = drawdown[drawdown < 0]
        avg_dd = np.mean(dd_negative) if len(dd_negative) > 0 else 0.0
        
        # Duration of current drawdown
        duration = 0
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown[i] < 0:
                duration += 1
            else:
                break
        
        return max_dd, avg_dd, duration
    
    def _value_at_risk(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate historical VaR.
        
        Args:
            returns: Array of returns
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            VaR (negative value for losses)
        """
        quantile = 1.0 - confidence
        var = np.quantile(returns, quantile)
        return var
    
    def _conditional_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        Average of all returns worse than VaR level.
        
        Args:
            returns: Array of returns
            confidence: Confidence level
            
        Returns:
            CVaR (negative value for losses)
        """
        var = self._value_at_risk(returns, confidence)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        cvar = np.mean(tail_returns)
        return cvar
    
    def _recovery_factor(
        self,
        total_return: float,
        max_drawdown: float,
    ) -> float:
        """
        Recovery Factor = Total Return / |Max Drawdown|
        Higher is better (rapid recovery from drawdowns)
        
        Args:
            total_return: Total return as decimal
            max_drawdown: Max drawdown as decimal (negative)
            
        Returns:
            Recovery factor
        """
        if max_drawdown >= 0 or max_drawdown == 0:
            return 0.0
        
        recovery = total_return / abs(max_drawdown)
        return recovery
    
    def _tail_ratio(self, returns: np.ndarray) -> float:
        """
        Tail Ratio = |95th percentile| / |5th percentile|
        Higher is better (fat right tail > fat left tail)
        
        Args:
            returns: Array of returns
            
        Returns:
            Tail ratio
        """
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 >= 0:
            return 1.0
        
        tail_ratio = abs(p95) / abs(p5)
        
        return tail_ratio
    
    def _skewness(self, returns: np.ndarray) -> float:
        """
        Calculate skewness of returns.
        Positive = right tail (good)
        Negative = left tail (bad)
        """
        if len(returns) < 3:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        skew = np.mean(((returns - mean) / std) ** 3)
        return skew
    
    def _kurtosis(self, returns: np.ndarray) -> float:
        """
        Calculate excess kurtosis.
        Positive = fat tails (risk)
        Negative = thin tails (safety)
        """
        if len(returns) < 4:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        kurt = np.mean(((returns - mean) / std) ** 4) - 3.0
        return kurt
    
    def underwater_plot_data(
        self,
        equity_curve: List[float],
    ) -> pd.DataFrame:
        """
        Generate data for underwater plot (drawdown over time).
        
        Args:
            equity_curve: List of portfolio values
            
        Returns:
            DataFrame with Date and Drawdown columns
        """
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        
        df = pd.DataFrame({
            'Period': range(len(drawdown)),
            'Drawdown': drawdown,
        })
        
        return df
    
    def drawdown_periods(
        self,
        equity_curve: List[float],
    ) -> List[Dict]:
        """
        Identify all drawdown periods and their characteristics.
        
        Args:
            equity_curve: List of portfolio values
            
        Returns:
            List of dicts with start, end, depth, recovery_days
        """
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        
        drawdown_periods_list = []
        in_drawdown = False
        start_idx = 0
        peak_value = equity_array[0]
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                peak_value = equity_array[i - 1] if i > 0 else equity_array[0]
            
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                end_idx = i
                
                drawdown_periods_list.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'depth': np.min(drawdown[start_idx:end_idx + 1]),
                    'duration': end_idx - start_idx,
                    'recovery_days': end_idx - start_idx,
                })
        
        # Handle ongoing drawdown
        if in_drawdown:
            drawdown_periods_list.append({
                'start_idx': start_idx,
                'end_idx': len(equity_array) - 1,
                'depth': np.min(drawdown[start_idx:]),
                'duration': len(equity_array) - 1 - start_idx,
                'recovery_days': None,  # Still ongoing
            })
        
        return drawdown_periods_list
