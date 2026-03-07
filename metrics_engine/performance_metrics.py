"""
Performance Metrics - Returns and Risk-Adjusted Returns

Calculates:
- Total return
- Annualized return
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Information ratio
- Jenson's alpha
- Treynor ratio
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Performance metrics dataclass"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: Optional[float]
    max_drawdown: float
    starting_value: float
    ending_value: float
    num_periods: int
    
    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(\n"
            f"  total_return={self.total_return:.2%}\n"
            f"  annualized_return={self.annualized_return:.2%}\n"
            f"  sharpe_ratio={self.sharpe_ratio:.3f}\n"
            f"  sortino_ratio={self.sortino_ratio:.3f}\n"
            f"  calmar_ratio={self.calmar_ratio:.3f}\n"
            f"  max_drawdown={self.max_drawdown:.2%}\n"
            f")"
        )


class PerformanceAnalyzer:
    """
    Calculates performance metrics for a trading strategy.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.04,  # 4% annual
        trading_days_per_year: int = 252,
        benchmark_returns: Optional[List[float]] = None,
    ):
        """
        Args:
            risk_free_rate: Risk-free rate for Sharpe/Treynor calculations
            trading_days_per_year: Number of trading days per year
            benchmark_returns: Optional benchmark returns for information ratio
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.benchmark_returns = benchmark_returns or []
    
    def calculate_returns(
        self,
        equity_curve: List[float],
    ) -> List[float]:
        """
        Calculate period returns from equity curve.
        
        Args:
            equity_curve: List of portfolio values
            
        Returns:
            List of returns (as decimals)
        """
        if len(equity_curve) < 2:
            return []
        
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        return returns.tolist()
    
    def calculate_metrics(
        self,
        equity_curve: List[float],
        risk_free_rate: Optional[float] = None,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: List of portfolio values over time
            risk_free_rate: Override risk_free_rate if provided
            
        Returns:
            PerformanceMetrics object
        """
        if len(equity_curve) < 2:
            raise ValueError("Need at least 2 data points")
        
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        equity_array = np.array(equity_curve)
        starting_value = equity_array[0]
        ending_value = equity_array[-1]
        
        # Total return
        total_return = (ending_value / starting_value) - 1.0
        
        # Calculate returns
        returns = self.calculate_returns(equity_curve)
        returns_array = np.array(returns)
        
        # Annualized return
        num_periods = len(equity_curve) - 1
        years = num_periods / self.trading_days_per_year
        annualized_return = (ending_value / starting_value) ** (1.0 / years) - 1.0
        
        # Volatility (annualized)
        daily_volatility = np.std(returns_array)
        annualized_volatility = daily_volatility * np.sqrt(self.trading_days_per_year)
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_array)
        
        # Sharpe ratio
        excess_return = annualized_return - rf
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        annualized_downside_vol = downside_volatility * np.sqrt(self.trading_days_per_year)
        sortino_ratio = excess_return / annualized_downside_vol if annualized_downside_vol > 0 else 0.0
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
        
        # Information ratio (if benchmark provided)
        information_ratio = None
        if self.benchmark_returns and len(self.benchmark_returns) == len(returns):
            excess_returns = returns_array - np.array(self.benchmark_returns)
            tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
            mean_excess = np.mean(excess_returns) * self.trading_days_per_year
            information_ratio = mean_excess / tracking_error if tracking_error > 0 else 0.0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            max_drawdown=max_drawdown,
            starting_value=starting_value,
            ending_value=ending_value,
            num_periods=num_periods,
        )
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        return np.min(drawdowns)
    
    def rolling_sharpe(
        self,
        equity_curve: List[float],
        window: int = 60,
        risk_free_rate: Optional[float] = None,
    ) -> List[float]:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            equity_curve: List of portfolio values
            window: Rolling window size in periods
            risk_free_rate: Override if provided
            
        Returns:
            List of rolling Sharpe ratios
        """
        if len(equity_curve) < window:
            return []
        
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        returns = self.calculate_returns(equity_curve)
        returns_array = np.array(returns)
        
        rolling_sharpes = []
        
        for i in range(window, len(returns_array) + 1):
            window_returns = returns_array[i - window:i]
            
            # Annualized return for this window
            window_return = np.mean(window_returns) * self.trading_days_per_year
            
            # Annualized volatility
            window_vol = np.std(window_returns) * np.sqrt(self.trading_days_per_year)
            
            # Sharpe ratio
            sharpe = (window_return - rf) / window_vol if window_vol > 0 else 0.0
            rolling_sharpes.append(sharpe)
        
        return rolling_sharpes
    
    def rolling_volatility(
        self,
        equity_curve: List[float],
        window: int = 20,
    ) -> List[float]:
        """
        Calculate rolling volatility.
        
        Args:
            equity_curve: List of portfolio values
            window: Rolling window size
            
        Returns:
            List of rolling volatilities (annualized)
        """
        if len(equity_curve) < window:
            return []
        
        returns = self.calculate_returns(equity_curve)
        returns_array = np.array(returns)
        
        rolling_vols = []
        
        for i in range(window, len(returns_array) + 1):
            window_returns = returns_array[i - window:i]
            vol = np.std(window_returns) * np.sqrt(self.trading_days_per_year)
            rolling_vols.append(vol)
        
        return rolling_vols
    
    def monthly_returns(
        self,
        equity_curve: List[float],
        dates: Optional[List] = None,
    ) -> pd.DataFrame:
        """
        Calculate monthly returns.
        
        Args:
            equity_curve: List of daily portfolio values
            dates: List of dates (if available)
            
        Returns:
            DataFrame with monthly returns
        """
        if dates is None:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=len(equity_curve), freq='D')
        
        df = pd.DataFrame({
            'Date': dates,
            'Value': equity_curve,
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['YearMonth'] = df['Date'].dt.to_period('M')
        
        # Get last value for each month
        monthly = df.groupby('YearMonth')['Value'].last()
        
        # Calculate returns
        monthly_returns = monthly.pct_change().dropna()
        
        return pd.DataFrame({
            'Month': monthly_returns.index.astype(str),
            'Return': monthly_returns.values,
        })
    
    def calculate_tail_ratio(self, returns: List[float]) -> float:
        """
        Calculate tail ratio = |95th percentile| / |5th percentile|
        Higher is better (positive tail > negative tail)
        
        Args:
            returns: List of returns
            
        Returns:
            Tail ratio
        """
        returns_array = np.array(returns)
        
        percentile_95 = np.percentile(returns_array, 95)
        percentile_5 = np.percentile(returns_array, 5)
        
        if percentile_5 == 0:
            return 1.0
        
        tail_ratio = abs(percentile_95) / abs(percentile_5)
        
        return tail_ratio
