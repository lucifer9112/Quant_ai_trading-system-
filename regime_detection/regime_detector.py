"""
Market Regime Detection - Identify market conditions and regimes

Detects:
- Volatility regimes (low/medium/high)
- Trend regimes (uptrend/downtrend/sideways)
- Correlation regimes (low/high correlation environment)
- VIX regimes for systematic risk
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class VolatilityRegime(Enum):
    """Volatility regime classifications."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class TrendRegime(Enum):
    """Trend regime classifications."""
    UPTREND = 1
    SIDEWAYS = 0
    DOWNTREND = -1


class CorrelationRegime(Enum):
    """Correlation regime classifications."""
    LOW_CORRELATION = 0  # Diversification beneficial
    HIGH_CORRELATION = 1  # Contagion risk elevated


@dataclass
class RegimeState:
    """Current market regime state."""
    timestamp: int
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    correlation_regime: CorrelationRegime
    volatility_value: float
    trend_strength: float  # 0-1, strength of trend
    correlation_value: float  # 0-1, average correlation
    risk_score: float  # Overall risk score (0-1)


class VolatilityRegimeDetector:
    """Detect volatility regimes using KMeans clustering."""
    
    def __init__(self, window: int = 60, n_regimes: int = 3):
        """
        Args:
            window: Rolling window for volatility calculation
            n_regimes: Number of volatility regimes (typically 3)
        """
        self.window = window
        self.n_regimes = n_regimes
        self.kmeans = None
        self.regime_centers = None
        self.scaler = StandardScaler()
    
    def fit(self, returns: np.ndarray) -> None:
        """
        Fit volatility regime detector.
        
        Args:
            returns: Array of daily returns
        """
        # Calculate rolling volatility
        rolling_vol = pd.Series(returns).rolling(self.window).std().values
        rolling_vol = rolling_vol[self.window:]  # Remove NaNs
        
        # Reshape for clustering
        X = rolling_vol.reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit KMeans
        self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        self.kmeans.fit(X_scaled)
        
        # Store regime centers (sorted by volatility)
        centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self.regime_centers = np.sort(centers.flatten())
    
    def predict(self, returns: np.ndarray) -> List[VolatilityRegime]:
        """
        Predict volatility regimes for returns.
        
        Args:
            returns: Array of daily returns
            
        Returns:
            List of VolatilityRegime values
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Calculate rolling volatility
        rolling_vol = pd.Series(returns).rolling(self.window).std().values
        
        # Predict regimes
        X = rolling_vol.reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        labels = self.kmeans.predict(X_scaled)
        
        # Map to regime enums
        regimes = [VolatilityRegime(label) for label in labels]
        return regimes
    
    def get_volatility_value(self, returns: np.ndarray, window: Optional[int] = None) -> float:
        """Get current annualized volatility."""
        if window is None:
            window = self.window
        
        if len(returns) < window:
            window = len(returns)
        
        return pd.Series(returns[-window:]).std() * np.sqrt(252)


class TrendRegimeDetector:
    """Detect trend regimes using price action and technical indicators."""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Args:
            short_window: Short-term trend window
            long_window: Long-term trend window
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def detect_trend(self, prices: np.ndarray) -> TrendRegime:
        """
        Detect trend direction.
        
        Args:
            prices: Array of prices
            
        Returns:
            TrendRegime classification
        """
        if len(prices) < self.long_window:
            return TrendRegime.SIDEWAYS
        
        # Calculate moving averages
        sma_short = pd.Series(prices).rolling(self.short_window).mean().iloc[-1]
        sma_long = pd.Series(prices).rolling(self.long_window).mean().iloc[-1]
        current_price = prices[-1]
        
        # Trend detection logic
        if current_price > sma_short > sma_long:
            return TrendRegime.UPTREND
        elif current_price < sma_short < sma_long:
            return TrendRegime.DOWNTREND
        else:
            return TrendRegime.SIDEWAYS
    
    def get_trend_strength(self, prices: np.ndarray) -> float:
        """
        Get trend strength (0-1).
        
        Args:
            prices: Array of prices
            
        Returns:
            Trend strength (0=no trend, 1=strong trend)
        """
        if len(prices) < self.long_window + 10:
            return 0.0
        
        # Calculate slopes of SMAs
        sma_200 = pd.Series(prices).rolling(200).mean()
        recent_slope = (sma_200.iloc[-1] - sma_200.iloc[-20]) / sma_200.iloc[-20]
        
        # Normalize to 0-1
        strength = min(abs(recent_slope) * 10, 1.0)
        return strength
    
    def detect_multiple_trends(
        self,
        prices: np.ndarray,
        periods: List[int] = None
    ) -> Dict[int, TrendRegime]:
        """
        Detect trends at multiple timeframes.
        
        Args:
            prices: Array of prices
            periods: List of periods to check (default: [20, 50, 200])
            
        Returns:
            Dict mapping period to TrendRegime
        """
        if periods is None:
            periods = [20, 50, 200]
        
        trends = {}
        for period in periods:
            if len(prices) >= period * 2:
                sma = pd.Series(prices).rolling(period).mean()
                current = prices[-1]
                sma_current = sma.iloc[-1]
                sma_past = sma.iloc[-period]
                
                if current > sma_current and sma_current > sma_past:
                    trends[period] = TrendRegime.UPTREND
                elif current < sma_current and sma_current < sma_past:
                    trends[period] = TrendRegime.DOWNTREND
                else:
                    trends[period] = TrendRegime.SIDEWAYS
        
        return trends


class CorrelationRegimeDetector:
    """Detect correlation regimes across assets."""
    
    def __init__(self, window: int = 60, threshold: float = 0.6):
        """
        Args:
            window: Rolling window for correlation
            threshold: Correlation threshold for regime switch
        """
        self.window = window
        self.threshold = threshold
    
    def detect_regime(self, price_data: pd.DataFrame) -> CorrelationRegime:
        """
        Detect correlation regime.
        
        Args:
            price_data: DataFrame with multiple asset prices (columns=assets)
            
        Returns:
            CorrelationRegime classification
        """
        if len(price_data) < self.window:
            return CorrelationRegime.LOW_CORRELATION
        
        # Calculate returns
        returns = price_data.pct_change().iloc[-self.window:]
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Get average correlation (exclude diagonal)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        avg_corr = np.abs(corr_matrix.values[mask]).mean()
        
        # Classify regime
        if avg_corr > self.threshold:
            return CorrelationRegime.HIGH_CORRELATION
        else:
            return CorrelationRegime.LOW_CORRELATION
    
    def get_correlation_value(self, price_data: pd.DataFrame) -> float:
        """
        Get average absolute correlation.
        
        Args:
            price_data: DataFrame with multiple asset prices
            
        Returns:
            Average absolute correlation (0-1)
        """
        if len(price_data) < self.window:
            return 0.0
        
        returns = price_data.pct_change().iloc[-self.window:]
        corr_matrix = returns.corr()
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        avg_corr = np.abs(corr_matrix.values[mask]).mean()
        
        return avg_corr
    
    def get_correlation_matrix(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Get current correlation matrix."""
        if len(price_data) < self.window:
            return price_data.pct_change().corr()
        
        returns = price_data.pct_change().iloc[-self.window:]
        return returns.corr()


class RegimeDetectionEngine:
    """Unified regime detection combining all detectors."""
    
    def __init__(
        self,
        vol_window: int = 60,
        trend_short: int = 20,
        trend_long: int = 50,
        corr_window: int = 60,
    ):
        """
        Args:
            vol_window: Volatility window
            trend_short: Short-term trend window
            trend_long: Long-term trend window
            corr_window: Correlation window
        """
        self.vol_detector = VolatilityRegimeDetector(window=vol_window)
        self.trend_detector = TrendRegimeDetector(trend_short, trend_long)
        self.corr_detector = CorrelationRegimeDetector(window=corr_window)
        self.is_fitted = False
    
    def fit(self, returns: np.ndarray) -> None:
        """Fit the regime detection model."""
        self.vol_detector.fit(returns)
        self.is_fitted = True
    
    def detect_regime(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        multi_asset_prices: Optional[pd.DataFrame] = None,
    ) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            prices: Array of prices
            returns: Array of returns
            multi_asset_prices: Optional DataFrame for correlation regime
            
        Returns:
            RegimeState object with all regime classifications
        """
        # Volatility regime
        vol_regimes = self.vol_detector.predict(returns)
        vol_regime = vol_regimes[-1] if vol_regimes else VolatilityRegime.MEDIUM
        vol_value = self.vol_detector.get_volatility_value(returns)
        
        # Trend regime
        trend_regime = self.trend_detector.detect_trend(prices)
        trend_strength = self.trend_detector.get_trend_strength(prices)
        
        # Correlation regime
        if multi_asset_prices is not None:
            corr_regime = self.corr_detector.detect_regime(multi_asset_prices)
            corr_value = self.corr_detector.get_correlation_value(multi_asset_prices)
        else:
            corr_regime = CorrelationRegime.LOW_CORRELATION
            corr_value = 0.0
        
        # Calculate overall risk score
        vol_score = vol_regime.value / 2  # 0-1
        trend_risk = 1.0 if trend_regime == TrendRegime.DOWNTREND else 0.0
        corr_risk = 0.5 if corr_regime == CorrelationRegime.HIGH_CORRELATION else 0.0
        risk_score = (vol_score * 0.5 + trend_risk * 0.3 + corr_risk * 0.2)
        
        return RegimeState(
            timestamp=len(prices) - 1,
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            correlation_regime=corr_regime,
            volatility_value=vol_value,
            trend_strength=trend_strength,
            correlation_value=corr_value,
            risk_score=risk_score,
        )
    
    def get_regime_history(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        window: int = 252,
    ) -> List[RegimeState]:
        """
        Get regime history for the last N periods.
        
        Args:
            prices: Array of prices
            returns: Array of returns
            window: Number of periods to analyze
            
        Returns:
            List of RegimeState objects
        """
        history = []
        start_idx = max(0, len(prices) - window)
        
        for i in range(start_idx, len(prices)):
            current_prices = prices[:i+1]
            current_returns = returns[:i+1]
            
            regime = self.detect_regime(current_prices, current_returns)
            history.append(regime)
        
        return history
