"""
Non-Linear & Interaction Features

Polynomial transformations, interaction terms, and derived features
that capture non-linear relationships.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class NonlinearFeatures:
    """Generate non-linear and interaction features."""
    
    def __init__(self, window: int = 20):
        """
        Args:
            window: Rolling window for calculations
        """
        self.window = window
    
    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add non-linear features to dataframe."""
        df = df.copy()
        
        # Polynomial transformations
        df = self._add_polynomial_features(df)
        
        # Interaction terms
        df = self._add_interaction_features(df)
        
        # Ratio features
        df = self._add_ratio_features(df)
        
        # Log transformations
        df = self._add_log_features(df)
        
        return df
    
    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Polynomial transformations of key features."""
        
        if "Close" not in df.columns:
            return df
        
        returns = df["Close"].pct_change()
        
        # 1. Squared returns (tail risk)
        df["returns_squared"] = returns ** 2
        
        # 2. Cubic returns (asymmetry)
        df["returns_cubic"] = returns ** 3
        
        # 3. Higher-order RSI transformation
        if "RSI" in df.columns:
            rsi = df["RSI"]
            df["rsi_squared"] = (rsi - 50) ** 2 / 2500  # Centered and normalized
            df["rsi_cubed"] = (rsi - 50) ** 3 / 125000
        
        # 4. Volatility squared (term for variance models)
        if "volatility" in df.columns:
            df["volatility_squared"] = df["volatility"] ** 2
        
        # 5. Price cubed relative to moving average
        if "SMA20" in df.columns:
            price_ratio = df["Close"] / df["SMA20"]
            df["price_ratio_to_sma_cubed"] = (price_ratio - 1) ** 3
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interaction terms between key indicators."""
        
        # RSI × Momentum (when RSI extreme AND prices accelerating)
        if "RSI" in df.columns and "momentum" in df.columns:
            rsi_extreme = (df["RSI"] > 70) | (df["RSI"] < 30)
            momentum_positive = df["momentum"] > 0
            df["rsi_momentum_interaction"] = (rsi_extreme.astype(float) * 
                                              momentum_positive.astype(float))
        
        # Trend × Volume (do volume and trend align)
        if "SMA20" in df.columns and "SMA50" in df.columns and "Volume" in df.columns:
            uptrend = df["SMA20"] > df["SMA50"]
            high_volume = df["Volume"] > df["Volume"].rolling(self.window).mean()
            df["trend_volume_agreement"] = (uptrend.astype(float) * 
                                            high_volume.astype(float))
        
        # Volatility × Price position (how much risk at current price)
        if "volatility" in df.columns and "Close" in df.columns:
            if "SMA50" in df.columns:
                high_price = df["Close"] > df["SMA50"]
                high_vol = df["volatility"] > df["volatility"].rolling(self.window).mean()
                df["high_vol_high_price"] = (high_vol.astype(float) * 
                                              high_price.astype(float))
        
        # MACD × RSI (filtered signals)
        if "MACD" in df.columns and "MACD_signal" in df.columns and "RSI" in df.columns:
            macd_bullish = df["MACD"] > df["MACD_signal"]
            rsi_not_overbought = df["RSI"] < 80
            df["macd_rsi_filtered_signal"] = (macd_bullish.astype(float) * 
                                               rsi_not_overbought.astype(float))
        
        # Price range interaction (expansion confirms strength)
        if "High" in df.columns and "Low" in df.columns:
            price_range = df["High"] - df["Low"]
            range_ma = price_range.rolling(self.window).mean()
            expanding_range = price_range > range_ma
            
            if "momentum" in df.columns:
                positive_momentum = df["momentum"] > 0
                df["expanding_range_momentum"] = (expanding_range.astype(float) * 
                                                  positive_momentum.astype(float))
        
        return df
    
    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ratio-based features."""
        
        if "Close" not in df.columns:
            return df
        
        # 1. Price relative to moving averages
        if "SMA20" in df.columns:
            df["price_to_sma20"] = df["Close"] / df["SMA20"]
        
        if "SMA50" in df.columns:
            df["price_to_sma50"] = df["Close"] / df["SMA50"]
        
        if "SMA200" in df.columns:
            df["price_to_sma200"] = df["Close"] / df["SMA200"]
        
        # 2. Volume ratio (current to average)
        if "Volume" in df.columns:
            volume_ma = df["Volume"].rolling(self.window).mean()
            df["volume_to_ma"] = df["Volume"] / volume_ma.replace(0, 1)
        
        # 3. Range ratio (today's range to average range)
        if "High" in df.columns and "Low" in df.columns:
            range_today = df["High"] - df["Low"]
            range_ma = range_today.rolling(self.window).mean()
            df["range_expansion_ratio"] = range_today / range_ma.replace(0, 1)
        
        # 4. ATR ratio
        if "ATR" in df.columns:
            atr_ma = df["ATR"].rolling(self.window).mean()
            df["atr_to_ma"] = df["ATR"] / atr_ma.replace(0, 1)
        
        # 5. Consecutive up/down days ratio
        if "Close" in df.columns:
            returns = df["Close"].pct_change()
            consecutive_ups = (returns > 0).rolling(self.window).sum()
            df["consecutive_ups_ratio"] = consecutive_ups / self.window
        
        return df
    
    def _add_log_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log transformations for skewed features."""
        
        if "Close" not in df.columns:
            return df
        
        close = df["Close"]
        
        # 1. Log returns (often more normally distributed)
        returns = close.pct_change()
        df["log_returns"] = np.log1p(returns)
        
        # 2. Log volume
        if "Volume" in df.columns:
            volume = df["Volume"] + 1  # Avoid log(0)
            df["log_volume"] = np.log(volume)
        
        # 3. Log of rolling max (normalization for long-term position)
        rolling_max = close.rolling(252, min_periods=1).max()
        df["log_price_to_rolling_max"] = np.log(close / rolling_max.replace(0, 1))
        
        # 4. Log price ratios
        if "SMA20" in df.columns:
            df["log_price_sma20"] = np.log(df["Close"] / df["SMA20"].replace(0, 1))
        
        return df
