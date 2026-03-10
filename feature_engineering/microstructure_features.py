"""
Microstructure Features - Order Flow & Market Structure Signals

Signals derived from:
- Order flow imbalance
- Volume distribution
- Market depth
- Liquidity measures
- Trading dynamics
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MicrostructureFeatures:
    """Generate market microstructure features."""
    
    def __init__(self, window: int = 20):
        """
        Args:
            window: Rolling window for calculations
        """
        self.window = window
    
    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all microstructure features to dataframe."""
        df = df.copy()
        
        # Volume-based features
        df = self._add_volume_features(df)
        
        # Price-volume features
        df = self._add_price_volume_features(df)
        
        # Liquidity features
        df = self._add_liquidity_features(df)
        
        # Trading dynamics
        df = self._add_trading_dynamics(df)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based microstructure signals."""
        
        if "Volume" not in df.columns:
            logger.warning("Volume column not found, skipping volume features")
            return df
        
        volume = df["Volume"].fillna(0)
        
        # 1. Volume ratio (current vs average)
        df["volume_ratio"] = volume / volume.rolling(self.window).mean()
        df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
        
        # 2. Volume trend (increasing or decreasing)
        df["volume_trend"] = volume.rolling(self.window).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )
        
        # 3. Volume standard deviation (volatility of volume)
        df["volume_volatility"] = volume.rolling(self.window).std()
        df["volume_volatility"] = df["volume_volatility"].fillna(0)
        
        # 4. On-Balance Volume (OBV) trend
        df["OBV"] = (np.sign(df["Close"].diff()) * volume).fillna(0).cumsum()
        df["OBV_sma"] = df["OBV"].rolling(self.window).mean()
        df["OBV_momentum"] = df["OBV"] - df["OBV_sma"]
        
        # 5. Volume-Weighted Average Price (VWAP)
        if "High" in df.columns and "Low" in df.columns:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            cumulative_vol = volume.cumsum()
            cumulative_typical_price_vol = (typical_price * volume).cumsum()
            df["vwap"] = cumulative_typical_price_vol / cumulative_vol.replace(0, 1)
            df["price_vwap_ratio"] = df["Close"] / df["vwap"]
            df["vwap_distance"] = (df["Close"] - df["vwap"]) / df["vwap"]
        
        return df
    
    def _add_price_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-volume interaction features."""
        
        if "Volume" not in df.columns or "Close" not in df.columns:
            return df
        
        volume = df["Volume"].fillna(0)
        close = df["Close"]
        returns = close.pct_change()
        
        # 1. Price-Volume trend (do volume and price move together)
        price_trend = close.rolling(self.window).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )
        volume_trend = volume.rolling(self.window).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )
        
        # Correlation of price change and volume change
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        df["price_volume_correlation"] = price_change.rolling(self.window).corr(volume_change)
        df["price_volume_correlation"] = df["price_volume_correlation"].fillna(0)
        
        # 2. Volume-weighted momentum
        df["volume_weighted_momentum"] = (returns * volume).rolling(self.window).sum() / volume.rolling(self.window).sum()
        df["volume_weighted_momentum"] = df["volume_weighted_momentum"].fillna(0)
        
        # 3. Accumulation/Distribution Line
        ad_line = (
            ((close - df["Low"]) - (df["High"] - close)) / 
            (df["High"] - df["Low"]).replace(0, 1)
        ) * volume
        df["ad_line"] = ad_line.cumsum()
        
        # 4. Money Flow Index (MFI) - volume-weighted RSI
        if len(df) > self.window:
            typical_price = (df["High"] + df["Low"] + close) / 3
            positive_flow = (typical_price.diff() > 0) * typical_price * volume
            negative_flow = (typical_price.diff() < 0) * abs(typical_price) * volume
            
            positive_mf = positive_flow.rolling(self.window).sum()
            negative_mf = negative_flow.rolling(self.window).sum()
            
            mfi_ratio = positive_mf / negative_mf.replace(0, 1)
            df["mfi"] = 100 - (100 / (1 + mfi_ratio))
            df["mfi"] = df["mfi"].fillna(50)
        
        return df
    
    def _add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Liquidity-related features."""
        
        if "High" not in df.columns or "Low" not in df.columns:
            return df
        
        # 1. Bid-Ask Spread proxy (High - Low / Close)
        df["bid_ask_spread"] = (df["High"] - df["Low"]) / df["Close"]
        df["bid_ask_spread_ma"] = df["bid_ask_spread"].rolling(self.window).mean()
        df["spread_ratio"] = df["bid_ask_spread"] / df["bid_ask_spread_ma"]
        
        # 2. Average True Range (ATR) as volatility/liquidity measure
        high_low = df["High"] - df["Low"]
        high_close = abs(df["High"] - df["Close"].shift())
        low_close = abs(df["Low"] - df["Close"].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr"] = true_range.rolling(self.window).mean()
        
        # 3. Liquidity ratio (Volume / ATR)
        if "Volume" in df.columns:
            df["liquidity_ratio"] = df["Volume"] / (df["atr"] + 1e-8)
        
        # 4. Price range relative to typical price
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        df["range_to_price"] = (df["High"] - df["Low"]) / typical_price
        
        return df
    
    def _add_trading_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trading behavior indicators."""
        
        if "Close" not in df.columns:
            return df
        
        close = df["Close"]
        
        # 1. Momentum acceleration (price change acceleration)
        momentum = close.pct_change()
        df["momentum"] = momentum
        df["momentum_acceleration"] = momentum.diff()
        
        # 2. Price position in range (0-1, where 0=low, 1=high)
        high = df.get("High", close.rolling(self.window).max())
        low = df.get("Low", close.rolling(self.window).min())
        
        range_high_low = (high - low).replace(0, 1)
        df["position_in_range"] = (close - low) / range_high_low
        df["position_in_range"] = df["position_in_range"].clip(0, 1)
        
        # 3. Close relative to high/low
        if "High" in df.columns and "Low" in df.columns:
            df["close_high_ratio"] = close / df["High"]
            df["close_low_ratio"] = close / df["Low"]
        
        # 4. Intrabar range (High - Low) momentum
        if "High" in df.columns and "Low" in df.columns:
            df["intrabar_range"] = df["High"] - df["Low"]
            df["intrabar_range_ma"] = df["intrabar_range"].rolling(self.window).mean()
            df["intrabar_expansion"] = df["intrabar_range"] / df["intrabar_range_ma"]
        
        # 5. Sequential closes (how many consecutive up/down closes)
        df["returns"] = close.pct_change()
        df["up_close"] = (df["returns"] > 0).astype(int)
        
        # Consecutive up closes
        consecutive_ups = df["up_close"].rolling(self.window).sum()
        df["consecutive_ups_ratio"] = consecutive_ups / self.window
        
        return df
