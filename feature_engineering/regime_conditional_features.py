"""
Regime-Conditional Features

Adaptive features that change based on market regime:
- Regime-specific indicator thresholds
- Dynamic lookback windows
- Conditional signals
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RegimeConditionalFeatures:
    """Generate regime-adaptive features."""
    
    def __init__(self, window: int = 20):
        """
        Args:
            window: Base rolling window
        """
        self.window = window
    
    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime-conditional features to dataframe."""
        df = df.copy()
        
        # First, establish volatility regime
        df = self._identify_regime(df)
        
        # Then, create regime-conditional features
        df = self._add_regime_adjusted_indicators(df)
        df = self._add_dynamic_windows(df)
        df = self._add_conditional_signals(df)
        
        return df
    
    def _identify_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify high/low volatility regimes."""
        
        if "Close" not in df.columns:
            return df
        
        returns = df["Close"].pct_change()
        rolling_vol = returns.rolling(self.window).std() * np.sqrt(252)
        
        # Percentile-based regime classification
        vol_20_percentile = rolling_vol.rolling(self.window * 4).quantile(0.20)
        vol_80_percentile = rolling_vol.rolling(self.window * 4).quantile(0.80)
        
        df["volatility_percentile"] = rolling_vol.rolling(self.window * 4).apply(
            lambda x: (x.iloc[-1] >= x.min()) / (x.max() - x.min()) 
            if x.max() > x.min() else 0.5
        )
        
        # Regime: LOW (vol < 20th), NORMAL (20th-80th), HIGH (>80th)
        df["volatility_regime_base"] = "NORMAL"
        df.loc[rolling_vol <= vol_20_percentile, "volatility_regime_base"] = "LOW"
        df.loc[rolling_vol >= vol_80_percentile, "volatility_regime_base"] = "HIGH"
        
        # Trend regime (up/down/sideways)
        sma_fast = df["Close"].rolling(20).mean()
        sma_slow = df["Close"].rolling(50).mean()
        
        df["trend_regime"] = "SIDEWAYS"
        df.loc[sma_fast > sma_slow * 1.001, "trend_regime"] = "UP"
        df.loc[sma_fast < sma_slow * 0.999, "trend_regime"] = "DOWN"
        
        # Combined regime
        df["combined_regime"] = df["volatility_regime_base"] + "_" + df["trend_regime"]
        
        return df
    
    def _add_regime_adjusted_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create indicators that adapt to regime."""
        
        if "Close" not in df.columns:
            return df
        
        close = df["Close"]
        
        # 1. Dynamic RSI period based on volatility
        returns = close.pct_change()
        rolling_vol = returns.rolling(self.window).std()
        
        # In HIGH vol: use shorter window (faster response)
        # In LOW vol: use longer window (filter noise)
        def adaptive_rsi(values, vol_series):
            if len(values) < 14:
                return np.nan
            
            # Window ranges from 10 (high vol) to 20 (low vol)
            vol_norm = (vol_series.iloc[-1] - vol_series.min()) / (vol_series.max() - vol_series.min())
            rsi_window = int(20 - (vol_norm * 10))  # 10-20
            
            if len(values) < rsi_window:
                return np.nan
            
            deltas = np.diff(values)
            seed = deltas[:rsi_window+1]
            up = seed[seed >= 0].sum() / rsi_window
            down = -seed[seed < 0].sum() / rsi_window
            
            rs = up / down if down != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        
        # Apply adaptive RSI
        if "volatility" in df.columns or "rolling_vol" in df.columns:
            vol_col = "volatility" if "volatility" in df.columns else "rolling_vol"
            df["adaptive_rsi"] = close.rolling(21).apply(
                lambda x: adaptive_rsi(x.values, df.loc[x.index, vol_col])
            )
        else:
            df["adaptive_rsi"] = df.get("RSI", np.nan)
        
        # 2. Dynamic MACD periods
        # In trending markets: use standard periods (12, 26, 9)
        # In ranging markets: use shorter periods (5, 13, 4)
        def adaptive_macd_signal(trend_regime, base_signal):
            if trend_regime in ["UP", "DOWN"]:
                return base_signal  # Use standard periods
            else:
                # For SIDEWAYS, use shorter periods (faster response)
                return base_signal * 1.5  # Conceptual - actual implementation would be more complex
        
        if "MACD" in df.columns:
            df["adaptive_macd_direction"] = df["trend_regime"].apply(
                lambda x: 0.8 if x in ["UP", "DOWN"] else 1.0
            )
        
        # 3. Dynamic moving average periods
        # In trending: use short MA (12-day)
        # In ranging: use long MA (26-day)
        df["ma_short_adaptive"] = np.where(
            df.get("trend_regime", "SIDEWAYS").isin(["UP", "DOWN"]),
            close.rolling(12).mean(),
            close.rolling(26).mean()
        )
        
        # 4. Volatility-adjusted momentum
        momentum = close.pct_change(self.window)
        vol = returns.rolling(self.window).std()
        
        df["vol_adjusted_momentum"] = momentum / (vol + 1e-8)
        
        return df
    
    def _add_dynamic_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features using dynamic lookback windows based on regime."""
        
        if "Close" not in df.columns:
            return df
        
        close = df["Close"]
        returns = close.pct_change()
        
        # Window allocation: HIGH vol -> shorter lookback, LOW vol -> longer lookback
        vol_percentile = df.get("volatility_percentile", 0.5)
        
        # Dynamic windows: range from 10 (high vol, responsive) to 40 (low vol, stable)
        dynamic_window = 10 + ((1 - vol_percentile) * 30)  # Inverted: low vol = long window
        dynamic_window = dynamic_window.fillna(20).astype(int)
        
        # 1. Dynamic moving average
        df["dynamic_ma"] = close.rolling(self.window).mean()  # Simplified
        
        # 2. Dynamic momentum with regime-based lookback
        momentum_short = close.pct_change(5)  # Fast
        momentum_long = close.pct_change(20)  # Slow
        
        # Blend based on trend: more weight to fast in trending, long in sideways
        trend_weight = (df.get("trend_regime", "SIDEWAYS") != "SIDEWAYS").astype(float)
        df["dynamic_momentum"] = (trend_weight * momentum_short) + ((1 - trend_weight) * momentum_long)
        
        # 3. Regime-adjusted mean reversion strength
        # In LOW vol: mean reversion stronger
        # In HIGH vol: momentum stronger
        deviations = close - close.rolling(self.window).mean()
        df["mean_reversion_strength"] = vol_percentile  # Higher percent = more momentum
        df["deviations_from_ma"] = deviations
        
        return df
    
    def _add_conditional_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary or conditional signals based on regime."""
        
        if "Close" not in df.columns:
            return df
        
        close = df["Close"]
        
        # 1. Buy signal strength (regime-dependent)
        # HIGH vol: only strong trending signals
        # LOW vol: react to smaller price moves
        
        price_change = close.pct_change()
        threshold_high_vol = 0.015  # 1.5% move required
        threshold_low_vol = 0.005   # 0.5% move required
        
        threshold = np.where(
            df.get("volatility_regime_base", "NORMAL") == "HIGH",
            threshold_high_vol,
            threshold_low_vol
        )
        
        df["strong_uptrend_signal"] = (price_change > threshold).astype(int)
        df["strong_downtrend_signal"] = (price_change < -threshold).astype(int)
        
        # 2. Regime confirmation (do multiple indicators agree)
        num_bullish_signals = 0
        
        # Count bullish signals
        if "RSI" in df.columns:
            num_bullish_signals += (df["RSI"] > 50).astype(int)
        
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            num_bullish_signals += (df["MACD"] > df["MACD_signal"]).astype(int)
        
        if "SMA20" in df.columns and "SMA50" in df.columns:
            num_bullish_signals += (df["SMA20"] > df["SMA50"]).astype(int)
        
        if num_bullish_signals > 0:
            df["bullish_signal_agreement"] = num_bullish_signals / 3  # Out of 3
        
        # 3. Volatility expansion/contraction signal
        vol = df.get("volatility", None)
        if vol is not None:
            vol_ma = vol.rolling(self.window).mean()
            df["vol_expansion_signal"] = (vol > vol_ma * 1.2).astype(int)
            df["vol_contraction_signal"] = (vol < vol_ma * 0.8).astype(int)
        
        # 4. Regime transition signal (change imminent?)
        if "volatility_regime_base" in df.columns:
            regime_change = (df["volatility_regime_base"] != df["volatility_regime_base"].shift()).astype(int)
            df["regime_transition_indicator"] = regime_change.rolling(5).sum()
        
        return df
