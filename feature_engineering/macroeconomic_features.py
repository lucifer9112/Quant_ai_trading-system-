"""
Macroeconomic & Market Structure Features

Signals derived from:
- Index performance
- Volatility environment
- Sector rotation
- Market breadth
- Cross-market relationships
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MacroeconomicFeatures:
    """Generate macroeconomic and market-wide features."""
    
    def __init__(self, window: int = 20):
        """
        Args:
            window: Rolling window for calculations
        """
        self.window = window
    
    def add(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        sector_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Add all macroeconomic features to dataframe.
        
        Args:
            df: Asset price dataframe
            index_df: Index (Nifty/Sensex) data, must be aligned with df
            sector_data: Dict of sector dataframes {sector_name -> df}
        """
        df = df.copy()
        
        # Market-wide features (from asset level)
        df = self._add_market_structure(df)
        
        # Index-relative features (if index data available)
        if index_df is not None:
            df = self._add_index_features(df, index_df)
        
        # Sector-relative features (if sector data available)
        if sector_data is not None:
            df = self._add_sector_features(df, sector_data)
        
        # Volatility regime features
        df = self._add_volatility_regime(df)
        
        # Momentum and trend at market level
        df = self._add_market_momentum(df)
        
        return df
    
    def _add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features capturing market structure from asset data."""
        
        if "Close" not in df.columns:
            return df
        
        close = df["Close"]
        
        # 1. Market regime identification (high/low volatility)
        returns = close.pct_change()
        rolling_vol = returns.rolling(self.window).std() * np.sqrt(252)
        vol_median = rolling_vol.rolling(self.window * 2).median()
        
        df["volatility"] = rolling_vol
        df["vol_percentile"] = vol_median.rolling(self.window).apply(
            lambda x: (rolling_vol.iloc[-1] > np.percentile(x, 75)) * 1.0
            if len(x) > 0 else 0.5
        )
        
        # 2. Volatility of volatility (VoV)
        df["vol_of_vol"] = rolling_vol.rolling(self.window).std()
        df["vol_of_vol"] = df["vol_of_vol"].fillna(0)
        
        # 3. Volatility trend (increasing/decreasing)
        df["vol_trend"] = rolling_vol - rolling_vol.rolling(self.window).mean()
        df["vol_trend"] = df["vol_trend"].fillna(0)
        
        # 4. Return distribution skewness (asymmetry)
        def skewness(x):
            if len(x) < 3:
                return 0
            return ((x - x.mean()) ** 3).mean() / (x.std() ** 3) if x.std() > 0 else 0
        
        df["return_skewness"] = returns.rolling(self.window).apply(skewness)
        df["return_skewness"] = df["return_skewness"].fillna(0)
        
        # 5. Return distribution kurtosis (tail risk)
        def kurtosis(x):
            if len(x) < 4:
                return 3  # Normal distribution kurtosis
            return ((x - x.mean()) ** 4).mean() / (x.std() ** 4) if x.std() > 0 else 3
        
        df["return_kurtosis"] = returns.rolling(self.window).apply(kurtosis)
        df["return_kurtosis"] = df["return_kurtosis"].fillna(3)
        
        # 6. Serial correlation (mean reversion vs momentum)
        def autocorrelation(x):
            if len(x) < 2:
                return 0
            return x.autocorr()
        
        df["serial_correlation"] = returns.rolling(self.window).apply(autocorrelation)
        df["serial_correlation"] = df["serial_correlation"].fillna(0)
        
        return df
    
    def _add_index_features(
        self,
        df: pd.DataFrame,
        index_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Features relative to market index."""
        
        if "Close" not in df.columns or "Close" not in index_df.columns:
            logger.warning("Close column not found in index or asset data")
            return df
        
        # Align index data with asset data by date
        if "Date" in df.columns and "Date" in index_df.columns:
            asset_dates = pd.to_datetime(df["Date"])
            index_dates = pd.to_datetime(index_df["Date"])
            
            # Merge on dates
            temp = df[["Date", "Close"]].copy()
            temp = temp.merge(
                index_df[["Date", "Close"]].rename(columns={"Close": "Index_Close"}),
                on="Date",
                how="left"
            )
            index_close = temp["Index_Close"].fillna(method="ffill")
        else:
            # Assume aligned indices
            index_close = index_df["Close"].copy()
            if len(index_close) != len(df):
                logger.warning("Index and asset data have different lengths, skipping index features")
                return df
        
        # 1. Index relative strength (stock return vs index return)
        asset_returns = df["Close"].pct_change()
        index_returns = index_close.pct_change()
        
        df["index_relative_return"] = asset_returns - index_returns
        
        # 2. Beta (correlation of returns)
        def compute_beta(returns_pair):
            if len(returns_pair) < 2:
                return 1.0
            stock_ret = returns_pair[:, 0]
            market_ret = returns_pair[:, 1]
            
            if np.std(market_ret) == 0:
                return 1.0
            
            covariance = np.cov(stock_ret, market_ret)[0, 1]
            market_variance = np.var(market_ret)
            return covariance / market_variance if market_variance > 0 else 1.0
        
        returns_pair = np.column_stack([asset_returns, index_returns])
        df["beta"] = pd.Series(returns_pair).rolling(self.window).apply(
            lambda x: compute_beta(x.values.reshape(-1, 2))
            if len(x) == (2 * self.window) else 1.0
        )
        df["beta"] = df["beta"].fillna(1.0)
        
        # 3. Correlation with index
        def compute_correlation(returns_pair):
            if len(returns_pair) < 2:
                return 0
            stock_ret = returns_pair[:, 0]
            market_ret = returns_pair[:, 1]
            return np.corrcoef(stock_ret, market_ret)[0, 1]
        
        df["index_correlation"] = pd.Series(returns_pair).rolling(self.window).apply(
            lambda x: compute_correlation(x.values.reshape(-1, 2))
            if len(x) == (2 * self.window) else 0
        )
        df["index_correlation"] = df["index_correlation"].fillna(0)
        
        # 4. Alpha (excess return)
        df["alpha"] = df["index_relative_return"]
        df["alpha_cumulative"] = df["alpha"].cumsum()
        
        # 5. Index momentum
        df["index_momentum"] = index_returns.rolling(self.window).mean()
        
        # 6. Relative to index close (overvalued/undervalued)
        df["relative_to_index"] = df["Close"] / index_close
        df["relative_to_index_ma"] = df["relative_to_index"].rolling(self.window).mean()
        df["relative_valuation"] = df["relative_to_index"] / df["relative_to_index_ma"]
        
        return df
    
    def _add_sector_features(
        self,
        df: pd.DataFrame,
        sector_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Features relative to sector peers."""
        
        if "sector" not in df.columns or not sector_data:
            return df
        
        sector = df["sector"].iloc[0] if len(df) > 0 else None
        if not sector or sector not in sector_data:
            return df
        
        sector_df = sector_data[sector]
        
        if "Close" not in df.columns or "Close" not in sector_df.columns:
            return df
        
        # 1. Sector momentum
        sector_returns = sector_df["Close"].pct_change()
        df["sector_momentum"] = sector_returns.rolling(self.window).mean()
        
        # 2. Relative to sector (outperformance)
        asset_returns = df["Close"].pct_change()
        df["sector_relative_return"] = asset_returns - sector_returns
        
        # 3. Relative strength within sector
        df["sector_relative_strength"] = (
            df["Close"] / sector_df["Close"].rolling(self.window).mean()
        )
        
        # 4. Sector rotation score (is this sector hot or cold)
        df["sector_rotation_score"] = sector_returns.rolling(self.window * 2).mean()
        
        return df
    
    def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility regime classification and features."""
        
        if "Close" not in df.columns:
            return df
        
        returns = df["Close"].pct_change()
        rolling_vol = returns.rolling(self.window).std() * np.sqrt(252)
        
        # Historical volatility percentiles
        vol_50 = rolling_vol.rolling(self.window * 4).quantile(0.5)
        vol_75 = rolling_vol.rolling(self.window * 4).quantile(0.75)
        vol_90 = rolling_vol.rolling(self.window * 4).quantile(0.90)
        
        # Regime labels
        df["volatility_regime"] = "NORMAL"
        df.loc[rolling_vol > vol_75, "volatility_regime"] = "HIGH"
        df.loc[rolling_vol < vol_50, "volatility_regime"] = "LOW"
        
        # Regime numeric
        df["volatility_regime_numeric"] = 1  # Normal
        df.loc[rolling_vol > vol_75, "volatility_regime_numeric"] = 2  # High
        df.loc[rolling_vol < vol_50, "volatility_regime_numeric"] = 0  # Low
        
        # Days in current regime
        regime_numeric = df["volatility_regime_numeric"]
        df["days_in_regime"] = (regime_numeric != regime_numeric.shift()).cumsum()
        df["days_in_regime"] = df.groupby("days_in_regime").cumcount() + 1
        
        return df
    
    def _add_market_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market-wide momentum indicators."""
        
        if "Close" not in df.columns:
            return df
        
        close = df["Close"]
        returns = close.pct_change()
        
        # 1. Short-term momentum (1M, 3M)
        df["momentum_1m"] = (close / close.shift(20) - 1) * 100
        df["momentum_3m"] = (close / close.shift(60) - 1) * 100
        
        # 2. Medium-term trend (6M, 12M)
        df["trend_6m"] = np.sign((close / close.shift(120) - 1))
        df["trend_12m"] = np.sign((close / close.shift(252) - 1))
        
        # 3. Rate of change (ROC) at multiple scales
        df["roc_5"] = (close / close.shift(5) - 1) * 100
        df["roc_10"] = (close / close.shift(10) - 1) * 100
        
        # 4. Moving average alignment (all MAs in uptrend/downtrend)
        if "SMA20" in df.columns and "SMA50" in df.columns and "SMA200" in df.columns:
            df["ma_aligned"] = (
                ((df["SMA20"] > df["SMA50"]) & (df["SMA50"] > df["SMA200"])).astype(int)
            )
        
        return df
