"""
Cross-Asset Features

Relationships between assets:
- Correlation dynamics
- Pair trading signals
- Sector relationships
- Lead/lag relationships
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class CrossAssetFeatures:
    """Generate features based on relationships between assets."""
    
    def __init__(self, window: int = 20):
        """
        Args:
            window: Rolling window for calculations
        """
        self.window = window
    
    def add(
        self,
        df: pd.DataFrame,
        universe_data: Optional[Dict[str, pd.DataFrame]] = None,
        sector_peers: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Add cross-asset features.
        
        Args:
            df: Asset dataframe (must have 'symbol' column)
            universe_data: Dict of all assets {symbol -> df}
            sector_peers: Dict {symbol -> list of peer symbols}
        """
        df = df.copy()
        
        if universe_data:
            df = self._add_pair_features(df, universe_data)
            df = self._add_correlation_features(df, universe_data)
        
        if sector_peers:
            df = self._add_sector_peer_features(df, sector_peers, universe_data)
        
        return df
    
    def _add_pair_features(
        self,
        df: pd.DataFrame,
        universe_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Features relative to specific peer pairs."""
        
        if "Close" not in df.columns or "symbol" not in df.columns:
            return df
        
        symbol = df["symbol"].iloc[0] if len(df) > 0 else None
        if not symbol or symbol not in universe_data:
            return df
        
        # Find correlations with all other assets
        asset_returns = df["Close"].pct_change()
        
        correlations = {}
        for other_symbol, other_df in universe_data.items():
            if other_symbol == symbol or "Close" not in other_df.columns:
                continue
            
            # Align by index position (assume same dates)
            other_returns = other_df["Close"].pct_change()
            
            if len(other_returns) == len(asset_returns):
                # Compute rolling correlation
                corr = asset_returns.rolling(self.window).corr(other_returns)
                correlations[other_symbol] = corr.fillna(0)
        
        # Add top 3 most correlated and most anti-correlated assets
        if correlations:
            mean_corrs = {k: v.mean() for k, v in correlations.items()}
            
            # Sort by absolute correlation
            sorted_corrs = sorted(mean_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for i, (peer_symbol, _) in enumerate(sorted_corrs[:3]):
                df[f"corr_with_peer_{i+1}"] = correlations[peer_symbol]
            
            # Most positively and negatively correlated
            positive_corrs = {k: v for k, v in mean_corrs.items() if v > 0}
            negative_corrs = {k: v for k, v in mean_corrs.items() if v < 0}
            
            if positive_corrs:
                best_positive = max(positive_corrs.items(), key=lambda x: x[1])
                df["corr_best_positive"] = correlations[best_positive[0]]
                df["peer_best_positive"] = best_positive[0]
            
            if negative_corrs:
                best_negative = max(negative_corrs.items(), key=lambda x: abs(x[1]))
                df["corr_best_negative"] = correlations[best_negative[0]]
                df["peer_best_negative"] = best_negative[0]
        
        return df
    
    def _add_correlation_features(
        self,
        df: pd.DataFrame,
        universe_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Features describing correlation environment."""
        
        if "Close" not in df.columns:
            return df
        
        asset_returns = df["Close"].pct_change()
        
        # Compute correlations with all other assets
        correlations = []
        for other_symbol, other_df in universe_data.items():
            if "Close" not in other_df.columns or len(other_df) != len(df):
                continue
            
            other_returns = other_df["Close"].pct_change()
            corr = asset_returns.corr(other_returns)
            if not np.isnan(corr):
                correlations.append(corr)
        
        if correlations:
            # 1. Average correlation with universe
            df["avg_correlation_universe"] = np.mean(correlations)
            
            # 2. Correlation volatility (how stable are correlations)
            df["correlation_volatility"] = np.std(correlations)
            
            # 3. Highest/lowest correlation
            df["max_correlation"] = np.max(correlations)
            df["min_correlation"] = np.min(correlations)
            
            # 4. Number of assets with positive correlation
            positive_count = sum(1 for c in correlations if c > 0)
            df["positive_correlation_count"] = positive_count / len(correlations)
        
        return df
    
    def _add_sector_peer_features(
        self,
        df: pd.DataFrame,
        sector_peers: Dict[str, List[str]],
        universe_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Features relative to sector peers."""
        
        if "Close" not in df.columns or "symbol" not in df.columns:
            return df
        
        symbol = df["symbol"].iloc[0] if len(df) > 0 else None
        if not symbol or symbol not in sector_peers:
            return df
        
        peers = sector_peers[symbol]
        peer_returns_list = []
        
        # Collect returns from all peers
        if universe_data:
            for peer in peers:
                if peer in universe_data and "Close" in universe_data[peer].columns:
                    peer_df = universe_data[peer]
                    if len(peer_df) == len(df):
                        peer_returns_list.append(peer_df["Close"].pct_change())
        
        if not peer_returns_list:
            return df
        
        asset_returns = df["Close"].pct_change()
        
        # 1. Outperformance vs sector average
        sector_avg_returns = pd.concat(peer_returns_list, axis=1).mean(axis=1)
        df["outperformance_vs_sector"] = (asset_returns - sector_avg_returns).cumsum()
        
        # 2. Relative strength ranking (how strong vs peers)
        all_returns = pd.concat([pd.DataFrame({symbol: asset_returns})] + 
                               [pd.DataFrame({f'peer_{i}': r}) for i, r in enumerate(peer_returns_list)],
                               axis=1)
        
        cumulative_returns = (1 + all_returns).cumprod()
        
        # Rank this asset vs peers (0-1, where 1 is best)
        def rank_percentile(row):
            if pd.isna(row[symbol]):
                return 0.5
            rank = sum(row[1:] > row[symbol]) / (len(row) - 1) if len(row) > 1 else 0.5
            return rank / len(row)
        
        df["sector_rank_percentile"] = cumulative_returns.apply(rank_percentile, axis=1)
        
        # 3. Correlation with sector average
        sector_avg_corr = asset_returns.rolling(self.window).corr(sector_avg_returns)
        df["correlation_with_sector"] = sector_avg_corr.fillna(0)
        
        # 4. Beta to sector (how much does asset move with sector)
        sector_returns_std = sector_avg_returns.std()
        sector_avg_var = sector_avg_returns.var()
        
        covariance = asset_returns.rolling(self.window).cov(sector_avg_returns)
        sector_beta = covariance / sector_avg_var if sector_avg_var > 0 else 1.0
        df["sector_beta"] = sector_beta.fillna(1.0)
        
        # 5. Idiosyncratic risk (residual after sector movement)
        sector_predicted_return = sector_beta * sector_avg_returns
        idiosyncratic_return = asset_returns - sector_predicted_return
        df["idiosyncratic_volatility"] = idiosyncratic_return.rolling(self.window).std()
        
        return df
