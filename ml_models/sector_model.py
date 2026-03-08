"""
Sector-Level ML Models - Train Models Per Sector

Captures sector-specific dynamics and inter-stock relationships within sectors.
Improves predictions through sector context.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class SectorPrediction:
    """Prediction at sector level"""
    sector: str
    symbol: str
    prediction: float
    confidence: float
    sector_strength: float  # How strong the sector signal is
    relative_strength: float  # Stock strength relative to sector


class SectorModel:
    """
    Train separate models for each sector.
    
    Benefits:
    - Sector-specific patterns (e.g., healthcare vs tech)
    - Intra-sector correlations
    - Reduced noise from unrelated sectors
    """
    
    def __init__(
        self,
        sector_universe: Dict[str, List[str]],  # sector -> list of symbols
        model_type: str = "xgboost",
        lookback_period: int = 60,
    ):
        """
        Args:
            sector_universe: Mapping of sector to members
            model_type: Type of ML model
            lookback_period: Lookback period for features
        """
        self.sector_universe = sector_universe
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.sector_models = {}  # sector -> model
        self.sector_scalers = {}  # sector -> scaler
        self.sector_features = {}  # sector -> feature columns
    
    def train_sector_model(
        self,
        data: pd.DataFrame,
        sector: str,
        symbols: List[str],
        feature_columns: List[str],
    ) -> Dict:
        """
        Train model for a specific sector.
        
        Args:
            data: Multi-asset DataFrame
            sector: Sector name
            symbols: Symbols in sector
            feature_columns: Feature columns to use
            
        Returns:
            Training metrics
        """
        # Filter data for sector symbols
        sector_data = data[data['symbol'].isin(symbols)].copy()
        
        if len(sector_data) < 100:
            return {'status': 'insufficient_data'}
        
        # Prepare features
        X_list = []
        y_list = []
        
        for symbol in symbols:
            symbol_data = sector_data[sector_data['symbol'] == symbol].sort_values('Date')
            
            if len(symbol_data) < self.lookback_period + 1:
                continue
            
            # Create lagged features
            for col in feature_columns:
                if col in symbol_data.columns:
                    for lag in range(1, self.lookback_period + 1):
                        symbol_data[f'{col}_lag{lag}'] = symbol_data[col].shift(lag)
            
            # Add sector features (intra-sector relationships)
            symbol_data = self._add_intra_sector_features(
                symbol_data, sector_data, sector, symbols, feature_columns
            )
            
            # Target: Returns
            symbol_data['target'] = (symbol_data['Close'].shift(-1) / symbol_data['Close'] - 1.0)
            
            # Get features
            feature_cols = [c for c in symbol_data.columns if 'lag' in c or 'sector_' in c]
            
            symbol_data = symbol_data.dropna()
            
            if len(symbol_data) > 0:
                X_list.append(symbol_data[feature_cols].values)
                y_list.append(symbol_data['target'].values)
        
        if not X_list:
            return {'status': 'no_valid_data'}
        
        # Combine data
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        if self.model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        model.fit(X_scaled, y)
        
        # Store
        self.sector_models[sector] = model
        self.sector_scalers[sector] = scaler
        self.sector_features[sector] = feature_cols
        
        # Metrics
        train_score = model.score(X_scaled, y)
        
        return {
            'status': 'success',
            'sector': sector,
            'n_samples': len(y),
            'r2_score': train_score,
            'symbols': symbols,
        }
    
    def _add_intra_sector_features(
        self,
        symbol_data: pd.DataFrame,
        sector_data: pd.DataFrame,
        sector: str,
        symbols: List[str],
        feature_columns: List[str],
    ) -> pd.DataFrame:
        """Add intra-sector features."""
        symbol = symbol_data['symbol'].iloc[0]
        
        # Sector momentum
        sector_returns = []
        for sym in symbols:
            if sym == symbol:
                continue
            other = sector_data[sector_data['symbol'] == sym]
            if len(other) > 0:
                ret = (other['Close'].iloc[-1] / other['Close'].iloc[0] - 1.0)
                sector_returns.append(ret)
        
        if sector_returns:
            symbol_data[f'sector_momentum_{sector}'] = np.mean(sector_returns)
            symbol_data[f'sector_std_{sector}'] = np.std(sector_returns)
        
        # Relative strength
        if 'Close' in symbol_data.columns and len(sector_returns) > 0:
            symbol_returns = (symbol_data['Close'].iloc[-1] / symbol_data['Close'].iloc[0] - 1.0)
            symbol_data[f'relative_strength_{sector}'] = symbol_returns - np.mean(sector_returns)
        
        return symbol_data
    
    def predict(
        self,
        data: pd.DataFrame,
        symbol: str,
        sector: str,
    ) -> Optional[SectorPrediction]:
        """
        Make prediction using sector model.
        
        Args:
            data: Feature data for symbol
            symbol: Target symbol
            sector: Sector of symbol
            
        Returns:
            SectorPrediction or None if sector not trained
        """
        if sector not in self.sector_models:
            return None
        
        model = self.sector_models[sector]
        scaler = self.sector_scalers[sector]
        features = self.sector_features[sector]
        
        # Get features
        X = data[features].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        
        # Confidence
        if hasattr(model, 'feature_importances_'):
            top_feature_importance = np.max(model.feature_importances_)
            confidence = min(1.0, top_feature_importance * 10)
        else:
            confidence = 0.5
        
        # Sector strength (based on sector momentum feature)
        sector_strength = 0.0
        if f'sector_momentum_{sector}' in data.columns:
            sector_strength = data[f'sector_momentum_{sector}'].values[0] if len(data) > 0 else 0.0
        
        # Relative strength
        relative_strength = 0.0
        if f'relative_strength_{sector}' in data.columns:
            relative_strength = data[f'relative_strength_{sector}'].values[0] if len(data) > 0 else 0.0
        
        return SectorPrediction(
            sector=sector,
            symbol=symbol,
            prediction=pred,
            confidence=confidence,
            sector_strength=sector_strength,
            relative_strength=relative_strength,
        )
    
    def train_all_sectors(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
    ) -> Dict[str, Dict]:
        """Train models for all sectors."""
        results = {}
        
        for sector, symbols in self.sector_universe.items():
            result = self.train_sector_model(
                data, sector, symbols, feature_columns
            )
            results[sector] = result
        
        return results
    
    def get_sector_importance(self, sector: str) -> Dict[str, float]:
        """Get feature importance for sector."""
        if sector not in self.sector_models:
            return {}
        
        model = self.sector_models[sector]
        features = self.sector_features[sector]
        
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        return dict(zip(features, model.feature_importances_))


class CrossSectorAnalysis:
    """Analyze relationships between sectors."""
    
    def __init__(self, sector_universe: Dict[str, List[str]]):
        self.sector_universe = sector_universe
        self.sector_correlations = {}
        self.sector_betas = {}
    
    def calculate_sector_correlations(self, data: pd.DataFrame):
        """Calculate correlations between sectors."""
        sector_returns = {}
        
        for sector, symbols in self.sector_universe.items():
            sector_data = data[data['symbol'].isin(symbols)]
            returns = sector_data.groupby('Date')['Close'].mean().pct_change()
            sector_returns[sector] = returns
        
        # Correlation matrix
        returns_df = pd.DataFrame(sector_returns)
        self.sector_correlations = returns_df.corr().to_dict()
        
        return self.sector_correlations
    
    def calculate_sector_betas(
        self,
        data: pd.DataFrame,
        market_returns: pd.Series,
    ):
        """Calculate beta of each sector relative to market."""
        for sector, symbols in self.sector_universe.items():
            sector_data = data[data['symbol'].isin(symbols)]
            sector_rets = sector_data.groupby('Date')['Close'].mean().pct_change()
            
            # Align with market returns
            aligned = pd.DataFrame({
                'sector': sector_rets,
                'market': market_returns
            }).dropna()
            
            if len(aligned) > 20:
                beta = aligned.cov().iloc[0, 1] / aligned['market'].var()
                self.sector_betas[sector] = beta
        
        return self.sector_betas
    
    def find_sector_divergence(self) -> Dict[str, float]:
        """Identify sectors diverging from market trend."""
        if not self.sector_betas:
            return {}
        
        avg_beta = np.mean(list(self.sector_betas.values()))
        divergence = {
            sector: beta - avg_beta
            for sector, beta in self.sector_betas.items()
        }
        
        return divergence
