"""
Cross-Asset ML Model - Train on Multiple Stocks Simultaneously

Improves prediction accuracy by learning shared patterns across assets.
Enables transfer learning and intra-asset correlations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


@dataclass
class CrossAssetPrediction:
    """Prediction from cross-asset model"""
    symbol: str
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    contribution_by_asset: Dict[str, float]


class CrossAssetModel:
    """
    Train ML model on multiple assets to capture cross-asset relationships.
    
    Approach:
    1. Combine features from all assets
    2. Learn shared patterns (cross-correlations)
    3. Asset-specific embeddings (transfer learning)
    4. Unified predictions for all assets
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",  # xgboost, lightgbm, mlp
        lookback_period: int = 60,
        prediction_horizon: int = 1,
        include_correlation_features: bool = True,
    ):
        """
        Args:
            model_type: Type of model to use
            lookback_period: Lookback for features
            prediction_horizon: Prediction horizon
            include_correlation_features: Include cross-asset correlation features
        """
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.include_correlation_features = include_correlation_features
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.symbols = []
        self.asset_embeddings = {}
    
    def prepare_cross_asset_features(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        feature_columns: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features from multiple assets.
        
        Args:
            data: DataFrame with multi-asset data (symbol column required)
            symbols: List of symbols to include
            feature_columns: Base feature columns
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        features_list = []
        targets_list = []
        all_features = []
        
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].copy()
            
            if len(symbol_data) < self.lookback_period + self.prediction_horizon:
                continue
            
            # Create lagged features for this asset
            for col in feature_columns:
                if col in symbol_data.columns:
                    for lag in range(1, self.lookback_period + 1):
                        symbol_data[f'{col}_lag{lag}_{symbol}'] = symbol_data[col].shift(lag)
            
            # Add cross-asset correlation features
            if self.include_correlation_features:
                symbol_data = self._add_correlation_features(
                    symbol_data, data, symbol, symbols, feature_columns
                )
            
            # Remove NaN rows
            symbol_data = symbol_data.dropna()
            
            if len(symbol_data) == 0:
                continue
            
            # Target: next day return
            if 'Close' in symbol_data.columns:
                target = (symbol_data['Close'].shift(-self.prediction_horizon) / 
                          symbol_data['Close'] - 1.0).dropna()
            else:
                continue
            
            # Get feature columns
            feature_cols = [c for c in symbol_data.columns if 'lag' in c or 'corr' in c]
            
            X = symbol_data[feature_cols].iloc[:-self.prediction_horizon].values
            y = target.iloc[:-self.prediction_horizon].values
            
            features_list.append(X)
            targets_list.append(y)
            
            if not all_features:
                all_features = feature_cols
        
        # Combine all assets
        X_combined = np.vstack(features_list)
        y_combined = np.concatenate(targets_list)
        
        self.feature_columns = all_features
        self.symbols = symbols
        
        return X_combined, y_combined, all_features
    
    def _add_correlation_features(
        self,
        symbol_data: pd.DataFrame,
        full_data: pd.DataFrame,
        symbol: str,
        all_symbols: List[str],
        feature_columns: List[str],
    ) -> pd.DataFrame:
        """Add correlation features between assets."""
        for other_symbol in all_symbols:
            if other_symbol == symbol:
                continue
            
            other_data = full_data[full_data['symbol'] == other_symbol]
            
            for col in ['Close', 'Volume']:
                if col in other_data.columns and col in symbol_data.columns:
                    # Correlation with other asset
                    corr = symbol_data[col].rolling(10).corr(
                        other_data[col].rolling(10).mean()
                    )
                    symbol_data[f'{col}_corr_{other_symbol}'] = corr
        
        return symbol_data
    
    def train(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        feature_columns: List[str],
        test_size: float = 0.2,
    ):
        """
        Train cross-asset model.
        
        Args:
            data: Multi-asset DataFrame
            symbols: List of symbols
            feature_columns: Feature columns to use
            test_size: Test set fraction
        """
        # Prepare features
        X, y, _ = self.prepare_cross_asset_features(data, symbols, feature_columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, shuffle=False
        )
        
        # Train model
        if self.model_type == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        else:  # MLP
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                random_state=42,
            )
        
        self.model.fit(X_train, y_train)
        
        # Score on test set
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_r2': train_score,
            'test_r2': test_score,
            'test_predictions': self.model.predict(X_test),
            'test_targets': y_test,
        }
    
    def predict(
        self,
        data: pd.DataFrame,
        symbol: str,
    ) -> CrossAssetPrediction:
        """
        Make prediction for a symbol using cross-asset model.
        
        Args:
            data: Feature data
            symbol: Target symbol
            
        Returns:
            CrossAssetPrediction with confidence and feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        X = data[self.feature_columns].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        pred = self.model.predict(X_scaled)[0]
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
        else:
            importance = {}
        
        # Confidence based on prediction magnitude and sharpness
        confidence = min(1.0, abs(pred) / 0.05)  # Higher confidence for stronger signals
        
        return CrossAssetPrediction(
            symbol=symbol,
            prediction=pred,
            confidence=confidence,
            feature_importance=importance,
            contribution_by_asset={},  # TODO: Implement attribution
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        return dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
    
    def save(self, path: str):
        """Save model and scaler."""
        joblib.dump(self.model, f"{path}_model.pkl")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")
    
    def load(self, path: str):
        """Load model and scaler."""
        self.model = joblib.load(f"{path}_model.pkl")
        self.scaler = joblib.load(f"{path}_scaler.pkl")


class TransferLearningModel:
    """
    Transfer learning from pre-trained cross-asset model to new assets.
    
    Use case: Apply knowledge from trained stocks to new/smaller stocks.
    """
    
    def __init__(self, base_model: CrossAssetModel):
        """
        Args:
            base_model: Pre-trained cross-asset model
        """
        self.base_model = base_model
        self.adapter_models = {}  # Symbol -> adapter model
    
    def train_adapter(
        self,
        data: pd.DataFrame,
        symbol: str,
        n_samples: int = 100,
    ):
        """
        Train small adapter model for new symbol using features from base model.
        
        Args:
            data: Feature data for new symbol
            symbol: New symbol to adapt for
            n_samples: Number of samples for adaptation (can be small due to transfer learning)
        """
        # Get base model features
        X = data[self.base_model.feature_columns[-50:]].tail(n_samples).values  # Last 50 features
        
        if X.shape[0] == 0:
            return
        
        # Scale with base model scaler
        X_scaled = self.base_model.scaler.transform(X)
        
        # Train lightweight adapter
        from sklearn.linear_model import Ridge
        adapter = Ridge(alpha=1.0)
        
        # Generate synthetic targets from base model
        y_base = self.base_model.model.predict(X_scaled)
        
        # Adapt to actual data
        if 'Close' in data.columns:
            y_actual = (data['Close'].shift(-1) / data['Close'] - 1.0).tail(n_samples).values
            
            # Remove NaNs
            mask = ~np.isnan(y_actual)
            if np.sum(mask) > 5:
                adapter.fit(X_scaled[mask], y_actual[mask])
                self.adapter_models[symbol] = adapter
    
    def predict(self, data: pd.DataFrame, symbol: str) -> float:
        """Predict using base model + adapter."""
        if symbol not in self.adapter_models:
            # Use base model directly
            X = data[self.base_model.feature_columns[-50:]].values.reshape(1, -1)
            X_scaled = self.base_model.scaler.transform(X)
            return self.base_model.model.predict(X_scaled)[0]
        
        # Use adapter
        X = data[self.base_model.feature_columns[-50:]].values.reshape(1, -1)
        X_scaled = self.base_model.scaler.transform(X)
        return self.adapter_models[symbol].predict(X_scaled)[0]
