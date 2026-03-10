"""
Comprehensive Feature Pipeline - Phase 1

Orchestrates:
- Technical indicators (existing)
- Microstructure features (NEW)
- Macroeconomic features (NEW)
- Cross-asset features (NEW)
- Regime-conditional features (NEW)
- Non-linear features (NEW)
"""

import pandas as pd
from typing import Optional, Dict, List
import logging

from .indicators.trend_indicators import TrendIndicators
from .indicators.momentum_indicators import MomentumIndicators
from .indicators.volatility_indicators import VolatilityIndicators
from .indicators.volume_indicators import VolumeIndicators

from .price_action.support_resistance import SupportResistance
from .price_action.breakout_detector import BreakoutDetector
from .price_action.candlestick_patterns import CandlePatterns

from .regime_detection.trend_classifier import TrendClassifier
from .regime_detection.volatility_regime import VolatilityRegime
from .regime_detection.momentum_score import MomentumScore

# NEW: Phase 1 expanded features
from .microstructure_features import MicrostructureFeatures
from .macroeconomic_features import MacroeconomicFeatures
from .cross_asset_features import CrossAssetFeatures
from .regime_conditional_features import RegimeConditionalFeatures
from .nonlinear_features import NonlinearFeatures
from .feature_analyzer import FeatureAnalyzer

logger = logging.getLogger(__name__)


class ComprehensiveFeaturePipeline:
    """
    Full feature engineering pipeline with 100+ features.
    
    Phases:
    1. Technical indicators (30 features)
    2. Price action (15 features)
    3. Microstructure (25+ features)
    4. Macroeconomic (30+ features) - optional with market data
    5. Cross-asset (20+ features) - optional with universe data
    6. Regime-conditional (15 features)
    7. Non-linear (20+ features)
    """
    
    def __init__(self, include_macro: bool = True, include_cross_asset: bool = True):
        """
        Args:
            include_macro: Include macroeconomic features
            include_cross_asset: Include cross-asset features
        """
        self.include_macro = include_macro
        self.include_cross_asset = include_cross_asset
    
    def run(
        self,
        df: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        sector_data: Optional[Dict[str, pd.DataFrame]] = None,
        universe_data: Optional[Dict[str, pd.DataFrame]] = None,
        sector_peers: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Run complete feature pipeline.
        
        Args:
            df: Asset price dataframe
            market_data: Optional index/market data for macro features
            sector_data: Optional sector dataframes
            universe_data: Optional dict of all assets for cross-asset features
            sector_peers: Optional dict mapping symbols to peer lists
            
        Returns:
            DataFrame with all features
        """
        df = df.copy()
        
        logger.info("=" * 60)
        logger.info(f"Starting comprehensive feature pipeline: {df.shape[1]} input columns")
        logger.info("=" * 60)
        
        # Phase 1: Technical indicators
        logger.info("Phase 1: Computing technical indicators...")
        df = TrendIndicators().add(df)
        df = MomentumIndicators().add(df)
        df = VolatilityIndicators().add(df)
        df = VolumeIndicators().add(df)
        logger.info(f"  → After technical indicators: {df.shape[1]} columns")
        
        # Phase 2: Price action
        logger.info("Phase 2: Detecting price action patterns...")
        df = SupportResistance().add(df)
        df = BreakoutDetector().detect(df)
        df = CandlePatterns().detect(df)
        logger.info(f"  → After price action: {df.shape[1]} columns")
        
        # Phase 3: Regime detection
        logger.info("Phase 3: Classifying regimes...")
        df = TrendClassifier().classify(df)
        df = VolatilityRegime().classify(df)
        df = MomentumScore().compute(df)
        logger.info(f"  → After regime detection: {df.shape[1]} columns")
        
        # Phase 4: Microstructure (NEW)
        logger.info("Phase 4: Computing microstructure features...")
        df = MicrostructureFeatures().add(df)
        logger.info(f"  → After microstructure: {df.shape[1]} columns")
        
        # Phase 5: Macroeconomic (NEW - optional)
        if self.include_macro:
            logger.info("Phase 5: Computing macroeconomic features...")
            try:
                df = MacroeconomicFeatures().add(
                    df,
                    index_df=market_data,
                    sector_data=sector_data,
                )
                logger.info(f"  → After macroeconomic: {df.shape[1]} columns")
            except Exception as e:
                logger.warning(f"  → Macroeconomic features skipped: {e}")
        
        # Phase 6: Cross-asset (NEW - optional)
        if self.include_cross_asset:
            logger.info("Phase 6: Computing cross-asset features...")
            try:
                df = CrossAssetFeatures().add(
                    df,
                    universe_data=universe_data,
                    sector_peers=sector_peers,
                )
                logger.info(f"  → After cross-asset: {df.shape[1]} columns")
            except Exception as e:
                logger.warning(f"  → Cross-asset features skipped: {e}")
        
        # Phase 7: Regime-conditional (NEW)
        logger.info("Phase 7: Computing regime-conditional features...")
        df = RegimeConditionalFeatures().add(df)
        logger.info(f"  → After regime-conditional: {df.shape[1]} columns")
        
        # Phase 8: Non-linear (NEW)
        logger.info("Phase 8: Computing non-linear and interaction features...")
        df = NonlinearFeatures().add(df)
        logger.info(f"  → After non-linear: {df.shape[1]} columns")
        
        # Final cleanup
        logger.info("Phase 9: Final cleanup...")
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info("=" * 60)
        logger.info(f"Feature pipeline complete: {df.shape[1]} output features")
        logger.info(f"Features by type:")
        logger.info(f"  - Technical indicators")
        logger.info(f"  - Price action")
        logger.info(f"  - Microstructure (volume, liquidity, trading dynamics)")
        logger.info(f"  - Macroeconomic (vol regime, index relative, sector)")
        logger.info(f"  - Cross-asset (correlations, peer relative)")
        logger.info(f"  - Regime-conditional (adaptive, dynamic)")
        logger.info(f"  - Non-linear (interactions, polynomials, log)")
        logger.info("=" * 60)
        
        return df
    
    def select_best_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        n_features: int = 60,
    ) -> tuple[pd.DataFrame, list]:
        """
        Select top N features by importance.
        
        Args:
            df: Feature dataframe
            target: Optional target variable
            n_features: Number of features to keep
            
        Returns:
            (selected_df, selected_feature_names)
        """
        logger.info(f"Analyzing {df.shape[1]} features for selection...")
        
        analyzer = FeatureAnalyzer()
        
        selected = analyzer.select_features(
            df.select_dtypes(include=['number']),
            y=target,
            n_features=n_features,
        )
        
        logger.info(f"Selected {len(selected)} features out of {df.shape[1]}")
        
        return df[selected], selected
