"""
Regime Detection Module - Market condition identification and adaptive allocation

Modules:
- regime_detector: Volatility, trend, correlation regime detection
- adaptive_allocator: Regime-aware portfolio allocation and rebalancing

Usage:
    from regime_detection import RegimeDetectionEngine, RegimeAwareAllocator
    
    # Detect market regime
    engine = RegimeDetectionEngine()
    engine.fit(historical_returns)
    regime = engine.detect_regime(prices, returns)
    
    # Adjust allocations
    allocator = RegimeAwareAllocator(base_allocations={'SPY': 0.9, 'CASH': 0.1})
    adaptive_weights = allocator.allocate_by_regime(regime, asset_vols)
"""

from .regime_detector import (
    VolatilityRegime,
    TrendRegime,
    CorrelationRegime,
    RegimeState,
    VolatilityRegimeDetector,
    TrendRegimeDetector,
    CorrelationRegimeDetector,
    RegimeDetectionEngine,
)

from .adaptive_allocator import (
    AllocationWeights,
    AllocationSignal,
    RegimeAwareAllocator,
    OpportunityDetector,
    RiskAdjustedPositionSizer,
)

__all__ = [
    'VolatilityRegime',
    'TrendRegime',
    'CorrelationRegime',
    'RegimeState',
    'VolatilityRegimeDetector',
    'TrendRegimeDetector',
    'CorrelationRegimeDetector',
    'RegimeDetectionEngine',
    'AllocationWeights',
    'AllocationSignal',
    'RegimeAwareAllocator',
    'OpportunityDetector',
    'RiskAdjustedPositionSizer',
]

__version__ = '1.0.0'
