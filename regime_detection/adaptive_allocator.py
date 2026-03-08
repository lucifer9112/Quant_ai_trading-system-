"""
Adaptive Portfolio Allocation - Adjust allocations based on market regimes

Implements:
- Regime-aware position sizing
- Dynamic rebalancing triggers
- Risk adjustment by regime
- Opportunity identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .regime_detector import RegimeState, VolatilityRegime, TrendRegime, CorrelationRegime


@dataclass
class AllocationWeights:
    """Portfolio allocation weights."""
    symbol: str
    base_weight: float  # Base tactical allocation
    regime_weight: float  # Regime-adjusted weight
    volatility_multiplier: float  # Volatility adjustment factor
    final_weight: float  # Final allocation weight


@dataclass
class AllocationSignal:
    """Signal for portfolio rebalancing."""
    rebalance_trigger: bool
    urgency: float  # 0-1, urgency of rebalancing
    recommendation: str
    expected_impact: float


class RegimeAwareAllocator:
    """Adjust portfolio allocation based on market regime."""
    
    def __init__(
        self,
        base_allocations: Dict[str, float],
        risk_budget: float = 0.15,  # Target portfolio volatility
    ):
        """
        Args:
            base_allocations: Base allocation weights by symbol
            risk_budget: Target portfolio volatility
        """
        self.base_allocations = base_allocations
        self.risk_budget = risk_budget
        self.allocation_history = []
    
    def allocate_by_regime(
        self,
        regime: RegimeState,
        asset_volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate portfolio allocation based on regime.
        
        Args:
            regime: Current RegimeState
            asset_volatilities: Current volatility by asset
            
        Returns:
            Dict of symbol -> allocation weight
        """
        allocations = self.base_allocations.copy()
        
        # Volatility regime adjustment
        if regime.volatility_regime == VolatilityRegime.LOW:
            # Low vol: Can take more risk, increase equity allocation
            equity_multiplier = 1.15
            cash_reduction = 0.85
        elif regime.volatility_regime == VolatilityRegime.HIGH:
            # High vol: Reduce risk, increase cash
            equity_multiplier = 0.80
            cash_reduction = 1.30
        else:
            # Medium vol: Keep base allocations
            equity_multiplier = 1.0
            cash_reduction = 1.0
        
        # Trend regime adjustment
        if regime.trend_regime == TrendRegime.DOWNTREND:
            # Downtrend: Reduce equity beta
            equity_multiplier *= 0.85
            # Increase defensive positions
            if 'SPY' in allocations:
                allocations['SPY'] *= 0.75
            if 'CASH' in allocations:
                allocations['CASH'] *= cash_reduction
        elif regime.trend_regime == TrendRegime.UPTREND:
            # Uptrend: Can increase equity exposure
            equity_multiplier *= 1.1
        
        # Correlation regime adjustment
        if regime.correlation_regime == CorrelationRegime.HIGH_CORRELATION:
            # High correlation: Concentrate in highest quality
            # Reduce diversification benefit from smaller positions
            quality_boost = 1.1
            junk_reduction = 0.85
            
            if 'SPY' in allocations:
                allocations['SPY'] *= quality_boost
            # Would reduce lower-quality holdings
        
        # Volatility-based sizing
        target_volatility = self.risk_budget
        current_portfolio_vol = self._estimate_portfolio_volatility(
            allocations, asset_volatilities
        )
        
        if current_portfolio_vol > 1e-6:
            vol_adjustment = target_volatility / current_portfolio_vol
        else:
            vol_adjustment = 1.0
        
        # Apply adjustments
        adjusted = {}
        for symbol, weight in allocations.items():
            adjusted[symbol] = weight * equity_multiplier * vol_adjustment
        
        # Normalize to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def _estimate_portfolio_volatility(
        self,
        weights: Dict[str, float],
        asset_vols: Dict[str, float],
    ) -> float:
        """
        Estimate portfolio volatility (simplified, ignoring correlations).
        
        Args:
            weights: Portfolio weights
            asset_vols: Asset volatilities
            
        Returns:
            Estimated portfolio volatility
        """
        vol_squared = 0
        for symbol, weight in weights.items():
            if symbol in asset_vols:
                vol_squared += (weight * asset_vols[symbol]) ** 2
        
        return np.sqrt(vol_squared)
    
    def get_rebalancing_signal(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.05,
    ) -> AllocationSignal:
        """
        Determine if rebalancing is needed.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights from regime allocation
            threshold: Deviation threshold for rebalancing
            
        Returns:
            AllocationSignal with rebalancing recommendation
        """
        max_deviation = 0
        max_symbol = None
        
        for symbol in target_weights:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            deviation = abs(current - target)
            
            if deviation > max_deviation:
                max_deviation = deviation
                max_symbol = symbol
        
        # Determine if rebalance needed
        rebalance_trigger = max_deviation > threshold
        urgency = min(max_deviation / threshold, 1.0) if threshold > 0 else 0.5
        
        if rebalance_trigger:
            recommendation = f"Rebalance {max_symbol}: {max_deviation:.2%} deviation"
            expected_impact = max_deviation * 0.5  # Rough estimate
        else:
            recommendation = "No rebalancing needed"
            expected_impact = 0.0
        
        return AllocationSignal(
            rebalance_trigger=rebalance_trigger,
            urgency=urgency,
            recommendation=recommendation,
            expected_impact=expected_impact,
        )


class OpportunityDetector:
    """Identify trading opportunities based on regime changes."""
    
    def __init__(self, lookback: int = 60):
        """
        Args:
            lookback: Historical periods to analyze
        """
        self.lookback = lookback
        self.regime_transitions = []
    
    def detect_opportunities(
        self,
        current_regime: RegimeState,
        previous_regime: Optional[RegimeState],
    ) -> List[Dict]:
        """
        Detect opportunities from regime transitions.
        
        Args:
            current_regime: Current market regime
            previous_regime: Previous market regime
            
        Returns:
            List of identified opportunities
        """
        opportunities = []
        
        if previous_regime is None:
            return opportunities
        
        # Volatility expansion opportunity
        if (previous_regime.volatility_regime == VolatilityRegime.LOW and
            current_regime.volatility_regime == VolatilityRegime.HIGH):
            opportunities.append({
                'type': 'volatility_expansion',
                'description': 'Volatility spike - opportunity to sell vol or reduce beta',
                'strength': current_regime.risk_score,
                'action': 'REDUCE_EQUITY_INCREASE_CASH',
            })
        
        # Volatility compression opportunity
        if (previous_regime.volatility_regime == VolatilityRegime.HIGH and
            current_regime.volatility_regime == VolatilityRegime.LOW):
            opportunities.append({
                'type': 'volatility_compression',
                'description': 'Volatility low - opportunity to increase equity',
                'strength': 1 - current_regime.risk_score,
                'action': 'INCREASE_EQUITY_REDUCE_CASH',
            })
        
        # Trend reversal opportunity
        if previous_regime.trend_regime != current_regime.trend_regime:
            opportunities.append({
                'type': 'trend_reversal',
                'description': f'Trend change: {previous_regime.trend_regime} -> {current_regime.trend_regime}',
                'strength': current_regime.trend_strength,
                'action': 'REPOSITION_FOR_NEW_TREND',
            })
        
        # Correlation dislocations
        if (previous_regime.correlation_regime == CorrelationRegime.LOW_CORRELATION and
            current_regime.correlation_regime == CorrelationRegime.HIGH_CORRELATION):
            opportunities.append({
                'type': 'correlation_increase',
                'description': 'Correlation increased - diversification diminished',
                'strength': current_regime.correlation_value,
                'action': 'CONCENTRATE_QUALITY_REDUCE_DIVERSIFICATION',
            })
        
        if (previous_regime.correlation_regime == CorrelationRegime.HIGH_CORRELATION and
            current_regime.correlation_regime == CorrelationRegime.LOW_CORRELATION):
            opportunities.append({
                'type': 'correlation_decrease',
                'description': 'Correlation decreased - diversification improved',
                'strength': 1 - current_regime.correlation_value,
                'action': 'INCREASE_DIVERSIFICATION',
            })
        
        return opportunities


class RiskAdjustedPositionSizer:
    """Size positions based on regime and risk tolerance."""
    
    def __init__(self, base_position_size: float = 0.02):
        """
        Args:
            base_position_size: Base position size as % of portfolio
        """
        self.base_position_size = base_position_size
    
    def calculate_position_size(
        self,
        regime: RegimeState,
        signal_strength: float,  # 0-1, confidence in signal
        asset_volatility: float,
        portfolio_value: float,
    ) -> float:
        """
        Calculate position size adjusting for regime.
        
        Args:
            regime: Current RegimeState
            signal_strength: Signal confidence (0-1)
            asset_volatility: Asset volatility
            portfolio_value: Total portfolio value
            
        Returns:
            Position size in units/shares
        """
        # Base position size
        position_size = self.base_position_size
        
        # Regime adjustment
        risk_adjustment = 1.0 - (regime.risk_score * 0.5)  # Reduce in high risk
        position_size *= risk_adjustment
        
        # Signal strength adjustment
        position_size *= signal_strength
        
        # Volatility adjustment (smaller in high vol)
        vol_adjustment = 1.0 / (1.0 + asset_volatility)
        position_size *= vol_adjustment
        
        # Trend adjustment (larger in uptrend)
        if regime.trend_regime == TrendRegime.UPTREND:
            position_size *= 1.1
        elif regime.trend_regime == TrendRegime.DOWNTREND:
            position_size *= 0.8
        
        return position_size * portfolio_value
    
    def get_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        regime: RegimeState,
    ) -> float:
        """
        Calculate Kelly Criterion fraction adjusted for regime.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount
            regime: Current RegimeState
            
        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss == 0:
            return 0.0
        
        # Base Kelly Criterion
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly = (b * p - q) / b if b > 0 else 0
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%
        
        # Regime adjustment (reduce Kelly in uncertain regimes)
        if regime.volatility_regime == VolatilityRegime.HIGH:
            kelly *= 0.7
        
        if regime.trend_regime == TrendRegime.SIDEWAYS:
            kelly *= 0.8
        
        return kelly
