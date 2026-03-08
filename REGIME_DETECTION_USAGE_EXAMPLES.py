"""
Phase 4: Regime Detection & Adaptive Allocation - Usage Examples

Demonstrates:
- Volatility regime detection
- Trend regime detection
- Correlation regime detection
- Regime-aware allocations
- Opportunity identification
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from regime_detection import (
    RegimeDetectionEngine,
    RegimeAwareAllocator,
    OpportunityDetector,
    RiskAdjustedPositionSizer,
    VolatilityRegime,
    TrendRegime,
)


# ============================================================================
# Example 1: Volatility Regime Detection
# ============================================================================

def example_volatility_regimes():
    """Demonstrate volatility regime detection."""
    print("=" * 60)
    print("EXAMPLE 1: Volatility Regime Detection")
    print("=" * 60)
    
    # Generate synthetic returns with varying volatility
    n_periods = 500
    returns = []
    
    # Period 1: Low volatility
    returns.extend(np.random.normal(0.0005, 0.008, 100))
    
    # Period 2: Normal volatility
    returns.extend(np.random.normal(0.0005, 0.015, 150))
    
    # Period 3: High volatility (market stress)
    returns.extend(np.random.normal(-0.001, 0.025, 100))
    
    # Period 4: Normal volatility recovering
    returns.extend(np.random.normal(0.0005, 0.015, 150))
    
    returns = np.array(returns)
    
    # Fit regime detector
    engine = RegimeDetectionEngine()
    engine.fit(returns)
    
    print("\n1. Fitting volatility regime detector...")
    print(f"   Trained on {len(returns)} periods")
    
    # Get predictions
    vol_detector = engine.vol_detector
    regimes = vol_detector.predict(returns)
    
    print("\n2. Regime Classification Results:")
    print(f"   Low Volatility: {sum(1 for r in regimes if r == VolatilityRegime.LOW)}")
    print(f"   Medium Volatility: {sum(1 for r in regimes if r == VolatilityRegime.MEDIUM)}")
    print(f"   High Volatility: {sum(1 for r in regimes if r == VolatilityRegime.HIGH)}")
    
    # Current volatility
    current_vol = vol_detector.get_volatility_value(returns)
    print(f"\n3. Current Annualized Volatility: {current_vol:.2%}")
    
    # Analyze regime transitions
    transitions = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            transitions.append((i, regimes[i-1], regimes[i]))
    
    print(f"\n4. Regime Transitions in Period: {len(transitions)}")
    if len(transitions) > 0:
        for idx, from_regime, to_regime in transitions[:5]:
            print(f"   Period {idx}: {from_regime.name} -> {to_regime.name}")
    
    return returns, regimes


# ============================================================================
# Example 2: Trend Regime Detection
# ============================================================================

def example_trend_regimes():
    """Demonstrate trend regime detection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Trend Regime Detection")
    print("=" * 60)
    
    # Generate synthetic price with clear trends
    n_periods = 300
    prices = [100]
    
    # Uptrend phase 1
    for _ in range(75):
        prices.append(prices[-1] + np.random.normal(0.5, 1.5))
    
    # Sideways phase
    for _ in range(75):
        prices.append(prices[-1] + np.random.normal(0, 1.5))
    
    # Downtrend phase
    for _ in range(75):
        prices.append(prices[-1] + np.random.normal(-0.5, 1.5))
    
    # Recovery phase
    for _ in range(75):
        prices.append(prices[-1] + np.random.normal(0.3, 1.5))
    
    prices = np.array(prices)
    
    # Detect trends
    trend_detector = RegimeDetectionEngine().trend_detector
    
    print("\n1. Analyzing price trends (75-period phases):")
    
    # Analyze each phase
    phases = [
        ('Uptrend Phase', prices[0:75]),
        ('Sideways Phase', prices[75:150]),
        ('Downtrend Phase', prices[150:225]),
        ('Recovery Phase', prices[225:300]),
    ]
    
    for phase_name, phase_prices in phases:
        trend = trend_detector.detect_trend(phase_prices)
        strength = trend_detector.get_trend_strength(phase_prices)
        print(f"\n   {phase_name}:")
        print(f"      Trend: {trend.name}")
        print(f"      Strength: {strength:.2%}")
    
    # Get multi-timeframe trends
    print("\n2. Multi-Timeframe Trend Analysis (current):")
    trends_multi = trend_detector.detect_multiple_trends(prices, [20, 50, 200])
    for period, trend in trends_multi.items():
        print(f"   {period}-period: {trend.name}")
    
    return prices


# ============================================================================
# Example 3: Correlation Regime Detection
# ============================================================================

def example_correlation_regimes():
    """Demonstrate correlation regime detection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Correlation Regime Detection")
    print("=" * 60)
    
    # Create synthetic multi-asset data
    n_periods = 200
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='D')
    
    # Create low correlation period
    asset1_low_corr = 100 + np.cumsum(np.random.normal(0.0005, 0.012, 100))
    asset2_low_corr = 100 + np.cumsum(np.random.normal(0.0008, 0.015, 100))
    asset3_low_corr = 100 + np.cumsum(np.random.normal(0.0003, 0.010, 100))
    
    # Create high correlation period (risk-off event)
    base_movement = np.cumsum(np.random.normal(-0.001, 0.015, 100))
    asset1_high_corr = 120 + base_movement + np.random.normal(0, 0.005, 100)
    asset2_high_corr = 115 + base_movement + np.random.normal(0, 0.005, 100)
    asset3_high_corr = 110 + base_movement + np.random.normal(0, 0.005, 100)
    
    # Combine
    prices_data = pd.DataFrame({
        'Asset1': np.concatenate([asset1_low_corr, asset1_high_corr]),
        'Asset2': np.concatenate([asset2_low_corr, asset2_high_corr]),
        'Asset3': np.concatenate([asset3_low_corr, asset3_high_corr]),
    }, index=dates)
    
    # Detect correlation regimes
    corr_detector = RegimeDetectionEngine().corr_detector
    
    print("\n1. Correlation Analysis:")
    
    # Low correlation period
    corr_low = corr_detector.get_correlation_value(prices_data.iloc[:100])
    regime_low = corr_detector.detect_regime(prices_data.iloc[:100])
    print(f"\n   Period 1 (Diversified Environment):")
    print(f"      Avg Correlation: {corr_low:.3f}")
    print(f"      Regime: {regime_low.name}")
    
    # High correlation period
    corr_high = corr_detector.get_correlation_value(prices_data.iloc[100:])
    regime_high = corr_detector.detect_regime(prices_data.iloc[100:])
    print(f"\n   Period 2 (Risk-Off Event):")
    print(f"      Avg Correlation: {corr_high:.3f}")
    print(f"      Regime: {regime_high.name}")
    
    print(f"\n2. Correlation Matrix (Low Correlation Period):")
    corr_matrix = corr_detector.get_correlation_matrix(prices_data.iloc[:100])
    print(corr_matrix.round(3))
    
    return prices_data


# ============================================================================
# Example 4: Regime-Aware Portfolio Allocation
# ============================================================================

def example_regime_aware_allocation():
    """Demonstrate regime-aware portfolio allocation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Regime-Aware Portfolio Allocation")
    print("=" * 60)
    
    # Base allocations
    base_allocations = {
        'SPY': 0.60,    # Large-cap equities
        'QQQ': 0.20,    # Tech/growth
        'AGG': 0.15,    # Bonds
        'CASH': 0.05,   # Cash
    }
    
    allocator = RegimeAwareAllocator(base_allocations, risk_budget=0.12)
    
    print("\n1. Base Allocations:")
    for symbol, weight in base_allocations.items():
        print(f"   {symbol}: {weight:.1%}")
    
    # Simulate different regimes
    print("\n2. Adaptive Allocations Under Different Regimes:")
    
    # Create test regimes
    from regime_detection import RegimeState, VolatilityRegime, TrendRegime, CorrelationRegime
    
    # Regime 1: Low vol, uptrend (bull market)
    regime_bull = RegimeState(
        timestamp=0,
        volatility_regime=VolatilityRegime.LOW,
        trend_regime=TrendRegime.UPTREND,
        correlation_regime=CorrelationRegime.LOW_CORRELATION,
        volatility_value=0.08,
        trend_strength=0.8,
        correlation_value=0.3,
        risk_score=0.2,
    )
    
    # Regime 2: High vol, downtrend (bear market)
    regime_bear = RegimeState(
        timestamp=100,
        volatility_regime=VolatilityRegime.HIGH,
        trend_regime=TrendRegime.DOWNTREND,
        correlation_regime=CorrelationRegime.HIGH_CORRELATION,
        volatility_value=0.25,
        trend_strength=0.7,
        correlation_value=0.75,
        risk_score=0.85,
    )
    
    # Regime 3: Medium vol, sideways
    regime_neutral = RegimeState(
        timestamp=200,
        volatility_regime=VolatilityRegime.MEDIUM,
        trend_regime=TrendRegime.SIDEWAYS,
        correlation_regime=CorrelationRegime.LOW_CORRELATION,
        volatility_value=0.15,
        trend_strength=0.2,
        correlation_value=0.4,
        risk_score=0.5,
    )
    
    # Asset volatilities
    asset_vols = {'SPY': 0.15, 'QQQ': 0.22, 'AGG': 0.04, 'CASH': 0.0}
    
    regimes = [
        ('BULL MARKET (Low Vol, Uptrend)', regime_bull),
        ('BEAR MARKET (High Vol, Downtrend)', regime_bear),
        ('NEUTRAL (Medium Vol, Sideways)', regime_neutral),
    ]
    
    for regime_name, regime in regimes:
        weights = allocator.allocate_by_regime(regime, asset_vols)
        print(f"\n   {regime_name}:")
        print(f"      Risk Score: {regime.risk_score:.2f}")
        for symbol, weight in weights.items():
            base = base_allocations.get(symbol, 0)
            change = weight - base
            print(f"      {symbol}: {weight:.1%} (Δ {change:+.1%})")
    
    return base_allocations, allocator


# ============================================================================
# Example 5: Opportunity Detection
# ============================================================================

def example_opportunity_detection():
    """Demonstrate opportunity detection from regime changes."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Opportunity Detection from Regime Transitions")
    print("=" * 60)
    
    from regime_detection import RegimeState, VolatilityRegime, TrendRegime, CorrelationRegime
    
    detector = OpportunityDetector()
    
    # Transition 1: Volatility spike (opportunity to reduce risk)
    print("\n1. Volatility Expansion Opportunity:")
    regime_before = RegimeState(
        timestamp=0,
        volatility_regime=VolatilityRegime.LOW,
        trend_regime=TrendRegime.UPTREND,
        correlation_regime=CorrelationRegime.LOW_CORRELATION,
        volatility_value=0.08,
        trend_strength=0.7,
        correlation_value=0.3,
        risk_score=0.2,
    )
    
    regime_after = RegimeState(
        timestamp=1,
        volatility_regime=VolatilityRegime.HIGH,
        trend_regime=TrendRegime.UPTREND,
        correlation_regime=CorrelationRegime.LOW_CORRELATION,
        volatility_value=0.25,
        trend_strength=0.7,
        correlation_value=0.3,
        risk_score=0.75,
    )
    
    opportunities = detector.detect_opportunities(regime_after, regime_before)
    for opp in opportunities:
        print(f"   Type: {opp['type']}")
        print(f"   Description: {opp['description']}")
        print(f"   Strength: {opp['strength']:.2%}")
        print(f"   Action: {opp['action']}")
    
    # Transition 2: Trend reversal
    print("\n2. Trend Reversal Opportunity:")
    regime_uptrend = RegimeState(
        timestamp=50,
        volatility_regime=VolatilityRegime.MEDIUM,
        trend_regime=TrendRegime.UPTREND,
        correlation_regime=CorrelationRegime.LOW_CORRELATION,
        volatility_value=0.15,
        trend_strength=0.8,
        correlation_value=0.4,
        risk_score=0.3,
    )
    
    regime_downtrend = RegimeState(
        timestamp=51,
        volatility_regime=VolatilityRegime.MEDIUM,
        trend_regime=TrendRegime.DOWNTREND,
        correlation_regime=CorrelationRegime.LOW_CORRELATION,
        volatility_value=0.15,
        trend_strength=0.7,
        correlation_value=0.4,
        risk_score=0.55,
    )
    
    opportunities = detector.detect_opportunities(regime_downtrend, regime_uptrend)
    for opp in opportunities:
        print(f"   Type: {opp['type']}")
        print(f"   Description: {opp['description']}")
        print(f"   Strength: {opp['strength']:.2%}")
        print(f"   Action: {opp['action']}")


# ============================================================================
# Example 6: Risk-Adjusted Position Sizing
# ============================================================================

def example_position_sizing():
    """Demonstrate risk-adjusted position sizing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Risk-Adjusted Position Sizing")
    print("=" * 60)
    
    from regime_detection import RegimeState, VolatilityRegime, TrendRegime, CorrelationRegime
    
    sizer = RiskAdjustedPositionSizer(base_position_size=0.02)
    
    # Portfolio and trade parameters
    portfolio_value = 100000
    asset_volatility = 0.18
    signal_strength = 0.85
    
    print("\n1. Base Parameters:")
    print(f"   Portfolio Value: ${portfolio_value:,.0f}")
    print(f"   Asset Volatility: {asset_volatility:.1%}")
    print(f"   Signal Strength: {signal_strength:.0%}")
    
    # Regime 1: Bull market
    print("\n2. Position Sizes by Regime:")
    regime_bull = RegimeState(
        timestamp=0,
        volatility_regime=VolatilityRegime.LOW,
        trend_regime=TrendRegime.UPTREND,
        correlation_regime=CorrelationRegime.LOW_CORRELATION,
        volatility_value=0.10,
        trend_strength=0.8,
        correlation_value=0.3,
        risk_score=0.2,
    )
    
    size_bull = sizer.calculate_position_size(
        regime_bull, signal_strength, asset_volatility, portfolio_value
    )
    print(f"\n   BULL MARKET:")
    print(f"      Position Size: ${size_bull:,.0f}")
    print(f"      % of Portfolio: {size_bull/portfolio_value:.2%}")
    
    # Regime 2: Bear market
    regime_bear = RegimeState(
        timestamp=100,
        volatility_regime=VolatilityRegime.HIGH,
        trend_regime=TrendRegime.DOWNTREND,
        correlation_regime=CorrelationRegime.HIGH_CORRELATION,
        volatility_value=0.30,
        trend_strength=0.7,
        correlation_value=0.75,
        risk_score=0.85,
    )
    
    size_bear = sizer.calculate_position_size(
        regime_bear, signal_strength, asset_volatility, portfolio_value
    )
    print(f"\n   BEAR MARKET:")
    print(f"      Position Size: ${size_bear:,.0f}")
    print(f"      % of Portfolio: {size_bear/portfolio_value:.2%}")
    
    # Kelly Criterion adjustment
    print("\n3. Kelly Criterion by Regime:")
    win_rate = 0.55
    avg_win = 1000
    avg_loss = 800
    
    kelly_bull = sizer.get_kelly_fraction(win_rate, avg_win, avg_loss, regime_bull)
    kelly_bear = sizer.get_kelly_fraction(win_rate, avg_win, avg_loss, regime_bear)
    
    print(f"\n   Bull Market Kelly Fraction: {kelly_bull:.2%}")
    print(f"   Bear Market Kelly Fraction: {kelly_bear:.2%}")
    print(f"   Reduction in Bear Market: {(kelly_bull - kelly_bear)/kelly_bull:.1%}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("PHASE 4: REGIME DETECTION & ADAPTIVE ALLOCATION")
    print("=" * 60)
    
    # Run all examples
    example_volatility_regimes()
    example_trend_regimes()
    example_correlation_regimes()
    example_regime_aware_allocation()
    example_opportunity_detection()
    example_position_sizing()
    
    print("\n")
    print("=" * 60)
    print("✓ PHASE 4 EXAMPLES COMPLETED!")
    print("=" * 60)
    print("\nKey Features:")
    print("  ✓ Volatility regime detection (3-state: Low/Medium/High)")
    print("  ✓ Trend regime detection (3-state: Up/Down/Sideways)")
    print("  ✓ Correlation regime detection (2-state: Low/High)")
    print("  ✓ Regime-aware portfolio rebalancing")
    print("  ✓ Opportunity detection from regime transitions")
    print("  ✓ Risk-adjusted position sizing by regime")
    print("  ✓ Kelly Criterion adjusted for regime risk")
    print("\nNext Steps:")
    print("  1. Integrate regime detection into main pipeline")
    print("  2. Phase 5: Enhanced Professional Backtester")
    print("  3. Add execution models and realistic order handling")
