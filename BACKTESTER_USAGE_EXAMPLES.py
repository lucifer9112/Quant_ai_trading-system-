"""
Phase 5: Enhanced Professional Backtester - Usage Examples

Demonstrates:
- Different order execution models
- Slippage and market impact
- Portfolio rebalancing
- Corporate actions handling
- Margin and leverage management
- Realistic backtesting workflow
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backtesting.execution import (
    OrderType,
    OrderSide,
    ExecutionManager,
    SlippageModel,
    PortfolioRebalancer,
    PortfolioConstraints,
    MarginManager,
    DividendHandler,
    StockSplitHandler,
)


# ============================================================================
# Example 1: Order Execution Models
# ============================================================================

def example_execution_models():
    """Demonstrate different order execution models."""
    print("=" * 60)
    print("EXAMPLE 1: Order Execution Models")
    print("=" * 60)
    
    # Setup execution manager with realistic slippage
    slippage = SlippageModel(
        fixed_bps=1.5,  # 1.5 bps fixed
        volume_impact=0.1,  # 0.1 bps per % of volume
        volatility_impact=0.5,  # 0.5 bps per volatility %
    )
    exec_manager = ExecutionManager(slippage)
    
    # Order parameters
    symbol = 'AAPL'
    quantity = 1000
    current_price = 150.0
    daily_volume = 50000000
    volatility = 0.18
    
    print(f"\nOrder Parameters:")
    print(f"  Symbol: {symbol}")
    print(f"  Quantity: {quantity:,} shares")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Daily Volume: {daily_volume:,}")
    print(f"  Volatility: {volatility:.1%}")
    
    # Test different execution types
    print(f"\n1. Market Order Execution:")
    market_result = exec_manager.execute_order(
        OrderType.MARKET, 1, symbol, OrderSide.BUY, quantity,
        current_price, daily_volume, volatility
    )
    print(f"   Execution Price: ${market_result.execution_price:.2f}")
    print(f"   Slippage: {market_result.slippage:.2%}")
    print(f"   Total Cost: ${market_result.cost:,.2f}")
    print(f"   Cost per Share: ${market_result.cost/quantity:.4f}")
    
    # VWAP with price history
    prices_history = np.linspace(148, 152, 20)
    volumes_history = np.random.randint(1000000, 3000000, 20)
    
    print(f"\n2. VWAP (Volume-Weighted Avg Price) Execution:")
    vwap_result = exec_manager.execute_order(
        OrderType.VWAP, 2, symbol, OrderSide.BUY, quantity,
        current_price, daily_volume, volatility,
        prices_history=prices_history.tolist(),
        volumes_history=volumes_history.tolist()
    )
    print(f"   Execution Price: ${vwap_result.execution_price:.2f}")
    print(f"   Slippage: {vwap_result.slippage:.2%}")
    print(f"   Total Cost: ${vwap_result.cost:,.2f}")
    print(f"   Cost per Share: ${vwap_result.cost/quantity:.4f}")
    
    # TWAP
    print(f"\n3. TWAP (Time-Weighted Avg Price) Execution:")
    twap_result = exec_manager.execute_order(
        OrderType.TWAP, 3, symbol, OrderSide.BUY, quantity,
        current_price, daily_volume, volatility,
        prices_history=prices_history.tolist()
    )
    print(f"   Execution Price: ${twap_result.execution_price:.2f}")
    print(f"   Slippage: {twap_result.slippage:.2%}")
    print(f"   Total Cost: ${twap_result.cost:,.2f}")
    print(f"   Cost per Share: ${twap_result.cost/quantity:.4f}")
    
    # Limit order
    print(f"\n4. Limit Order Execution (Limit=150.50):")
    limit_result = exec_manager.execute_order(
        OrderType.LIMIT, 4, symbol, OrderSide.BUY, quantity,
        current_price, daily_volume, volatility,
        limit_price=150.50
    )
    print(f"   Filled Quantity: {limit_result.filled_quantity:.0f}")
    print(f"   Fill Rate: {limit_result.filled_quantity/quantity:.1%}")
    print(f"   Execution Price: ${limit_result.execution_price:.2f}")
    print(f"   Status: {limit_result.status.name}")
    
    # Execution statistics
    print(f"\n5. Execution Statistics:")
    stats = exec_manager.get_execution_statistics()
    print(f"   Total Orders: {stats['total_orders']}")
    print(f"   Filled Orders: {stats['filled_orders']}")
    print(f"   Fill Rate: {stats['fill_rate']:.1%}")
    print(f"   Avg Slippage: {stats['avg_slippage_bps']:.2f} bps")
    print(f"   Total Costs: ${stats['total_costs']:,.2f}")


# ============================================================================
# Example 2: Slippage Impact Analysis
# ============================================================================

def example_slippage_analysis():
    """Analyze impact of slippage on order execution costs."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Slippage Impact Analysis")
    print("=" * 60)
    
    base_price = 100.0
    daily_volume = 1000000
    volatility = 0.15
    
    print(f"\nAnalyzing slippage impact across position sizes...")
    print(f"Base Price: ${base_price:.2f}")
    print(f"Daily Volume: {daily_volume:,}")
    print(f"Volatility: {volatility:.1%}\n")
    
    # Different position sizes
    sizes = [100, 500, 1000, 5000, 10000]
    
    slippage_model = SlippageModel()
    
    print(f"{'Size':>8} {'% of Vol':>12} {'Slippage (bps)':>16} {'Cost Impact':>15}")
    print(f"{'-'*8} {'-'*12} {'-'*16} {'-'*15}")
    
    for size in sizes:
        pct_volume = (size / daily_volume) * 100
        slippage_bps = slippage_model.calculate_slippage(
            size, daily_volume, volatility, is_buy=True
        ) * 10000
        cost_impact = slippage_bps * base_price / 10000
        
        print(f"{size:>8,} {pct_volume:>11.3f}% {slippage_bps:>15.2f} ${cost_impact:>13.2f}")


# ============================================================================
# Example 3: Portfolio Rebalancing
# ============================================================================

def example_portfolio_rebalancing():
    """Demonstrate portfolio rebalancing logic."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Portfolio Rebalancing")
    print("=" * 60)
    
    # Create rebalancer
    rebalancer = PortfolioRebalancer(
        rebalance_frequency='monthly',
        threshold_pct=0.05  # Rebalance if drift > 5%
    )
    
    # Portfolio state
    portfolio_value = 1000000
    current_weights = {
        'AAPL': 0.35,
        'MSFT': 0.25,
        'GOOGL': 0.20,
        'NVDA': 0.15,
        'CASH': 0.05,
    }
    
    target_weights = {
        'AAPL': 0.30,
        'MSFT': 0.25,
        'GOOGL': 0.25,
        'NVDA': 0.15,
        'CASH': 0.05,
    }
    
    print(f"\nCurrent vs Target Allocations:")
    print(f"{'Symbol':<10} {'Current':>12} {'Target':>12} {'Deviation':>12}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    
    for symbol in target_weights:
        current = current_weights.get(symbol, 0)
        target = target_weights.get(symbol, 0)
        deviation = current - target
        print(f"{symbol:<10} {current:>11.1%} {target:>11.1%} {deviation:>+11.1%}")
    
    # Check rebalance trigger
    now = datetime.now()
    last_rebalance = now - timedelta(days=35)  # 35 days ago
    
    should_rebalance, reason = rebalancer.check_rebalance_trigger(
        current_weights, target_weights, last_rebalance, now
    )
    
    print(f"\nRebalance Decision:")
    print(f"  Trigger: {should_rebalance}")
    print(f"  Reason: {reason}")
    
    # Calculate rebalance trades
    if should_rebalance:
        current_values = {sym: portfolio_value * w for sym, w in current_weights.items()}
        trades = rebalancer.calculate_rebalance_trades(current_values, target_weights)
        
        print(f"\nRebalance Trades Required:")
        print(f"{'Symbol':<10} {'Trade Value':>15} {'Type':>10}")
        print(f"{'-'*10} {'-'*15} {'-'*10}")
        
        total_buy = 0
        total_sell = 0
        for symbol, trade_value in trades.items():
            trade_type = 'BUY' if trade_value > 0 else 'SELL'
            if trade_value > 0:
                total_buy += trade_value
            else:
                total_sell += abs(trade_value)
            print(f"{symbol:<10} ${trade_value:>13,.0f} {trade_type:>10}")
        
        print(f"\n  Total Buy Value: ${total_buy:,.0f}")
        print(f"  Total Sell Value: ${total_sell:,.0f}")
        print(f"  Transaction Cost: {(total_buy + total_sell) * 0.001:.2f} bps")


# ============================================================================
# Example 4: Portfolio Constraints and Risk Controls
# ============================================================================

def example_constraints():
    """Demonstrate portfolio constraint enforcement."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Portfolio Constraints & Risk Controls")
    print("=" * 60)
    
    # Create constraints
    constraints = PortfolioConstraints(
        max_position_pct=0.20,
        max_sector_concentration=0.40,
        max_leverage=2.0,
        min_cash_pct=0.05,
    )
    
    # Test portfolio
    position_weights = {
        'AAPL': 0.18,
        'MSFT': 0.17,
        'GOOGL': 0.15,
        'NVDA': 0.22,  # Exceeds limit!
        'CASH': 0.03,  # Below minimum!
    }
    
    sector_allocation = {
        'Technology': 0.72,  # Exceeds limit!
        'Healthcare': 0.15,
        'Financials': 0.10,
    }
    
    cash_pct = 0.03
    leverage = 1.5
    
    print(f"\nPortfolio Constraints:")
    print(f"  Max Position: {constraints.max_position_pct:.1%}")
    print(f"  Max Sector Concentration: {constraints.max_sector_concentration:.1%}")
    print(f"  Max Leverage: {constraints.max_leverage:.2f}x")
    print(f"  Min Cash: {constraints.min_cash_pct:.1%}")
    
    # Check constraints
    is_valid, violations = constraints.check_constraints(
        position_weights, sector_allocation, cash_pct, leverage
    )
    
    print(f"\nConstraint Validation Results:")
    print(f"  Valid: {is_valid}")
    print(f"  Violations: {len(violations)}")
    
    for violation in violations:
        print(f"    - {violation}")


# ============================================================================
# Example 5: Margin Management
# ============================================================================

def example_margin_management():
    """Demonstrate margin and leverage management."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Margin & Leverage Management")
    print("=" * 60)
    
    margin_manager = MarginManager(
        initial_margin_req=0.50,
        maintenance_margin_req=0.30,
        margin_call_threshold=0.05,
    )
    
    # Scenarios
    scenarios = [
        ('Healthy Position', 100000, 80000, 50000, 10000),
        ('Moderate Leverage', 100000, 60000, 80000, 20000),
        ('High Leverage', 100000, 40000, 120000, 30000),
        ('Margin Call', 100000, 20000, 150000, 50000),
    ]
    
    print(f"\nMargin Status Analysis:")
    print(f"{'Scenario':<20} {'Long':>12} {'Short':>12} {'Margin':>12} {'Status':>20}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*20}")
    
    for scenario_name, portfolio, cash, long_val, short_val in scenarios:
        margin_pct, is_valid, status = margin_manager.check_margin_status(
            portfolio, cash, long_val, short_val
        )
        
        margin_display = f"{margin_pct:.2f}x" if margin_pct != float('inf') else "N/A"
        print(f"{scenario_name:<20} ${long_val:>11,} ${short_val:>11,} {margin_display:>12} {status:>20}")


# ============================================================================
# Example 6: Corporate Actions
# ============================================================================

def example_corporate_actions():
    \"\"\"Demonstrate handling of corporate actions.\"\"\"
    print(\"\\n\" + \"=\" * 60)
    print(\"EXAMPLE 6: Corporate Actions\")
    print(\"=\" * 60)
    
    # Dividend handling
    div_handler = DividendHandler(reinvest=True)
    
    print(f\"\\n1. Dividend Reinvestment:\")
    print(f\"   Initial Position: 100 shares @ $150\")
    
    qty, cash = div_handler.process_dividend(
        'AAPL', 100, 0.92, 150.0, datetime.now()
    )
    
    print(f\"   Dividend: $0.92/share = $92 total\")
    print(f\"   After Reinvestment: {qty:.2f} shares, ${cash:.2f} cash\")
    
    # Stock split handling
    split_handler = StockSplitHandler()
    
    print(f\"\\n2. Stock Split (3:1):\")
    print(f\"   Before Split: 100 shares @ $450\")
    
    new_qty, new_price = split_handler.process_split(
        'NVDA', 100, 3.0, 450.0, datetime.now()
    )
    
    print(f\"   After Split: {new_qty:.0f} shares @ ${new_price:.2f}\")
    print(f\"   Portfolio Value: ${100 * 450:.0f} -> ${new_qty * new_price:.0f} (unchanged)\")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == \"__main__\":
    print(\"\\n\")
    print(\"=\" * 60)
    print(\"PHASE 5: ENHANCED PROFESSIONAL BACKTESTER\")
    print(\"=\" * 60)
    
    # Run all examples
    example_execution_models()
    example_slippage_analysis()
    example_portfolio_rebalancing()
    example_constraints()
    example_margin_management()
    example_corporate_actions()
    
    print(\"\\n\")
    print(\"=\" * 60)
    print(\"✓ PHASE 5 EXAMPLES COMPLETED!\")
    print(\"=\" * 60)
    print(\"\\nKey Features:\")
    print(\"  ✓ Market, VWAP, TWAP, Limit order execution\")
    print(\"  ✓ Realistic slippage and market impact\")
    print(\"  ✓ Portfolio rebalancing (periodic & threshold-based)\")
    print(\"  ✓ Position concentration limits\")
    print(\"  ✓ Leverage and margin tracking\")
    print(\"  ✓ Dividend handling and reinvestment\")
    print(\"  ✓ Stock split and corporate action processing\")
    print(\"\\nIntegration Path:\")
    print(\"  1. Connect execution models to backtester\")
    print(\"  2. Replace simple execution with realistic models\")
    print(\"  3. Add margin calls and position liquidation\")
    print(\"  4. Full production backtesting with institutional quality\")
