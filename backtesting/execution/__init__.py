"""
Execution Module - Professional order execution and portfolio management

Modules:
- order_execution: Market, limit, VWAP, TWAP execution models
- portfolio_management: Rebalancing, corporate actions, constraints, margin

Usage:
    from backtesting.execution import ExecutionManager, PortfolioRebalancer
    
    # Execute order with realistic slippage
    exec_manager = ExecutionManager()
    result = exec_manager.execute_order(
        order_type=OrderType.MARKET,
        order_id=1,
        symbol='AAPL',
        side=OrderSide.BUY,
        quantity=100,
        current_price=150.0,
    )
    
    # Rebalance portfolio
    rebalancer = PortfolioRebalancer(rebalance_frequency='monthly')
    should_rebalance, reason = rebalancer.check_rebalance_trigger(...)
"""

from .order_execution import (
    OrderType,
    OrderSide,
    OrderStatus,
    ExecutionResult,
    SlippageModel,
    MarketOrderExecutor,
    VWAPExecutor,
    TWAPExecutor,
    LimitOrderExecutor,
    ExecutionManager,
)

from .portfolio_management import (
    CorporateAction,
    DividendHandler,
    StockSplitHandler,
    PortfolioRebalancer,
    PortfolioConstraints,
    MarginManager,
)

__all__ = [
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'ExecutionResult',
    'SlippageModel',
    'MarketOrderExecutor',
    'VWAPExecutor',
    'TWAPExecutor',
    'LimitOrderExecutor',
    'ExecutionManager',
    'CorporateAction',
    'DividendHandler',
    'StockSplitHandler',
    'PortfolioRebalancer',
    'PortfolioConstraints',
    'MarginManager',
]

__version__ = '1.0.0'
