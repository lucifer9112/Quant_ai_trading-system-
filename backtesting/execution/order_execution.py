"""
Order Execution Models - Realistic order handling and execution

Implements:
- Market orders
- Limit orders
- VWAP (Volume-Weighted Average Price)
- TWAP (Time-Weighted Average Price)
- Slippage modeling
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass


class OrderType(Enum):
    """Order types."""
    MARKET = 1
    LIMIT = 2
    VWAP = 3
    TWAP = 4


class OrderSide(Enum):
    """Order side."""
    BUY = 1
    SELL = -1


class OrderStatus(Enum):
    """Order status."""
    PENDING = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    CANCELLED = 3


@dataclass
class ExecutionResult:
    """Result of order execution."""
    order_id: int
    symbol: str
    side: OrderSide
    quantity: float
    execution_price: float  # Realized price
    filled_quantity: float
    status: OrderStatus
    slippage: float  # Deviation from target
    cost: float  # Total trade cost including slippage/commission


class SlippageModel:
    """Model realistic market impact and slippage."""
    
    def __init__(
        self,
        fixed_bps: float = 1.0,  # Fixed basis points
        volume_impact: float = 0.1,  # Impact per % of volume
        volatility_impact: float = 0.5,  # Impact per volatility point
    ):
        """
        Args:
            fixed_bps: Fixed slippage in basis points
            volume_impact: Slippage increase per 1% of daily volume
            volatility_impact: Slippage increase per 1% volatility
        """
        self.fixed_bps = fixed_bps
        self.volume_impact = volume_impact
        self.volatility_impact = volatility_impact
    
    def calculate_slippage(
        self,
        quantity: float,
        daily_volume: float,
        volatility: float,
        is_buy: bool = True,
    ) -> float:
        """
        Calculate execution slippage.
        
        Args:
            quantity: Order quantity
            daily_volume: Daily trading volume
            volatility: Current volatility
            is_buy: True for buy orders, False for sell
            
        Returns:
            Slippage in basis points
        """
        # Fixed slippage
        slippage = self.fixed_bps
        
        # Volume impact (larger trades worse)
        order_pct = (quantity / daily_volume) * 100
        volume_slippage = order_pct * self.volume_impact
        slippage += volume_slippage
        
        # Volatility impact
        vol_slippage = volatility * 100 * self.volatility_impact
        slippage += vol_slippage
        
        # Direction (buy slippage typically positive, sell negative)
        if is_buy:
            slippage_bps = slippage
        else:
            slippage_bps = -slippage
        
        return slippage_bps / 10000  # Convert bps to decimal


class MarketOrderExecutor:
    """Execute market orders immediately at market price."""
    
    def __init__(self, slippage_model: Optional[SlippageModel] = None):
        """
        Args:
            slippage_model: Optional slippage model
        """
        self.slippage_model = slippage_model or SlippageModel()
    
    def execute(
        self,
        order_id: int,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        daily_volume: float,
        volatility: float,
        commission: float = 0.001,  # Trade commission
    ) -> ExecutionResult:
        """
        Execute market order.
        
        Args:
            order_id: Order ID
            symbol: Symbol
            side: Buy or Sell
            quantity: Order quantity
            current_price: Current market price
            daily_volume: Daily trading volume
            volatility: Current volatility
            commission: Commission per share
            
        Returns:
            ExecutionResult
        """
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(
            quantity, daily_volume, volatility, side == OrderSide.BUY
        )
        
        # Execution price
        is_buy = side == OrderSide.BUY
        execution_price = current_price * (1 + slippage)
        
        # Trade cost
        trade_value = quantity * execution_price
        commission_cost = quantity * commission
        total_cost = trade_value + commission_cost
        
        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            execution_price=execution_price,
            filled_quantity=quantity,
            status=OrderStatus.FILLED,
            slippage=slippage,
            cost=total_cost,
        )


class VWAPExecutor:
    """Execute orders using VWAP (Volume-Weighted Average Price)."""
    
    def __init__(
        self,
        hours: float = 4.0,  # Execute over 4 hours
        slippage_model: Optional[SlippageModel] = None,
    ):
        """
        Args:
            hours: Time window to execute order (hours)
            slippage_model: Optional slippage model
        """
        self.hours = hours
        self.slippage_model = slippage_model or SlippageModel()
    
    def execute(
        self,
        order_id: int,
        symbol: str,
        side: OrderSide,
        quantity: float,
        prices: List[float],
        volumes: List[float],
        current_price: float,
        volatility: float,
        commission: float = 0.001,
    ) -> ExecutionResult:
        """
        Execute order using VWAP algorithm.
        
        Args:
            order_id: Order ID
            symbol: Symbol
            side: Buy or Sell
            quantity: Order quantity
            prices: List of historical prices
            volumes: List of historical volumes
            current_price: Current market price
            volatility: Current volatility
            commission: Commission per share
            
        Returns:
            ExecutionResult
        """
        # Calculate VWAP
        prices = np.array(prices)
        volumes = np.array(volumes)
        
        vwap = np.sum(prices * volumes) / np.sum(volumes)
        
        # Slippage increases with order size relative to volume
        daily_vol = np.mean(volumes)
        slippage = self.slippage_model.calculate_slippage(
            quantity, daily_vol, volatility, side == OrderSide.BUY
        )
        
        # Execution price targets VWAP with slippage
        execution_price = vwap * (1 + slippage)
        
        # Trade cost
        trade_value = quantity * execution_price
        commission_cost = quantity * commission
        total_cost = trade_value + commission_cost
        
        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            execution_price=execution_price,
            filled_quantity=quantity,
            status=OrderStatus.FILLED,
            slippage=slippage,
            cost=total_cost,
        )


class TWAPExecutor:
    """Execute orders using TWAP (Time-Weighted Average Price)."""
    
    def __init__(
        self,
        time_intervals: int = 20,  # Split into 20 intervals
        slippage_model: Optional[SlippageModel] = None,
    ):
        """
        Args:
            time_intervals: Number of time intervals for execution
            slippage_model: Optional slippage model
        """
        self.time_intervals = time_intervals
        self.slippage_model = slippage_model or SlippageModel()
    
    def execute(
        self,
        order_id: int,
        symbol: str,
        side: OrderSide,
        quantity: float,
        prices: List[float],
        current_price: float,
        volatility: float,
        commission: float = 0.001,
    ) -> ExecutionResult:
        """
        Execute order using TWAP algorithm.
        
        Args:
            order_id: Order ID
            symbol: Symbol
            side: Buy or Sell
            quantity: Order quantity
            prices: List of historical prices
            current_price: Current market price
            volatility: Current volatility
            commission: Commission per share
            
        Returns:
            ExecutionResult
        """
        # Calculate TWAP
        prices = np.array(prices)
        twap = np.mean(prices)
        
        # Slippage
        daily_vol = np.std(prices) * 100  # Rough volume estimate
        slippage = self.slippage_model.calculate_slippage(
            quantity, daily_vol, volatility, side == OrderSide.BUY
        )
        
        # Execution price
        execution_price = twap * (1 + slippage)
        
        # Trade cost
        trade_value = quantity * execution_price
        commission_cost = quantity * commission
        total_cost = trade_value + commission_cost
        
        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            execution_price=execution_price,
            filled_quantity=quantity,
            status=OrderStatus.FILLED,
            slippage=slippage,
            cost=total_cost,
        )


class LimitOrderExecutor:
    """Execute limit orders at specified price."""
    
    def __init__(self, slippage_model: Optional[SlippageModel] = None):
        """
        Args:
            slippage_model: Optional slippage model
        """
        self.slippage_model = slippage_model or SlippageModel()
    
    def execute(
        self,
        order_id: int,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        current_price: float,
        daily_volume: float,
        volatility: float,
        commission: float = 0.001,
    ) -> ExecutionResult:
        """
        Execute limit order.
        
        Args:
            order_id: Order ID
            symbol: Symbol
            side: Buy or Sell
            quantity: Order quantity
            limit_price: Limit price
            current_price: Current market price
            daily_volume: Daily trading volume
            volatility: Current volatility
            commission: Commission per share
            
        Returns:
            ExecutionResult (may be partially filled or unfilled)
        """
        is_buy = side == OrderSide.BUY
        
        # Check if limit order can be filled
        if is_buy and current_price <= limit_price:
            # Buy order can be filled
            filled_pct = 0.8 + 0.2 * (1 - min(current_price / limit_price, 1.0))
            filled_quantity = quantity * filled_pct
            execution_price = current_price
            status = OrderStatus.FILLED if filled_pct >= 0.95 else OrderStatus.PARTIALLY_FILLED
        elif not is_buy and current_price >= limit_price:
            # Sell order can be filled
            filled_pct = 0.8 + 0.2 * (1 - min(limit_price / current_price, 1.0))
            filled_quantity = quantity * filled_pct
            execution_price = current_price
            status = OrderStatus.FILLED if filled_pct >= 0.95 else OrderStatus.PARTIALLY_FILLED
        else:
            # Order not filled
            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                execution_price=limit_price,
                filled_quantity=0,
                status=OrderStatus.PENDING,
                slippage=0,
                cost=0,
            )
        
        # Slippage on filled portion
        slippage = self.slippage_model.calculate_slippage(
            filled_quantity, daily_volume, volatility, is_buy
        ) * 0.5  # Reduce slippage for limit orders
        
        execution_price = execution_price * (1 + slippage)
        
        # Trade cost
        trade_value = filled_quantity * execution_price
        commission_cost = filled_quantity * commission
        total_cost = trade_value + commission_cost
        
        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            execution_price=execution_price,
            filled_quantity=filled_quantity,
            status=status,
            slippage=slippage,
            cost=total_cost,
        )


class ExecutionManager:
    """Manage multiple execution strategies."""
    
    def __init__(self, slippage_model: Optional[SlippageModel] = None):
        """
        Args:
            slippage_model: Optional slippage model
        """
        self.slippage_model = slippage_model or SlippageModel()
        self.market_executor = MarketOrderExecutor(slippage_model)
        self.vwap_executor = VWAPExecutor(slippage_model=slippage_model)
        self.twap_executor = TWAPExecutor(slippage_model=slippage_model)
        self.limit_executor = LimitOrderExecutor(slippage_model)
        self.execution_history = []
    
    def execute_order(
        self,
        order_type: OrderType,
        order_id: int,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        daily_volume: float = 1000000,
        volatility: float = 0.15,
        limit_price: Optional[float] = None,
        prices_history: Optional[List[float]] = None,
        volumes_history: Optional[List[float]] = None,
        commission: float = 0.001,
    ) -> ExecutionResult:
        """
        Execute order using specified execution type.
        
        Args:
            order_type: OrderType enum
            order_id: Order ID
            symbol: Symbol
            side: Buy or Sell
            quantity: Order quantity
            current_price: Current market price
            daily_volume: Daily trading volume
            volatility: Current volatility
            limit_price: Limit price (for limit orders)
            prices_history: Historical prices (for VWAP/TWAP)
            volumes_history: Historical volumes (for VWAP)
            commission: Commission per share
            
        Returns:
            ExecutionResult
        """
        if order_type == OrderType.MARKET:
            result = self.market_executor.execute(
                order_id, symbol, side, quantity, current_price,
                daily_volume, volatility, commission
            )
        elif order_type == OrderType.LIMIT:
            result = self.limit_executor.execute(
                order_id, symbol, side, quantity, limit_price or current_price,
                current_price, daily_volume, volatility, commission
            )
        elif order_type == OrderType.VWAP:
            if prices_history is None or volumes_history is None:
                raise ValueError("VWAP requires price and volume history")
            result = self.vwap_executor.execute(
                order_id, symbol, side, quantity, prices_history,
                volumes_history, current_price, volatility, commission
            )
        elif order_type == OrderType.TWAP:
            if prices_history is None:
                raise ValueError("TWAP requires price history")
            result = self.twap_executor.execute(
                order_id, symbol, side, quantity, prices_history,
                current_price, volatility, commission
            )
        else:
            raise ValueError(f"Unknown order type: {order_type}")
        
        self.execution_history.append(result)
        return result
    
    def get_execution_statistics(self) -> dict:
        """Get execution statistics from history."""
        if not self.execution_history:
            return {}
        
        results = self.execution_history
        total_orders = len(results)
        filled_orders = sum(1 for r in results if r.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED])
        total_slippage = sum(abs(r.slippage) for r in results) / total_orders
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'avg_slippage_bps': total_slippage * 10000,
            'total_costs': sum(r.cost for r in results),
        }
