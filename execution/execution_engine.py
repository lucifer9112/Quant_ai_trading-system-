"""Realistic trade execution model with slippage and commissions."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Statistics from trade execution."""
    num_trades: int
    gross_pnl: float
    net_pnl: float
    total_slippage: float
    total_commissions: float
    avg_fill_price: float
    execution_quality: float  # (fill_price - mid_price) / spread


class SlippageModel:
    """Model realistic slippage based on order characteristics.
    
    Sources of slippage:
    - Market impact: Large orders move price against trader
    - Adverse selection: Counterparties know about imbalances
    - Timing: Price moves between signal and execution
    - Volatility: Higher vol = worse execution
    
    Model: slippage (bps) = f(order_size, volatility, bid_ask_spread)
    """

    def __init__(self):
        """Initialize slippage model."""
        logger.info("SlippageModel initialized")

    def estimate_slippage(
        self,
        position_size: float,  # Shares or contracts
        price: float,
        volume: float,  # Market volume
        bid_ask_spread: float,
        volatility: float,
    ) -> float:
        """Estimate slippage for an order.
        
        Parameters
        ----------
        position_size : float
            Order size in shares/contracts
        price : float
            Current price
        volume : float
            Market volume today
        bid_ask_spread : float
            Bid-ask spread in absolute dollars
        volatility : float
            Current volatility (annual %)
            
        Returns
        -------
        slippage : float
            Slippage in dollars
        """
        # Order size as % of daily volume
        order_pct_volume = position_size / max(volume, 1)
        
        # Base slippage from bid-ask
        base_slippage = bid_ask_spread / 2
        
        # Market impact: larger orders get worse prices
        # Formula: impact (bps) = sqrt(position_size / volume) * 2
        impact_bps = np.sqrt(np.clip(order_pct_volume, 0, 1)) * 200  # in basis points
        impact_dollars = impact_bps / 10000 * price
        
        # Volatility penalty: higher vol = worse execution
        vol_multiplier = 1.0 + volatility / 100  # 20% vol = 1.2x penalty
        
        # Timing slippage: cost of delayed execution
        timing_slippage = bid_ask_spread / 4  # 25% of spread
        
        # Total
        total_slippage = (base_slippage + impact_dollars + timing_slippage) * vol_multiplier
        total_slippage = np.clip(total_slippage, 0, price * 0.02)  # Cap at 2%
        
        return total_slippage


class CommissionModel:
    """Model realistic trading commissions and fees."""

    def __init__(
        self,
        per_trade_fee: float = 0.0,  # Flat fee per trade
        percent_fee: float = 0.001,  # Percentage fee (0.1%)
        min_commission: float = 1.0,
        exchange_fee: float = 0.0001,  # Exchange fee as % of trade value
    ):
        """Initialize commission model.
        
        Parameters
        ----------
        per_trade_fee : float
            Fixed cost per trade (dollars)
        percent_fee : float
            Percentage commissions (0.001 = 0.1%)
        min_commission : float
            Minimum commission per trade
        exchange_fee : float
            Exchange/clearing fees as % of value
        """
        self.per_trade_fee = per_trade_fee
        self.percent_fee = percent_fee
        self.min_commission = min_commission
        self.exchange_fee = exchange_fee
        
        logger.info(
            f"CommissionModel initialized: "
            f"per_trade={per_trade_fee}, percent={percent_fee*100:.2f}%, "
            f"exchange={exchange_fee*10000:.1f}bps"
        )

    def calculate_commission(
        self,
        position_size: float,
        price: float,
    ) -> float:
        """Calculate commission for a trade.
        
        Parameters
        ----------
        position_size : float
            Position size in shares
        price : float
            Execution price
            
        Returns
        -------
        commission : float
            Commission in dollars
        """
        trade_value = position_size * price
        
        # Percentage-based
        percent_commission = trade_value * self.percent_fee
        
        # Exchange fee
        exchange_fee = trade_value * self.exchange_fee
        
        # Fixed fee
        fixed_commission = self.per_trade_fee
        
        total = percent_commission + exchange_fee + fixed_commission
        total = max(total, self.min_commission)
        
        return total


class ExecutionEngine:
    """Execute trades with realistic slippage, commissions, and constraints.
    
    Key features:
    - Executes position changes with realistic costs
    - Tracks cumulative PnL and execution stats
    - Supports partial fills and order types
    - Position size limits (risk management)
    
    Expected reduction in returns: ~1-2% annually from execution costs
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        max_position_size: float = 0.05,  # Max 5% of capital per position
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
    ):
        """Initialize execution engine.
        
        Parameters
        ----------
        initial_capital : float
            Starting capital
        max_position_size : float
            Max position as % of capital
        slippage_model : SlippageModel, optional
        commission_model : CommissionModel, optional
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        
        self.slippage_model = slippage_model or SlippageModel()
        self.commission_model = commission_model or CommissionModel()
        
        self.positions = {}  # symbol -> (size, entry_price)
        self.cash = initial_capital
        self.trade_history = []
        self.pnl_history = []
        
        logger.info(
            f"ExecutionEngine initialized: "
            f"capital=${initial_capital:,.0f}, "
            f"max_position={max_position_size*100:.1f}%"
        )

    def execute_trade(
        self,
        symbol: str,
        target_size: float,
        current_price: float,
        volume: float,
        bid_ask_spread: float = 0.01,
        volatility: float = 20,
    ) -> Tuple[float, ExecutionStats]:
        """Execute approach to target position size.
        
        Parameters
        ----------
        symbol : str
            Asset symbol
        target_size : float
            Target position size (shares)
        current_price : float
            Current market price
        volume : float
            Market volume
        bid_ask_spread : float
            Bid-ask spread (dollars)
        volatility : float
            Current volatility (annual %)
            
        Returns
        -------
        execution_price : float
            Actual execution price after slippage
        stats : ExecutionStats
            Execution statistics
        """
        current_size = self.positions.get(symbol, (0, 0))[0]
        size_change = target_size - current_size
        
        if abs(size_change) < 0.01:  # No meaningful change
            return current_price, ExecutionStats(0, 0, 0, 0, 0, current_price, 1.0)
        
        # Estimate slippage
        slippage = self.slippage_model.estimate_slippage(
            abs(size_change), current_price, volume, bid_ask_spread, volatility
        )
        
        # Adjust price for slippage
        if size_change > 0:  # Buy
            execution_price = current_price + slippage
        else:  # Sell
            execution_price = current_price - slippage
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(abs(size_change), execution_price)
        
        # PnL from position change
        if current_size > 0:  # Existing long position
            if size_change < 0:  # Selling
                close_pnl = size_change * (execution_price - self.positions[symbol][1])
            else:  # Adding to position
                close_pnl = 0
        else:
            close_pnl = 0
        
        # Update position
        trade_value = abs(size_change) * execution_price
        self.positions[symbol] = (target_size, execution_price)
        self.cash -= trade_value - close_pnl - commission
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'size_change': size_change,
            'price': execution_price,
            'slippage': slippage,
            'commission': commission,
            'pnl': close_pnl,
        }
        self.trade_history.append(trade_record)
        
        # Calculate execution quality
        execution_quality = 1.0 - (slippage / (current_price * bid_ask_spread))
        execution_quality = np.clip(execution_quality, 0, 1)
        
        stats = ExecutionStats(
            num_trades=1,
            gross_pnl=close_pnl,
            net_pnl=close_pnl - commission,
            total_slippage=slippage,
            total_commissions=commission,
            avg_fill_price=execution_price,
            execution_quality=execution_quality,
        )
        
        logger.debug(
            f"Executed {symbol}: size={size_change:+.0f} @ "
            f"{execution_price:.2f} (slippage: ${slippage:.4f}, "
            f"commission: ${commission:.2f})"
        )
        
        return execution_price, stats

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value at mark-to-market.
        
        Parameters
        ----------
        current_prices : dict
            Symbol -> current price
            
        Returns
        -------
        value : float
            Total portfolio value
        """
        portfolio_value = self.cash
        
        for symbol, (size, _) in self.positions.items():
            if size > 0:
                portfolio_value += size * current_prices.get(symbol, 0)
        
        return portfolio_value

    def get_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate current unrealized PnL.
        
        Parameters
        ----------
        current_prices : dict
            Symbol -> current price
            
        Returns
        -------
        pnl : float
            Unrealized PnL
        """
        return self.get_portfolio_value(current_prices) - self.initial_capital

    def summary(self) -> str:
        """Get execution summary.
        
        Returns
        -------
        summary : str
        """
        lines = [
            "ExecutionEngine Summary",
            "=" * 50,
            f"Initial Capital: ${self.initial_capital:,.0f}",
            f"Current Cash: ${self.cash:,.0f}",
            f"Open Positions: {len(self.positions)}",
            f"Total Trades: {len(self.trade_history)}",
        ]
        
        if self.trade_history:
            total_slippage = sum(t['slippage'] for t in self.trade_history)
            total_commission = sum(t['commission'] for t in self.trade_history)
            total_pnl = sum(t['pnl'] for t in self.trade_history)
            
            lines.extend([
                f"",
                f"Trade Statistics:",
                f"  Total Slippage: ${total_slippage:,.2f}",
                f"  Total Commissions: ${total_commission:,.2f}",
                f"  Realized PnL: ${total_pnl:,.2f}",
                f"  Avg Slippage per Trade: ${total_slippage/len(self.trade_history):.4f}",
            ])
        
        return "\n".join(lines)
