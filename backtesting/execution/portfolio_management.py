"""
Portfolio Management - Rebalancing, corporate actions, and constraints

Implements:
- Portfolio rebalancing (periodic, threshold-based)
- Corporate actions (dividends, stock splits, spin-offs)
- Leverage and margin constraints
- Position concentration limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CorporateAction:
    """Corporate action event."""
    date: datetime
    symbol: str
    action_type: str  # 'dividend', 'split', 'spin_off'
    ratio: float  # For splits: new_shares/old_shares
    amount: float  # For dividends: amount per share
    description: str


class DividendHandler:
    """Handle dividend payments and reinvestment."""
    
    def __init__(self, reinvest: bool = True):
        """
        Args:
            reinvest: Whether to reinvest dividends (True) or pay to cash (False)
        """
        self.reinvest = reinvest
        self.dividend_history = []
    
    def process_dividend(
        self,
        symbol: str,
        quantity: float,
        dividend_per_share: float,
        current_price: float,
        date: datetime,
    ) -> Tuple[float, float]:
        """
        Process dividend payment.
        
        Args:
            symbol: Symbol
            quantity: Shares held
            dividend_per_share: Dividend amount per share
            current_price: Current stock price (for reinvestment)
            date: Dividend date
            
        Returns:
            (new_quantity, cash_received) - if reinvesting, cash_received=0
        """
        dividend_amount = quantity * dividend_per_share
        
        self.dividend_history.append({
            'date': date,
            'symbol': symbol,
            'quantity': quantity,
            'dividend_per_share': dividend_per_share,
            'total_dividend': dividend_amount,
            'reinvested': self.reinvest,
        })
        
        if self.reinvest and current_price > 0:
            # Reinvest dividend into shares
            new_shares = dividend_amount / current_price
            return quantity + new_shares, 0.0
        else:
            # Pay dividend to cash
            return quantity, dividend_amount


class StockSplitHandler:
    """Handle stock splits and consolidations."""
    
    def __init__(self):
        self.split_history = []
    
    def process_split(
        self,
        symbol: str,
        current_quantity: float,
        split_ratio: float,  # New shares per old share
        current_price: float,
        date: datetime,
    ) -> Tuple[float, float]:
        """
        Process stock split.
        
        Args:
            symbol: Symbol
            current_quantity: Current shares held
            split_ratio: New shares / old shares (e.g., 2.0 for 2:1 split)
            current_price: Current share price
            date: Split date
            
        Returns:
            (new_quantity, adjusted_price)
        """
        new_quantity = current_quantity * split_ratio
        adjusted_price = current_price / split_ratio
        
        self.split_history.append({
            'date': date,
            'symbol': symbol,
            'old_quantity': current_quantity,
            'new_quantity': new_quantity,
            'ratio': split_ratio,
            'old_price': current_price,
            'new_price': adjusted_price,
        })
        
        return new_quantity, adjusted_price


class PortfolioRebalancer:
    """Rebalance portfolio based on various triggers."""
    
    def __init__(
        self,
        rebalance_frequency: str = 'monthly',  # 'daily', 'weekly', 'monthly', 'quarterly'
        threshold_pct: float = 0.05,  # Rebalance if drift > 5%
    ):
        """
        Args:
            rebalance_frequency: Rebalancing frequency
            threshold_pct: Threshold for threshold-based rebalancing
        """
        self.rebalance_frequency = rebalance_frequency
        self.threshold_pct = threshold_pct
        self.rebalance_history = []
    
    def check_rebalance_trigger(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        last_rebalance_date: datetime,
        current_date: datetime,
    ) -> Tuple[bool, str]:
        """
        Check if rebalancing is needed.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights
            last_rebalance_date: Date of last rebalancing
            current_date: Current date
            
        Returns:
            (should_rebalance, reason)
        """
        # Check threshold-based trigger
        max_deviation = 0
        for symbol in target_weights:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            deviation = abs(current - target) / (target + 1e-6)
            max_deviation = max(max_deviation, deviation)
        
        if max_deviation > self.threshold_pct:
            return True, f"Threshold exceeded: {max_deviation:.2%} > {self.threshold_pct:.2%}"
        
        # Check frequency-based trigger
        if self.rebalance_frequency == 'daily':
            days_since = (current_date - last_rebalance_date).days
            if days_since >= 1:
                return True, "Daily rebalance schedule"
        elif self.rebalance_frequency == 'weekly':
            days_since = (current_date - last_rebalance_date).days
            if days_since >= 7:
                return True, "Weekly rebalance schedule"
        elif self.rebalance_frequency == 'monthly':
            months_since = (current_date.year - last_rebalance_date.year) * 12 + \
                          (current_date.month - last_rebalance_date.month)
            if months_since >= 1:
                return True, "Monthly rebalance schedule"
        elif self.rebalance_frequency == 'quarterly':
            months_since = (current_date.year - last_rebalance_date.year) * 12 + \
                          (current_date.month - last_rebalance_date.month)
            if months_since >= 3:
                return True, "Quarterly rebalance schedule"
        
        return False, "No rebalance trigger"
    
    def calculate_rebalance_trades(
        self,
        current_values: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance.
        
        Args:
            current_values: Current value by symbol
            target_weights: Target weights
            
        Returns:
            Dict of symbol -> trade amount (positive=buy, negative=sell)
        """
        total_value = sum(current_values.values())
        trades = {}
        
        for symbol, target_weight in target_weights.items():
            current_value = current_values.get(symbol, 0)
            target_value = total_value * target_weight
            trade_value = target_value - current_value
            trades[symbol] = trade_value
        
        return trades


class PortfolioConstraints:
    """Enforce portfolio constraints."""
    
    def __init__(
        self,
        max_position_pct: float = 0.20,  # Max position size as % of portfolio
        max_sector_concentration: float = 0.40,  # Max sector as % of portfolio
        max_leverage: float = 1.0,  # Max leverage (1.0 = no leverage)
        min_cash_pct: float = 0.05,  # Min cash as % of portfolio
    ):
        """
        Args:
            max_position_pct: Maximum position size as % of total portfolio
            max_sector_concentration: Maximum sector concentration
            max_leverage: Maximum leverage ratio
            min_cash_pct: Minimum cash as % of portfolio
        """
        self.max_position_pct = max_position_pct
        self.max_sector_concentration = max_sector_concentration
        self.max_leverage = max_leverage
        self.min_cash_pct = min_cash_pct
        self.violations = []
    
    def check_constraints(
        self,
        portfolio_weights: Dict[str, float],
        sector_allocation: Dict[str, float],
        cash_pct: float,
        gross_leverage: float,
    ) -> Tuple[bool, List[str]]:
        """
        Check if portfolio meets all constraints.
        
        Args:
            portfolio_weights: Position weights
            sector_allocation: Sector allocation
            cash_pct: Cash as % of portfolio
            gross_leverage: Gross leverage ratio
            
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Check position limits
        for symbol, weight in portfolio_weights.items():
            if weight > self.max_position_pct:
                violations.append(
                    f"{symbol}: {weight:.1%} exceeds max {self.max_position_pct:.1%}"
                )
        
        # Check sector limits
        for sector, weight in sector_allocation.items():
            if weight > self.max_sector_concentration:
                violations.append(
                    f"Sector {sector}: {weight:.1%} exceeds max {self.max_sector_concentration:.1%}"
                )
        
        # Check leverage
        if gross_leverage > self.max_leverage:
            violations.append(
                f"Leverage {gross_leverage:.2f}x exceeds max {self.max_leverage:.2f}x"
            )
        
        # Check minimum cash
        if cash_pct < self.min_cash_pct:
            violations.append(
                f"Cash {cash_pct:.1%} below minimum {self.min_cash_pct:.1%}"
            )
        
        is_valid = len(violations) == 0
        self.violations = violations
        
        return is_valid, violations
    
    def adjust_positions(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Adjust positions to comply with constraints.
        
        Args:
            positions: Current positions by symbol
            portfolio_value: Total portfolio value
            
        Returns:
            Adjusted positions
        """
        adjusted = {}
        
        # Cap positions to max size
        max_position_value = portfolio_value * self.max_position_pct
        
        for symbol, position_value in positions.items():
            if position_value > max_position_value:
                adjusted[symbol] = max_position_value
            else:
                adjusted[symbol] = position_value
        
        return adjusted


class MarginManager:
    """Manage margin and leverage requirements."""
    
    def __init__(
        self,
        initial_margin_req: float = 0.50,  # 50% of short value
        maintenance_margin_req: float = 0.30,  # 30% maintenance
        margin_call_threshold: float = 0.05,  # Warn at 5% above requirement
    ):
        """
        Args:
            initial_margin_req: Initial margin requirement
            maintenance_margin_req: Maintenance margin requirement
            margin_call_threshold: Threshold for margin calls
        """
        self.initial_margin_req = initial_margin_req
        self.maintenance_margin_req = maintenance_margin_req
        self.margin_call_threshold = margin_call_threshold
        self.margin_history = []
    
    def calculate_margin_requirement(
        self,
        long_value: float,
        short_value: float,
    ) -> float:
        """
        Calculate margin requirement.
        
        Args:
            long_value: Value of long positions
            short_value: Value of short positions
            
        Returns:
            Required margin amount
        """
        requirement = short_value * self.maintenance_margin_req
        return requirement
    
    def check_margin_status(
        self,
        portfolio_value: float,
        cash_available: float,
        long_value: float,
        short_value: float,
    ) -> Tuple[float, bool, str]:
        """
        Check margin status.
        
        Args:
            portfolio_value: Total portfolio value
            cash_available: Available cash
            long_value: Value of long positions
            short_value: Value of short positions
            
        Returns:
            (margin_pct, is_valid, status_message)
        """
        requirement = self.calculate_margin_requirement(long_value, short_value)
        
        if requirement > 0:
            margin_pct = cash_available / requirement
        else:
            margin_pct = float('inf')
        
        is_valid = margin_pct >= 1.0
        
        if margin_pct >= 1.5:
            status = "Healthy margin position"
        elif margin_pct >= 1.0:
            status = f"Adequate margin ({margin_pct:.2f}x requirement)"
        elif margin_pct >= (1.0 - self.margin_call_threshold):
            status = f"WARNING: Close to margin call ({margin_pct:.2f}x)"
        else:
            status = f"MARGIN CALL: {margin_pct:.2f}x requirement"
        
        return margin_pct, is_valid, status
