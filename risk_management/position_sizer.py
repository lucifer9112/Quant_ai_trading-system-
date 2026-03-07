"""
Advanced Position Sizing - Multiple Sizing Strategies

Includes:
1. Volatility-adjusted sizing (inverse volatility weighting)
2. Risk-parity sizing (inverse volatility risk allocation)
3. Kelly criterion sizing
4. Volatility targeting (target portfolio volatility)
5. Signal-strength adjusted sizing
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PositionSizeResult:
    """Results from position sizing calculation"""
    symbol: str
    position_size: float
    portfolio_weight: float
    risk_amount: float
    leverage: float


class AdvancedPositionSizer:
    """
    Advanced position sizing using multiple approaches.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_weight: float = 0.25,
        max_leverage: float = 2.0,
        volatility_target: float = 0.15,  # 15% annual volatility
        risk_per_trade: float = 0.02,  # 2% of capital at risk
    ):
        """
        Args:
            initial_capital: Starting portfolio capital
            max_position_weight: Maximum weight per position (0.25 = 25%)
            max_leverage: Maximum portfolio leverage allowed
            volatility_target: Target portfolio volatility (annual)
            risk_per_trade: Risk per trade as fraction of capital
        """
        self.initial_capital = initial_capital
        self.max_position_weight = max_position_weight
        self.max_leverage = max_leverage
        self.volatility_target = volatility_target
        self.risk_per_trade = risk_per_trade
    
    def volatility_adjusted(
        self,
        prices: np.ndarray,
        signal_strength: float = 1.0,
        lookback_periods: int = 20,
    ) -> float:
        """
        Adjust position size based on volatility.
        Higher volatility → smaller position.
        
        Args:
            prices: Array of historical prices
            signal_strength: Signal conviction (-1 to 1)
            lookback_periods: Periods to calculate volatility
            
        Returns:
            Position size multiplier (0-1)
        """
        if len(prices) < lookback_periods:
            return abs(signal_strength)
        
        returns = np.diff(prices[-lookback_periods:]) / prices[-lookback_periods:-1]
        volatility = np.std(returns)
        
        if volatility <= 0:
            return abs(signal_strength)
        
        # Scale inversely with volatility
        # Target volatility / current volatility
        vol_scalar = self.volatility_target / volatility
        vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Limit adjustment
        
        return abs(signal_strength) * vol_scalar
    
    def risk_parity(
        self,
        volatilities: List[float],
        signals: List[float],
    ) -> Dict[str, float]:
        """
        Risk parity position sizing.
        Allocate inversely to volatility so each position contributes
        roughly equal risk to the portfolio.
        
        Args:
            volatilities: List of volatilities per asset
            signals: List of signal strengths per asset
            
        Returns:
            Dict mapping signal index to position weight
        """
        if not volatilities or len(volatilities) != len(signals):
            return {}
        
        # Inverse volatility weighting
        inverse_vols = []
        for vol in volatilities:
            if vol is None or vol <= 0 or np.isnan(vol):
                inverse_vols.append(1.0)
            else:
                inverse_vols.append(1.0 / vol)
        
        total_inverse_vol = sum(inverse_vols)
        if total_inverse_vol <= 0:
            return {i: 0.0 for i in range(len(signals))}
        
        # Base weights from inverse volatility
        base_weights = [iv / total_inverse_vol for iv in inverse_vols]
        
        # Apply signal direction and strength
        adjusted_weights = {}
        for i, (weight, signal) in enumerate(zip(base_weights, signals)):
            direction = 1.0 if signal >= 0 else -1.0
            strength = abs(signal)
            adjusted_weights[i] = weight * direction * strength
        
        # Normalize
        total_weight = sum(abs(w) for w in adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        # Apply max position constraint
        adjusted_weights = {
            k: np.clip(v, -self.max_position_weight, self.max_position_weight)
            for k, v in adjusted_weights.items()
        }
        
        return adjusted_weights
    
    def kelly_based(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_price: float,
        stop_loss_price: float,
        available_capital: float,
    ) -> float:
        """
        Position size based on Kelly criterion.
        
        Args:
            win_rate: Probability of profit
            avg_win: Average profit amount
            avg_loss: Average loss amount
            current_price: Entry price
            stop_loss_price: Stop loss price
            available_capital: Capital available to risk
            
        Returns:
            Number of shares to buy
        """
        if available_capital <= 0 or current_price <= 0:
            return 0.0
        
        if win_rate <= 0 or avg_loss <= 0:
            return 0.0
        
        # Kelly fraction
        loss_prob = 1.0 - win_rate
        payoff_ratio = avg_win / avg_loss
        
        kelly = ((win_rate * payoff_ratio) - loss_prob) / payoff_ratio
        kelly = max(0.0, min(kelly, self.max_position_weight))
        
        # Safe Kelly (half Kelly)
        safe_kelly = kelly / 2.0
        
        # Risk amount
        risk_amount = available_capital * safe_kelly
        
        # Position size
        shares = risk_amount / current_price
        
        return shares
    
    def volatility_target_sizing(
        self,
        portfolio_value: float,
        signal_weights: Dict[str, float],
        volatilities: Dict[str, float],
        current_volatility: float = 0.15,
    ) -> Dict[str, float]:
        """
        Scale portfolio to match target volatility.
        
        Args:
            portfolio_value: Total portfolio value
            signal_weights: Weight for each position based on signal
            volatilities: Volatility for each position
            current_volatility: Current portfolio volatility
            
        Returns:
            Adjusted position weights
        """
        if current_volatility <= 0:
            return signal_weights
        
        # Calculate volatility multiplier
        vol_multiplier = self.volatility_target / current_volatility
        vol_multiplier = np.clip(vol_multiplier, 0.5, self.max_leverage)
        
        # Scale all positions
        scaled_weights = {}
        for symbol, weight in signal_weights.items():
            scaled = weight * vol_multiplier
            scaled = np.clip(scaled, -self.max_position_weight, self.max_position_weight)
            scaled_weights[symbol] = scaled
        
        return scaled_weights
    
    def dynamic_size_adjustment(
        self,
        base_position_size: float,
        confidence_score: float,  # 0-1
        volatility_regime: str,    # "low", "medium", "high"
        market_stress: float = 0.0, # 0-1, higher = more stress
    ) -> float:
        """
        Adjust position size based on multiple factors.
        
        Args:
            base_position_size: Initial calculated size
            confidence_score: Model confidence (0-1)
            volatility_regime: Current volatility environment
            market_stress: Market stress indicator (0-1)
            
        Returns:
            Adjusted position size
        """
        adjustment = 1.0
        
        # Apply confidence multiplier
        adjustment *= confidence_score
        
        # Apply volatility regime adjustment
        regime_multipliers = {
            "low": 1.2,
            "medium": 1.0,
            "high": 0.7,
        }
        adjustment *= regime_multipliers.get(volatility_regime, 1.0)
        
        # Reduce size in high stress
        adjustment *= (1.0 - (market_stress * 0.5))
        
        # Clip to reasonable range
        adjustment = np.clip(adjustment, 0.1, 2.0)
        
        return base_position_size * adjustment
    
    def portfolio_concentration_check(
        self,
        proposed_weights: Dict[str, float],
        concentration_limit: float = 0.5,  # Max 50% in top position
    ) -> Tuple[bool, str]:
        """
        Check if proposed weights violate concentration limits.
        
        Args:
            proposed_weights: Dict of symbol -> weight
            concentration_limit: Maximum weight for single position
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not proposed_weights:
            return True, "No positions"
        
        max_weight = max(abs(w) for w in proposed_weights.values())
        
        if abs(max_weight) > concentration_limit:
            return False, f"Concentration {max_weight:.2%} exceeds limit {concentration_limit:.2%}"
        
        gross_exposure = sum(abs(w) for w in proposed_weights.values())
        if gross_exposure > self.max_leverage:
            return False, f"Gross exposure {gross_exposure:.2f}x exceeds max {self.max_leverage}x"
        
        return True, "Weights valid"
    
    def position_from_risk_amount(
        self,
        risk_amount: float,
        entry_price: float,
        stop_price: float,
    ) -> float:
        """
        Calculate position size given a risk amount and stop loss level.
        
        Risk per share = entry_price - stop_price
        Position size = risk_amount / (risk per share)
        
        Args:
            risk_amount: Dollar amount to risk on the trade
            entry_price: Entry price
            stop_price: Stop loss price
            
        Returns:
            Number of shares to buy
        """
        if entry_price <= 0 or stop_price >= entry_price:
            return 0.0
        
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0.0
        
        shares = risk_amount / risk_per_share
        return shares
