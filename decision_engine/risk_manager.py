import math


class RiskManager:

    def __init__(
        self,
        capital=100000,
        risk_per_trade=0.02,
        max_drawdown_limit=0.20,
        max_position_weight=0.25,
        volatility_target=0.15,
    ):

        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown_limit = max_drawdown_limit
        self.max_position_weight = max_position_weight
        self.volatility_target = volatility_target

    def position_size(self, price, volatility=None, signal_strength=1.0):

        return self.dynamic_position_size(
            available_capital=self.capital,
            price=price,
            volatility=volatility,
            signal_strength=signal_strength,
        )

    def dynamic_position_size(self, available_capital, price, volatility=None, signal_strength=1.0):

        if price <= 0:
            return 0.0

        risk_amount = available_capital * self.risk_per_trade * max(abs(signal_strength), 0.0)

        if volatility and volatility > 0:
            volatility_scale = min(1.0, self.volatility_target / volatility)
            risk_amount *= volatility_scale

        position_value = min(risk_amount / self.risk_per_trade, available_capital * self.max_position_weight)

        return position_value / price

    def kelly_fraction(self, win_rate, win_loss_ratio):

        if win_loss_ratio <= 0:
            return 0.0

        fraction = win_rate - ((1 - win_rate) / win_loss_ratio)

        return max(0.0, min(fraction, self.max_position_weight))

    def risk_parity_weights(self, volatilities):

        if not volatilities:
            return []

        inverse_volatility = []
        for volatility in volatilities:
            if volatility is None or math.isnan(volatility) or volatility <= 0:
                inverse_volatility.append(1.0)
            else:
                inverse_volatility.append(1.0 / volatility)

        total = sum(inverse_volatility)
        if total == 0:
            return [0.0 for _ in inverse_volatility]

        return [value / total for value in inverse_volatility]

    def apply_exit_rules(self, entry_price, current_price, stop_loss_pct=0.05, take_profit_pct=0.10):

        if entry_price <= 0 or current_price <= 0:
            return False

        pnl_ratio = current_price / entry_price - 1.0

        return pnl_ratio <= -stop_loss_pct or pnl_ratio >= take_profit_pct

    def enforce_drawdown_limit(self, current_value, peak_value):

        if peak_value <= 0:
            return False

        drawdown = 1.0 - current_value / peak_value

        return drawdown >= self.max_drawdown_limit
