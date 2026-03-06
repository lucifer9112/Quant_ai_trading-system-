class RiskManager:

    def __init__(self, capital=100000, risk_per_trade=0.02):

        self.capital = capital
        self.risk_per_trade = risk_per_trade

    def position_size(self, price):

        risk_amount = self.capital * self.risk_per_trade

        size = risk_amount / price

        return size