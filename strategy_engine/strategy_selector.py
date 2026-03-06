class StrategySelector:

    def select(self, row):

        if row["Trend"] == "Bullish":
            return "trend"

        elif row["Trend"] == "Bearish":
            return "trend"

        elif row["Volatility_Regime"] == "Low":
            return "mean_reversion"

        elif row["Volatility_Regime"] == "High":
            return "breakout"

        else:
            return "momentum"