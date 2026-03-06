import pandas as pd


class StrategyBacktester:

    def backtest(self, df):

        capital = 100000
        position = 0

        portfolio = []

        for i in range(len(df)):

            signal = df["strategy_score"].iloc[i]
            price = df["Close"].iloc[i]

            if signal > 0.5 and position == 0:

                position = capital / price
                capital = 0

            elif signal < -0.5 and position > 0:

                capital = position * price
                position = 0

            value = capital + position * price

            portfolio.append(value)

        df["portfolio_value"] = portfolio

        return df