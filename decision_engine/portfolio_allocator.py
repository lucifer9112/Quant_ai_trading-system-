class PortfolioAllocator:

    def allocate(self, df, capital=100000):

        portfolio = []

        cash = capital
        position = 0

        for _, row in df.iterrows():

            signal = row["final_signal"]
            price = row["Close"]

            if signal == "BUY" and position == 0:

                position = cash / price
                cash = 0

            elif signal == "SELL" and position > 0:

                cash = position * price
                position = 0

            value = cash + position * price

            portfolio.append(value)

        df["portfolio_value"] = portfolio

        return df