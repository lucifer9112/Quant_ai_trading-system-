import numpy as np
import pandas as pd

from decision_engine.risk_manager import RiskManager


class PortfolioAllocator:

    def __init__(
        self,
        risk_manager=None,
        sentiment_tilt_strength=0.20,
        ml_tilt_strength=0.15,
        max_position_weight=0.25,
        max_gross_exposure=1.0,
    ):

        self.risk_manager = risk_manager or RiskManager(max_position_weight=max_position_weight)
        self.sentiment_tilt_strength = sentiment_tilt_strength
        self.ml_tilt_strength = ml_tilt_strength
        self.max_position_weight = max_position_weight
        self.max_gross_exposure = max_gross_exposure

    @staticmethod
    def _signal_to_direction(signal, score=0.0):

        if isinstance(signal, str):
            mapping = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}
            return mapping.get(signal.upper(), 0.0)

        if score > 0:
            return 1.0
        if score < 0:
            return -1.0

        return 0.0

    def _build_daily_weights(self, day_slice):

        active = day_slice.copy()
        active["strategy_score"] = (
            active["strategy_score"] if "strategy_score" in active.columns else 0.0
        )
        active["strategy_score"] = active["strategy_score"].fillna(0.0)
        active["ml_prediction"] = (
            active["ml_prediction"] if "ml_prediction" in active.columns else 0.0
        )
        active["ml_prediction"] = active["ml_prediction"].fillna(0.0)
        active["sentiment_composite"] = (
            active["sentiment_composite"] if "sentiment_composite" in active.columns else 0.0
        )
        active["sentiment_composite"] = active["sentiment_composite"].fillna(0.0)
        active["signal_direction"] = [
            self._signal_to_direction(signal, score)
            for signal, score in zip(
                active["final_signal"] if "final_signal" in active.columns else pd.Series(index=active.index, dtype=object),
                active["strategy_score"],
            )
        ]

        active = active.loc[active["signal_direction"] != 0].copy()

        if active.empty:
            return {}

        volatilities = active.get("rolling_vol_20", pd.Series(index=active.index, dtype=float))
        volatilities = volatilities.fillna(volatilities.median()).replace(0, np.nan)
        parity_weights = self.risk_manager.risk_parity_weights(volatilities.tolist())

        conviction = (
            active["strategy_score"].abs() +
            active["ml_prediction"].abs() * self.ml_tilt_strength +
            active["sentiment_composite"].abs() * self.sentiment_tilt_strength
        ).clip(lower=0.05)

        alignment = (
            1.0 +
            self.sentiment_tilt_strength * (
                active["signal_direction"] * active["sentiment_composite"]
            ).clip(lower=-1.0, upper=1.0)
        ).clip(lower=0.5, upper=1.5)

        raw_weights = []
        for parity_weight, conviction_score, alignment_score in zip(parity_weights, conviction, alignment):
            raw_weights.append(parity_weight * conviction_score * alignment_score)

        total_raw = sum(raw_weights)
        if total_raw <= 0:
            normalized = [1.0 / len(raw_weights) for _ in raw_weights]
        else:
            normalized = [weight / total_raw for weight in raw_weights]

        weights = {}
        for symbol, direction, base_weight in zip(active["symbol"], active["signal_direction"], normalized):
            weights[symbol] = float(np.clip(direction * base_weight, -self.max_position_weight, self.max_position_weight))

        gross_exposure = sum(abs(weight) for weight in weights.values())
        if gross_exposure > self.max_gross_exposure and gross_exposure > 0:
            scale = self.max_gross_exposure / gross_exposure
            weights = {symbol: weight * scale for symbol, weight in weights.items()}

        return weights

    def construct_portfolio(self, df, capital=100000):

        dataset = df.copy()

        if "symbol" not in dataset.columns:
            dataset["symbol"] = "DEFAULT"

        dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
        dataset = dataset.dropna(subset=["Date", "Close"]).sort_values(["Date", "symbol"]).reset_index(drop=True)
        dataset["portfolio_weight"] = 0.0
        dataset["target_position_units"] = 0.0

        daily_values = []
        portfolio_value = capital
        previous_weights = {}
        previous_prices = {}

        for date, day_slice in dataset.groupby("Date", sort=True):
            current_prices = dict(zip(day_slice["symbol"], day_slice["Close"]))
            portfolio_return = 0.0

            for symbol, weight in previous_weights.items():
                previous_price = previous_prices.get(symbol)
                current_price = current_prices.get(symbol)
                if previous_price in (None, 0) or current_price is None:
                    continue
                portfolio_return += weight * ((current_price / previous_price) - 1.0)

            portfolio_value *= (1.0 + portfolio_return)

            weights = self._build_daily_weights(day_slice)
            gross_exposure = sum(abs(weight) for weight in weights.values())

            for row in day_slice.itertuples():
                weight = weights.get(row.symbol, 0.0)
                dataset.at[row.Index, "portfolio_weight"] = weight
                dataset.at[row.Index, "target_position_units"] = 0.0 if row.Close == 0 else (portfolio_value * weight) / row.Close

            daily_values.append({
                "Date": date,
                "portfolio_value": portfolio_value,
                "gross_exposure": gross_exposure,
            })

            previous_weights = weights
            previous_prices = current_prices

        daily_values_df = pd.DataFrame(daily_values)
        dataset = dataset.merge(daily_values_df, on="Date", how="left")

        return dataset

    def allocate(self, df, capital=100000):

        dataset = df.copy()

        multi_asset = "symbol" in dataset.columns and dataset["symbol"].nunique() > 1

        if multi_asset:
            return self.construct_portfolio(dataset, capital=capital)

        portfolio = []

        cash = capital
        position = 0

        for _, row in dataset.iterrows():

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

        dataset["portfolio_value"] = portfolio

        return dataset
