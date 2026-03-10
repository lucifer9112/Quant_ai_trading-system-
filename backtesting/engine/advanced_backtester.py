from dataclasses import dataclass

import numpy as np
import pandas as pd

from decision_engine.risk_manager import RiskManager
from risk_management.confidence_position_sizer import ConfidencePositionSizer


@dataclass
class AdvancedBacktestResult:
    equity_curve: pd.DataFrame
    metrics: dict[str, float]


class AdvancedBacktester:

    def __init__(
        self,
        initial_capital=100000.0,
        transaction_cost_bps=5.0,
        slippage_bps=3.0,
        rebalance_frequency=1,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        max_drawdown_pct=0.20,
        max_position_weight=0.25,
        risk_manager=None,
        bias_safe=True,
        execution_delay_bars=1,
    ):

        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.rebalance_frequency = max(1, rebalance_frequency)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_weight = max_position_weight
        self.risk_manager = risk_manager or RiskManager(
            capital=initial_capital,
            max_drawdown_limit=max_drawdown_pct,
        )
        self.bias_safe = bias_safe
        self.execution_delay_bars = max(0, execution_delay_bars)
        self.confidence_sizer = ConfidencePositionSizer()

    def _normalize_input(self, df):

        dataset = df.copy()

        if "symbol" not in dataset.columns:
            dataset["symbol"] = "DEFAULT"

        dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
        dataset = dataset.dropna(subset=["Date", "Close"])

        return dataset.sort_values(["Date", "symbol"]).reset_index(drop=True)

    @staticmethod
    def _signal_to_direction(value):

        if isinstance(value, str):
            mapping = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}
            return mapping.get(value.upper(), 0.0)

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        if numeric > 0:
            return 1.0
        if numeric < 0:
            return -1.0
        return 0.0

    def _target_weights(self, day_slice, signal_column):

        precomputed_weight_column = None
        if "portfolio_weight" in day_slice.columns:
            precomputed_weight_column = "portfolio_weight"
        elif "target_weight" in day_slice.columns:
            precomputed_weight_column = "target_weight"

        if precomputed_weight_column is not None:
            weights = {
                symbol: float(weight)
                for symbol, weight in zip(day_slice["symbol"], day_slice[precomputed_weight_column].fillna(0.0))
                if float(weight) != 0.0
            }
            gross_exposure = sum(abs(weight) for weight in weights.values())
            if gross_exposure > 1.0 and gross_exposure > 0:
                weights = {symbol: weight / gross_exposure for symbol, weight in weights.items()}
            return weights

        directions = day_slice[signal_column].map(self._signal_to_direction)
        active = day_slice.loc[directions != 0].copy()

        if active.empty:
            return {}

        active["direction"] = directions.loc[active.index]
        volatilities = active.get("rolling_vol_20", pd.Series(index=active.index, dtype=float))
        volatilities = volatilities.fillna(volatilities.median()).replace(0, np.nan)
        raw_weights = self.risk_manager.risk_parity_weights(volatilities.tolist())

        weights = {}
        confidences = (
            active["prediction_confidence"]
            if "prediction_confidence" in active.columns
            else pd.Series(1.0, index=active.index)
        ).fillna(1.0)
        entropies = (
            active["prediction_entropy"]
            if "prediction_entropy" in active.columns
            else pd.Series(0.0, index=active.index)
        ).fillna(0.0)
        for symbol, direction, raw_weight, confidence, entropy in zip(
            active["symbol"],
            active["direction"],
            raw_weights,
            confidences,
            entropies,
        ):
            multiplier = self.confidence_sizer.confidence_multiplier(confidence, entropy=entropy)
            weights[symbol] = float(
                np.clip(
                    direction * raw_weight * multiplier,
                    -self.max_position_weight,
                    self.max_position_weight,
                )
            )

        gross_exposure = sum(abs(weight) for weight in weights.values())
        if gross_exposure > 1.0 and gross_exposure > 0:
            weights = {symbol: weight / gross_exposure for symbol, weight in weights.items()}

        return weights

    def backtest(self, df, signal_column="final_signal"):

        dataset = self._normalize_input(df)
        grouped = list(dataset.groupby("Date", sort=True))

        portfolio_value = self.initial_capital
        peak_value = portfolio_value
        previous_prices = {}
        current_weights = {}
        pending_weight_queue = [{} for _ in range(self.execution_delay_bars)] if self.bias_safe else []
        entry_prices = {}
        snapshots = []
        rebalances = 0

        for date, day_slice in grouped:
            day_slice = day_slice.copy()
            returns = {}

            for row in day_slice.itertuples(index=False):
                prev_price = previous_prices.get(row.symbol)
                returns[row.symbol] = 0.0 if prev_price in (None, 0) else (row.Close / prev_price) - 1.0
                previous_prices[row.symbol] = row.Close

            portfolio_return = sum(current_weights.get(symbol, 0.0) * returns[symbol] for symbol in returns)
            portfolio_value *= (1.0 + portfolio_return)
            peak_value = max(peak_value, portfolio_value)

            current_prices = dict(zip(day_slice["symbol"], day_slice["Close"]))
            exited_symbols = set()

            for symbol, entry_price in list(entry_prices.items()):
                current_price = current_prices.get(symbol)
                if current_price is None:
                    continue
                if self.risk_manager.apply_exit_rules(
                    entry_price,
                    current_price,
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                ):
                    current_weights[symbol] = 0.0
                    exited_symbols.add(symbol)

            for symbol in exited_symbols:
                entry_prices.pop(symbol, None)

            if self.risk_manager.enforce_drawdown_limit(portfolio_value, peak_value):
                current_weights = {}

            if rebalances % self.rebalance_frequency == 0:
                target_weights = self._target_weights(day_slice, signal_column)
                executable_weights = target_weights
                if self.bias_safe and self.execution_delay_bars > 0:
                    pending_weight_queue.append(target_weights)
                    executable_weights = pending_weight_queue.pop(0)

                turnover = sum(
                    abs(executable_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
                    for symbol in set(executable_weights) | set(current_weights)
                )
                cost_rate = (self.transaction_cost_bps + self.slippage_bps) / 10000.0
                transaction_cost = portfolio_value * turnover * cost_rate
                portfolio_value -= transaction_cost
                current_weights = {symbol: weight for symbol, weight in executable_weights.items() if weight != 0}

                for symbol, weight in current_weights.items():
                    if symbol not in entry_prices or weight == 0:
                        entry_prices[symbol] = current_prices.get(symbol)
            else:
                turnover = 0.0
                transaction_cost = 0.0

            gross_exposure = sum(abs(weight) for weight in current_weights.values())
            peak_value = max(peak_value, portfolio_value)
            drawdown = 0.0 if peak_value == 0 else 1.0 - portfolio_value / peak_value

            snapshots.append({
                "Date": date,
                "portfolio_value": portfolio_value,
                "drawdown": drawdown,
                "gross_exposure": gross_exposure,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
            })

            rebalances += 1

        equity_curve = pd.DataFrame(snapshots)
        metrics = self._compute_metrics(equity_curve)

        return AdvancedBacktestResult(equity_curve=equity_curve, metrics=metrics)

    @staticmethod
    def _compute_metrics(equity_curve):

        if equity_curve.empty:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

        returns = equity_curve["portfolio_value"].pct_change().dropna()

        if returns.empty:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": float(equity_curve["drawdown"].max()),
                "win_rate": 0.0,
            }

        mean_return = returns.mean()
        std_return = returns.std()
        downside = returns[returns < 0]
        downside_std = downside.std()

        total_return = equity_curve["portfolio_value"].iloc[-1] / equity_curve["portfolio_value"].iloc[0] - 1.0
        annualized_return = (1.0 + total_return) ** (252 / max(len(returns), 1)) - 1.0

        sharpe_ratio = 0.0 if pd.isna(std_return) or std_return == 0 else (mean_return / std_return) * np.sqrt(252)
        sortino_ratio = 0.0 if pd.isna(downside_std) or downside_std == 0 else (mean_return / downside_std) * np.sqrt(252)

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": float(equity_curve["drawdown"].max()),
            "win_rate": float((returns > 0).mean()),
        }
