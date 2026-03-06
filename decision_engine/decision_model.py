import math
from numbers import Number


class DecisionModel:

    def decide(self, row):

        score = row["strategy_score"]

        ml_prediction = row.get("ml_prediction") if hasattr(row, "get") else None

        if isinstance(ml_prediction, Number) and math.isfinite(ml_prediction):
            score += ml_prediction * 0.3

        if score > 0.5:
            return "BUY"

        elif score < -0.5:
            return "SELL"

        else:
            return "HOLD"
