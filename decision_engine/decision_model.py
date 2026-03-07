import math
from numbers import Number


class DecisionModel:

    def decide(self, row):

        score = row["strategy_score"]

        ml_prediction = row.get("ml_prediction") if hasattr(row, "get") else None
        sentiment_composite = row.get("sentiment_composite") if hasattr(row, "get") else None
        sentiment_confidence = row.get("sentiment_confidence", 1.0) if hasattr(row, "get") else 1.0

        if isinstance(ml_prediction, Number) and math.isfinite(ml_prediction):
            score += ml_prediction * 0.3

        if isinstance(sentiment_composite, Number) and math.isfinite(sentiment_composite):
            confidence = 1.0
            if isinstance(sentiment_confidence, Number) and math.isfinite(sentiment_confidence):
                confidence = sentiment_confidence
            score += sentiment_composite * 0.2 * confidence

        if score > 0.5:
            return "BUY"

        elif score < -0.5:
            return "SELL"

        else:
            return "HOLD"
