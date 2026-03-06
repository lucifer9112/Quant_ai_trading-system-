from decision_engine.decision_model import DecisionModel


def test_nan_ml_prediction_does_not_override_buy_signal():

    model = DecisionModel()

    row = {
        "strategy_score": 0.6,
        "ml_prediction": float("nan")
    }

    assert model.decide(row) == "BUY"


def test_nan_ml_prediction_does_not_override_sell_signal():

    model = DecisionModel()

    row = {
        "strategy_score": -0.6,
        "ml_prediction": float("nan")
    }

    assert model.decide(row) == "SELL"


def test_finite_ml_prediction_adjusts_final_decision():

    model = DecisionModel()

    row = {
        "strategy_score": 0.45,
        "ml_prediction": 0.5
    }

    assert model.decide(row) == "BUY"
