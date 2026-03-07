import pandas as pd

from models.rl.trading_env import TradingEnvironment


def test_trading_environment_step_returns_valid_transition():

    frame = pd.DataFrame({
        "Close": [100.0, 101.0, 103.0],
        "feature_a": [0.1, 0.2, 0.3],
        "feature_b": [1.0, 0.5, -0.2],
    })

    env = TradingEnvironment(frame, ["feature_a", "feature_b"])

    observation, info = env.reset()
    next_observation, reward, terminated, truncated, next_info = env.step(1)

    assert observation.shape == (2,)
    assert next_observation.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "portfolio_value" in info
    assert "portfolio_value" in next_info
