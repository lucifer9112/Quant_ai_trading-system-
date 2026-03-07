from features.store.online_feature_state import OnlineFeatureState


def test_online_feature_state_generates_live_strategy_features():

    state = OnlineFeatureState(window_size=30)
    latest = None

    for index in range(25):
        latest = state.update({
            "Date": f"2024-01-{index + 1:02d}",
            "symbol": "RELIANCE",
            "Open": 100 + index,
            "High": 101 + index,
            "Low": 99 + index,
            "Close": 100.5 + index,
            "Volume": 1000 + index * 5,
        })

    assert latest is not None
    assert latest["symbol"] == "RELIANCE"
    assert "strategy_score" in latest
    assert "rolling_vol_20" in latest
