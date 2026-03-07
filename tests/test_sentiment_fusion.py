import pandas as pd

from features.sentiment.sentiment_fusion import SentimentFusionPipeline


def test_sentiment_fusion_enriches_panel_with_composite_features():

    panel = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "symbol": ["AAA", "AAA"],
        "sector": ["Tech", "Tech"],
        "trend_regime_code": [1, 1],
    })

    news = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "symbol": ["AAA", "AAA"],
        "sentiment": [0.6, 0.2],
    })
    twitter = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "symbol": ["AAA", "AAA"],
        "sentiment": [0.3, -0.1],
    })
    sector = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "sector": ["Tech", "Tech"],
        "sentiment": [0.4, 0.5],
    })

    pipeline = SentimentFusionPipeline(weights={"news": 0.5, "twitter": 0.3, "sector": 0.2}, smoothing_window=1)
    result = pipeline.enrich(panel, news_df=news, twitter_df=twitter, sector_df=sector)

    assert {"sentiment_composite", "sentiment_divergence", "sentiment_confidence", "sentiment_momentum"}.issubset(result.columns)
    assert round(float(result.loc[0, "sentiment_composite"]), 4) == 0.47


def test_sentiment_fusion_defaults_to_neutral_when_inputs_absent():

    panel = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01"]),
        "symbol": ["AAA"],
        "sector": ["Tech"],
    })

    result = SentimentFusionPipeline().enrich(panel)

    assert float(result.loc[0, "sentiment_composite"]) == 0.0
    assert float(result.loc[0, "sentiment_confidence"]) == 0.0


def test_sentiment_fusion_normalizes_timezone_aware_sentiment_dates_before_merge():

    panel = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01"]),
        "symbol": ["AAA"],
        "sector": ["Tech"],
    })

    news = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01T08:30:00+05:30"]),
        "symbol": ["AAA"],
        "sentiment": [0.6],
    })

    result = SentimentFusionPipeline(smoothing_window=1).enrich(panel, news_df=news)

    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert result["Date"].dt.tz is None
    assert float(result.loc[0, "news_sentiment"]) == 0.6
