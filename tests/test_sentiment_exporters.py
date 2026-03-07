import pandas as pd

from core.schemas import AssetMetadata
from core.universe import UniverseDefinition
from data_pipeline.news_data.news_sentiment_exporter import NewsSentimentExporter
from data_pipeline.twitter_data.twitter_sentiment_exporter import TwitterSentimentExporter


class StubNewsCollector:

    def collect(self):

        return [
            {
                "title": "Reliance expands refining business",
                "link": "https://example.com/reliance",
                "published": "2024-01-01",
            },
            {
                "title": "Macro headline without ticker mention",
                "link": "https://example.com/macro",
                "published": "2024-01-01",
            },
        ]


class StubNewsScorer:

    def score(self, text):

        return 0.5


class StubTwitterCollector:

    def search(self, query, limit=100):

        return [
            {
                "date": "2024-01-01",
                "text": f"{query} is trending strongly",
            }
        ]


class StubTwitterScorer:

    def score(self, text):

        return 0.25


class FailingTwitterCollector:

    def search(self, query, limit=100):

        raise RuntimeError("snscrape could not be imported")


def test_news_sentiment_exporter_emits_symbol_shaped_rows():

    universe = UniverseDefinition(
        name="demo",
        assets=(AssetMetadata(symbol="RELIANCE", sector="Energy"),),
    )

    exporter = NewsSentimentExporter(
        collector=StubNewsCollector(),
        scorer=StubNewsScorer(),
    )
    result = exporter.build_records(universe)

    assert list(result.columns) == ["Date", "symbol", "sector", "sentiment", "source", "text", "link"]
    assert len(result) == 1
    assert result.loc[0, "symbol"] == "RELIANCE"
    assert float(result.loc[0, "sentiment"]) == 0.5


def test_news_sentiment_exporter_normalizes_timezone_aware_dates():

    class TimezoneAwareNewsCollector:

        def collect(self):

            return [
                {
                    "title": "RELIANCE publishes earnings",
                    "link": "https://example.com/reliance-earnings",
                    "published": "2024-01-01T09:15:00+05:30",
                }
            ]

    universe = UniverseDefinition(
        name="demo",
        assets=(AssetMetadata(symbol="RELIANCE", sector="Energy"),),
    )

    exporter = NewsSentimentExporter(
        collector=TimezoneAwareNewsCollector(),
        scorer=StubNewsScorer(),
    )
    result = exporter.build_records(universe)

    assert pd.api.types.is_datetime64_any_dtype(result["Date"])
    assert result["Date"].dt.tz is None
    assert result.loc[0, "Date"] == pd.Timestamp("2024-01-01")


def test_twitter_sentiment_exporter_emits_symbol_shaped_rows():

    universe = UniverseDefinition(
        name="demo",
        assets=(AssetMetadata(symbol="INFY", sector="Information Technology"),),
    )

    exporter = TwitterSentimentExporter(
        collector=StubTwitterCollector(),
        scorer=StubTwitterScorer(),
    )
    result = exporter.build_records(universe, limit_per_symbol=1)

    assert {"Date", "symbol", "sector", "sentiment", "source", "text"}.issubset(result.columns)
    assert len(result) == 1
    assert result.loc[0, "symbol"] == "INFY"
    assert float(result.loc[0, "sentiment"]) == 0.25


def test_twitter_sentiment_exporter_returns_empty_frame_when_collection_unavailable():

    universe = UniverseDefinition(
        name="demo",
        assets=(AssetMetadata(symbol="TCS", sector="Information Technology"),),
    )

    exporter = TwitterSentimentExporter(
        collector=FailingTwitterCollector(),
        scorer=StubTwitterScorer(),
    )
    result = exporter.build_records(universe, limit_per_symbol=1)

    assert list(result.columns) == TwitterSentimentExporter.OUTPUT_COLUMNS
    assert result.empty
