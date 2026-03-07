import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.universe import UniverseManager
from data_pipeline.news_data.news_sentiment_exporter import NewsSentimentExporter
from data_pipeline.twitter_data.twitter_sentiment_exporter import TwitterSentimentExporter
from utils.config_loader import ConfigLoader
from utils.logger import get_logger


logger = get_logger()


def parse_args():

    parser = argparse.ArgumentParser(
        description="Build symbol-level and sector-level sentiment input files for the fusion pipeline."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the config file.")
    parser.add_argument("--output-dir", default="data/sentiment", help="Directory for sentiment CSV outputs.")
    parser.add_argument("--twitter-limit", type=int, default=50, help="Tweets to fetch per symbol.")

    return parser.parse_args()


def build_sector_sentiment(news_df, twitter_df):

    frames = [frame for frame in [news_df, twitter_df] if frame is not None and not frame.empty]

    if not frames:
        return pd.DataFrame(columns=["Date", "sector", "sentiment"])

    combined = pd.concat(frames, ignore_index=True)

    return (
        combined.groupby(["Date", "sector"], as_index=False)["sentiment"]
        .mean()
        .sort_values(["Date", "sector"])
        .reset_index(drop=True)
    )


def main():

    args = parse_args()
    config = ConfigLoader(args.config).load()
    universe = UniverseManager(base_dir=PROJECT_ROOT).from_mapping(config)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building news sentiment inputs")
    news_df = NewsSentimentExporter().build_records(universe)

    logger.info("Building Twitter sentiment inputs")
    twitter_df = TwitterSentimentExporter().build_records(universe, limit_per_symbol=args.twitter_limit)

    logger.info("Building sector sentiment inputs")
    sector_df = build_sector_sentiment(news_df, twitter_df)

    news_path = output_dir / "news_sentiment.csv"
    twitter_path = output_dir / "twitter_sentiment.csv"
    sector_path = output_dir / "sector_sentiment.csv"

    for frame, path in [
        (news_df, news_path),
        (twitter_df, twitter_path),
        (sector_df, sector_path),
    ]:
        if frame is None:
            frame = pd.DataFrame()
        frame.to_csv(path, index=False)

    logger.info("Wrote sentiment inputs to %s", output_dir)


if __name__ == "__main__":
    main()
