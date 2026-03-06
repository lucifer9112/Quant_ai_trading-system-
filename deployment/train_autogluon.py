import argparse
import numpy as np
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.market_data.nse_downloader import NSEDownloader
from feature_engineering.feature_pipeline import FeaturePipeline
from ml_models.autogluon.autogluon_trainer import AutoGluonTrainer
from strategy_engine.strategy_scoring import StrategyScoring
from utils.config_loader import ConfigLoader
from utils.logger import get_logger


logger = get_logger()


def build_training_frame(df, horizon=1, threshold=0.002):

    if "Close" not in df.columns:
        raise ValueError("Input dataframe must include a 'Close' column.")

    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    dataset = df.copy()

    forward_return = dataset["Close"].shift(-horizon) / dataset["Close"] - 1.0

    dataset["target_return"] = np.where(
        forward_return > threshold,
        1,
        np.where(forward_return < -threshold, -1, 0)
    )

    dataset = dataset.iloc[:-horizon].copy()

    numeric_columns = dataset.select_dtypes(include=["number", "bool"]).columns.tolist()
    feature_columns = [column for column in numeric_columns if column != "target_return"]

    training_frame = dataset[feature_columns + ["target_return"]].dropna().copy()
    training_frame["target_return"] = training_frame["target_return"].astype(int)

    return training_frame


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train an AutoGluon model for quant trading signals."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the config file (default: config.yaml)."
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Override symbol from config (example: RELIANCE)."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model output path."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in trading days (default: 1)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.002,
        help="Neutral-zone threshold for forward return labels (default: 0.002)."
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="AutoGluon fit time limit in seconds (default: 600)."
    )
    parser.add_argument(
        "--presets",
        default="medium_quality_faster_train",
        help="AutoGluon presets value (default: medium_quality_faster_train)."
    )

    return parser.parse_args()


def main():

    args = parse_args()
    config = ConfigLoader(args.config).load()

    symbol = args.symbol or config["symbol"]
    model_path = args.model_path or config.get("model", {}).get(
        "autogluon_path",
        "models/autogluon"
    )

    logger.info("Downloading market data for training: %s", symbol)
    df = NSEDownloader(symbol).download()

    logger.info("Generating features")
    df = FeaturePipeline().run(df)

    logger.info("Computing strategy signals/scores")
    df = StrategyScoring().compute_score(df)

    logger.info("Building supervised training dataset")
    training_frame = build_training_frame(
        df,
        horizon=args.horizon,
        threshold=args.threshold
    )

    if training_frame.empty:
        raise ValueError("Training dataset is empty after preprocessing.")

    label_counts = training_frame["target_return"].value_counts().to_dict()
    if len(label_counts) < 2:
        raise ValueError(
            "Training labels contain fewer than 2 classes. "
            f"Current distribution: {label_counts}"
        )

    logger.info("Training rows: %d", len(training_frame))
    logger.info("Target distribution: %s", label_counts)
    logger.info("Training AutoGluon model at: %s", model_path)

    trainer = AutoGluonTrainer(
        label="target_return",
        model_path=model_path,
        problem_type="multiclass",
        eval_metric="accuracy"
    )
    trainer.train(
        training_frame,
        time_limit=args.time_limit,
        presets=args.presets
    )

    logger.info("Training complete. Model saved to: %s", model_path)


if __name__ == "__main__":
    main()
