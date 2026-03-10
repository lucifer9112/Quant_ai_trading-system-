import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.common.input_loaders import load_optional_dataframe
from core.universe import UniverseManager
from data.providers.market.multi_asset_loader import MultiAssetMarketLoader
from data.storage.gold.panel_dataset_builder import PanelDatasetBuilder
from feature_engineering.feature_analyzer import FeatureAnalyzer
from ml_models.autogluon.autogluon_trainer import AutoGluonTrainer
from monitoring.model_drift import ModelDriftDetector
from utils.config_loader import ConfigLoader
from utils.logger import get_logger


logger = get_logger()


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train a pooled AutoGluon model for a single asset or multi-asset universe."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the config file.")
    parser.add_argument("--symbol", default=None, help="Train on a single symbol override.")
    parser.add_argument("--universe-path", default=None, help="Universe YAML path override.")
    parser.add_argument("--model-path", default=None, help="Override model output path.")
    parser.add_argument("--start-date", default=None, help="Override start date.")
    parser.add_argument("--end-date", default=None, help="Override end date.")
    parser.add_argument("--horizon", type=int, default=1, help="Label horizon in trading days.")
    parser.add_argument("--threshold", type=float, default=0.002, help="Neutral return threshold.")
    parser.add_argument("--news-sentiment-path", default=None, help="Optional symbol-level news sentiment file.")
    parser.add_argument("--twitter-sentiment-path", default=None, help="Optional symbol-level Twitter sentiment file.")
    parser.add_argument("--sector-sentiment-path", default=None, help="Optional sector sentiment file.")
    parser.add_argument("--macro-path", default=None, help="Optional macro dataframe path.")
    parser.add_argument("--benchmark-path", default=None, help="Optional benchmark dataframe path.")
    parser.add_argument("--time-limit", type=int, default=600, help="AutoGluon training time limit.")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation before final fit.")
    parser.add_argument("--prune-correlated", action="store_true", help="Prune highly correlated features.")
    parser.add_argument("--max-features", type=int, default=None, help="Optional cap on selected features.")
    parser.add_argument("--save-drift-baseline", action="store_true", help="Save a feature drift baseline.")
    parser.add_argument("--save-shap-report", action="store_true", help="Save feature importance diagnostics.")
    parser.add_argument(
        "--presets",
        default="medium_quality_faster_train",
        help="AutoGluon presets value."
    )

    return parser.parse_args()


def load_universe(config, args):

    manager = UniverseManager(base_dir=PROJECT_ROOT)

    if args.universe_path:
        return manager.from_yaml(args.universe_path)

    if args.symbol:
        single_config = {
            "symbol": args.symbol,
            "data": config.get("data", {}),
        }
        return manager.from_mapping(single_config)

    return manager.from_mapping(config)


def main():

    args = parse_args()
    config = ConfigLoader(args.config).load()
    universe = load_universe(config, args)

    logger.info("Loading universe '%s' with %d assets", universe.name, len(universe.assets))

    loader = MultiAssetMarketLoader()
    market_panel = loader.load(
        universe,
        start=args.start_date or universe.start_date,
        end=args.end_date or universe.end_date,
    )

    sentiment_config = config.get("sentiment", {})
    news_sentiment_df = load_optional_dataframe(
        args.news_sentiment_path or sentiment_config.get("news_path"),
        base_dir=PROJECT_ROOT,
    )
    twitter_sentiment_df = load_optional_dataframe(
        args.twitter_sentiment_path or sentiment_config.get("twitter_path"),
        base_dir=PROJECT_ROOT,
    )
    sector_sentiment_df = load_optional_dataframe(
        args.sector_sentiment_path or sentiment_config.get("sector_path"),
        base_dir=PROJECT_ROOT,
    )
    macro_df = load_optional_dataframe(args.macro_path, base_dir=PROJECT_ROOT)
    benchmark_df = load_optional_dataframe(args.benchmark_path, base_dir=PROJECT_ROOT)

    logger.info("Building feature panel")
    builder = PanelDatasetBuilder()
    feature_panel = builder.build_panel(
        market_panel,
        news_sentiment_df=news_sentiment_df,
        twitter_sentiment_df=twitter_sentiment_df,
        sector_sentiment_df=sector_sentiment_df,
        macro_df=macro_df,
        benchmark_df=benchmark_df,
    )

    logger.info("Building supervised training frame")
    training_frame = builder.build_training_frame(
        feature_panel,
        horizon=args.horizon,
        threshold=args.threshold,
        include_metadata=args.walk_forward,
        prune_correlated=args.prune_correlated,
        max_features=args.max_features,
    )

    if training_frame.empty:
        raise ValueError("Training dataset is empty after preprocessing.")

    label_counts = training_frame["target_return"].value_counts().to_dict()
    if len(label_counts) < 2:
        raise ValueError(
            "Training labels contain fewer than 2 classes. "
            f"Current distribution: {label_counts}"
        )

    model_path = args.model_path or config.get("model", {}).get("autogluon_path", "models/autogluon")

    logger.info("Training rows: %d", len(training_frame))
    logger.info("Target distribution: %s", label_counts)
    logger.info("Training AutoGluon model at: %s", model_path)

    trainer = AutoGluonTrainer(
        label="target_return",
        model_path=model_path,
        problem_type="multiclass",
        eval_metric="accuracy",
    )
    if args.walk_forward:
        logger.info("Running walk-forward validation")
        walk_forward_summary = trainer.walk_forward_validate(
            training_frame,
            date_col="Date",
            time_limit=max(60, min(args.time_limit, 300)),
            presets=args.presets,
            excluded_columns=["symbol", "sector"],
        )
        summary_path = Path(model_path) / "walk_forward_metrics.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(walk_forward_summary, indent=2), encoding="utf-8")
        logger.info("Walk-forward metrics saved to: %s", summary_path)

    final_training_frame = training_frame.drop(columns=["Date", "symbol", "sector"], errors="ignore")
    predictor = trainer.train(final_training_frame, time_limit=args.time_limit, presets=args.presets)

    feature_columns = [column for column in final_training_frame.columns if column != "target_return"]
    feature_frame = final_training_frame[feature_columns]

    if args.save_drift_baseline:
        logger.info("Saving drift baseline")
        drift_detector = ModelDriftDetector().fit(feature_frame)
        drift_path = Path(model_path) / "drift_baseline.json"
        drift_detector.save(drift_path)
        logger.info("Drift baseline saved to: %s", drift_path)

    if args.save_shap_report:
        logger.info("Saving explainability report")
        analyzer = FeatureAnalyzer()
        shap_importance = analyzer.compute_shap_importance(predictor, feature_frame)
        shap_path = Path(model_path) / "shap_feature_importance.csv"
        shap_importance.to_csv(shap_path, index=False)
        logger.info("Explainability report saved to: %s", shap_path)

    logger.info("Training complete. Model saved to: %s", model_path)


if __name__ == "__main__":
    main()
