from utils.config_loader import ConfigLoader
from utils.logger import get_logger

from data_pipeline.market_data.nse_downloader import NSEDownloader
from feature_engineering.feature_pipeline import FeaturePipeline

from strategy_engine.strategy_scoring import StrategyScoring
from decision_engine.signal_generator import SignalGenerator
from decision_engine.portfolio_allocator import PortfolioAllocator

from ml_models.autogluon.autogluon_predictor import AutoGluonPredictor


logger = get_logger()


class QuantTradingSystem:

    def __init__(self, config_path="config.yaml"):

        self.config = ConfigLoader(config_path).load()

    def run(self):

        logger.info("Downloading market data")

        downloader = NSEDownloader(self.config["symbol"])

        df = downloader.download()

        logger.info("Generating features")

        df = FeaturePipeline().run(df)

        logger.info("Running strategies")

        df = StrategyScoring().compute_score(df)

        logger.info("ML prediction")

        model_path = self.config.get("model", {}).get(
            "autogluon_path",
            "models/autogluon"
        )

        try:
            predictor = AutoGluonPredictor(model_path=model_path)

            df = predictor.predict(df)

        except Exception as exc:
            logger.warning("ML model not available: %s", exc)

        logger.info("Generating signals")

        df = SignalGenerator().generate(df)

        logger.info("Portfolio simulation")

        df = PortfolioAllocator().allocate(df)

        return df


if __name__ == "__main__":

    system = QuantTradingSystem()

    result = system.run()

    print(result.tail())
