from pathlib import Path

from apps.common.input_loaders import load_optional_dataframe
from backtesting.engine.advanced_backtester import AdvancedBacktester
from core.universe import UniverseManager
from data.providers.market.multi_asset_loader import MultiAssetMarketLoader
from data.storage.gold.panel_dataset_builder import PanelDatasetBuilder
from utils.config_loader import ConfigLoader
from utils.logger import get_logger

from data_pipeline.market_data.nse_downloader import NSEDownloader

from strategy_engine.strategy_scoring import StrategyScoring
from decision_engine.signal_generator import SignalGenerator
from decision_engine.portfolio_allocator import PortfolioAllocator

from ml_models.autogluon.autogluon_predictor import AutoGluonPredictor


logger = get_logger()


class QuantTradingSystem:

    def __init__(self, config_path="config.yaml"):

        self.project_root = Path(__file__).resolve().parent
        self.config_path = config_path
        self.config = ConfigLoader(config_path).load()

    def _pipeline_config(self):

        return self.config.get("pipeline", {})

    def _backtest_config(self):

        return self.config.get("backtesting", {})

    def _portfolio_config(self):

        return self.config.get("portfolio", {})

    def _sentiment_inputs(self):

        sentiment_config = self.config.get("sentiment", {})

        return {
            "news_sentiment_df": load_optional_dataframe(
                sentiment_config.get("news_path"),
                base_dir=self.project_root,
            ),
            "twitter_sentiment_df": load_optional_dataframe(
                sentiment_config.get("twitter_path"),
                base_dir=self.project_root,
            ),
            "sector_sentiment_df": load_optional_dataframe(
                sentiment_config.get("sector_path"),
                base_dir=self.project_root,
            ),
        }

    def _apply_model_predictions(self, df):

        model_path = self.config.get("model", {}).get(
            "autogluon_path",
            "models/autogluon"
        )

        try:
            predictor = AutoGluonPredictor(model_path=model_path)

            return predictor.predict(df)

        except Exception as exc:
            logger.warning("ML model not available: %s", exc)
            return df

    def _build_portfolio_allocator(self):

        portfolio_config = self._portfolio_config()
        trading_config = self.config.get("trading", {})

        return PortfolioAllocator(
            sentiment_tilt_strength=portfolio_config.get("sentiment_tilt_strength", 0.20),
            ml_tilt_strength=portfolio_config.get("ml_tilt_strength", 0.15),
            max_position_weight=portfolio_config.get("max_position_weight", 0.25),
            max_gross_exposure=portfolio_config.get("max_gross_exposure", 1.0),
        )

    def _build_advanced_backtester(self):

        trading_config = self.config.get("trading", {})
        backtest_config = self._backtest_config()

        return AdvancedBacktester(
            initial_capital=trading_config.get("initial_capital", 100000),
            transaction_cost_bps=backtest_config.get("transaction_cost_bps", 5.0),
            slippage_bps=backtest_config.get("slippage_bps", 3.0),
            rebalance_frequency=backtest_config.get("rebalance_frequency", 1),
            stop_loss_pct=backtest_config.get("stop_loss_pct", 0.05),
            take_profit_pct=backtest_config.get("take_profit_pct", 0.10),
            max_drawdown_pct=backtest_config.get("max_drawdown_pct", 0.20),
            max_position_weight=self._portfolio_config().get("max_position_weight", 0.25),
        )

    def _run_backtest(self, df):

        backtester = self._build_advanced_backtester()
        result = backtester.backtest(df)
        overlapping_columns = [
            column
            for column in result.equity_curve.columns
            if column != "Date" and column in df.columns
        ]
        merged = df.drop(columns=overlapping_columns, errors="ignore").merge(
            result.equity_curve,
            on="Date",
            how="left",
        )
        merged.attrs["backtest_metrics"] = result.metrics

        return merged

    def _run_single_asset(self):

        logger.info("Downloading market data")

        downloader = NSEDownloader(self.config["symbol"])

        df = downloader.download()

        logger.info("Generating features")

        if self._pipeline_config().get("use_research_features", False):
            from features.pipelines.research_feature_pipeline import ResearchFeaturePipeline

            pipeline_class = ResearchFeaturePipeline
        else:
            from feature_engineering.feature_pipeline import FeaturePipeline

            pipeline_class = FeaturePipeline

        df = pipeline_class().run(df)

        logger.info("Running strategies")

        df = StrategyScoring().compute_score(df)

        logger.info("ML prediction")
        df = self._apply_model_predictions(df)

        logger.info("Generating signals")

        df = SignalGenerator().generate(df)

        logger.info("Portfolio simulation")

        df = self._build_portfolio_allocator().allocate(
            df,
            capital=self.config.get("trading", {}).get("initial_capital", 100000),
        )

        if self._backtest_config().get("enabled", False):
            logger.info("Running advanced backtest")
            df = self._run_backtest(df)

        return df

    def _run_multi_asset(self):

        logger.info("Loading multi-asset universe")

        universe = UniverseManager(base_dir=self.project_root).from_mapping(self.config)
        loader = MultiAssetMarketLoader()
        market_panel = loader.load(
            universe,
            start=self.config.get("data", {}).get("start_date", universe.start_date),
            end=self.config.get("data", {}).get("end_date", universe.end_date),
        )

        logger.info("Generating multi-asset feature panel")
        panel_builder = PanelDatasetBuilder()
        panel = panel_builder.build_panel(
            market_panel,
            **self._sentiment_inputs(),
        )

        logger.info("ML prediction")
        panel = self._apply_model_predictions(panel)

        logger.info("Generating signals")
        panel = SignalGenerator().generate(panel)

        logger.info("Constructing portfolio")
        panel = self._build_portfolio_allocator().construct_portfolio(
            panel,
            capital=self.config.get("trading", {}).get("initial_capital", 100000),
        )

        if self._backtest_config().get("enabled", True):
            logger.info("Running advanced backtest")
            panel = self._run_backtest(panel)

        return panel

    def run(self):

        if self._pipeline_config().get("mode", "single_asset") == "multi_asset":
            return self._run_multi_asset()

        return self._run_single_asset()



if __name__ == "__main__":

    system = QuantTradingSystem()

    result = system.run()

    print(result.tail())
