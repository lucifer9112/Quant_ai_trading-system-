from pathlib import Path

import numpy as np
import pandas as pd

from apps.common.input_loaders import load_optional_dataframe
from backtesting.engine.advanced_backtester import AdvancedBacktester
from core.universe import UniverseManager
from data.providers.market.multi_asset_loader import MultiAssetMarketLoader
from data.storage.gold.panel_dataset_builder import PanelDatasetBuilder
from utils.config_loader import ConfigLoader
from utils.logger import get_logger

from data_pipeline.market_data.nse_downloader import NSEDownloader
from data_pipeline.data_merger import DataMerger

from strategy_engine.strategy_scoring import StrategyScoring
from decision_engine.signal_generator import SignalGenerator
from decision_engine.portfolio_allocator import PortfolioAllocator

# Phase 1: Risk Management & Metrics Engine
from risk_management.kelly_criterion import KellyCriterion
from risk_management.position_sizer import AdvancedPositionSizer
from risk_management.portfolio_risk import PortfolioRiskManager
from metrics_engine.metrics_aggregator import MetricsAggregator
from metrics_engine.drawdown_analysis import DrawdownAnalyzer

# Phase 2: Multi-Asset ML
from ml_models.autogluon.autogluon_predictor import AutoGluonPredictor
from ml_models.ensemble import EnsemblePredictor
from ml_models.cross_asset_model import CrossAssetModel
from ml_models.sector_model import SectorModel
from ml_models.calibration import ProbabilityCalibrator

# Phase 3: Visualization
from visualization.report_generator import ReportGenerator
from visualization.equity_curves import EquityCurveVisualizer
from visualization.trade_signals import TradeSignalVisualizer
from visualization.performance_dashboard import PerformanceDashboard

# Phase 4: Regime Detection
from regime_detection.regime_detector import RegimeDetectionEngine
from regime_detection.adaptive_allocator import RegimeAwareAllocator

# Phase 5: Execution
from backtesting.execution.order_execution import ExecutionManager
from backtesting.execution.portfolio_management import PortfolioConstraints, PortfolioRebalancer

# Database & Experiment Tracking
from database.storage_manager import StorageManager
from database.duckdb.duckdb_engine import DuckDBEngine
from experiment_tracking.mlflow_tracker import MLFlowTracker


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
        """Apply Phase 2 ML predictions (AutoGluon if available)"""

        try:
            ml_ensemble = self._build_ml_ensemble()

            # Apply AutoGluon if available
            if ml_ensemble["autogluon"] is not None:
                df = ml_ensemble["autogluon"].predict(df)

            # Apply ensemble: combine available component predictions
            ensemble = ml_ensemble["ensemble"]
            if ensemble is not None and "ml_prediction" in df.columns:
                # Use ensemble to combine available predictions into a
                # confidence-weighted prediction stored alongside raw score
                predictions = []
                for _, row in df.iterrows():
                    result = ensemble.combine_predictions(
                        autogluon_pred=row.get("ml_prediction"),
                        technical_pred=row.get("strategy_score"),
                        sentiment_pred=row.get("sentiment_composite"),
                        symbol=row.get("symbol", "DEFAULT"),
                    )
                    predictions.append(result.prediction)
                df["ensemble_prediction"] = predictions

            return df

        except Exception as exc:
            logger.warning("ML ensemble not available: %s", exc)
            return df

    def _build_portfolio_allocator(self):

        portfolio_config = self._portfolio_config()

        return PortfolioAllocator(
            sentiment_tilt_strength=portfolio_config.get("sentiment_tilt_strength", 0.20),
            ml_tilt_strength=portfolio_config.get("ml_tilt_strength", 0.15),
            max_position_weight=portfolio_config.get("max_position_weight", 0.25),
            max_gross_exposure=portfolio_config.get("max_gross_exposure", 1.0),
        )

    def _build_risk_manager(self):
        """Build Phase 1 risk management components"""

        risk_config = self.config.get("risk_management", {})
        kelly_config = risk_config.get("kelly_criterion", {})

        kelly = KellyCriterion(
            max_kelly_fraction=kelly_config.get("kelly_fraction", 0.25),
            safety_factor=kelly_config.get("safety_factor", 2.0),
        )

        position_sizer = AdvancedPositionSizer(
            initial_capital=self.config.get("trading", {}).get("initial_capital", 100000),
            max_position_weight=risk_config.get("max_position_weight", 0.25),
            volatility_target=risk_config.get("volatility_target", 0.15),
            risk_per_trade=risk_config.get("fixed_fraction", 0.02),
        )

        portfolio_risk = PortfolioRiskManager(
            initial_capital=self.config.get("trading", {}).get("initial_capital", 100000),
            max_drawdown_pct=risk_config.get("max_drawdown_pct", 0.20),
        )

        return {
            "kelly": kelly,
            "position_sizer": position_sizer,
            "portfolio_risk": portfolio_risk,
        }

    def _build_metrics_engine(self):
        """Build Phase 1 metrics engine"""

        metrics_config = self.config.get("metrics", {})

        return MetricsAggregator(
            risk_free_rate=metrics_config.get("risk_free_rate", 0.04),
            trading_days_per_year=metrics_config.get("trading_days_per_year", 252),
            benchmark_returns=None,
        )

    def _build_regime_detector(self):
        """Build Phase 4 regime detection engine"""

        regime_config = self.config.get("regime_detection", {})

        detector = RegimeDetectionEngine(
            vol_window=regime_config.get("volatility_window", 60),
            trend_short=regime_config.get("trend_window", 20),
            trend_long=regime_config.get("trend_long", 50),
            corr_window=regime_config.get("correlation_window", 60),
        )

        base_weights = self.config.get("portfolio", {}).get("base_weights", {})
        if not base_weights:
            base_weights = {"DEFAULT": 1.0}

        allocator = RegimeAwareAllocator(
            base_allocations=base_weights,
            risk_budget=regime_config.get("risk_budget", 0.15),
        )

        return {
            "detector": detector,
            "allocator": allocator,
        }

    def _build_ml_ensemble(self):
        """Build Phase 2 ML ensemble"""

        ml_config = self.config.get("ml_models", {})
        model_path = self.config.get("model", {}).get("autogluon_path", "models/autogluon")

        # Initialize base predictors
        autogluon = None
        try:
            autogluon = AutoGluonPredictor(model_path=model_path)
        except Exception as e:
            logger.warning("AutoGluon model not available: %s", e)

        # Initialize ensemble with correct constructor args
        ensemble = EnsemblePredictor(
            ensemble_method=ml_config.get("ensemble_method", "weighted_avg"),
            update_frequency=ml_config.get("update_frequency", 10),
        )

        return {
            "autogluon": autogluon,
            "ensemble": ensemble,
        }

    def _build_visualization_engine(self):
        """Build Phase 3 visualization engine"""

        viz_config = self.config.get("visualization", {})

        return {
            "report": ReportGenerator(
                output_dir=viz_config.get("output_dir", "reports"),
            ),
            "equity_curves": EquityCurveVisualizer(),
            "trade_signals": TradeSignalVisualizer(),
            "dashboard": PerformanceDashboard(),
        }

    def _build_execution_manager(self):
        """Build Phase 5 execution manager"""

        backtest_config = self._backtest_config()
        execution_config = backtest_config.get("execution", {})

        # ExecutionManager accepts an optional SlippageModel, not string params.
        # Use default constructor which creates a standard SlippageModel.
        return ExecutionManager()

    def _build_portfolio_constraints(self):
        """Build Phase 5 portfolio constraints"""

        backtest_config = self._backtest_config()
        portfolio_config = self._portfolio_config()

        return PortfolioConstraints(
            max_position_pct=backtest_config.get("max_position_weight", portfolio_config.get("max_position_weight", 0.25)),
            max_sector_concentration=backtest_config.get("max_sector_weight", 0.40),
            max_leverage=backtest_config.get("max_leverage", 1.0),
            min_cash_pct=backtest_config.get("min_cash_pct", 0.05),
        )

    def _build_portfolio_rebalancer(self):
        """Build Phase 5 portfolio rebalancer"""

        backtest_config = self._backtest_config()
        
        # Map frequency: 1 (daily) -> "daily", 5 -> "weekly", 20 -> "monthly"
        frequency_int = backtest_config.get("rebalance_frequency", 1)
        if frequency_int == 1:
            frequency_str = "daily"
        elif frequency_int <= 5:
            frequency_str = "weekly"
        elif frequency_int <= 20:
            frequency_str = "monthly"
        else:
            frequency_str = "quarterly"

        return PortfolioRebalancer(
            rebalance_frequency=frequency_str,
            threshold_pct=backtest_config.get("drift_tolerance", 0.05),
        )

    def _build_storage_manager(self):
        """Build database storage manager for data persistence"""

        try:
            db_config = self.config.get("database", {})
            db_path = db_config.get("duckdb", "data/market.db")
            engine = DuckDBEngine(db_path=db_path)
            return StorageManager(duckdb_engine=engine, influx_client=None)
        except Exception as exc:
            logger.warning("Storage manager not available: %s", exc)
            return None

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

        # Phase 1: Calculate comprehensive metrics using MetricsAggregator
        try:
            metrics_engine = self._build_metrics_engine()
            equity_values = result.equity_curve["portfolio_value"].tolist()
            if len(equity_values) > 1:
                detailed_report = metrics_engine.calculate_all_metrics(
                    equity_curve=equity_values,
                    trades=None,
                )
                merged.attrs["detailed_metrics"] = detailed_report.to_dict()
                logger.info(
                    "Metrics calculated: Sharpe=%.2f, Max DD=%.2f%%",
                    detailed_report.performance.sharpe_ratio,
                    detailed_report.risk.max_drawdown * 100,
                )
        except Exception as exc:
            logger.warning("Metrics calculation failed: %s", exc)

        # Phase 3: Generate reports and visualizations
        self._generate_reports(result)

        # Log metrics to experiment tracker if available
        self._log_experiment_metrics(result.metrics)

        # Persist data to database if available
        self._persist_results(merged)

        return merged

    def _generate_reports(self, backtest_result):
        """Generate Phase 3 visualizations and reports"""

        try:
            viz_engines = self._build_visualization_engine()
            figures = []

            # Generate equity curve figures if available
            if hasattr(backtest_result, "equity_curve") and not backtest_result.equity_curve.empty:
                try:
                    # Plot main equity curve
                    fig = viz_engines["equity_curves"].plot_equity_curve(
                        backtest_result.equity_curve,
                        title="Strategy Equity Curve"
                    )
                    if fig is not None:
                        figures.append(fig)
                    
                    # Plot equity with drawdown
                    fig2 = viz_engines["equity_curves"].plot_equity_with_drawdown(
                        backtest_result.equity_curve,
                        title="Equity Curve with Drawdown"
                    )
                    if fig2 is not None:
                        figures.append(fig2)
                    
                    # Plot underwater chart (drawdown analysis)
                    fig3 = viz_engines["equity_curves"].plot_underwater(
                        backtest_result.equity_curve
                    )
                    if fig3 is not None:
                        figures.append(fig3)
                        
                except Exception as exc:
                    logger.warning("Equity curve visualization failed: %s", exc)

            # Generate trade signals figures if available
            if hasattr(backtest_result, "trades") and backtest_result.trades is not None:
                try:
                    # Plot trade analysis
                    fig = viz_engines["trade_signals"].plot_trade_analysis(
                        backtest_result.trades,
                        title="Trade Analysis"
                    )
                    if fig is not None:
                        figures.append(fig)
                        
                except Exception as exc:
                    logger.warning("Trade signals visualization failed: %s", exc)

            # Generate performance dashboard figures if available
            if hasattr(backtest_result, "metrics") and backtest_result.metrics:
                try:
                    # Plot metrics grid
                    fig = viz_engines["dashboard"].plot_metrics_grid(
                        backtest_result.metrics
                    )
                    if fig is not None:
                        figures.append(fig)
                        
                except Exception as exc:
                    logger.warning("Performance dashboard visualization failed: %s", exc)

            # Generate comprehensive PDF report from all figures
            if figures:
                summary = getattr(backtest_result, "metrics", {}) if hasattr(backtest_result, "metrics") else {}
                report_path = viz_engines["report"].generate_strategy_report(
                    figures=figures,
                    title="Strategy Backtest Report",
                    summary=summary,
                    filename="backtest_report.pdf"
                )
                logger.info(f"Report generated: {report_path}")
            else:
                logger.warning("No figures generated for report")

        except Exception as exc:
            logger.warning("Report generation failed: %s", exc)

    def _log_experiment_metrics(self, metrics):
        """Log metrics to MLFlow experiment tracker"""

        try:
            tracker = MLFlowTracker()
            tracker.start_run()
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    tracker.log_metric(name, value)
            tracker.end_run()
            logger.info("Experiment metrics logged to MLFlow")
        except Exception as exc:
            logger.debug("MLFlow tracking not available: %s", exc)

    def _persist_results(self, df):
        """Persist results to database"""

        try:
            storage = self._build_storage_manager()
            if storage is not None:
                storage.store_market_data(df)
                logger.info("Results persisted to database")
        except Exception as exc:
            logger.debug("Database persistence not available: %s", exc)

    def _apply_regime_aware_allocation(self, df, portfolio_allocator):
        """Apply Phase 4 regime detection to allocation"""

        try:
            regime_engines = self._build_regime_detector()
            detector = regime_engines["detector"]
            allocator = regime_engines["allocator"]

            # Prepare price and return arrays for regime detection
            if "Close" in df.columns:
                prices = df["Close"].values.astype(float)
                returns = np.diff(prices) / prices[:-1]
                returns = np.insert(returns, 0, 0.0)

                # Fit the detector on historical returns
                if len(returns) > 60:
                    detector.fit(returns)

                    # Detect current market regime
                    regime = detector.detect_regime(prices, returns)

                    # Store regime info in dataframe
                    df["market_regime_vol"] = regime.volatility_regime.name
                    df["market_regime_trend"] = regime.trend_regime.name
                    df["market_risk_score"] = regime.risk_score

                    # Get asset volatilities for allocation
                    asset_vols = {}
                    if "symbol" in df.columns:
                        for symbol in df["symbol"].unique():
                            symbol_data = df[df["symbol"] == symbol]
                            if len(symbol_data) > 20:
                                symbol_returns = symbol_data["Close"].pct_change().dropna()
                                asset_vols[symbol] = float(symbol_returns.std() * np.sqrt(252))
                            else:
                                asset_vols[symbol] = 0.15
                    else:
                        asset_vols["DEFAULT"] = float(pd.Series(returns).std() * np.sqrt(252))

                    # Get regime-aware allocation weights
                    regime_weights = allocator.allocate_by_regime(regime, asset_vols)

                    # Apply regime weights as adjustments to existing portfolio weights
                    if "portfolio_weight" in df.columns:
                        for symbol, weight in regime_weights.items():
                            mask = df["symbol"] == symbol if "symbol" in df.columns else slice(None)
                            df.loc[mask, "portfolio_weight"] *= weight

                    logger.info(
                        "Regime-aware allocation applied (vol=%s, trend=%s, risk=%.2f)",
                        regime.volatility_regime.name,
                        regime.trend_regime.name,
                        regime.risk_score,
                    )
                else:
                    logger.warning("Insufficient data for regime detection, need >60 bars")

            return df

        except Exception as exc:
            logger.warning("Regime detection failed, using standard allocation: %s", exc)
            return df

    def _merge_sentiment_data(self, df):
        """Merge news and twitter sentiment data into market dataframe"""

        try:
            merger = DataMerger()
            sentiment_config = self.config.get("sentiment", {})

            # Merge news sentiment if available
            news_path = sentiment_config.get("news_path")
            if news_path:
                news_df = load_optional_dataframe(news_path, base_dir=self.project_root)
                if news_df is not None:
                    df = merger.merge_market_news(df, news_df)

            # Merge twitter sentiment if available
            twitter_path = sentiment_config.get("twitter_path")
            if twitter_path:
                twitter_df = load_optional_dataframe(twitter_path, base_dir=self.project_root)
                if twitter_df is not None:
                    df = merger.merge_twitter(df, twitter_df)

            return df

        except Exception as exc:
            logger.warning("Sentiment data merge failed: %s", exc)
            return df

    def _run_single_asset(self):

        logger.info("Downloading market data")

        downloader = NSEDownloader(self.config["symbol"])

        df = downloader.download()

        # Merge sentiment data if available
        df = self._merge_sentiment_data(df)

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

        # Phase 4: Apply regime-aware allocation if enabled
        if self.config.get("regime_detection", {}).get("enabled", False):
            df = self._build_portfolio_allocator().allocate(
                df,
                capital=self.config.get("trading", {}).get("initial_capital", 100000),
            )
            df = self._apply_regime_aware_allocation(
                df,
                self._build_portfolio_allocator()
            )
        else:
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

        # Phase 4: Apply regime-aware allocation if enabled
        if self.config.get("regime_detection", {}).get("enabled", False):
            panel = self._build_portfolio_allocator().construct_portfolio(
                panel,
                capital=self.config.get("trading", {}).get("initial_capital", 100000),
            )
            panel = self._apply_regime_aware_allocation(
                panel,
                self._build_portfolio_allocator()
            )
        else:
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
