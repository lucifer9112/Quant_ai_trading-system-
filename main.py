"""Unified Quantitative Trading System - All Phases Integrated

Complete system combining:
- Phase 1: Feature engineering (100+ features), walk-forward validation, bias detection
- Phase 2: ML ensemble stacking (5 base learners), dynamic weighting, probability calibration
- Phase 3: Production execution with realistic costs, online learning, performance monitoring

Single unified entry point for all functionality.

Usage:
    # Run complete system (all phases)
    system = QuantTradingSystem(config_path="config.yaml")
    results = system.run_phase1_phase2_phase3(df)
    
    # Or run individual phases
    results1 = system.run_phase1_only(df)
    results2 = system.run_phase2_only(df)
    results3 = system.run_phase3_only(df)
    
    # Or legacy single-asset/multi-asset mode
    results = system.run()
"""

from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
import pickle

import numpy as np
import pandas as pd

# Original system imports
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

# Risk Management & Metrics
from risk_management.kelly_criterion import KellyCriterion
from risk_management.position_sizer import AdvancedPositionSizer
from risk_management.portfolio_risk import PortfolioRiskManager
from metrics_engine.metrics_aggregator import MetricsAggregator
from metrics_engine.drawdown_analysis import DrawdownAnalyzer

# ML Models (existing)
from ml_models.autogluon.autogluon_predictor import AutoGluonPredictor
from ml_models.ensemble import EnsemblePredictor
from ml_models.cross_asset_model import CrossAssetModel
from ml_models.sector_model import SectorModel
from ml_models.calibration import ProbabilityCalibrator

# Phase 2-3 New Imports (conditional - may be None if not yet implemented)
try:
    from ml_models.phase2_ml_ensemble import Phase2MLEnsemble
except ImportError:
    Phase2MLEnsemble = None

try:
    from execution.phase3_production import Phase3ProductionExecution
except ImportError:
    Phase3ProductionExecution = None

# Phase 1 Validation & Feature Engineering (conditional)
try:
    from validation.walk_forward_validator import ExpandingWindowValidator
except ImportError:
    ExpandingWindowValidator = None

try:
    from feature_engineering.comprehensive_pipeline import ComprehensiveFeaturePipeline
    from feature_engineering.feature_analyzer import FeatureAnalyzer
except ImportError:
    ComprehensiveFeaturePipeline = None
    FeatureAnalyzer = None

try:
    from backtesting.bias_detector import BacktestingBiasDetector
except ImportError:
    BacktestingBiasDetector = None

# Visualization & Execution (existing)
from visualization.report_generator import ReportGenerator
from visualization.equity_curves import EquityCurveVisualizer
from visualization.trade_signals import TradeSignalVisualizer
from visualization.performance_dashboard import PerformanceDashboard

from regime_detection.regime_detector import RegimeDetectionEngine
from regime_detection.adaptive_allocator import RegimeAwareAllocator

from backtesting.execution.order_execution import ExecutionManager
from backtesting.execution.portfolio_management import PortfolioConstraints, PortfolioRebalancer

from database.storage_manager import StorageManager
from database.duckdb.duckdb_engine import DuckDBEngine
from experiment_tracking.mlflow_tracker import MLFlowTracker

logger = get_logger()


class QuantTradingSystem:
    """Unified Quantitative Trading System with Phases 1, 2, and 3 integrated.
    
    This single system provides:
    - Phase 1: Feature engineering (100+ features), walk-forward validation, bias detection
    - Phase 2: ML ensemble with stacking, dynamic weighting, probability calibration
    - Phase 3: Production execution with realistic costs and monitoring
    
    Can be used as:
    1. Complete system: run_phase1_phase2_phase3(df)
    2. Individual phases: run_phase1_only(df), run_phase2_only(df), run_phase3_only(df)
    3. Legacy system: All original functionality unchanged via run()
    """

    def __init__(self, config_path="config.yaml"):
        """Initialize unified trading system.
        
        Parameters
        ----------
        config_path : str
            Path to configuration YAML file
        """
        self.project_root = Path(__file__).resolve().parent
        self.config_path = config_path
        self.config = ConfigLoader(config_path).load()
        
        # Phase 1-3 components (initialized as needed)
        self.feature_pipeline = None
        self.feature_analyzer = None
        self.validator = None
        self.bias_detector = None
        
        # Phase 2 components
        self.ml_ensemble = None
        
        # Phase 3 components
        self.execution_engine = None
        self.online_learner = None
        self.monitor = None
        
        # Results storage
        self.phase1_results = None
        self.phase2_results = None
        self.phase3_results = None
        
        logger.info("QuantTradingSystem initialized (unified Phases 1-3)")

    # ==================== PHASE 1: Feature Engineering & Validation ====================

    def _initialize_phase1(self):
        """Initialize Phase 1 components."""
        if ComprehensiveFeaturePipeline is not None and self.feature_pipeline is None:
            self.feature_pipeline = ComprehensiveFeaturePipeline()
            self.feature_analyzer = FeatureAnalyzer(method='f_score')
            self.validator = ExpandingWindowValidator(initial_window=252, step_size=63, forecast_horizon=21)
            self.bias_detector = BacktestingBiasDetector()
            logger.info("Phase 1 components initialized")
        else:
            logger.warning("Phase 1 components not available (missing imports)")

    def run_phase1_only(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run Phase 1: Feature engineering, validation, bias detection.
        
        Parameters
        ----------
        df : DataFrame
            Input data with columns: Date, symbol (optional), OHLCV
            
        Returns
        -------
        results : dict
            Phase 1 results with accuracy, features, biases
        """
        logger.info("=" * 70)
        logger.info("PHASE 1: FEATURE ENGINEERING & VALIDATION")
        logger.info("=" * 70)
        
        if ComprehensiveFeaturePipeline is None:
            logger.error("Phase 1 components not available. Install required modules.")
            return {}
        
        self._initialize_phase1()
        
        # Prepare data
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Engineer features (100+ features)
        logger.info("Engineering 100+ features from market data...")
        df_eng = self.feature_pipeline.run(df)
        
        # Create target
        df_eng['forward_returns'] = df_eng['Close'].pct_change().shift(-1)
        df_eng['signal'] = (df_eng['forward_returns'] > 0).astype(int)
        
        # Select best features for modeling
        exclude_cols = {'Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'forward_returns', 'signal', 'returns'}
        feature_cols = [c for c in df_eng.columns if c not in exclude_cols and not c.startswith(('_', 'Unnamed'))]
        df_eng = df_eng.dropna()
        
        if len(feature_cols) == 0:
            logger.error("No features available after engineering")
            return {}
        
        X = df_eng[feature_cols].values
        y = df_eng['signal'].values
        
        logger.info(f"Engineered {X.shape[1]} features from {X.shape[0]} samples")
        
        # Feature selection (top 60 features)
        logger.info("Selecting best 60 features using feature importance...")
        best_features = self.feature_analyzer.select_features(
            df_eng[feature_cols], y, n_features=min(60, len(feature_cols))
        )
        X = df_eng[best_features].values
        
        # Walk-forward validation
        logger.info("Running walk-forward validation (expanding window strategy)...")
        fold_results = []
        from sklearn.ensemble import RandomForestClassifier
        
        folds = list(self.validator.split(df_eng))
        logger.info(f"Walk-forward folds: {len(folds)}")
        
        for fold_idx, fold in enumerate(folds):
            X_train, y_train = X[fold.train_idx], y[fold.train_idx]
            X_val, y_val = X[fold.val_idx], y[fold.val_idx]
            
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            accuracy = np.mean(preds == y_val)
            fold_results.append({'fold': fold_idx, 'accuracy': accuracy})
            logger.info(f"  Fold {fold_idx}: Accuracy = {accuracy:.2%}")
        
        mean_acc = np.mean([r['accuracy'] for r in fold_results])
        std_acc = np.std([r['accuracy'] for r in fold_results])
        
        # Bias detection audit
        logger.info("Running comprehensive bias detection audit...")
        audit = {}
        if self.bias_detector is not None:
            audit = self.bias_detector.run_full_audit(df_eng[feature_cols + ['signal']])
        
        self.phase1_results = {
            'phase': 1,
            'num_engineered_features': len(feature_cols),
            'num_selected_features': len(best_features),
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_results': fold_results,
            'bias_audit': audit,
            'best_features': best_features,
            'X': X,
            'y': y,
            'df': df_eng,
        }
        
        logger.info(f"✅ Phase 1 Complete")
        logger.info(f"   Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
        logger.info(f"   Features selected: {len(best_features)}/{len(feature_cols)}")
        
        return self.phase1_results

    # ==================== PHASE 2: ML Ensemble ====================

    def run_phase2_only(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run Phase 2: ML ensemble with stacking, weighting, calibration.
        
        Requires Phase 1 to be run first for feature engineering.
        
        Parameters
        ----------
        df : DataFrame
            Input data
            
        Returns
        -------
        results : dict
            Phase 2 results with ensemble accuracy
        """
        logger.info("=" * 70)
        logger.info("PHASE 2: ML ENSEMBLE WITH STACKING & DYNAMIC WEIGHTING")
        logger.info("=" * 70)
        
        if Phase2MLEnsemble is None:
            logger.error("Phase 2 components not available. Install required modules.")
            return {}
        
        if self.phase1_results is None:
            logger.warning("Phase 1 not run yet. Running Phase 1 first...")
            self.run_phase1_only(df)
        
        # Initialize Phase 2 ML Ensemble
        self.ml_ensemble = Phase2MLEnsemble(
            use_calibration=True,
            use_dynamic_weighting=True,
        )
        
        logger.info("Running walk-forward ensemble training...")
        logger.info("  Base learners: RF, GB, LGB, XGB, Ridge")
        logger.info("  Meta-learner: Logistic Regression")
        logger.info("  Dynamic weighting: Enabled")
        logger.info("  Probability calibration: Enabled")
        
        results = self.ml_ensemble.run_walk_forward_phase2(df)
        
        self.phase2_results = results
        
        logger.info(f"✅ Phase 2 Complete")
        logger.info(f"   Accuracy: {results.get('mean_accuracy', 0):.2%}")
        logger.info(f"   Base learners: {', '.join(results.get('base_learners', []))}")
        
        return results

    # ==================== PHASE 3: Production Execution ====================

    def run_phase3_only(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run Phase 3: Production execution with realistic costs and monitoring.
        
        Requires Phase 2 to be run first for ML ensemble.
        
        Parameters
        ----------
        df : DataFrame
            Input data
            
        Returns
        -------
        results : dict
            Phase 3 results with realistic returns
        """
        logger.info("=" * 70)
        logger.info("PHASE 3: PRODUCTION EXECUTION WITH REALISTIC COSTS")
        logger.info("=" * 70)
        
        if Phase3ProductionExecution is None:
            logger.error("Phase 3 components not available. Install required modules.")
            return {}
        
        if self.ml_ensemble is None:
            logger.warning("Phase 2 not run yet. Running Phase 2 first...")
            self.run_phase2_only(df)
        
        # Initialize Phase 3 execution
        initial_capital = self.config.get('trading', {}).get('initial_capital', 1000000)
        self.execution_engine = Phase3ProductionExecution(
            initial_capital=initial_capital,
            ml_ensemble=self.ml_ensemble,
            use_online_learning=True,
            use_monitoring=True,
        )
        
        logger.info(f"Generating trading signals (initial capital: ${initial_capital:,.0f})...")
        X, y = self.ml_ensemble.prepare_features(df)
        signals, confidence = self.execution_engine.generate_trading_signals(X, df['Date'])
        
        df_exec = df.iloc[-len(signals):].copy()
        df_exec['signal'] = signals
        df_exec['confidence'] = confidence
        
        logger.info("Executing strategy with realistic costs...")
        logger.info("  Transaction costs: 5 bps")
        logger.info("  Slippage: 3 bps")
        logger.info("  Bid-ask spread: 2 bps")
        
        results = self.execution_engine.execute_strategy(
            df_exec,
            position_sizing='confidence',
            base_position_size=100,
        )
        
        self.phase3_results = results
        
        logger.info(f"✅ Phase 3 Complete")
        logger.info(f"   Total Return: {results.get('total_return', 0):.2%}")
        logger.info(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"   Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        logger.info(f"   Win Rate: {results.get('win_rate', 0):.2%}")
        
        return results

    # ==================== COMPLETE PIPELINE ====================

    def run_phase1_phase2_phase3(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete pipeline: Phase 1 → Phase 2 → Phase 3.
        
        This is the primary entry point for the unified system.
        
        Parameters
        ----------
        df : DataFrame
            Input market data with Date, OHLCV columns
            
        Returns
        -------
        results : dict
            Complete results from all three phases
        """
        logger.info("=" * 70)
        logger.info("UNIFIED TRADING SYSTEM: PHASE 1 + PHASE 2 + PHASE 3")
        logger.info("=" * 70)
        logger.info("Starting complete quantitative trading pipeline...")
        logger.info("")
        
        start_time = datetime.now()
        
        # Phase 1: Feature Engineering & Validation
        logger.info(">>> Starting Phase 1 (Feature Engineering)...")
        phase1 = self.run_phase1_only(df)
        logger.info(f"    ✅ Phase 1 Accuracy: {phase1.get('mean_accuracy', 0):.2%}")
        logger.info(f"    ✅ Features Selected: {phase1.get('num_selected_features', 0)}")
        logger.info("")
        
        # Phase 2: ML Ensemble
        logger.info(">>> Starting Phase 2 (ML Ensemble)...")
        phase2 = self.run_phase2_only(df)
        logger.info(f"    ✅ Phase 2 Accuracy: {phase2.get('mean_accuracy', 0):.2%}")
        logger.info(f"    ✅ Base Learners: {', '.join(phase2.get('base_learners', []))}")
        logger.info("")
        
        # Phase 3: Production Execution
        logger.info(">>> Starting Phase 3 (Production Execution)...")
        phase3 = self.run_phase3_only(df)
        logger.info(f"    ✅ Phase 3 Return: {phase3.get('total_return', 0):.2%}")
        logger.info(f"    ✅ Sharpe Ratio: {phase3.get('sharpe_ratio', 0):.2f}")
        logger.info("")
        
        # Aggregate all results
        elapsed = (datetime.now() - start_time).total_seconds()
        results = {
            'timestamp': datetime.now(),
            'elapsed_seconds': elapsed,
            'phase1': phase1,
            'phase2': phase2,
            'phase3': phase3,
        }
        
        logger.info("=" * 70)
        logger.info("✅ UNIFIED SYSTEM EXECUTION FINISHED")
        logger.info(f"   Total time: {elapsed:.1f} seconds")
        logger.info("=" * 70)
        
        return results

    # ==================== LEGACY SYSTEM (Original functionality) ====================

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
        style = viz_config.get("style", "whitegrid")

        return {
            "report": ReportGenerator(
                output_dir=viz_config.get("output_dir", "reports"),
            ),
            "equity_curves": EquityCurveVisualizer(style=style),
            "trade_signals": TradeSignalVisualizer(style=style),
            "dashboard": PerformanceDashboard(style=style),
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
                equity_series = backtest_result.equity_curve.set_index("Date")["portfolio_value"]
                # Plot main equity curve separately so a failure doesn't stop the others
                try:
                    fig = viz_engines["equity_curves"].plot_equity_curve(
                        equity_series,
                        title="Strategy Equity Curve"
                    )
                    if fig is not None:
                        figures.append(fig)
                except Exception:
                    logger.exception("plot_equity_curve threw an exception")

                try:
                    fig2 = viz_engines["equity_curves"].plot_equity_with_drawdown(
                        equity_series,
                        title="Equity Curve with Drawdown"
                    )
                    if fig2 is not None:
                        figures.append(fig2)
                except Exception:
                    logger.exception("plot_equity_with_drawdown threw an exception")

                try:
                    fig3 = viz_engines["equity_curves"].plot_underwater(
                        equity_series
                    )
                    if fig3 is not None:
                        figures.append(fig3)
                except Exception:
                    logger.exception("plot_underwater threw an exception")

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
        """Run legacy system (original single-asset or multi-asset mode)."""
        if self._pipeline_config().get("mode", "single_asset") == "multi_asset":
            return self._run_multi_asset()
        return self._run_single_asset()

    def summary(self) -> str:
        """Get complete system summary."""
        lines = [
            "",
            "=" * 80,
            "UNIFIED QUANTITATIVE TRADING SYSTEM - EXECUTION SUMMARY",
            "=" * 80,
        ]
        
        if self.phase1_results:
            lines.extend([
                "",
                "PHASE 1: FEATURE ENGINEERING & VALIDATION",
                f"  Status: ✅ Complete",
                f"  Mean Accuracy: {self.phase1_results.get('mean_accuracy', 0):.2%} ± {self.phase1_results.get('std_accuracy', 0):.2%}",
                f"  Features Engineered: {self.phase1_results.get('num_engineered_features', 0)}",
                f"  Features Selected: {self.phase1_results.get('num_selected_features', 0)}",
            ])
        
        if self.phase2_results:
            lines.extend([
                "",
                "PHASE 2: ML ENSEMBLE WITH STACKING",
                f"  Status: ✅ Complete",
                f"  Mean Accuracy: {self.phase2_results.get('mean_accuracy', 0):.2%}",
                f"  Base Learners: {', '.join(self.phase2_results.get('base_learners', []))}",
            ])
        
        if self.phase3_results:
            lines.extend([
                "",
                "PHASE 3: PRODUCTION EXECUTION",
                f"  Status: ✅ Complete",
                f"  Total Return: {self.phase3_results.get('total_return', 0):.2%}",
                f"  Sharpe Ratio: {self.phase3_results.get('sharpe_ratio', 0):.2f}",
                f"  Max Drawdown: {self.phase3_results.get('max_drawdown', 0):.2%}",
                f"  Win Rate: {self.phase3_results.get('win_rate', 0):.2%}",
                f"  Number of Trades: {self.phase3_results.get('num_trades', 0)}",
            ])
        
        lines.extend(["", "=" * 80, ""])
        
        return "\n".join(lines)


# ==================== COMMAND-LINE ENTRY POINT ====================

if __name__ == "__main__":
    import sys
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    parser = argparse.ArgumentParser(
        description="Unified Quantitative Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete system (Phase 1 + 2 + 3)
  python main.py --mode complete --data data.csv
  
  # Run Phase 1 only
  python main.py --mode phase1 --data data.csv
  
  # Run Phase 2 only
  python main.py --mode phase2 --data data.csv
  
  # Run Phase 3 only
  python main.py --mode phase3 --data data.csv
  
  # Run legacy system (single-asset or multi-asset)
  python main.py --mode legacy --config config.yaml
        """,
    )
    
    parser.add_argument(
        "--mode",
        choices=["complete", "phase1", "phase2", "phase3", "legacy"],
        default="complete",
        help="Execution mode (default: complete)"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to CSV data file (for Phase 1-3)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = QuantTradingSystem(config_path=args.config)
    
    if args.mode == "legacy":
        # Run legacy system
        logger.info("Running legacy trading system...")
        result = system.run()
        print(result.tail())
    else:
        # Run phase-based system
        if not args.data:
            print("ERROR: --data argument required for phase-based execution")
            sys.exit(1)
        
        # Load data
        df = pd.read_csv(args.data)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Execute selected mode
        if args.mode == "complete":
            results = system.run_phase1_phase2_phase3(df)
        elif args.mode == "phase1":
            results = system.run_phase1_only(df)
        elif args.mode == "phase2":
            results = system.run_phase2_only(df)
        elif args.mode == "phase3":
            results = system.run_phase3_only(df)
        
        # Print summary
        print(system.summary())
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
        style = viz_config.get("style", "whitegrid")

        return {
            "report": ReportGenerator(
                output_dir=viz_config.get("output_dir", "reports"),
            ),
            "equity_curves": EquityCurveVisualizer(style=style),
            "trade_signals": TradeSignalVisualizer(style=style),
            "dashboard": PerformanceDashboard(style=style),
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
                equity_series = backtest_result.equity_curve.set_index("Date")["portfolio_value"]
                # Plot main equity curve separately so a failure doesn't stop the others
                try:
                    fig = viz_engines["equity_curves"].plot_equity_curve(
                        equity_series,
                        title="Strategy Equity Curve"
                    )
                    if fig is not None:
                        figures.append(fig)
                except Exception:
                    logger.exception("plot_equity_curve threw an exception")

                try:
                    fig2 = viz_engines["equity_curves"].plot_equity_with_drawdown(
                        equity_series,
                        title="Equity Curve with Drawdown"
                    )
                    if fig2 is not None:
                        figures.append(fig2)
                except Exception:
                    logger.exception("plot_equity_with_drawdown threw an exception")

                try:
                    fig3 = viz_engines["equity_curves"].plot_underwater(
                        equity_series
                    )
                    if fig3 is not None:
                        figures.append(fig3)
                except Exception:
                    logger.exception("plot_underwater threw an exception")

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


