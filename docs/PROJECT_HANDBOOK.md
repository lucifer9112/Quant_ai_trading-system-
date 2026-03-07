# Quant AI Trading System Handbook

## 1. Purpose

This document is the technical reference for the `quant_ai_trading` repository. It explains:

- the current architecture
- the end-to-end dataflow
- the main code paths for training, inference, sentiment, backtesting, and live signals
- the role of each major folder and important module
- how configuration controls runtime behavior
- what is production-ready, what is scaffolded, and what remains legacy code

The codebase is currently in a hybrid state:

- the original project layout is still present under folders such as `data_pipeline/`, `feature_engineering/`, `strategy_engine/`, `decision_engine/`, and `ml_models/`
- a newer architecture has been added in parallel under `core/`, `data/`, `features/`, `backtesting/`, `apps/`, and `models/rl/`

This was intentional. It allowed the platform to evolve toward a professional multi-asset research stack without breaking the original working pipeline in one large rewrite.

## 2. Platform Goal

The system is an AI-driven quantitative research and trading platform for Indian equities, currently centered on NSE symbols. It supports:

- batch market data collection
- feature engineering for technical, regime, and sentiment signals
- pooled multi-asset tabular ML training using AutoGluon
- portfolio construction with risk-aware and sentiment-aware weighting
- advanced backtesting with costs, slippage, and risk controls
- near real-time signal generation scaffolding
- reinforcement learning environment scaffolding for future policy learning
- experiment tracking and model artifact management

The target architecture is closer to a hedge-fund research platform than a single-strategy script.

## 3. High-Level Architecture

The repository now follows a layered architecture.

### 3.1 Core layer

Path: `core/`

Responsibilities:

- domain contracts
- event objects
- universe definition and parsing
- future foundation for typed platform-wide interfaces

Important files:

- `core/schemas.py`
- `core/events.py`
- `core/universe.py`

### 3.2 Data layer

Paths:

- `data/providers/`
- `data/storage/`
- legacy providers under `data_pipeline/`

Responsibilities:

- source-specific data access
- multi-asset market data loading
- construction of panel datasets for training and research
- sentiment CSV generation for feature fusion

### 3.3 Feature layer

Paths:

- `feature_engineering/` for legacy feature modules
- `features/` for the new composable feature architecture

Responsibilities:

- technical features
- market microstructure features
- regime-aware features
- sentiment fusion features
- online rolling feature state for live signals

### 3.4 Model layer

Paths:

- `ml_models/` for tabular and classical model wrappers
- `models/rl/` for RL scaffolding
- `nlp_models/` for FinBERT, embeddings, and text modeling

Responsibilities:

- tabular prediction and training
- future deep learning and RL agents
- sentiment scoring helpers

### 3.5 Strategy, decision, and risk layer

Paths:

- `strategy_engine/`
- `decision_engine/`

Responsibilities:

- handcrafted strategy signals
- signal scoring and final trade decisions
- portfolio construction
- position sizing and risk controls

### 3.6 Backtesting layer

Path: `backtesting/engine/`

Responsibilities:

- transaction-cost-aware simulation
- slippage-aware simulation
- rebalancing logic
- risk-limit enforcement
- performance metrics

### 3.7 Application layer

Paths:

- `main.py`
- `apps/batch_train/`
- `apps/live_trading/`
- `apps/data/`
- `deployment/`

Responsibilities:

- runtime entrypoints
- training CLI
- live signal CLI
- sentiment-input generation CLI
- deployment helpers

## 4. Current Repository Map

This section explains the practical role of each folder in the repository.

### 4.1 Root files

- `main.py`: primary runtime entrypoint. Can run single-asset or multi-asset mode depending on `config.yaml`.
- `config.yaml`: default runtime configuration. Currently defaults to multi-asset mode with advanced backtesting enabled.
- `requirements.txt`: Python dependencies.
- `.python-version`: declares Python 3.11 as the target runtime for local environment managers.
- `runtime.txt`: declares Python 3.11 for deployment platforms that honor runtime files.
- `README.md`: quick-start document.
- `docs/PROJECT_HANDBOOK.md`: this full technical handbook.

### 4.1.1 Python runtime policy

The project target runtime is Python 3.11.

Reason:

- the full market, training, and backtesting stack is broadly portable
- the Twitter sentiment collection path depends on `snscrape`
- `snscrape` is not reliable on Python 3.12+ in the environments used for this project

Practical consequence:

- Python 3.12+ may still run the core pipeline
- Twitter sentiment collection is only considered supported on Python 3.11
- `apps/data/build_sentiment_inputs.py --require-twitter` now fails fast on unsupported runtimes with a clear message
- plain `apps/data/build_sentiment_inputs.py` still falls back gracefully to empty Twitter sentiment inputs when compatibility is not available

### 4.2 `apps/`

- `apps/batch_train/train_tabular.py`: pooled tabular model training entrypoint.
- `apps/data/build_sentiment_inputs.py`: builds `news_sentiment.csv`, `twitter_sentiment.csv`, and `sector_sentiment.csv` for the sentiment fusion pipeline.
- `apps/data/build_sentiment_inputs.py --require-twitter`: enforces Python 3.11 compatibility for Twitter sentiment generation.
- `apps/live_trading/live_signal_engine.py`: combines online feature state and the signal generator to produce live trading signals.
- `apps/live_trading/run_live_signals.py`: CLI wrapper for live signal streaming.
- `apps/common/input_loaders.py`: helper to read optional CSV, Parquet, or JSON inputs.

### 4.3 `core/`

- `schemas.py`: dataclasses for `AssetMetadata`, `FeatureRow`, `TradeSignal`, `Position`, `PortfolioSnapshot`.
- `events.py`: market, signal, order, and fill event dataclasses.
- `universe.py`: parses universe definitions from config or YAML and standardizes symbol metadata.

### 4.4 `configs/`

- `configs/universes/nse_largecap.yaml`: starter multi-asset universe for NSE large-cap symbols.

### 4.5 `data/`

- `data/providers/market/multi_asset_loader.py`: downloads multiple symbols and combines them into a normalized market panel.
- `data/storage/gold/panel_dataset_builder.py`: transforms raw multi-asset market data into a feature-enriched research/training panel.
- `data/sentiment/*.csv`: default sentiment input files used by the fusion pipeline. These are lightweight CSV placeholders or generated artifacts.

### 4.6 `data_pipeline/`

Legacy provider and preprocessing modules remain here.

- `market_data/nse_downloader.py`: single-symbol historical downloader using `yfinance`. Used by the legacy single-asset path and also indirectly by the multi-asset loader.
- `market_data/realtime_stream.py`: polling-based live market stream wrapper.
- `news_data/news_collector.py`, `rss_parser.py`, `news_cleaner.py`: collect and clean RSS-based news.
- `twitter_data/twitter_collector.py`, `tweet_cleaner.py`: collect and clean tweets.
- `news_data/news_sentiment_exporter.py`: converts collected news into symbol-level sentiment rows.
- `twitter_data/twitter_sentiment_exporter.py`: converts collected tweets into symbol-level sentiment rows.
- `data_merger.py`: older merge helper for market, news, and Twitter data.

### 4.7 `feature_engineering/`

Legacy feature modules, still used as the base layer in the new research pipeline.

- `feature_pipeline.py`: original feature pipeline composition.
- `indicators/`: trend, momentum, volatility, and volume indicators.
- `price_action/`: support/resistance, breakout, and candlestick pattern features.
- `regime_detection/`: trend, volatility, and momentum regime modules.
- `sentiment_features/`: older sentiment utility modules.

Important note:

- The new runtime does not primarily depend on `feature_engineering/sentiment_features/sentiment_aggregator.py` for multi-asset modeling.
- The current fused sentiment path is implemented in `features/sentiment/sentiment_fusion.py`.

### 4.8 `features/`

This is the new feature-system layer.

- `registry.py`: simple feature transform registry.
- `pipelines/research_feature_pipeline.py`: wraps the legacy base pipeline and appends advanced feature families.
- `technical/advanced_features.py`: VWAP, ATR ratio, Bollinger width, rolling volatility, and multi-horizon returns.
- `market_microstructure/advanced_features.py`: gap ratio, range efficiency, close location, volume z-score, price impact proxy, overnight reversal.
- `regime/regime_aware_features.py`: numeric regime encoding and regime-adjusted feature interactions.
- `sentiment/sentiment_fusion.py`: symbol-level plus sector-level sentiment fusion and smoothing.
- `store/online_feature_state.py`: rolling feature state for live signal generation.

### 4.9 `ml_models/`

- `autogluon/autogluon_trainer.py`: trains AutoGluon models from a supervised feature frame.
- `autogluon/autogluon_predictor.py`: loads `predictor.pkl` and adds `ml_prediction` to a dataframe.
- `classical_models/`: classical model wrappers.
- `feature_selector.py`: feature-selection helper.

### 4.10 `models/`

- `models/rl/trading_env.py`: Gymnasium-compatible RL trading environment scaffold.
- `models/rl/stable_baselines_agent.py`: wrapper around Stable-Baselines3 algorithms.

Note:

- `models/autogluon/` is intentionally ignored in git. Generated model artifacts remain local and should not be committed.

### 4.11 `strategy_engine/`

- `strategies/`: trend-following, mean reversion, breakout, and momentum strategy modules.
- `strategy_scoring.py`: combines strategy outputs into `strategy_score`.
- `strategy_backtester.py`: legacy facade that now delegates to the advanced backtester.
- `strategy_selector.py`: older strategy-selection helper.

### 4.12 `decision_engine/`

- `decision_model.py`: converts strategy, ML, and fused sentiment into final trade decisions.
- `signal_generator.py`: adds `final_signal` to a dataframe and supports latest-row live signal generation.
- `risk_manager.py`: dynamic sizing, risk parity, capped Kelly sizing, stop-loss/take-profit checks, and max drawdown protection.
- `portfolio_allocator.py`: single-asset allocator plus newer multi-asset, sentiment-aware portfolio constructor.

### 4.13 `backtesting/`

- `engine/advanced_backtester.py`: advanced event-like portfolio simulation with turnover, costs, slippage, rebalancing, exit rules, and metrics.

### 4.14 `experiment_tracking/`

- `mlflow_tracker.py`: lazy-load MLflow tracker wrapper.
- `model_registry.py`: registry placeholder.
- `experiment_logger.py`: basic experiment logger.

### 4.15 `database/`

- `duckdb/duckdb_engine.py`: DuckDB wrapper with table-name validation and dataframe registration.
- `influxdb/`: live time-series storage support.
- `storage_manager.py`: storage helper.

### 4.16 `dashboards/`

- `trading_dashboard.py`
- `monitoring_dashboard.py`

These remain part of the application surface, but they are not yet deeply wired into the new architecture.

### 4.17 `research/`

- notebooks, Lux-based exploratory analysis, and visualization utilities.

### 4.18 `tests/`

Contains both original and newly added tests. The project currently includes tests for:

- universe handling
- DuckDB safety
- feature pipelines
- decision model behavior
- AutoGluon trainer/predictor wrappers
- panel dataset building
- advanced backtesting
- online feature state
- RL trading environment scaffold
- sentiment fusion and sentiment exporters
- multi-asset `main.py` path

## 5. Offline Research and Training Dataflow

The main research and training flow is now multi-asset by default.

### 5.1 Training flow summary

1. Load config from `config.yaml`.
2. Build a universe using `UniverseManager`.
3. Download market data for every symbol in the universe using `MultiAssetMarketLoader`.
4. Build a feature panel using `PanelDatasetBuilder`.
5. Optionally load sentiment CSV inputs and fuse them into the panel.
6. Generate supervised labels using forward returns.
7. Train AutoGluon on the pooled multi-asset frame.

### 5.2 Training entrypoint

Primary entrypoint:

- `apps/batch_train/train_tabular.py`

Compatibility entrypoint:

- `deployment/train_autogluon.py`

### 5.3 Training CLI example

```bash
python deployment/train_autogluon.py
```

Multi-asset with explicit universe and sentiment inputs:

```bash
python apps/batch_train/train_tabular.py \
  --config config.yaml \
  --universe-path configs/universes/nse_largecap.yaml \
  --news-sentiment-path data/sentiment/news_sentiment.csv \
  --twitter-sentiment-path data/sentiment/twitter_sentiment.csv \
  --sector-sentiment-path data/sentiment/sector_sentiment.csv
```

### 5.4 Label generation

`PanelDatasetBuilder.build_training_frame()` currently creates a multiclass label named `target_return` based on forward return over a configurable horizon.

Logic:

- `1` if forward return > threshold
- `-1` if forward return < negative threshold
- `0` otherwise

This turns the training problem into a pooled multiclass classification task.

## 6. Runtime Inference Dataflow

`main.py` is the current main application entrypoint.

### 6.1 Mode selection

`config.yaml` controls which runtime path is used:

- `pipeline.mode: single_asset` runs the older single-symbol path
- `pipeline.mode: multi_asset` runs the newer research-grade multi-asset path

Current default:

- `multi_asset`

### 6.2 Multi-asset runtime flow

1. Read config.
2. Resolve the configured universe.
3. Download market data for all symbols.
4. Build the research feature panel.
5. Load optional sentiment CSV inputs.
6. Fuse sentiment into the panel.
7. Add ML predictions if `models/autogluon/predictor.pkl` exists.
8. Generate `final_signal` from strategy score, ML output, and sentiment.
9. Construct a portfolio using risk-aware and sentiment-aware weights.
10. Run advanced backtesting if enabled.

### 6.3 Single-asset runtime flow

The older path still exists for compatibility:

1. Download one symbol using `NSEDownloader`
2. Run `FeaturePipeline`
3. Compute `strategy_score`
4. Optionally add `ml_prediction`
5. Generate `final_signal`
6. Allocate capital using the legacy single-position allocator
7. Optionally run the advanced backtester on top of the single-symbol output

## 7. Market Dataflow

### 7.1 Historical market data

Provider:

- `data_pipeline/market_data/nse_downloader.py`

Behavior:

- uses `yfinance` lazily
- configures a local cache in `.cache/yfinance`
- supports single-symbol historical downloads
- normalizes columns to `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

### 7.2 Multi-asset market panel

Provider:

- `data/providers/market/multi_asset_loader.py`

Behavior:

- iterates across every asset in the universe
- downloads each symbol separately
- appends `symbol` and `sector`
- returns one combined dataframe sorted by `Date` and `symbol`

## 8. Feature Engineering Details

### 8.1 Legacy base features

The new research feature pipeline starts from the old `feature_engineering/feature_pipeline.py`, which still computes:

- moving averages
- MACD
- ADX
- RSI
- stochastic indicators
- CCI
- ROC
- Bollinger Bands
- ATR
- OBV
- VWAP
- support and resistance
- breakout and candlestick patterns
- trend classification
- volatility regime classification
- momentum score

### 8.2 Advanced technical features

Added in `features/technical/advanced_features.py`:

- `VWAP` if not already present
- `ATR` if not already present
- `bollinger_band_width`
- `atr_to_close`
- `rolling_vol_5`
- `rolling_vol_20`
- `return_1d`
- `return_3d`
- `return_5d`
- `return_10d`
- `vwap_deviation`

### 8.3 Market microstructure features

Added in `features/market_microstructure/advanced_features.py`:

- `gap_ratio`
- `intraday_range_efficiency`
- `close_location_value`
- `volume_zscore_20`
- `price_impact_proxy`
- `overnight_reversal`

### 8.4 Regime-aware features

Added in `features/regime/regime_aware_features.py`:

- `trend_regime_code`
- `volatility_regime_code`
- `regime_adjusted_momentum`
- `regime_adjusted_return_5d`
- `regime_pressure_score`

### 8.5 Cross-sectional features

Added in `PanelDatasetBuilder.build_panel()`:

- `sector_return_mean_1d`
- `sector_relative_return_1d`
- `cross_sectional_momentum_rank`
- `symbol_code`
- `sector_code`

## 9. Sentiment System

The sentiment system has two layers:

- legacy sentiment scoring helpers in `feature_engineering/sentiment_features/`
- new multi-asset fusion and ingestion pipeline in `features/sentiment/` and `apps/data/`

### 9.1 Raw sentiment sources

News:

- RSS feeds are parsed by `data_pipeline/news_data/rss_parser.py`
- items are collected by `data_pipeline/news_data/news_collector.py`
- text is cleaned by `data_pipeline/news_data/news_cleaner.py`
- news polarity is scored by `feature_engineering/sentiment_features/news_sentiment.py`

Twitter:

- tweets are scraped by `data_pipeline/twitter_data/twitter_collector.py`
- text is cleaned by `data_pipeline/twitter_data/tweet_cleaner.py`
- tweet polarity is scored by `feature_engineering/sentiment_features/twitter_sentiment.py`

### 9.2 Sentiment exporters

`apps/data/build_sentiment_inputs.py` builds three CSV files:

- `data/sentiment/news_sentiment.csv`
- `data/sentiment/twitter_sentiment.csv`
- `data/sentiment/sector_sentiment.csv`

Expected schemas:

`news_sentiment.csv`

| Column | Meaning |
| --- | --- |
| `Date` | publication timestamp |
| `symbol` | mapped asset symbol |
| `sector` | asset sector |
| `sentiment` | polarity score |
| `source` | source label, usually `news` |
| `text` | cleaned text/title |
| `link` | source URL |

`twitter_sentiment.csv`

| Column | Meaning |
| --- | --- |
| `Date` | tweet timestamp |
| `symbol` | mapped asset symbol |
| `sector` | asset sector |
| `sentiment` | polarity score |
| `source` | source label, usually `twitter` |
| `text` | cleaned tweet text |

`sector_sentiment.csv`

| Column | Meaning |
| --- | --- |
| `Date` | daily timestamp |
| `sector` | sector name |
| `sentiment` | average sector sentiment |

### 9.3 Sentiment fusion logic

`features/sentiment/sentiment_fusion.py` does the following:

1. validates the input schemas
2. aggregates symbol-level sentiment by `Date` and `symbol`
3. aggregates sector-level sentiment by `Date` and `sector`
4. merges all sentiment sources into the market panel
5. fills missing values with neutral defaults
6. smooths symbol and sector sentiment using a rolling window
7. computes fused features including:
   - `news_sentiment`
   - `twitter_sentiment`
   - `sector_sentiment`
   - `news_sentiment_volume`
   - `twitter_sentiment_volume`
   - `sector_sentiment_volume`
   - `sentiment_divergence`
   - `sentiment_alignment`
   - `sentiment_confidence`
   - `sentiment_composite`
   - `sentiment_momentum`
   - `sentiment_regime_interaction`

Default fusion weights:

- news: `0.45`
- twitter: `0.35`
- sector: `0.20`

## 10. ML Layer

### 10.1 AutoGluon training

File:

- `ml_models/autogluon/autogluon_trainer.py`

Behavior:

- lazy-loads AutoGluon
- validates that the label exists
- creates the model directory if missing
- fits a `TabularPredictor`
- supports multiclass training by default

### 10.2 AutoGluon inference

File:

- `ml_models/autogluon/autogluon_predictor.py`

Behavior:

- checks for `predictor.pkl`
- lazy-loads the AutoGluon predictor
- appends predictions into the dataframe as `ml_prediction`

### 10.3 Current modeling style

The system currently treats AutoGluon as a pooled multi-asset tabular classifier trained on a combined feature panel. It is not yet a market-neutral ranking model or a full alpha model with explicit train, validation, and out-of-sample time partitions beyond AutoGluon's internal split.

## 11. Signal and Decision Flow

### 11.1 Strategy scores

`strategy_engine/strategy_scoring.py` combines four underlying strategy outputs into a single `strategy_score`:

- trend following: weight `0.4`
- mean reversion: weight `0.2`
- breakout: weight `0.2`
- momentum: weight `0.2`

### 11.2 Decision model

`decision_engine/decision_model.py` builds the final decision score from:

- `strategy_score`
- `ml_prediction` if present and finite
- `sentiment_composite` if present and finite, scaled by `sentiment_confidence`

Decision rule:

- score > `0.5` -> `BUY`
- score < `-0.5` -> `SELL`
- otherwise -> `HOLD`

### 11.3 Signal generator

`decision_engine/signal_generator.py` adds `final_signal` row by row and supports latest-row generation for live mode.

## 12. Portfolio Construction and Risk

### 12.1 Risk manager

`decision_engine/risk_manager.py` currently supports:

- dynamic position sizing
- volatility-adjusted sizing
- capped Kelly fraction
- risk parity weights
- stop-loss / take-profit exit checks
- max drawdown protection

Key parameters:

- `risk_per_trade`
- `max_drawdown_limit`
- `max_position_weight`
- `volatility_target`

### 12.2 Portfolio allocator

`decision_engine/portfolio_allocator.py` has two behaviors.

Single-asset mode:

- simple buy, sell, hold simulation for one instrument

Multi-asset mode:

- converts `final_signal` into directional intent
- uses risk parity over estimated volatility
- applies conviction based on `strategy_score`, `ml_prediction`, and `sentiment_composite`
- adds sentiment alignment tilt
- clips per-position weights
- caps gross exposure
- computes `portfolio_weight` and `target_position_units`

## 13. Advanced Backtesting

`backtesting/engine/advanced_backtester.py` is the main research simulator.

Supported behaviors:

- portfolio-level daily simulation over grouped dates
- transaction costs in basis points
- slippage in basis points
- periodic rebalancing
- drawdown-based circuit breaker behavior
- stop-loss and take-profit checks
- risk-parity allocation fallback if explicit portfolio weights are not provided

If a dataframe already contains `portfolio_weight` or `target_weight`, the backtester uses those instead of recomputing target weights. This allows portfolio construction and backtesting to remain separate stages.

Computed metrics:

- `total_return`
- `annualized_return`
- `sharpe_ratio`
- `sortino_ratio`
- `max_drawdown`
- `win_rate`

Backtest output columns include:

- `portfolio_value`
- `drawdown`
- `gross_exposure`
- `turnover`
- `transaction_cost`

## 14. Live Signal Path

The live path is a scaffold for near real-time signal generation.

Important files:

- `data_pipeline/market_data/realtime_stream.py`
- `features/store/online_feature_state.py`
- `apps/live_trading/live_signal_engine.py`
- `apps/live_trading/run_live_signals.py`

Flow:

1. `RealtimeStreamer` polls recent intraday bars.
2. `OnlineFeatureState` maintains a rolling buffer per symbol.
3. It computes lightweight rolling features such as returns, VWAP deviation, volatility, rolling highs/lows, and temporary strategy signals.
4. `LiveSignalEngine` passes the latest row into `SignalGenerator.generate_latest()`.
5. The CLI prints the resulting signal payload.

Important note:

- This path is near real-time polling, not broker-grade event streaming.
- There is no broker execution adapter yet.
- This is a research/live-signal scaffold, not full automated execution.

## 15. Reinforcement Learning Scaffold

The RL layer is intentionally early-stage.

### 15.1 Environment

File:

- `models/rl/trading_env.py`

Behavior:

- Gymnasium-compatible environment if `gymnasium` is installed
- observation is a vector of selected feature columns
- action space is discrete with 3 actions
  - `0`: flat or no action
  - `1`: long
  - `2`: short
- reward is based on change in portfolio value minus transaction cost

### 15.2 Agent wrapper

File:

- `models/rl/stable_baselines_agent.py`

Behavior:

- lazy-loads Stable-Baselines3
- instantiates a selected algorithm such as `PPO`
- trains on the provided environment
- supports inference with `predict()`

Important note:

- This is scaffolding only.
- There is not yet a full RL training pipeline, reward engineering framework, benchmark suite, or policy evaluation framework.

## 16. Configuration Reference

Current runtime config is defined in `config.yaml`.

### 16.1 Core keys

`symbol`

- single-symbol fallback used by the legacy path

`universe.path`

- path to a universe YAML file

`pipeline.mode`

- `single_asset` or `multi_asset`

`pipeline.use_research_features`

- enables the new research feature pipeline for the single-asset path

`data.start_date`, `data.end_date`

- historical market-data window

`model.autogluon_path`

- model artifact directory, must contain `predictor.pkl` for inference

### 16.2 Portfolio and risk keys

`trading.initial_capital`

- starting portfolio value

`trading.risk_per_trade`

- risk scaling input for sizing functions

`portfolio.sentiment_tilt_strength`

- how strongly sentiment influences portfolio weights

`portfolio.ml_tilt_strength`

- how strongly ML predictions influence portfolio weights

`portfolio.max_position_weight`

- cap per symbol

`portfolio.max_gross_exposure`

- gross leverage cap

### 16.3 Backtesting keys

- `backtesting.enabled`
- `backtesting.transaction_cost_bps`
- `backtesting.slippage_bps`
- `backtesting.rebalance_frequency`
- `backtesting.stop_loss_pct`
- `backtesting.take_profit_pct`
- `backtesting.max_drawdown_pct`

### 16.4 Sentiment keys

- `sentiment.news_path`
- `sentiment.twitter_path`
- `sentiment.sector_path`

These files are optional. If absent and `missing_ok=True`, the pipeline falls back to neutral sentiment.

## 17. Command Reference

### 17.1 Run the default application

```bash
python main.py
```

### 17.2 Train AutoGluon

```bash
python deployment/train_autogluon.py
```

### 17.3 Build sentiment inputs

```bash
python deployment/build_sentiment_inputs.py
```

### 17.4 Run the live signal scaffold

```bash
python apps/live_trading/run_live_signals.py --symbol RELIANCE --interval 10 --limit 5
```

### 17.5 Run tests

```bash
python -m pytest -q
```

## 18. Dependency Groups

From `requirements.txt`, the main dependency categories are:

Core data and modeling:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `autogluon`

Market and storage:

- `yfinance`
- `duckdb`
- `influxdb-client`

Feature and sentiment tooling:

- `ta`
- `textblob`
- `feedparser`
- `snscrape`

NLP and embeddings:

- `transformers`
- `sentence-transformers`

Research and dashboards:

- `streamlit`
- `matplotlib`
- `lux-api`

Experiment and operations:

- `mlflow`
- `psutil`

RL:

- `gymnasium`
- `stable-baselines3`

## 19. Testing and Quality

The repository includes regression tests for the recent refactor work. This means:

- architecture additions are not just placeholders
- multi-asset path is covered in tests
- new sentiment fusion behavior is covered in tests
- advanced backtesting behavior is covered in tests
- live feature state and RL environment scaffolding are covered in tests

At the time of the latest refactor pass, the test suite passed locally with:

```bash
python -m pytest -q
```

## 20. Known Limitations

This project is significantly stronger than the original version, but it is still a research platform rather than a production trading stack. Important limitations include:

1. Execution is not broker-integrated.
2. Real-time streaming is polling-based, not event-bus-based.
3. Data providers are still lightweight and public-source driven.
4. Universe management is static and YAML-based.
5. Train/validation/test protocol is still simpler than institutional walk-forward research.
6. ML predictions are class labels, not calibrated probability-aware portfolio scores.
7. Experiment tracking and model registry are still minimal wrappers.
8. Some legacy modules coexist with newer ones and may overlap in purpose.
9. `models/autogluon/` is local artifact storage and should not be treated as source code.

## 21. Legacy vs New Architecture

The repository currently has both old and new layers.

### Legacy-first modules

- `feature_engineering/`
- `data_pipeline/`
- `strategy_engine/`
- `decision_engine/`
- `ml_models/`

### Newer architecture modules

- `core/`
- `data/`
- `features/`
- `backtesting/`
- `apps/`
- `models/rl/`

Current design intent:

- keep legacy code operational
- move new research-grade work into the new architecture
- gradually rewire entrypoints toward the new stack
- avoid a destructive rewrite

## 22. Recommended Next Refactor Priorities

If development continues, the highest-value next steps are:

1. Replace remaining eager dependencies and legacy import coupling.
2. Introduce explicit train, validation, and out-of-sample walk-forward splits.
3. Add model probability support and confidence-aware signal scaling.
4. Add broker adapters and order-state management.
5. Expand sentiment entity linking beyond simple symbol string matching.
6. Add richer experiment tracking and model lineage.
7. Add production data quality checks and schema validation.
8. Push more runtime code from `main.py` into `apps/` services.

## 23. Quick Mental Model

If you need one short summary of how the platform works today, it is this:

- market data is collected per symbol
- symbols are combined into a panel
- the panel is enriched with legacy features plus advanced technical, regime, microstructure, and sentiment features
- strategy signals and ML predictions are combined into final trading decisions
- portfolio weights are constructed with risk and sentiment awareness
- the resulting portfolio is simulated through an advanced backtester
- optional live-mode code can generate near real-time signals from rolling market state

This is the current operational center of the project.
