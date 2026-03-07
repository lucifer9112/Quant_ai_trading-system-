# Quant AI Trading System

This project is a modular AI-driven trading research platform.

## Python Version

- The repository target runtime is Python `3.11`.
- The core market, training, and backtesting flows can run on newer versions in some environments.
- The full Twitter sentiment pipeline is only supported on Python `3.11` because `snscrape` is not reliable on Python `3.12+`.

Recommended setup:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Features

- Market data ingestion
- News & Twitter sentiment
- Technical indicators
- Strategy engine
- ML predictions
- Decision engine
- Portfolio simulation
- Dashboards
- Docker deployment

## Full Documentation

See docs/PROJECT_HANDBOOK.md for the full architecture, dataflow, module, configuration, and workflow reference.

## Run

python main.py

The default config now uses the multi-asset research path with advanced backtesting enabled.

## Train AutoGluon Model

Train and save the model artifacts expected by `main.py`:

python deployment/train_autogluon.py

Optional arguments:

- `--symbol RELIANCE`
- `--horizon 1`
- `--threshold 0.002`
- `--time-limit 600`
- `--model-path models/autogluon`

After training, run:

python main.py

## Build Sentiment Inputs

Generate the sentiment CSV files consumed by the fusion pipeline:

python apps/data/build_sentiment_inputs.py

This writes:

- `data/sentiment/news_sentiment.csv`
- `data/sentiment/twitter_sentiment.csv`
- `data/sentiment/sector_sentiment.csv`

If Twitter collection is unavailable or `snscrape` is incompatible with the current Python runtime, the command will still complete and write empty Twitter sentiment inputs instead of failing the whole pipeline.

If you want Twitter sentiment to be mandatory, use:

```bash
python apps/data/build_sentiment_inputs.py --require-twitter
```

On unsupported runtimes this fails early with an explicit Python 3.11 requirement instead of silently falling back.

