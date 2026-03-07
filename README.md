# Quant AI Trading System

This project is a modular AI-driven trading research platform.

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
