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
