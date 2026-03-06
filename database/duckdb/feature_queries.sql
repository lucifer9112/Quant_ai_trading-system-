-- SQL for feature extraction queries
SELECT
    Date,
    Close,
    SMA20,
    RSI,
    MACD
FROM market_features
WHERE Date > '2023-01-01'