-- SQL for analytics queries
SELECT
    AVG(Close) as avg_price,
    MAX(Volume) as max_volume
FROM market_features