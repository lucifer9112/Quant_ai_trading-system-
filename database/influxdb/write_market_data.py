try:
    from .influx_client import InfluxDBClientManager
except ImportError:
    from database.influxdb.influx_client import InfluxDBClientManager


class MarketDataWriter:

    def __init__(self, influx_client):

        self.client = influx_client

    def write(self, df):

        self.client.write_dataframe(
            df,
            measurement="market_prices"
        )
