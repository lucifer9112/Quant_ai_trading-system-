from influxdb_client import InfluxDBClient, Point, WritePrecision
import pandas as pd


class InfluxDBClientManager:

    def __init__(self, url, token, org, bucket):

        self.bucket = bucket
        self.org = org

        self.client = InfluxDBClient(
            url=url,
            token=token,
            org=org
        )

        self.write_api = self.client.write_api()

    def write_dataframe(self, df, measurement="market_data"):

        for _, row in df.iterrows():

            point = (
                Point(measurement)
                .tag("symbol", row.get("symbol", "UNKNOWN"))
                .field("close", float(row["Close"]))
                .field("volume", float(row["Volume"]))
            )

            self.write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=point
            )