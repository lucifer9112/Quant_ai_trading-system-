from influxdb_client import Point


class SentimentWriter:

    def __init__(self, influx_client):

        self.client = influx_client

    def write(self, df):

        for _, row in df.iterrows():

            point = (
                Point("sentiment")
                .tag("symbol", row["symbol"])
                .field("sentiment", float(row["sentiment"]))
            )

            self.client.write_api.write(
                bucket=self.client.bucket,
                org=self.client.org,
                record=point
            )