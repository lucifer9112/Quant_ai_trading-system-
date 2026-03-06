class StorageManager:

    def __init__(self, duckdb_engine, influx_client):

        self.duckdb = duckdb_engine
        self.influx = influx_client

    def store_market_data(self, df):

        self.duckdb.store_dataframe(df, "market_data")

        if self.influx:
            self.influx.write_dataframe(df)

    def query(self, sql):

        return self.duckdb.query(sql)