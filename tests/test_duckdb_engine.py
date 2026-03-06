import pytest

from database.duckdb.duckdb_engine import DuckDBEngine


class FakeConnection:

    def __init__(self):

        self.registered = []
        self.executed = []
        self.unregistered = []

    def register(self, name, dataframe):

        self.registered.append((name, dataframe))

    def execute(self, sql):

        self.executed.append(sql)
        return self

    def unregister(self, name):

        self.unregistered.append(name)

    def close(self):

        return None


def test_store_dataframe_rejects_invalid_table_name():

    engine = DuckDBEngine(connection=FakeConnection())

    with pytest.raises(ValueError):
        engine.store_dataframe([{"Close": 10}], "prices;DROP")


def test_store_dataframe_uses_registered_temp_view():

    connection = FakeConnection()
    engine = DuckDBEngine(connection=connection)

    dataframe = [{"Close": 10}]

    engine.store_dataframe(dataframe, "daily_prices")

    assert connection.registered == [("__input_df", dataframe)]
    assert connection.executed == [
        'CREATE OR REPLACE TABLE "daily_prices" AS SELECT * FROM "__input_df"'
    ]
    assert connection.unregistered == ["__input_df"]


def test_append_dataframe_uses_safe_table_reference():

    connection = FakeConnection()
    engine = DuckDBEngine(connection=connection)

    engine.append_dataframe([{"Close": 20}], "daily_prices")

    assert connection.executed == [
        'INSERT INTO "daily_prices" SELECT * FROM "__input_df"'
    ]
