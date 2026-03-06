import re


_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name):

    if not isinstance(name, str) or not _VALID_IDENTIFIER.fullmatch(name):
        raise ValueError(
            "Invalid table name. Use only letters, numbers, and underscores, "
            "and do not start with a number."
        )

    return name


class DuckDBEngine:

    def __init__(self, db_path="market_data.db", connection=None):

        if connection is not None:
            self.conn = connection
            return

        try:
            import duckdb
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "duckdb is not installed. Install it with `pip install duckdb`."
            ) from exc

        self.conn = duckdb.connect(db_path)

    def _execute_with_dataframe(self, df, table_name, statement):

        safe_table_name = _validate_identifier(table_name)

        temp_view = "__input_df"

        self.conn.register(temp_view, df)

        try:
            self.conn.execute(
                statement.format(
                    table=f'"{safe_table_name}"',
                    view=temp_view
                )
            )
        finally:
            if hasattr(self.conn, "unregister"):
                self.conn.unregister(temp_view)

    def store_dataframe(self, df, table_name):

        self._execute_with_dataframe(
            df,
            table_name,
            'CREATE OR REPLACE TABLE {table} AS SELECT * FROM "{view}"'
        )

    def append_dataframe(self, df, table_name):

        self._execute_with_dataframe(
            df,
            table_name,
            'INSERT INTO {table} SELECT * FROM "{view}"'
        )

    def query(self, sql):

        return self.conn.execute(sql).df()

    def close(self):

        self.conn.close()
