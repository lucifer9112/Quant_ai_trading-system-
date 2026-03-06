class MLFlowTracker:

    def __init__(self):

        self._mlflow = None

    def _client(self):

        if self._mlflow is None:
            try:
                import mlflow
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "mlflow is not installed. Install it with `pip install mlflow`."
                ) from exc

            self._mlflow = mlflow

        return self._mlflow

    def start_run(self):

        self._client().start_run()

    def log_metric(self, name, value):

        self._client().log_metric(name, value)

    def end_run(self):

        self._client().end_run()
