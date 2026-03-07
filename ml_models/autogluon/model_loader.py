class ModelLoader:

    def load(self, path):

        try:
            from autogluon.tabular import TabularPredictor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "autogluon is not installed. Install it with `pip install autogluon.tabular`."
            ) from exc

        return TabularPredictor.load(path)
