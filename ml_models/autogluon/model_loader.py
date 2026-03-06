from autogluon.tabular import TabularPredictor


class ModelLoader:

    def load(self, path):

        return TabularPredictor.load(path)