import pandas as pd

from explainability.shap_explainer import ShapExplainer


class FakeAutoGluonPredictor:

    def feature_importance(self, X):

        return pd.DataFrame({
            "feature": list(X.columns),
            "importance": [0.7, 0.3],
        })


def test_shap_explainer_falls_back_for_non_callable_predictor():

    explainer = ShapExplainer()
    frame = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0],
        "feature_b": [10.0, 11.0, 12.0],
    })

    report = explainer.explain(FakeAutoGluonPredictor(), frame)

    assert not report.feature_importance.empty
    assert list(report.feature_importance["feature"])[:2] == ["feature_a", "feature_b"]
