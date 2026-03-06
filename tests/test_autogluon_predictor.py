import builtins
import sys
import types

import pytest

from ml_models.autogluon.autogluon_predictor import AutoGluonPredictor


class FakeFrame(dict):

    def copy(self):

        return FakeFrame(self)


def test_predict_adds_ml_prediction_and_caches_loaded_model(monkeypatch, tmp_path):

    load_calls = []

    class FakeLoadedPredictor:

        def predict(self, df):

            return [0.2, -0.1]

    class FakeTabularPredictor:

        @staticmethod
        def load(path):

            load_calls.append(path)
            return FakeLoadedPredictor()

    autogluon_module = types.ModuleType("autogluon")
    tabular_module = types.ModuleType("autogluon.tabular")
    tabular_module.TabularPredictor = FakeTabularPredictor
    autogluon_module.tabular = tabular_module

    monkeypatch.setitem(sys.modules, "autogluon", autogluon_module)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular_module)

    model_path = tmp_path / "test-model"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "predictor.pkl").write_text("stub", encoding="utf-8")

    predictor = AutoGluonPredictor(model_path=str(model_path))
    input_frame = FakeFrame({"Close": [100, 102]})

    result1 = predictor.predict(input_frame)
    result2 = predictor.predict(input_frame)

    assert load_calls == [str(model_path)]
    assert result1["ml_prediction"] == [0.2, -0.1]
    assert result2["ml_prediction"] == [0.2, -0.1]


def test_predict_raises_clear_error_when_autogluon_missing(monkeypatch, tmp_path):

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "autogluon.tabular":
            raise ModuleNotFoundError("No module named 'autogluon'")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "autogluon", raising=False)
    monkeypatch.delitem(sys.modules, "autogluon.tabular", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    model_path = tmp_path / "test-model"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "predictor.pkl").write_text("stub", encoding="utf-8")

    predictor = AutoGluonPredictor(model_path=str(model_path))

    with pytest.raises(RuntimeError, match="autogluon is not installed"):
        predictor.predict(FakeFrame({"Close": [100]}))


def test_predict_raises_clear_error_when_model_artifact_missing(tmp_path):

    predictor = AutoGluonPredictor(model_path=str(tmp_path / "missing-model"))

    with pytest.raises(FileNotFoundError, match="AutoGluon model not found"):
        predictor.predict(FakeFrame({"Close": [100]}))
