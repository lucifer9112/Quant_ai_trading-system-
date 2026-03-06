import builtins
import sys
import types

import pytest

from ml_models.autogluon.autogluon_trainer import AutoGluonTrainer


class FakeFrame:

    def __init__(self, columns):

        self.columns = columns


def test_trainer_fits_and_creates_model_directory(monkeypatch, tmp_path):

    calls = {}

    class FakeTabularPredictor:

        def __init__(self, label, path, problem_type, eval_metric):

            calls["init"] = {
                "label": label,
                "path": path,
                "problem_type": problem_type,
                "eval_metric": eval_metric,
            }

        def fit(self, train_data, time_limit, presets):

            calls["fit"] = {
                "train_data": train_data,
                "time_limit": time_limit,
                "presets": presets,
            }
            return self

    autogluon_module = types.ModuleType("autogluon")
    tabular_module = types.ModuleType("autogluon.tabular")
    tabular_module.TabularPredictor = FakeTabularPredictor
    autogluon_module.tabular = tabular_module

    monkeypatch.setitem(sys.modules, "autogluon", autogluon_module)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular_module)

    model_path = tmp_path / "autogluon-model"
    trainer = AutoGluonTrainer(model_path=str(model_path))

    frame = FakeFrame(columns=["feature_a", "target_return"])
    trainer.train(frame, time_limit=30, presets="best_quality")

    assert model_path.exists()
    assert calls["init"] == {
        "label": "target_return",
        "path": str(model_path),
        "problem_type": "multiclass",
        "eval_metric": "accuracy",
    }
    assert calls["fit"] == {
        "train_data": frame,
        "time_limit": 30,
        "presets": "best_quality",
    }


def test_trainer_raises_on_missing_label_column(tmp_path):

    trainer = AutoGluonTrainer(model_path=str(tmp_path / "autogluon-model"))

    with pytest.raises(ValueError, match="label column"):
        trainer.train(FakeFrame(columns=["feature_a"]))


def test_trainer_raises_clear_error_when_autogluon_missing(monkeypatch, tmp_path):

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "autogluon.tabular":
            raise ModuleNotFoundError("No module named 'autogluon'")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "autogluon", raising=False)
    monkeypatch.delitem(sys.modules, "autogluon.tabular", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    trainer = AutoGluonTrainer(model_path=str(tmp_path / "autogluon-model"))

    with pytest.raises(RuntimeError, match="autogluon is not installed"):
        trainer.train(FakeFrame(columns=["target_return"]))
