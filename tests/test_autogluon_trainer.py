import builtins
import sys
import types

import pandas as pd
import pytest

from ml_models.autogluon.autogluon_trainer import AutoGluonTrainer
from validation.walk_forward_validator import WalkForwardValidator


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


def test_walk_forward_validate_handles_non_contiguous_source_index(tmp_path):

    class FakeTabularPredictor:

        def __init__(self, label, path, problem_type, eval_metric):

            self.label = label

        def fit(self, train_data, time_limit, presets):

            return self

        def predict(self, data):

            return pd.Series([0] * len(data), index=data.index)

    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-04", "2024-01-02"]
            ),
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "target_return": [0, 0, 0, 0],
        },
        index=[100, 200, 300, 400],
    )

    trainer = AutoGluonTrainer(model_path=str(tmp_path / "autogluon-model"))
    trainer._tabular_predictor = FakeTabularPredictor

    summary = trainer.walk_forward_validate(
        frame,
        validator=WalkForwardValidator(initial_window=2, forecast_horizon=1, step_size=1),
        time_limit=1,
        presets="medium_quality_faster_train",
    )

    assert len(summary["folds"]) == 2
    assert summary["mean_accuracy"] == 1.0
