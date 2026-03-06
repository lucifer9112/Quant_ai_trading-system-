import builtins
import sys
import types

import pytest

from experiment_tracking.mlflow_tracker import MLFlowTracker


def test_mlflow_tracker_calls_mlflow_api(monkeypatch):

    calls = []
    mlflow_module = types.ModuleType("mlflow")

    def start_run():
        calls.append("start")

    def log_metric(name, value):
        calls.append(("metric", name, value))

    def end_run():
        calls.append("end")

    mlflow_module.start_run = start_run
    mlflow_module.log_metric = log_metric
    mlflow_module.end_run = end_run

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)

    tracker = MLFlowTracker()
    tracker.start_run()
    tracker.log_metric("accuracy", 0.9)
    tracker.end_run()

    assert calls == [
        "start",
        ("metric", "accuracy", 0.9),
        "end",
    ]


def test_mlflow_tracker_raises_clear_error_when_dependency_missing(monkeypatch):

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ModuleNotFoundError("No module named 'mlflow'")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    tracker = MLFlowTracker()

    with pytest.raises(RuntimeError, match="mlflow is not installed"):
        tracker.start_run()
