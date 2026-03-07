import pytest

from core.runtime import ensure_twitter_runtime_supported, is_target_python


def test_is_target_python_accepts_python_311():

    assert is_target_python((3, 11, 9))


def test_is_target_python_rejects_python_312():

    assert not is_target_python((3, 12, 0))


def test_ensure_twitter_runtime_supported_raises_for_python_312():

    with pytest.raises(RuntimeError, match="Python 3.11"):
        ensure_twitter_runtime_supported((3, 12, 12))
