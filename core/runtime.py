import sys


TARGET_PYTHON_MAJOR = 3
TARGET_PYTHON_MINOR = 11
MIN_TWITTER_PYTHON_MAJOR = 3
MIN_TWITTER_PYTHON_MINOR = 10


def version_tuple(version_info=None):

    info = version_info or sys.version_info

    if hasattr(info, "major"):
        return info.major, info.minor, info.micro

    values = tuple(info)
    if len(values) < 3:
        raise ValueError("version_info must provide at least major, minor, and micro values.")

    return int(values[0]), int(values[1]), int(values[2])


def format_version(version_info=None):

    major, minor, micro = version_tuple(version_info)

    return f"{major}.{minor}.{micro}"


def is_target_python(version_info=None):

    major, minor, _ = version_tuple(version_info)

    return (major, minor) == (TARGET_PYTHON_MAJOR, TARGET_PYTHON_MINOR)


def ensure_twitter_runtime_supported(version_info=None):

    major, minor, _ = version_tuple(version_info)

    if (major, minor) >= (MIN_TWITTER_PYTHON_MAJOR, MIN_TWITTER_PYTHON_MINOR):
        return

    raise RuntimeError(
        "Twitter sentiment collection requires Python 3.10+ so the scraping "
        "dependencies can be imported reliably. "
        f"Current runtime: Python {format_version(version_info)}. "
        "Use Python 3.10 or newer to enable Twitter sentiment inputs."
    )
