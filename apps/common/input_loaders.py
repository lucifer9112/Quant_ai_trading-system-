from pathlib import Path

import pandas as pd


def load_optional_dataframe(path, base_dir=".", missing_ok=True):

    if not path:
        return None

    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (Path(base_dir) / file_path).resolve()

    if not file_path.exists():
        if missing_ok:
            return None
        raise FileNotFoundError(f"Optional input file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)

    raise ValueError(f"Unsupported optional input format: {file_path.suffix}")
