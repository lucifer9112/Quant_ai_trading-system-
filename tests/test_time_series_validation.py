import pandas as pd

from validation.time_series_cv import PurgedWalkForwardSplitter, TimeSeriesSplitConfig


def test_purged_walk_forward_splitter_splits_on_unique_dates_for_panel_data():

    frame = pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-01-02", "2024-01-02",
            "2024-01-03", "2024-01-03",
            "2024-01-04", "2024-01-04",
        ]),
        "symbol": ["AAA", "BBB"] * 4,
        "Close": [100, 200, 101, 201, 102, 202, 103, 203],
    })

    splitter = PurgedWalkForwardSplitter(
        TimeSeriesSplitConfig(min_train_size=2, test_size=1, step_size=1)
    )

    folds = splitter.split(frame)

    assert len(folds) == 2
    assert set(frame.iloc[folds[0].val_idx]["Date"]) == {pd.Timestamp("2024-01-03")}
    assert set(frame.iloc[folds[1].val_idx]["Date"]) == {pd.Timestamp("2024-01-04")}
    assert frame.iloc[folds[0].train_idx]["Date"].max() < frame.iloc[folds[0].val_idx]["Date"].min()
