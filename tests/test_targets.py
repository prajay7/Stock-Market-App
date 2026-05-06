import pandas as pd

from src.features.targets import add_classification_target, add_regression_target


def test_targets_are_future_shifted():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "close": [100, 102, 101, 103, 104],
        }
    )
    cdf = add_classification_target(df, horizon_days=1)
    rdf = add_regression_target(df, horizon_days=1)

    assert "target_up_1d" in cdf.columns
    assert "target_return_1d" in rdf.columns
    assert "target_close_1d" in rdf.columns
    # Last row has no forward value
    assert pd.isna(cdf.iloc[-1]["target_up_1d"])
    assert pd.isna(rdf.iloc[-1]["target_return_1d"])
    assert pd.isna(rdf.iloc[-1]["target_close_1d"])
