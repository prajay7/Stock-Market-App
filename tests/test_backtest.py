import pandas as pd

from src.backtesting.metrics import summary_metrics


def test_summary_metrics_sanity():
    strategy = pd.Series([0.01, -0.005, 0.02])
    benchmark = pd.Series([0.005, -0.002, 0.01])
    out = summary_metrics(strategy, benchmark)
    assert "cumulative_return" in out
    assert "max_drawdown" in out
    assert -1.0 <= out["max_drawdown"] <= 0.0
