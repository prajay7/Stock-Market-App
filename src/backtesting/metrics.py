from __future__ import annotations

import numpy as np
import pandas as pd


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0


def summary_metrics(returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, float]:
    if returns.empty:
        return {"cumulative_return": 0.0, "max_drawdown": 0.0, "hit_ratio": 0.0, "benchmark_cumulative_return": 0.0, "sharpe_like": 0.0}

    equity = (1 + returns).cumprod()
    bench_equity = (1 + benchmark_returns.fillna(0.0)).cumprod()
    dd = compute_drawdown(equity)

    sharpe_like = 0.0
    if returns.std(ddof=1) > 0:
        sharpe_like = float(np.sqrt(252) * returns.mean() / returns.std(ddof=1))

    return {
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "hit_ratio": float((returns > 0).mean()),
        "benchmark_cumulative_return": float(bench_equity.iloc[-1] - 1.0),
        "sharpe_like": sharpe_like,
    }
