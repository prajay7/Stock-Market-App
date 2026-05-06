from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_equity_plot(backtest_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(backtest_df["date"], backtest_df["portfolio_equity"], label="Strategy")
    plt.plot(backtest_df["date"], backtest_df["benchmark_equity"], label="Benchmark", alpha=0.8)
    plt.title("Backtest Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
