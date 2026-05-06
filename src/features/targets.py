from __future__ import annotations

import numpy as np
import pandas as pd


def add_classification_target(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    out = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    future_return = out.groupby("symbol")["close"].shift(-horizon_days) / out["close"] - 1.0
    out[f"target_up_{horizon_days}d"] = (future_return > 0).where(future_return.notna(), pd.NA)
    out[f"forward_return_{horizon_days}d"] = future_return
    return out


def add_regression_target(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    out = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    out[f"target_return_{horizon_days}d"] = out.groupby("symbol")["close"].shift(-horizon_days) / out["close"] - 1.0
    out[f"target_close_{horizon_days}d"] = out.groupby("symbol")["close"].shift(-horizon_days)
    return out


def add_movement_target(
    df: pd.DataFrame,
    horizon_days: int = 1,
    up_threshold: float = 0.01,
    down_threshold: float = 0.005,
) -> pd.DataFrame:
    out = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    target_col = f"target_movement_{horizon_days}n"
    values: list[object] = [pd.NA] * len(out)

    for _, group in out.groupby("symbol", sort=False):
        indices = list(group.index)
        highs = pd.to_numeric(group["high"], errors="coerce").to_numpy()
        lows = pd.to_numeric(group["low"], errors="coerce").to_numpy()
        closes = pd.to_numeric(group["close"], errors="coerce").to_numpy()

        for offset, idx in enumerate(indices):
            entry = closes[offset]
            if not np.isfinite(entry) or entry <= 0:
                continue

            upper = entry * (1.0 + float(up_threshold))
            lower = entry * (1.0 - float(down_threshold))
            future_highs = highs[offset + 1 : offset + 1 + horizon_days]
            future_lows = lows[offset + 1 : offset + 1 + horizon_days]

            if len(future_highs) == 0:
                continue

            label = 0
            for future_high, future_low in zip(future_highs, future_lows):
                if np.isfinite(future_high) and np.isfinite(future_low):
                    if future_high >= upper and future_low <= lower:
                        label = 0
                        break
                    if future_high >= upper:
                        label = 1
                        break
                    if future_low <= lower:
                        label = 0
                        break
            values[idx] = label

    out[target_col] = values
    out[f"target_up_{horizon_days}d"] = out[target_col]
    out[f"forward_return_{horizon_days}d"] = out.groupby("symbol")["close"].shift(-horizon_days) / out["close"] - 1.0
    return out


def add_outperform_target(df: pd.DataFrame, benchmark_symbol: str, horizon_days: int = 3) -> pd.DataFrame:
    out = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    out[f"forward_return_{horizon_days}d"] = out.groupby("symbol")["close"].shift(-horizon_days) / out["close"] - 1.0

    benchmark = out[out["symbol"] == benchmark_symbol][["date", f"forward_return_{horizon_days}d"]].rename(
        columns={f"forward_return_{horizon_days}d": "benchmark_forward_return"}
    )
    out = out.merge(benchmark, on="date", how="left")
    out[f"target_outperform_{horizon_days}d"] = (out[f"forward_return_{horizon_days}d"] > out["benchmark_forward_return"]).astype(int)
    return out
