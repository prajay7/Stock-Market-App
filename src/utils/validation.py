from typing import Iterable

import pandas as pd


REQUIRED_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume", "symbol"]


def validate_ohlcv_frame(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")


def validate_non_empty(symbols: Iterable[str]) -> None:
    if not list(symbols):
        raise ValueError("At least one symbol is required")
