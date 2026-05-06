from pathlib import Path
from typing import Iterable, List

import pandas as pd


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(" ", "")


def normalize_symbols(symbols: Iterable[str]) -> List[str]:
    seen = set()
    normalized = []
    for symbol in symbols:
        cleaned = normalize_symbol(symbol)
        if cleaned and cleaned not in seen:
            normalized.append(cleaned)
            seen.add(cleaned)
    return normalized


def map_symbol_for_market(symbol: str, market: str = "us") -> str:
    cleaned = normalize_symbol(symbol)
    market_norm = market.strip().lower()

    if market_norm != "india":
        return cleaned

    if "." in cleaned:
        return cleaned

    # Default Indian NSE suffix if exchange not explicitly provided.
    return f"{cleaned}.NS"


def load_symbols_from_csv(
    csv_path: str,
    symbol_column: str | None = None,
    market: str = "us",
    series_filter: str | None = None,
    max_symbols: int | None = None,
) -> List[str]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV file is empty")

    work_df = df.copy()

    if series_filter:
        if "Series" not in work_df.columns:
            raise ValueError("series_filter provided but 'Series' column not found in CSV")
        allowed = {s.strip().upper() for s in series_filter.split(",") if s.strip()}
        work_df = work_df[work_df["Series"].astype(str).str.upper().isin(allowed)]

    if symbol_column:
        if symbol_column not in work_df.columns:
            raise ValueError(f"Column '{symbol_column}' not found in CSV. Available: {list(work_df.columns)}")
        source = work_df[symbol_column]
    else:
        candidates = ["symbol", "symbols", "ticker", "tickers", "stock", "stocks", "nse_code"]
        selected = next((c for c in candidates if c in work_df.columns), work_df.columns[0])
        source = work_df[selected]

    symbols = [map_symbol_for_market(str(v), market=market) for v in source.dropna().tolist()]
    normalized = normalize_symbols(symbols)
    if max_symbols is not None and max_symbols > 0:
        return normalized[:max_symbols]
    return normalized
