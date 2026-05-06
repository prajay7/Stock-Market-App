from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from src.data.db import SQLiteDataStore
from src.data.cache import symbol_news_path, symbol_price_path
from src.data.storage import read_parquet_if_exists, write_parquet
from src.features.pipeline import build_feature_pipeline
from src.utils.symbols import normalize_symbols


def load_price_data(raw_dir: Path, symbols: list[str], interval: str = "1d") -> pd.DataFrame:
    settings = get_settings()
    store = SQLiteDataStore(settings.db_path)
    db_frames = []
    db_df = store.read_candles(normalize_symbols(symbols), interval)
    if not db_df.empty:
        db_frames.append(db_df)
    frames = []
    for symbol in normalize_symbols(symbols):
        df = read_parquet_if_exists(symbol_price_path(raw_dir, symbol, interval))
        if not df.empty:
            frames.append(df)
    if db_frames or frames:
        parts = db_frames + frames
        merged = pd.concat(parts, ignore_index=True, sort=False)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.dropna(subset=["date", "symbol"]).drop_duplicates(subset=["symbol", "date"], keep="last")
        return merged.sort_values(["symbol", "date"])
    return pd.DataFrame()


def load_news_data(raw_dir: Path, symbols: list[str]) -> pd.DataFrame:
    frames = []
    for symbol in normalize_symbols(symbols):
        df = read_parquet_if_exists(symbol_news_path(raw_dir, symbol))
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(["symbol", "published_at"])


def build_and_save_dataset(raw_dir: Path, processed_dir: Path, symbols: list[str], interval: str = "1d", horizon_days: int = 1) -> Path:
    prices = load_price_data(raw_dir, symbols, interval)
    if prices.empty:
        raise ValueError("No price data found. Run historical ingestion first.")
    news = load_news_data(raw_dir, symbols)
    dataset = build_feature_pipeline(prices, news, horizon_days=horizon_days)

    output_path = processed_dir / f"dataset_{interval}_{horizon_days}d.parquet"
    write_parquet(dataset, output_path)
    return output_path
