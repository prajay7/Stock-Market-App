from pathlib import Path


def symbol_price_path(raw_dir: Path, symbol: str, interval: str) -> Path:
    return raw_dir / "prices" / interval / f"{symbol}.parquet"


def symbol_news_path(raw_dir: Path, symbol: str) -> Path:
    return raw_dir / "news" / f"{symbol}.parquet"
