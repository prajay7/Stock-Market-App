from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from src.data.cache import symbol_news_path
from src.data.storage import append_dedup_by_keys, read_parquet_if_exists, write_parquet
from src.data.metadata_store import MetadataStore
from src.features.news_scoring import enrich_news_scores
from src.utils.symbols import normalize_symbols

logger = logging.getLogger(__name__)


@dataclass
class AlphaVantageNewsLoader:
    raw_data_dir: Path
    api_key: str
    database_url: str = "sqlite:///./stock_ai.db"
    timeout_sec: float = 20.0

    @retry(wait=wait_exponential(min=1, max=32), stop=stop_after_attempt(4), reraise=True)
    def _fetch_symbol_news(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        if not self.api_key:
            return pd.DataFrame()

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "limit": str(limit),
            "apikey": self.api_key,
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

        items = payload.get("feed", [])
        records = []
        for item in items:
            ticker_sentiment = item.get("ticker_sentiment", [])
            symbol_payload = next((x for x in ticker_sentiment if x.get("ticker") == symbol), {})
            records.append(
                {
                    "symbol": symbol,
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "summary": item.get("summary"),
                    "url": item.get("url"),
                    "published_at": pd.to_datetime(item.get("time_published"), format="%Y%m%dT%H%M%S", errors="coerce"),
                    "sentiment_score": pd.to_numeric(symbol_payload.get("ticker_sentiment_score"), errors="coerce"),
                    "relevance_score": pd.to_numeric(symbol_payload.get("relevance_score"), errors="coerce"),
                    "overall_sentiment_score": pd.to_numeric(item.get("overall_sentiment_score"), errors="coerce"),
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df = df.dropna(subset=["published_at", "url"]).drop_duplicates(subset=["symbol", "url"], keep="last")
        df["published_date"] = df["published_at"].dt.floor("D")
        return df.sort_values("published_at").reset_index(drop=True)

    def ingest(self, symbols: list[str], limit_per_symbol: int = 100) -> dict[str, int]:
        symbols = normalize_symbols(symbols)
        summary: dict[str, int] = {}
        store = MetadataStore(self.database_url)

        for symbol in symbols:
            path = symbol_news_path(self.raw_data_dir, symbol)
            existing = read_parquet_if_exists(path)

            try:
                fresh = self._fetch_symbol_news(symbol, limit=limit_per_symbol)
            except Exception as exc:
                logger.warning("news_fetch_failed", extra={"symbol": symbol, "error": str(exc)})
                summary[symbol] = 0
                continue

            if fresh.empty:
                summary[symbol] = len(existing)
                continue

            combined = append_dedup_by_keys(existing, fresh, keys=["symbol", "url"])
            combined = combined.sort_values("published_at").reset_index(drop=True)
            write_parquet(combined, path)
            store.upsert_news_records(
                combined[["symbol", "source", "title", "summary", "url", "published_at", "sentiment_score", "relevance_score"]]
                .to_dict(orient="records")
            )
            summary[symbol] = len(combined)
            logger.info("news_ingested", extra={"symbol": symbol, "rows": len(combined)})

        return summary


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "sentiment_same_day",
                "sentiment_3d",
                "sentiment_7d",
                "sentiment_momentum_3d",
                "news_count",
                "news_count_3d",
                "news_count_7d",
                "positive_news_count",
                "negative_news_count",
                "news_impact_mean",
                "news_impact_max",
                "news_impact_3d",
                "news_signal_score",
                "news_signal_3d",
                "news_signal_7d",
                "news_buzz_3d",
                "relevance_mean",
            ]
        )

    df = enrich_news_scores(news_df)
    df["date"] = pd.to_datetime(df["published_at"]).dt.floor("D")
    df = df.dropna(subset=["symbol", "date"]).copy()
    if df.empty:
        return aggregate_daily_sentiment(pd.DataFrame())

    for column in ["sentiment_score", "relevance_score", "news_impact_score", "news_signal_score"]:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    if "news_sentiment_label" not in df.columns:
        df["news_sentiment_label"] = "neutral"
    df["positive_flag"] = (df["news_sentiment_label"].astype(str) == "positive").astype(int)
    df["negative_flag"] = (df["news_sentiment_label"].astype(str) == "negative").astype(int)

    daily = (
        df.groupby(["symbol", "date"], as_index=False)
        .agg(
            sentiment_same_day=("sentiment_score", "mean"),
            news_count=("url", "count"),
            relevance_mean=("relevance_score", "mean"),
            positive_news_count=("positive_flag", "sum"),
            negative_news_count=("negative_flag", "sum"),
            news_impact_mean=("news_impact_score", "mean"),
            news_impact_max=("news_impact_score", "max"),
            news_signal_score=("news_signal_score", "mean"),
        )
        .sort_values(["symbol", "date"])
    )

    daily["sentiment_3d"] = daily.groupby("symbol")["sentiment_same_day"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    daily["sentiment_7d"] = daily.groupby("symbol")["sentiment_same_day"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    daily["sentiment_momentum_3d"] = daily["sentiment_3d"] - daily["sentiment_7d"]
    daily["news_count_3d"] = daily.groupby("symbol")["news_count"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    daily["news_count_7d"] = daily.groupby("symbol")["news_count"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    daily["news_impact_3d"] = daily.groupby("symbol")["news_impact_mean"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    daily["news_signal_3d"] = daily.groupby("symbol")["news_signal_score"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    daily["news_signal_7d"] = daily.groupby("symbol")["news_signal_score"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    daily["news_buzz_3d"] = daily["news_count"] / daily["news_count_3d"].replace(0, pd.NA)
    daily["news_buzz_3d"] = pd.to_numeric(daily["news_buzz_3d"], errors="coerce").fillna(0.0)
    return daily
