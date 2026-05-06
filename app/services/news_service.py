from __future__ import annotations

import logging

from app.core.config import get_settings
from src.data.news_loader import AlphaVantageNewsLoader
from src.data.web_news_scraper import WebsiteNewsScraper, build_common_news_sources

logger = logging.getLogger(__name__)


class NewsService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = str(self.settings.news_provider or "alphavantage").strip().lower()
        self.loader = AlphaVantageNewsLoader(
            raw_data_dir=self.settings.raw_data_dir,
            api_key=self.settings.alpha_vantage_api_key,
            database_url=self.settings.database_url,
            timeout_sec=self.settings.request_timeout_sec,
        )
        self.web_scraper = WebsiteNewsScraper(
            raw_data_dir=self.settings.raw_data_dir,
            database_url=self.settings.database_url,
            timeout_sec=self.settings.request_timeout_sec,
        )

    def ingest_news(self, symbols: list[str], limit_per_symbol: int) -> dict[str, int]:
        logger.info(
            "news_ingest_started",
            extra={
                "provider": str(self.provider),
                "symbols_count": len(symbols),
                "limit_per_symbol": int(limit_per_symbol),
            },
        )
        if self.provider in {"google", "web", "scrape", "rss"}:
            summary: dict[str, int] = {}
            per_source_limit = max(1, int(limit_per_symbol // 3) if limit_per_symbol > 3 else int(limit_per_symbol))
            for symbol in symbols:
                query = f"{symbol} stock news"
                sources = build_common_news_sources(query)
                frame = self.web_scraper.ingest_many(sources=sources, symbol=symbol, limit_per_source=per_source_limit)
                summary[symbol] = len(frame)
            logger.info("news_ingest_completed", extra={"provider": str(self.provider), "result": summary})
            return summary
        result = self.loader.ingest(symbols=symbols, limit_per_symbol=limit_per_symbol)
        logger.info("news_ingest_completed", extra={"provider": str(self.provider), "result": result})
        return result


news_service = NewsService()
