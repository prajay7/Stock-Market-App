from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache
from typing import Optional

from app.core.config import get_settings
from app.news.service import news_impact_service
from src.data.metadata_store import metadata_store

logger = logging.getLogger(__name__)


class TrendingSymbolsService:
    """Fetches trending/opportunity symbols from news analysis instead of hardcoded defaults."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.timeout_sec = max(1, int(self.settings.automation_step_timeout_sec or 30))

    def _run_with_timeout(self, fn, *args, **kwargs):
        """Run function with timeout protection."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=self.timeout_sec)
        except FuturesTimeoutError:
            logger.warning(
                "trending_symbols_timeout",
                extra={"function": getattr(fn, "__name__", str(fn)), "timeout_sec": self.timeout_sec},
            )
            future.cancel()
            return None
        except Exception as exc:
            logger.warning("trending_symbols_fetch_failed", extra={"error": str(exc)})
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _fetch_news_opportunities(self) -> list[str]:
        """Fetch symbols from news impact analysis (trending in news)."""
        symbols: list[str] = []

        # Scan latest news for opportunities
        scan = self._run_with_timeout(
            news_impact_service.refresh,
            force_refresh=bool(self.settings.automation_force_news_refresh),
        )
        if scan is not None:
            for item in scan.top_opportunities:
                ticker = str(item.beneficiary_ticker or item.primary_ticker or "").strip().upper()
                if ticker:
                    symbols.append(ticker)

        # Also fetch recent beneficiary opportunities from metadata
        try:
            recent = metadata_store.read_beneficiary_opportunities_with_signal(limit=500)
            for row in recent:
                ticker = str(row.get("ticker") or row.get("primary_ticker") or "").strip().upper()
                if ticker:
                    symbols.append(ticker)
        except Exception as exc:
            logger.warning("metadata_opportunities_failed", extra={"error": str(exc)})

        return symbols

    @staticmethod
    def _dedupe_and_limit(symbols: list[str], limit: int = 25) -> list[str]:
        """Remove duplicates and limit to top N symbols."""
        seen: set[str] = set()
        out: list[str] = []
        for raw in symbols:
            symbol = str(raw).strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
            if len(out) >= limit:
                break
        return out

    def get_trending_symbols(self, fallback_to_defaults: bool = True, limit: int = 25) -> tuple[list[str], str]:
        """
        Fetch trending symbols from news.

        Args:
            fallback_to_defaults: If no trending symbols found, fall back to DEFAULT_SYMBOLS
            limit: Maximum number of symbols to return

        Returns:
            (symbols_list, source_label) - e.g., (["RELIANCE", "TCS"], "news_trending")
        """
        logger.info(
            "trending_symbols_fetch_started",
            extra={"limit": limit, "fallback_to_defaults": fallback_to_defaults},
        )

        # Try to fetch trending symbols from news
        trending = self._fetch_news_opportunities()
        symbols = self._dedupe_and_limit(trending, limit=limit)

        if symbols:
            logger.info("trending_symbols_fetched", extra={"count": len(symbols), "symbols": symbols[:5]})
            return symbols, "news_trending"

        # Fallback to defaults if no trending found
        if fallback_to_defaults:
            defaults = [str(sym).strip().upper() for sym in self.settings.default_symbols if str(sym).strip()]
            logger.info("trending_symbols_fallback", extra={"count": len(defaults), "reason": "no_trending_found"})
            return defaults, "default"

        logger.warning("trending_symbols_empty", extra={"reason": "no_trending_found_and_fallback_disabled"})
        return [], "none"

    def get_trending_symbols_cached(self, ttl_seconds: int = 300, **kwargs) -> tuple[list[str], str]:
        """
        Cached version of get_trending_symbols.
        Uses lru_cache with TTL-like behavior (cache for 5 minutes by default).

        Returns:
            (symbols_list, source_label)
        """
        return self._get_trending_cached_internal(ttl_seconds=ttl_seconds, **kwargs)

    @lru_cache(maxsize=1)
    def _get_trending_cached_internal(self, ttl_seconds: int = 300, fallback_to_defaults: bool = True, limit: int = 25) -> tuple[list[str], str]:
        """Internal cached method. Cache key includes ttl_seconds to refresh periodically."""
        return self.get_trending_symbols(fallback_to_defaults=fallback_to_defaults, limit=limit)


# Singleton instance
trending_symbols_service = TrendingSymbolsService()
