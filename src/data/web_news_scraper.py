from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
import pandas as pd

from src.data.cache import symbol_news_path
from src.data.metadata_store import MetadataStore
from src.data.storage import append_dedup_by_keys, read_parquet_if_exists, write_parquet
from src.features.news_scoring import enrich_news_scores

logger = logging.getLogger(__name__)


_BLOCKLIST = {
    "home",
    "login",
    "sign in",
    "subscribe",
    "register",
    "privacy policy",
    "terms of use",
    "cookie policy",
    "advertise",
    "contact",
    "about",
    "read more",
    "next",
    "previous",
}


def _clean_metadata_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    cleaned: list[dict[str, Any]] = []
    for row in records:
        out: dict[str, Any] = {}
        for key, value in row.items():
            out[key] = None if pd.isna(value) else value
        cleaned.append(out)
    return cleaned


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.title_parts: list[str] = []
        self.in_anchor = False
        self.anchor_href = ""
        self.anchor_parts: list[str] = []
        self.candidates: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): value for key, value in attrs if key}
        if tag.lower() == "title":
            self.in_title = True
        elif tag.lower() == "a":
            href = (attr_map.get("href") or "").strip()
            if href:
                self.in_anchor = True
                self.anchor_href = href
                self.anchor_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False
        elif tag.lower() == "a" and self.in_anchor:
            text = re.sub(r"\s+", " ", "".join(self.anchor_parts)).strip()
            href = self.anchor_href.strip()
            if text and href:
                self.candidates.append({"title": text, "href": href})
            self.in_anchor = False
            self.anchor_href = ""
            self.anchor_parts = []

    def handle_data(self, data: str) -> None:
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
        if self.in_anchor:
            self.anchor_parts.append(text)


def _site_source(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url


def _normalize_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip()
    cleaned = cleaned.strip(" -|:\t\r\n")
    return cleaned


def _is_valid_headline(text: str) -> bool:
    normalized = text.lower().strip()
    if len(normalized) < 18:
        return False
    if normalized in _BLOCKLIST:
        return False
    if normalized.endswith(".") and len(normalized) < 28:
        return False
    return True


def build_common_news_sources(query: str) -> list[dict[str, str]]:
    cleaned = query.strip()
    if not cleaned:
        return []

    encoded = quote_plus(cleaned)
    return [
        {"label": "Google News RSS", "url": f"https://news.google.com/rss/search?q={encoded}"},
        {"label": "Bing News", "url": f"https://www.bing.com/news/search?q={encoded}"},
        {"label": "Google Search", "url": f"https://www.google.com/search?q={quote_plus(cleaned + ' news')}"},
    ]


def _parse_rss_feed(xml_text: str, symbol: str, source_label: str, source_url: str, limit: int) -> pd.DataFrame:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = _normalize_title((item.findtext("title") or "").strip())
        link = (item.findtext("link") or "").strip()
        if not title or not link or not _is_valid_headline(title):
            continue

        pub_date = item.findtext("pubDate") or item.findtext("date") or ""
        parsed_date = pd.to_datetime(pub_date, errors="coerce") if pub_date else pd.NaT
        records.append(
            {
                "symbol": symbol,
                "source": source_label,
                "title": title,
                "summary": title,
                "url": link,
                "published_at": parsed_date,
                "sentiment_score": None,
                "relevance_score": None,
                "overall_sentiment_score": None,
                "scraped_from": source_url,
            }
        )
        if len(records) >= limit:
            break

    return enrich_news_scores(pd.DataFrame(records))


@dataclass
class WebsiteNewsScraper:
    raw_data_dir: Path
    database_url: str = "sqlite:///./stock_ai.db"
    timeout_sec: float = 20.0

    def _fetch_text(self, url: str) -> tuple[str, str]:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockAI/1.0)"}
        with httpx.Client(timeout=self.timeout_sec, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            return response.text, content_type

    def _scrape_single(self, url: str, symbol: str, limit: int = 20, source_label: str | None = None) -> pd.DataFrame:
        if not url:
            return pd.DataFrame()

        text, content_type = self._fetch_text(url)
        source = source_label or _site_source(url)

        if "xml" in content_type or url.lower().endswith((".rss", ".xml")) or "<rss" in text.lower():
            rss_df = _parse_rss_feed(text, symbol=symbol, source_label=source, source_url=url, limit=limit)
            if not rss_df.empty:
                return rss_df

        parser = _AnchorParser()
        parser.feed(text)

        records: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for candidate in parser.candidates:
            title = _normalize_title(candidate.get("title", ""))
            href = str(candidate.get("href") or "").strip()
            if not title or not href:
                continue
            if href.startswith(("javascript:", "mailto:", "#")):
                continue
            if not _is_valid_headline(title):
                continue

            resolved_url = urljoin(url, href)
            if resolved_url in seen_urls:
                continue
            seen_urls.add(resolved_url)

            records.append(
                {
                    "symbol": symbol,
                    "source": source,
                    "title": title,
                    "summary": title,
                    "url": resolved_url,
                    "published_at": None,
                    "sentiment_score": None,
                    "relevance_score": None,
                    "overall_sentiment_score": None,
                    "scraped_from": url,
                }
            )
            if len(records) >= limit:
                break

        if not records:
            return pd.DataFrame(columns=["symbol", "source", "title", "summary", "url", "published_at"])

        df = enrich_news_scores(pd.DataFrame(records))
        df = df.drop_duplicates(subset=["symbol", "url"], keep="last")
        return df.reset_index(drop=True)

    def scrape(self, url: str, symbol: str, limit: int = 20, source_label: str | None = None) -> pd.DataFrame:
        return self._scrape_single(url=url, symbol=symbol, limit=limit, source_label=source_label)

    def scrape_many(self, sources: list[dict[str, str]], symbol: str, limit_per_source: int = 20) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for source in sources:
            url = str(source.get("url") or "").strip()
            label = str(source.get("label") or _site_source(url))
            if not url:
                continue
            try:
                frame = self._scrape_single(url=url, symbol=symbol, limit=limit_per_source, source_label=label)
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                logger.warning("website_source_scrape_failed", extra={"symbol": symbol, "source_url": url, "error": str(exc)})
                continue

        if not frames:
            return pd.DataFrame()

        # Explicitly specify dtypes to avoid FutureWarning about concat with empty/all-NA columns
        combined = enrich_news_scores(pd.concat(frames, ignore_index=True, sort=False))
        return combined.drop_duplicates(subset=["symbol", "url"], keep="last").reset_index(drop=True)

    def ingest(self, url: str, symbol: str, limit: int = 20) -> pd.DataFrame:
        scraped = enrich_news_scores(self.scrape(url=url, symbol=symbol, limit=limit))
        if scraped.empty:
            return scraped

        path = symbol_news_path(self.raw_data_dir, symbol)
        existing = read_parquet_if_exists(path)
        combined = append_dedup_by_keys(existing, scraped, keys=["symbol", "url"])
        combined = combined.sort_values("title").reset_index(drop=True)
        write_parquet(combined, path)

        store = MetadataStore(self.database_url)
        metadata_rows = combined[["symbol", "source", "title", "summary", "url", "published_at", "sentiment_score", "relevance_score"]].copy()
        store.upsert_news_records(_clean_metadata_records(metadata_rows))

        logger.info("website_news_ingested", extra={"symbol": symbol, "source_url": url, "rows": len(combined)})
        return combined

    def ingest_many(self, sources: list[dict[str, str]], symbol: str, limit_per_source: int = 20) -> pd.DataFrame:
        scraped = enrich_news_scores(self.scrape_many(sources=sources, symbol=symbol, limit_per_source=limit_per_source))
        if scraped.empty:
            return scraped

        path = symbol_news_path(self.raw_data_dir, symbol)
        existing = read_parquet_if_exists(path)
        combined = append_dedup_by_keys(existing, scraped, keys=["symbol", "url"])
        combined = combined.sort_values(["published_at", "title"], na_position="last").reset_index(drop=True)
        write_parquet(combined, path)

        store = MetadataStore(self.database_url)
        metadata_rows = combined[["symbol", "source", "title", "summary", "url", "published_at", "sentiment_score", "relevance_score"]].copy()
        store.upsert_news_records(_clean_metadata_records(metadata_rows))

        logger.info("website_news_ingested_many", extra={"symbol": symbol, "rows": len(combined)})
        return combined
