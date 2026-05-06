from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import httpx
import pandas as pd

from app.news.models import NewsArticle

try:
    import feedparser  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    feedparser = None

logger = logging.getLogger(__name__)


def _clean_text(value: Any) -> str:
    return unescape(" ".join(str(value or "").split()).strip())


def _stable_article_hash(title: str, link: str) -> str:
    payload = f"{_clean_text(title).lower()}|{_clean_text(link).lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def load_rss_feeds(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    feeds: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        label = str(item.get("label") or "").strip() or url
        if url:
            feeds.append({"label": label, "url": url})
    return feeds


def save_rss_feeds(path: Path, feeds: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feeds, indent=2), encoding="utf-8")


@dataclass
class RSSFetcher:
    timeout_sec: float = 20.0

    def _fetch_text(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StockAI/1.0)"}
        with httpx.Client(timeout=self.timeout_sec, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text

    def _parse_with_feedparser(self, feed_text: str, source_label: str, limit: int) -> list[NewsArticle]:
        if feedparser is None:
            return []
        parsed = feedparser.parse(feed_text)
        items = getattr(parsed, "entries", []) or []
        articles: list[NewsArticle] = []
        for entry in items[:limit]:
            title = _clean_text(getattr(entry, "title", ""))
            link = _clean_text(getattr(entry, "link", ""))
            summary = _clean_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))
            if not title or not link:
                continue
            published = None
            if getattr(entry, "published", None):
                published = _parse_datetime(getattr(entry, "published"))
            elif getattr(entry, "updated", None):
                published = _parse_datetime(getattr(entry, "updated"))
            articles.append(
                NewsArticle(
                    article_hash=_stable_article_hash(title, link),
                    title=title,
                    summary=summary,
                    link=link,
                    source=source_label,
                    published_at=published,
                    raw_text=_clean_text(f"{title}. {summary}"),
                )
            )
        return articles

    def _parse_rss_xml(self, xml_text: str, source_label: str, limit: int) -> list[NewsArticle]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        articles: list[NewsArticle] = []
        for item in root.findall(".//item"):
            title = _clean_text(item.findtext("title"))
            link = _clean_text(item.findtext("link"))
            summary = _clean_text(item.findtext("description") or item.findtext("summary"))
            if not title or not link:
                continue
            published = _parse_datetime(item.findtext("pubDate") or item.findtext("date") or item.findtext("published"))
            articles.append(
                NewsArticle(
                    article_hash=_stable_article_hash(title, link),
                    title=title,
                    summary=summary,
                    link=link,
                    source=source_label,
                    published_at=published,
                    raw_text=_clean_text(f"{title}. {summary}"),
                )
            )
            if len(articles) >= limit:
                break
        return articles

    def fetch_feed(self, feed_url: str, source_label: str | None = None, limit: int = 20) -> list[NewsArticle]:
        label = source_label or feed_url
        try:
            xml_text = self._fetch_text(feed_url)
        except Exception as exc:
            logger.warning("rss_fetch_failed", extra={"source_url": feed_url, "error": str(exc)})
            return []

        if feedparser is not None:
            try:
                articles = self._parse_with_feedparser(xml_text, label, limit)
                if articles:
                    return articles
            except Exception as exc:
                logger.warning("rss_feedparser_failed", extra={"source_url": feed_url, "error": str(exc)})

        articles = self._parse_rss_xml(xml_text, label, limit)
        if articles:
            return articles

        return self._parse_html_fallback(xml_text, feed_url, label, limit)

    def _parse_html_fallback(self, html_text: str, source_url: str, source_label: str, limit: int) -> list[NewsArticle]:
        # Very small fallback for RSS pages that render links but not XML.
        from src.data.web_news_scraper import _AnchorParser, _is_valid_headline, _normalize_title

        parser = _AnchorParser()
        parser.feed(html_text)
        seen: set[str] = set()
        articles: list[NewsArticle] = []
        for candidate in parser.candidates:
            title = _normalize_title(candidate.get("title", ""))
            href = str(candidate.get("href") or "").strip()
            if not title or not href or not _is_valid_headline(title):
                continue
            link = urljoin(source_url, href)
            if link in seen:
                continue
            seen.add(link)
            articles.append(
                NewsArticle(
                    article_hash=_stable_article_hash(title, link),
                    title=title,
                    summary=title,
                    link=link,
                    source=source_label,
                    published_at=None,
                    raw_text=title,
                )
            )
            if len(articles) >= limit:
                break
        return articles

    def fetch_many(self, feeds: list[dict[str, str]], limit_per_feed: int = 20) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        seen_hashes: set[str] = set()
        for feed in feeds:
            feed_url = str(feed.get("url") or "").strip()
            label = str(feed.get("label") or feed_url).strip()
            if not feed_url:
                continue
            try:
                fetched = self.fetch_feed(feed_url, source_label=label, limit=limit_per_feed)
            except Exception as exc:
                logger.warning("rss_feed_failed", extra={"source_url": feed_url, "error": str(exc)})
                continue
            for article in fetched:
                if article.article_hash in seen_hashes:
                    continue
                seen_hashes.add(article.article_hash)
                articles.append(article)
        return articles
