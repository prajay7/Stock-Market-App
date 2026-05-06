from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests
import xml.etree.ElementTree as ET

from app.core.config import get_settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_symbols_from_csv(csv_path: Path) -> list[dict]:
    out = []
    if not csv_path.exists():
        return out
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return out
    for _, row in df.iterrows():
        out.append({
            "symbol": str(row.get("Symbol") or "").strip(),
            "security_name": str(row.get("Security Name") or "").strip(),
        })
    return out


def fetch_rss_items(feed_url: str, limit: int = 50) -> list[dict]:
    try:
        r = requests.get(feed_url, timeout=15)
        r.raise_for_status()
        text = r.text
    except Exception:
        return []
    try:
        root = ET.fromstring(text)
    except Exception:
        return []
    items = []
    # RSS/ATOM common paths
    for item in root.findall('.//item') or root.findall('.//entry'):
        title_el = item.find('title')
        link_el = item.find('link')
        pub_el = item.find('pubDate') or item.find('published') or item.find('updated')
        description_el = item.find('description') or item.find('summary')
        title = title_el.text if title_el is not None else ""
        link = ''
        if link_el is not None:
            link = link_el.text or link_el.get('href') or ''
        pub = pub_el.text if pub_el is not None else None
        description = description_el.text if description_el is not None else ""
        items.append({
            "title": str(title or "").strip(),
            "link": str(link or "").strip(),
            "published": str(pub) if pub is not None else None,
            "description": str(description or "").strip(),
        })
        if len(items) >= limit:
            break
    return items


def _ensure_db_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                path TEXT,
                source TEXT,
                metadata_json TEXT,
                created_at TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_dataset(
    csv_symbols_path: Path | None = None,
    save_folder: Path | None = None,
    feed_limit: int = 3,
    items_per_feed: int = 50,
    symbol_limit: int | None = None,
) -> dict:
    settings = get_settings()
    csv_path = csv_symbols_path or Path(settings.root_path) if hasattr(settings, 'root_path') else Path("./sec_list.csv")
    # fallback to repository root file
    if not csv_path.exists():
        csv_path = Path("./sec_list.csv")

    symbols = read_symbols_from_csv(csv_path)
    if symbol_limit:
        symbols = symbols[: int(symbol_limit)]

    feed_list = []
    try:
        feeds = settings.news_rss_feeds_path if hasattr(settings, 'news_rss_feeds_path') else None
        if feeds and Path(feeds).exists():
            feed_list = json.loads(Path(feeds).read_text(encoding='utf-8'))
        else:
            feed_list = settings.news_rss_feeds_default
    except Exception:
        feed_list = settings.news_rss_feeds_default

    feed_list = feed_list[:feed_limit]

    rows: list[dict] = []
    scraped_at = _now_iso()
    for feed in feed_list:
        url = str(feed.get('url') or '')
        label = str(feed.get('label') or url)
        items = fetch_rss_items(url, limit=items_per_feed)
        for it in items:
            for sym in symbols:
                symbol = sym.get('symbol')
                name = sym.get('security_name')
                title = str(it.get('title') or '').lower()
                description = str(it.get('description') or '').lower()
                if not symbol:
                    continue
                # simple filter: check symbol or company name in title/description
                if (symbol.lower() in title) or (symbol.lower() in description) or (name and name.lower() in title) or (name and name.lower() in description):
                    rows.append(
                        {
                            'symbol': symbol,
                            'security_name': name,
                            'feed_label': label,
                            'feed_url': url,
                            'title': it.get('title'),
                            'link': it.get('link'),
                            'published': it.get('published'),
                            'description': it.get('description'),
                            'scraped_at': scraped_at,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "symbol",
                "security_name",
                "feed_label",
                "feed_url",
                "title",
                "link",
                "published",
                "description",
                "scraped_at",
            ]
        )
    save_folder = save_folder or Path(settings.raw_data_dir) / 'datasets'
    save_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_path = save_folder / f'dataset_{timestamp}.csv'
    df.to_csv(file_path, index=False)

    # store metadata in DB
    db_path = Path(settings.db_path or Path('./data/stock_data.db'))
    _ensure_db_table(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            'INSERT INTO datasets(name, path, source, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)',
            (
                f'dataset_{timestamp}',
                str(file_path),
                'rss_feeds',
                json.dumps({'rows': len(df), 'feeds': [f.get('label') for f in feed_list]}),
                _now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {'path': str(file_path), 'rows': len(df)}


if __name__ == '__main__':
    print(create_dataset())
