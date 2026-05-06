import pandas as pd

from src.data.cache import symbol_news_path
from src.data.metadata_store import MetadataStore
from src.data.web_news_scraper import WebsiteNewsScraper, _parse_rss_feed


def test_website_scraper_parses_rss_items():
    xml = """
    <rss><channel>
      <item>
        <title>Reliance shares rise after strong quarterly results</title>
        <link>https://example.com/reliance-results</link>
        <pubDate>Mon, 20 Apr 2026 09:30:00 GMT</pubDate>
      </item>
      <item>
        <title>Short</title>
        <link>https://example.com/short</link>
      </item>
    </channel></rss>
    """

    df = _parse_rss_feed(
        xml,
        symbol="RELIANCE.NS",
        source_label="Test RSS",
        source_url="https://example.com/rss",
        limit=10,
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "RELIANCE.NS"
    assert row["source"] == "Test RSS"
    assert row["url"] == "https://example.com/reliance-results"
    assert pd.notna(row["published_at"])


def test_website_scraper_parses_html_links_and_resolves_urls(monkeypatch, tmp_path):
    html = """
    <html><body>
      <a href="/markets/reliance">Reliance shares rise after strong quarterly results</a>
      <a href="javascript:void(0)">Reliance invalid link headline that should be skipped</a>
      <a href="/markets/reliance">Reliance shares rise after strong quarterly results</a>
      <a href="/markets/infosys">Infosys signs major AI partnership with global bank</a>
    </body></html>
    """
    scraper = WebsiteNewsScraper(raw_data_dir=tmp_path)
    monkeypatch.setattr(scraper, "_fetch_text", lambda url: (html, "text/html"))

    df = scraper.scrape("https://example.com/news", symbol="RELIANCE.NS", limit=10)

    assert len(df) == 2
    assert set(df["url"]) == {
        "https://example.com/markets/reliance",
        "https://example.com/markets/infosys",
    }
    assert set(df["symbol"]) == {"RELIANCE.NS"}


def test_website_scraper_ingest_many_persists_parquet_and_metadata(monkeypatch, tmp_path):
    xml = """
    <rss><channel>
      <item>
        <title>Reliance shares rise after strong quarterly results</title>
        <link>https://example.com/reliance-results</link>
        <pubDate>Mon, 20 Apr 2026 09:30:00 GMT</pubDate>
      </item>
    </channel></rss>
    """
    db_url = f"sqlite:///{tmp_path / 'news.db'}"
    raw_dir = tmp_path / "raw"
    scraper = WebsiteNewsScraper(raw_data_dir=raw_dir, database_url=db_url)
    monkeypatch.setattr(scraper, "_fetch_text", lambda url: (xml, "application/rss+xml"))

    df = scraper.ingest_many(
        [{"label": "Test RSS", "url": "https://example.com/rss"}],
        symbol="RELIANCE.NS",
        limit_per_source=10,
    )

    assert len(df) == 1
    stored = pd.read_parquet(symbol_news_path(raw_dir, "RELIANCE.NS"))
    assert len(stored) == 1

    store = MetadataStore(db_url)
    with store.engine.begin() as conn:
        urls = conn.execute(store.news.select()).mappings().all()

    assert len(urls) == 1
    assert urls[0]["url"] == "https://example.com/reliance-results"
