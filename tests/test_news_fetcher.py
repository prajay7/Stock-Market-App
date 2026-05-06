from app.news.fetcher import RSSFetcher, _stable_article_hash
from app.news.models import NewsArticle


def test_rss_parser_extracts_articles_and_dates():
    xml = """
    <rss><channel>
      <item>
        <title>NTPC announces capacity expansion</title>
        <link>https://example.com/a1</link>
        <description>Expansion plan.</description>
        <pubDate>Mon, 18 Apr 2026 09:00:00 GMT</pubDate>
      </item>
      <item>
        <title>Power Grid gains on NTPC expansion</title>
        <link>https://example.com/a2</link>
        <description>Indirect beneficiary.</description>
        <pubDate>Mon, 18 Apr 2026 10:00:00 GMT</pubDate>
      </item>
    </channel></rss>
    """
    fetcher = RSSFetcher()
    articles = fetcher._parse_rss_xml(xml, source_label="Test Feed", limit=10)
    assert len(articles) == 2
    assert articles[0].title == "NTPC announces capacity expansion"
    assert articles[0].source == "Test Feed"
    assert articles[0].published_at is not None
    assert articles[0].article_hash == _stable_article_hash("NTPC announces capacity expansion", "https://example.com/a1")


def test_fetch_many_deduplicates_by_article_hash(monkeypatch):
    fetcher = RSSFetcher()
    article_a = NewsArticle(
        article_hash=_stable_article_hash("Headline", "https://example.com/a"),
        title="Headline",
        summary="Summary",
        link="https://example.com/a",
        source="Feed 1",
        raw_text="Headline Summary",
    )
    article_b = NewsArticle(
        article_hash=_stable_article_hash("Headline", "https://example.com/a"),
        title="Headline",
        summary="Summary",
        link="https://example.com/a",
        source="Feed 2",
        raw_text="Headline Summary",
    )

    def fake_fetch_feed(feed_url, source_label=None, limit=20):
        return [article_a] if "one" in feed_url else [article_b]

    monkeypatch.setattr(fetcher, "fetch_feed", fake_fetch_feed)
    items = fetcher.fetch_many(
        [
            {"label": "One", "url": "https://feed-one.example/rss"},
            {"label": "Two", "url": "https://feed-two.example/rss"},
        ],
        limit_per_feed=10,
    )
    assert len(items) == 1
    assert items[0].article_hash == article_a.article_hash
