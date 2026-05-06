from pathlib import Path

from app.news.analyzer import NewsAnalyzer
from app.news.fetcher import _stable_article_hash
from app.news.models import NewsArticle
from app.news.relations import CompanyRelations


class FakeProvider:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    def generate(self, article, candidates):
        self.calls += 1
        return self.payload


def _make_article(title: str, summary: str = "") -> NewsArticle:
    return NewsArticle(
        article_hash=_stable_article_hash(title, "https://example.com/article"),
        title=title,
        summary=summary,
        link="https://example.com/article",
        source="Test Feed",
        raw_text=f"{title}. {summary}",
    )


def test_rule_based_event_and_sentiment_and_beneficiaries(tmp_path: Path):
    relations_path = tmp_path / "relations.json"
    relations_path.write_text(
        """
        {
          "NTPC": {
            "ticker": "NTPC.NS",
            "aliases": ["National Thermal Power Corporation"],
            "sector": "Power",
            "relations": [
              {"company": "Power Grid", "ticker": "POWERGRID.NS", "relation": "transmission", "strength": 0.7},
              {"company": "BHEL", "ticker": "BHEL.NS", "relation": "equipment_supplier", "strength": 0.6}
            ]
          }
        }
        """,
        encoding="utf-8",
    )
    analyzer = NewsAnalyzer(relations=CompanyRelations(relations_path), provider_name="rule")
    article = _make_article("NTPC announces capacity expansion after strong demand", "New plant capacity to boost output")
    analysis = analyzer.analyze(article)

    assert analysis.primary_company == "NTPC"
    assert analysis.event_type == "capacity_expansion"
    assert analysis.sentiment_label == "positive"
    assert analysis.is_actionable is True
    assert analysis.beneficiary_companies
    assert analysis.beneficiary_companies[0].company in {"Power Grid", "BHEL"}


def test_malformed_llm_json_falls_back_to_rule_based(tmp_path: Path):
    relations_path = tmp_path / "relations.json"
    relations_path.write_text(
        "{\"NTPC\": [{\"company\": \"Power Grid\", \"relation\": \"transmission\", \"strength\": 0.7}]}",
        encoding="utf-8",
    )
    provider = FakeProvider("{not-valid-json")
    analyzer = NewsAnalyzer(relations=CompanyRelations(relations_path), provider_name="openai", llm_provider=provider)
    article = _make_article("NTPC beats estimates on strong demand")
    analysis = analyzer.analyze(article)

    assert provider.calls >= 2
    assert analysis.event_type == "earnings_beat"
    assert analysis.sentiment_label == "positive"
    assert analysis.primary_company == "NTPC"
