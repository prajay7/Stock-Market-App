from datetime import datetime, timedelta, timezone

import pandas as pd

from app.news.models import BeneficiaryCompany, NewsAnalysisResult, NewsArticle
from app.news.scorer import NewsScorer
from app.news.fetcher import _stable_article_hash


def test_beneficiary_ranking_prefers_higher_combined_score():
    scorer = NewsScorer()
    analysis = NewsAnalysisResult(
        primary_company="NTPC",
        sector="Power",
        event_type="capacity_expansion",
        sentiment_label="positive",
        sentiment_score=0.7,
        impact_score=0.8,
        is_actionable=True,
        confidence_score=0.9,
        summary="Expansion",
        beneficiary_companies=[
            BeneficiaryCompany(company="Power Grid", relation="transmission", reason="", benefit_score=0.7, ticker="POWERGRID.NS"),
            BeneficiaryCompany(company="BHEL", relation="equipment_supplier", reason="", benefit_score=0.6, ticker="BHEL.NS"),
        ],
    )
    ranked = scorer.rank_beneficiaries(analysis, freshness=0.9, price_scores={"Power Grid": 0.8, "BHEL": 0.2})
    assert ranked[0].company == "Power Grid"
    assert ranked[0].benefit_score >= ranked[1].benefit_score


def test_overall_score_reacts_to_freshness_and_price_opportunity():
    scorer = NewsScorer()
    article = NewsArticle(
        article_hash=_stable_article_hash("Headline", "https://example.com/a"),
        title="Headline",
        summary="Summary",
        link="https://example.com/a",
        source="Feed",
        published_at=datetime.now(timezone.utc) - timedelta(hours=1),
        raw_text="Headline Summary",
    )
    analysis = NewsAnalysisResult(
        primary_company="NTPC",
        sector="Power",
        event_type="capacity_expansion",
        sentiment_label="positive",
        sentiment_score=0.6,
        impact_score=0.9,
        is_actionable=True,
        confidence_score=0.8,
        summary="Summary",
        beneficiary_companies=[BeneficiaryCompany(company="Power Grid", relation="transmission", reason="", benefit_score=0.7, ticker="POWERGRID.NS")],
    )
    item = scorer.score_item(article, analysis, primary_ticker=None, beneficiary_scores={"Power Grid": 0.8})
    assert 0.0 <= item.overall_score <= 1.0
    assert item.freshness_score > 0.8
    assert item.price_opportunity_score >= 0.0


def test_price_reaction_filter_flags_extended_move(monkeypatch):
    scorer = NewsScorer(price_reaction_max_abs_pct=3.5)

    class _FakeTicker:
        def history(self, period="6d", interval="1d", auto_adjust=False):
            return pd.DataFrame({"Close": [100.0, 106.0]})

    monkeypatch.setattr("app.news.scorer.yf.Ticker", lambda ticker: _FakeTicker())

    score, meta = scorer.price_opportunity_score("RELIANCE.NS")
    assert meta["moved_too_much"] is True
    assert meta["price_reaction_ok"] is False
    assert score < 0.5


def test_helper_scoring_functions_cover_timing_buckets():
    scorer = NewsScorer(early_move_threshold_pct=1.5, late_move_threshold_pct=4.0)

    assert scorer.classify_timing_label(1.0) == "early"
    assert scorer.classify_timing_label(2.5) == "moderate"
    assert scorer.classify_timing_label(5.2) == "late"

    early_score = scorer.compute_price_opportunity_score(1.0)
    moderate_score = scorer.compute_price_opportunity_score(2.5)
    late_score = scorer.compute_price_opportunity_score(5.2)
    assert early_score > moderate_score > late_score

    freshness = scorer.compute_freshness_score(datetime.now(timezone.utc) - timedelta(hours=2))
    overall = scorer.compute_overall_opportunity_score(
        impact_score=0.8,
        relation_strength=0.7,
        freshness_score=freshness,
        price_opportunity_score=early_score,
    )
    assert 0.0 <= freshness <= 1.0
    assert 0.0 <= overall <= 1.0
