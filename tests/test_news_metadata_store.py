from datetime import datetime

from src.data.metadata_store import MetadataStore


def test_news_signal_history_persistence(tmp_path):
    db_path = tmp_path / "test_news.db"
    store = MetadataStore(f"sqlite:///{db_path}")

    generated_at = datetime.utcnow()
    store.upsert_news_impact_items(
        [
            {
                "generated_at": generated_at,
                "article_hash": "hash-1",
                "source": "Feed",
                "title": "Headline",
                "link": "https://example.com/a",
                "published_at": generated_at,
                "primary_company": "NTPC",
                "primary_ticker": "NTPC.NS",
                "event_type": "capacity_expansion",
                "sentiment_label": "positive",
                "impact_score": 0.8,
                "confidence_score": 0.7,
                "is_actionable": True,
                "freshness_score": 0.9,
                "relation_strength": 0.6,
                "price_opportunity_score": 0.75,
                "overall_score": 0.78,
                "payload": {"ok": True},
            }
        ]
    )

    store.write_news_impact_signals(
        [
            {
                "generated_at": generated_at,
                "article_hash": "hash-1",
                "headline": "Headline",
                "source": "Feed",
                "published_at": generated_at,
                "primary_company": "NTPC",
                "primary_ticker": "NTPC.NS",
                "beneficiary_company": "Power Grid",
                "beneficiary_ticker": "POWERGRID.NS",
                "relation": "transmission",
                "relation_strength": 0.7,
                "sentiment_label": "positive",
                "event_type": "capacity_expansion",
                "impact_score": 0.8,
                "freshness_score": 0.9,
                "price_change_pct_1d": 1.2,
                "price_reaction_ok": True,
                "price_opportunity_score": 0.85,
                "signal_score": 0.82,
                "is_early_opportunity": True,
                "reason": "Test",
                "payload": {"kind": "signal"},
            }
        ]
    )

    history = store.read_news_signal_history(limit=10)
    assert len(history) == 1
    assert history[0]["beneficiary_company"] == "Power Grid"
    assert history[0]["is_early_opportunity"] is True


def test_signal_outcome_upsert_prevents_duplicates(tmp_path):
    db_path = tmp_path / "test_outcomes.db"
    store = MetadataStore(f"sqlite:///{db_path}")

    created = datetime.utcnow()
    signal_id = store.upsert_analyzed_news_signal(
        {
            "article_hash": "hash-x",
            "title": "Headline",
            "link": "https://example.com/a",
            "source": "Feed",
            "published_at": created,
            "primary_company": "NTPC",
            "primary_ticker": "NTPC.NS",
            "sector": "Power",
            "event_type": "capacity_expansion",
            "sentiment_label": "positive",
            "sentiment_score": 0.6,
            "impact_score": 0.8,
            "confidence_score": 0.7,
            "is_actionable": True,
            "summary": "Summary",
            "created_at": created,
        }
    )
    store.write_beneficiary_opportunities(
        [
            {
                "signal_id": signal_id,
                "company": "Power Grid",
                "ticker": "POWERGRID.NS",
                "relation": "transmission",
                "relation_strength": 0.7,
                "benefit_score": 0.8,
                "freshness_score": 0.9,
                "price_change_pct": 1.0,
                "price_opportunity_score": 0.9,
                "overall_score": 0.85,
                "timing_label": "early",
                "reason": "Test",
                "signal_price": 100.0,
                "signal_timestamp": created,
                "price_source": "yfinance_close",
                "created_at": created,
            }
        ]
    )
    opportunity_id = int(store.read_beneficiary_opportunities_with_signal(limit=1)[0]["id"])

    row = {
        "opportunity_id": opportunity_id,
        "ticker": "POWERGRID.NS",
        "evaluation_horizon_days": 1,
        "target_date": created,
        "entry_price": 100.0,
        "exit_price": 102.0,
        "absolute_return": 2.0,
        "percent_return": 2.0,
        "benchmark_return": 1.0,
        "alpha_return": 1.0,
        "is_positive": True,
        "evaluation_status": "completed",
        "evaluated_at": created,
        "created_at": created,
    }
    store.upsert_signal_outcome(row)
    store.upsert_signal_outcome({**row, "exit_price": 103.0, "percent_return": 3.0, "absolute_return": 3.0})

    outcomes = [r for r in store.read_signal_outcomes(limit=10) if int(r["opportunity_id"]) == opportunity_id]
    assert len(outcomes) == 1
    assert float(outcomes[0]["percent_return"]) == 3.0
