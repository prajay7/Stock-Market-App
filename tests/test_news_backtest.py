from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from app.news.backtest_service import NewsSignalBacktestService
from src.data.metadata_store import MetadataStore


def _make_service(tmp_path, horizons=None, benchmark_ticker=""):
    db_path = tmp_path / "backtest.db"
    settings = SimpleNamespace(
        database_url=f"sqlite:///{db_path}",
        backtest_horizons=horizons or [1, 3, 5, 7],
        benchmark_ticker=benchmark_ticker,
        trading_day_fallback_mode="next_available_close",
        max_backtest_rows_in_ui=500,
    )
    return NewsSignalBacktestService(settings), MetadataStore(settings.database_url)


def _seed_opportunity(store: MetadataStore, created_at: datetime, score: float = 0.72) -> int:
    signal_id = store.upsert_analyzed_news_signal(
        {
            "article_hash": f"h-{created_at.timestamp()}",
            "title": "Signal headline",
            "link": "https://example.com/x",
            "source": "Feed",
            "published_at": created_at,
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
            "created_at": created_at,
        }
    )
    store.write_beneficiary_opportunities(
        [
            {
                "signal_id": signal_id,
                "company": "Power Grid",
                "ticker": "POWERGRID.NS",
                "relation": "transmission",
                "relation_strength": 0.75,
                "benefit_score": score,
                "freshness_score": 0.9,
                "price_change_pct": 1.2,
                "price_opportunity_score": 0.85,
                "overall_score": score,
                "timing_label": "early",
                "reason": "Test",
                "signal_price": 100.0,
                "signal_timestamp": created_at,
                "price_source": "yfinance_close",
                "created_at": created_at,
            }
        ]
    )
    rows = store.read_beneficiary_opportunities_with_signal(limit=200)
    matching = [r for r in rows if int(r.get("signal_id", -1)) == int(signal_id)]
    matching.sort(key=lambda r: int(r.get("id", 0)), reverse=True)
    return int(matching[0]["id"])


def test_forward_return_calculation():
    payload = NewsSignalBacktestService.compute_forward_return(100.0, 106.0)
    assert round(payload["absolute_return"], 2) == 6.0
    assert round(payload["percent_return"], 2) == 6.0
    assert payload["is_positive"] is True


def test_non_trading_day_fallback_uses_next_available_close(tmp_path, monkeypatch):
    service, _store = _make_service(tmp_path)

    class _FakeTicker:
        def history(self, start, end, interval="1d", auto_adjust=False):
            idx = pd.to_datetime(["2026-04-20", "2026-04-21"])  # Monday, Tuesday
            return pd.DataFrame({"Close": [101.0, 102.0]}, index=idx)

    monkeypatch.setattr("app.news.backtest_service.yf.Ticker", lambda _: _FakeTicker())
    # Target on weekend should fallback to Monday.
    price, price_dt = service.get_price_on_or_after("POWERGRID.NS", datetime(2026, 4, 19))
    assert price == 101.0
    assert price_dt is not None


def test_pending_and_completed_status_logic(tmp_path, monkeypatch):
    service, store = _make_service(tmp_path, horizons=[1, 3])
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    created = now - timedelta(days=2)
    opp_id = _seed_opportunity(store, created_at=created)

    monkeypatch.setattr(service, "_utc_now", lambda: now)
    monkeypatch.setattr(service, "get_price_on_or_after", lambda ticker, dt: (105.0, dt))

    outcomes = service.evaluate_opportunity_outcomes(opp_id)
    by_h = {int(o["evaluation_horizon_days"]): o for o in outcomes}
    assert by_h[1]["evaluation_status"] == "completed"
    assert by_h[3]["evaluation_status"] == "pending"


def test_duplicate_outcome_prevention_via_upsert(tmp_path, monkeypatch):
    service, store = _make_service(tmp_path, horizons=[1, 3])
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    created = now - timedelta(days=5)
    opp_id = _seed_opportunity(store, created_at=created)

    monkeypatch.setattr(service, "_utc_now", lambda: now)
    monkeypatch.setattr(service, "get_price_on_or_after", lambda ticker, dt: (110.0, dt))

    service.evaluate_opportunity_outcomes(opp_id)
    service.evaluate_opportunity_outcomes(opp_id)

    rows = [r for r in store.read_signal_outcomes(limit=50) if int(r["opportunity_id"]) == opp_id]
    assert len(rows) == 2  # one row per horizon only


def test_metrics_aggregation_and_score_bucket(tmp_path, monkeypatch):
    service, store = _make_service(tmp_path, horizons=[1])
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)

    opp_1 = _seed_opportunity(store, created_at=now - timedelta(days=4), score=0.35)
    opp_2 = _seed_opportunity(store, created_at=now - timedelta(days=4, minutes=1), score=0.85)

    monkeypatch.setattr(service, "_utc_now", lambda: now)

    price_map = {
        opp_1: 95.0,
        opp_2: 110.0,
    }

    def _mock_get_price_on_or_after(ticker, dt):
        # Return by inferred opportunity via entry date ordering from DB row calls.
        # For this test, any call after entry returns one of these values via horizon loops.
        if dt.date() >= (now - timedelta(days=3)).date():
            return 110.0, dt
        return 95.0, dt

    monkeypatch.setattr(service, "get_price_on_or_after", _mock_get_price_on_or_after)

    service.evaluate_opportunity_outcomes(opp_1)
    service.evaluate_opportunity_outcomes(opp_2)

    summary = service.summary()
    assert summary.totals["total_signals"] >= 2
    assert "hit_rate" in summary.totals
    buckets = {row.get("score_bucket") for row in summary.by_score_bucket}
    assert buckets


def test_benchmark_alpha_calculation(tmp_path, monkeypatch):
    service, store = _make_service(tmp_path, horizons=[1], benchmark_ticker="^NSEI")
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    opp_id = _seed_opportunity(store, created_at=now - timedelta(days=3))

    monkeypatch.setattr(service, "_utc_now", lambda: now)

    def _mock_get_price_on_or_after(ticker, dt):
        if ticker == "POWERGRID.NS":
            return 110.0, dt
        if ticker == "^NSEI":
            if dt.date() <= (now - timedelta(days=3)).date():
                return 100.0, dt
            return 102.0, dt
        return None, None

    monkeypatch.setattr(service, "get_price_on_or_after", _mock_get_price_on_or_after)

    outcomes = service.evaluate_opportunity_outcomes(opp_id)
    completed = next(o for o in outcomes if o["evaluation_status"] == "completed")
    assert completed["benchmark_return"] is not None
    assert completed["alpha_return"] is not None
