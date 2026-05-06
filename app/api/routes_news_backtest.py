from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.news.backtest_service import news_signal_backtest_service

router = APIRouter(prefix="/news/backtest", tags=["news-backtest"])


@router.get("/summary")
def backtest_summary(
    event_type: str | None = Query(None),
    sector: str | None = Query(None),
    sentiment: str | None = Query(None),
    timing_label: str | None = Query(None),
    relation: str | None = Query(None),
    horizon_days: int | None = Query(None),
    min_confidence: float | None = Query(None),
    min_overall_score: float | None = Query(None),
) -> dict:
    try:
        filters = {
            "event_type": [event_type] if event_type else None,
            "sector": [sector] if sector else None,
            "sentiment_label": [sentiment] if sentiment else None,
            "timing_label": [timing_label] if timing_label else None,
            "relation": [relation] if relation else None,
            "horizon_days": [horizon_days] if horizon_days else None,
            "min_confidence": min_confidence,
            "min_overall_score": min_overall_score,
        }
        return news_signal_backtest_service.summary(filters=filters).model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/signals")
def backtest_signals(
    horizon_days: int | None = Query(None),
    max_rows: int = Query(500, ge=1, le=5000),
) -> list[dict]:
    try:
        filters = {"horizon_days": [horizon_days] if horizon_days else None}
        return news_signal_backtest_service.signal_history(filters=filters, limit=max_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/run")
def run_backtest() -> dict:
    try:
        return news_signal_backtest_service.evaluate_pending_outcomes()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
