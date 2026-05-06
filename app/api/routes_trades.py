from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.news.models import PaperTradeClose, PaperTradeCreate
from app.services.paper_trading_service import paper_trading_service
from src.data.metadata_store import metadata_store

router = APIRouter(prefix="/trades", tags=["trades"])


@router.post("")
def create_trade(payload: PaperTradeCreate):
    try:
        trade_id = paper_trading_service.create_trade(payload)
        return {"id": trade_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/from-alert/{alert_id}")
def create_trade_from_alert(alert_id: int, payload: PaperTradeCreate):
    try:
        trade_id = paper_trading_service.create_trade_from_alert(
            alert_id,
            entry_price=payload.entry_price,
            quantity=payload.quantity,
            capital=payload.capital,
            notes=payload.notes,
            trade_reason=payload.trade_reason,
        )
        return {"id": trade_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/from-opportunity/{opportunity_id}")
def create_trade_from_opportunity(opportunity_id: int, payload: PaperTradeCreate):
    try:
        trade_id = paper_trading_service.create_trade_from_opportunity(
            opportunity_id,
            entry_price=payload.entry_price,
            quantity=payload.quantity,
            capital=payload.capital,
            notes=payload.notes,
            trade_reason=payload.trade_reason,
        )
        return {"id": trade_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("")
def list_trades(
    limit: int = Query(200, ge=1, le=5000),
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    source_type: Optional[str] = None,
):
    try:
        trades = metadata_store.read_paper_trades(limit=limit, status=status, symbol=symbol, source_type=source_type)
        return {"trades": trades}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{trade_id}")
def get_trade(trade_id: int):
    trade = metadata_store.read_paper_trade(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade


@router.put("/{trade_id}")
def update_trade(trade_id: int, payload: dict):
    try:
        metadata_store.update_paper_trade(trade_id, payload)
        return {"status": "updated"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{trade_id}/close")
def close_trade(trade_id: int, payload: PaperTradeClose):
    try:
        paper_trading_service.close_trade(trade_id, payload)
        return {"status": "closed"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/refresh")
def refresh_open_trades():
    try:
        updated = paper_trading_service.refresh_open_trades()
        return {"updated": updated}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/analytics")
def get_analytics():
    try:
        return paper_trading_service.get_analytics()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
