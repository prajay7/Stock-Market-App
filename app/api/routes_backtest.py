from fastapi import APIRouter, HTTPException, Query

from app.api.error_utils import build_error_envelope
from app.services.backtest_service import backtest_service

router = APIRouter(tags=["backtest"])


@router.get("/backtest")
def backtest(
    symbols: str = Query(..., description="Comma separated symbols"),
    start: str = Query(...),
    end: str = Query(...),
    horizon_days: int = Query(1, ge=1, le=30),
    top_n: int = Query(5, ge=1, le=100),
    model_name: str = Query("xgboost_classifier", min_length=2),
    mode: str = Query("static", pattern="^(static|walk_forward)$"),
    retrain_every_days: int = Query(20, ge=1, le=365),
    min_train_rows: int = Query(300, ge=30, le=100000),
    train_lookback_days: int = Query(0, ge=0, le=10000),
) -> dict:
    parsed_symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    try:
        return backtest_service.backtest(
            parsed_symbols,
            start,
            end,
            horizon_days,
            top_n,
            model_name,
            mode=mode,
            retrain_every_days=retrain_every_days,
            min_train_rows=min_train_rows,
            train_lookback_days=(train_lookback_days if train_lookback_days > 0 else None),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=build_error_envelope(
                code="BACKTEST_ERROR",
                message="Backtest failed",
                details={"reason": str(exc)},
            ),
        ) from exc
