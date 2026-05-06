from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from app.services.automation_service import automation_service

router = APIRouter(prefix="/automation", tags=["automation"])


@router.post("/run")
def run_automation_cycle(model: Optional[str] = None, interval_minutes: Optional[int] = None):
    try:
        interval = int(interval_minutes) if interval_minutes is not None else None
        return automation_service.run_cycle(model_override=model, interval_minutes_override=interval)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/latest")
def get_latest_automation_result():
    try:
        return automation_service.latest()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/summary")
def generate_ai_summary(predictions_limit: int = 20, news_limit: int = 25):
    try:
        return automation_service.generate_ai_summary(
            predictions_limit=int(predictions_limit),
            news_limit=int(news_limit),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/summary/latest")
def get_latest_ai_summary():
    try:
        return automation_service.latest_ai_summary()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
