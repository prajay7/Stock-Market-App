from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class IngestHistoricalRequest(BaseModel):
    symbols: List[str] = Field(min_length=1, max_length=500)
    interval: str = Field(default="1d", min_length=2, max_length=10)
    lookback_days: int = Field(default=3650, ge=30, le=20000)

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, value: List[str]) -> List[str]:
        cleaned = []
        seen = set()
        for item in value:
            sym = str(item or "").strip().upper()
            if not sym:
                continue
            if sym not in seen:
                cleaned.append(sym)
                seen.add(sym)
        if not cleaned:
            raise ValueError("symbols must contain at least one non-empty ticker")
        return cleaned


class IngestNewsRequest(BaseModel):
    symbols: List[str] = Field(min_length=1, max_length=500)
    limit_per_symbol: int = Field(default=100, ge=1, le=1000)

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, value: List[str]) -> List[str]:
        cleaned = []
        seen = set()
        for item in value:
            sym = str(item or "").strip().upper()
            if not sym:
                continue
            if sym not in seen:
                cleaned.append(sym)
                seen.add(sym)
        if not cleaned:
            raise ValueError("symbols must contain at least one non-empty ticker")
        return cleaned


class TrainRequest(BaseModel):
    symbols: List[str] = Field(min_length=1, max_length=500)
    horizon_days: int = Field(default=1, ge=1, le=30)
    task_type: str = Field(default="classification", pattern="^(classification|regression_return|regression_close|movement)$")

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, value: List[str]) -> List[str]:
        cleaned = []
        seen = set()
        for item in value:
            sym = str(item or "").strip().upper()
            if not sym:
                continue
            if sym not in seen:
                cleaned.append(sym)
                seen.add(sym)
        if not cleaned:
            raise ValueError("symbols must contain at least one non-empty ticker")
        return cleaned


class PredictRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, max_length=500)
    horizon_days: int = Field(default=1, ge=1, le=30)
    model_name: str = Field(default="xgboost_classifier", min_length=2, max_length=100)
    atr_multiplier: float = Field(default=1.0, ge=0.25, le=10.0)
    include_live_quote: bool = False
    use_trending: bool = Field(default=False, description="Use trending symbols from news instead of provided symbols")

    @field_validator("symbols", mode="before")
    @classmethod
    def _validate_symbols(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None or not value:
            return None
        cleaned = []
        seen = set()
        for item in value:
            sym = str(item or "").strip().upper()
            if not sym:
                continue
            if sym not in seen:
                cleaned.append(sym)
                seen.add(sym)
        if not cleaned:
            return None
        return cleaned


class PredictionItem(BaseModel):
    symbol: str
    interval: Optional[str] = None
    prediction_time: Optional[str] = None
    price_as_of: Optional[str] = None
    price_as_of_time: Optional[str] = None
    current_price: Optional[float] = None
    predicted_price: Optional[float] = None
    probability: Optional[float] = None
    signal: Optional[str] = None
    reason: Optional[str] = None
    risk_level: Optional[str] = None
    model_version: Optional[str] = None
    live_price: Optional[float] = None
    live_price_as_of: Optional[str] = None
    live_price_as_of_time: Optional[str] = None
    live_price_source: Optional[str] = None
    target_price: Optional[float] = None
    news_adjusted_target_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    prob_up: Optional[float] = None
    news_adjusted_prob_up: Optional[float] = None
    predicted_return: Optional[float] = None
    news_adjusted_predicted_return: Optional[float] = None
    confidence: Optional[float] = None
    news_adjusted_confidence: Optional[float] = None
    latest_sentiment: Optional[float] = None
    news_count: Optional[int] = None
    news_impact_score: Optional[float] = None
    news_signal_score: Optional[float] = None
    news_probability_boost: Optional[float] = None
    decision: str
    news_decision: Optional[str] = None
    news_reason: Optional[str] = None


class PredictResponse(BaseModel):
    generated_at: datetime
    predictions: List[PredictionItem]


class BacktestRequest(BaseModel):
    symbols: List[str] = Field(min_length=1, max_length=500)
    start: str = Field(min_length=10, max_length=32)
    end: str = Field(min_length=10, max_length=32)
    horizon_days: int = Field(default=1, ge=1, le=30)
    top_n: int = Field(default=5, ge=1, le=100)
    model_name: str = Field(default="xgboost_classifier", min_length=2, max_length=100)


class BacktestResponse(BaseModel):
    metrics: dict
    rows: int


class TrainResponse(BaseModel):
    trained_at: datetime
    task_type: str
    target: str
    best: dict
    models: dict


class ErrorResponse(BaseModel):
    detail: str


class ApiErrorPayload(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)
    trace_id: Optional[str] = None
    timestamp: datetime


class ApiErrorEnvelope(BaseModel):
    error: ApiErrorPayload
