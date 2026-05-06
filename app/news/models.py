from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class NewsArticle(BaseModel):
    article_hash: str
    title: str
    summary: str = ""
    link: str
    source: str
    published_at: Optional[datetime] = None
    raw_text: str = ""

    @field_validator("title", "summary", "link", "source", "raw_text", mode="before")
    @classmethod
    def _coerce_str(cls, value):
        return "" if value is None else str(value)


class BeneficiaryCompany(BaseModel):
    company: str
    relation: str
    reason: str
    benefit_score: float = Field(ge=0.0, le=1.0)
    ticker: Optional[str] = None
    relation_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    price_change_pct_1d: Optional[float] = None
    price_reaction_ok: Optional[bool] = None


class NewsAnalysisResult(BaseModel):
    primary_company: Optional[str] = None
    sector: Optional[str] = None
    event_type: str = "other"
    sentiment_label: str = Field(pattern="^(positive|neutral|negative)$")
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    impact_score: float = Field(ge=0.0, le=1.0)
    is_actionable: bool = False
    confidence_score: float = Field(ge=0.0, le=1.0)
    summary: str = ""
    beneficiary_companies: List[BeneficiaryCompany] = Field(default_factory=list)


class AnalyzedNewsItem(BaseModel):
    article: NewsArticle
    analysis: NewsAnalysisResult
    freshness_score: float = Field(ge=0.0, le=1.0)
    relation_strength: float = Field(ge=0.0, le=1.0)
    price_opportunity_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    ranked_beneficiaries: List[BeneficiaryCompany] = Field(default_factory=list)


class NewsImpactSignal(BaseModel):
    generated_at: datetime
    article_hash: str
    headline: str
    source: str
    published_at: Optional[datetime] = None
    primary_company: Optional[str] = None
    primary_ticker: Optional[str] = None
    beneficiary_company: str
    beneficiary_ticker: Optional[str] = None
    relation: str
    relation_strength: float = Field(ge=0.0, le=1.0)
    sentiment_label: str = Field(pattern="^(positive|neutral|negative)$")
    event_type: str = "other"
    impact_score: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    signal_price: Optional[float] = None
    price_change_pct_1d: Optional[float] = None
    price_reaction_ok: bool = True
    timing_label: str = Field(pattern="^(early|moderate|late)$")
    price_opportunity_score: float = Field(ge=0.0, le=1.0)
    signal_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    is_early_opportunity: bool = False
    reason: str = ""


class NewsImpactScanResult(BaseModel):
    generated_at: datetime
    articles: List[AnalyzedNewsItem]
    top_opportunities: List[NewsImpactSignal]
    signal_history: List[NewsImpactSignal] = Field(default_factory=list)


class SignalOutcome(BaseModel):
    opportunity_id: int
    ticker: str
    evaluation_horizon_days: int
    target_date: datetime
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    absolute_return: Optional[float] = None
    percent_return: Optional[float] = None
    benchmark_return: Optional[float] = None
    alpha_return: Optional[float] = None
    is_positive: Optional[bool] = None
    evaluation_status: str = Field(pattern="^(pending|completed|failed)$")
    evaluated_at: Optional[datetime] = None
    created_at: datetime


class SignalBacktestSummary(BaseModel):
    totals: dict[str, Any]
    by_horizon: list[dict[str, Any]] = Field(default_factory=list)
    by_event_type: list[dict[str, Any]] = Field(default_factory=list)
    by_sector: list[dict[str, Any]] = Field(default_factory=list)
    by_relation_type: list[dict[str, Any]] = Field(default_factory=list)
    by_timing_label: list[dict[str, Any]] = Field(default_factory=list)
    by_score_bucket: list[dict[str, Any]] = Field(default_factory=list)


# Watchlist and alert models

class WatchlistItem(BaseModel):
    id: Optional[int] = None
    watchlist_id: int
    item_type: str
    item_value: str
    normalized_value: str
    created_at: Optional[datetime] = None


class Watchlist(BaseModel):
    id: Optional[int] = None
    name: str
    description: str = ""
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    items: list[WatchlistItem] = Field(default_factory=list)


class AlertRule(BaseModel):
    id: Optional[int] = None
    name: str
    is_active: bool = True
    watchlist_id: Optional[int] = None
    sentiment_filter: Optional[str] = None  # positive/neutral/negative
    actionable_only: bool = False
    min_confidence_score: Optional[float] = None
    min_impact_score: Optional[float] = None
    min_overall_score: Optional[float] = None
    timing_labels: Optional[list[str]] = None
    event_types: Optional[list[str]] = None
    sectors: Optional[list[str]] = None
    max_price_move_pct: Optional[float] = None
    notification_channels: list[str] = Field(default_factory=lambda: ["in_app"])
    cooldown_minutes: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("notification_channels", mode="before")
    @classmethod
    def validate_notification_channels(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v) if v.startswith("[") else [v]
            except:
                return ["in_app"]
        return v or ["in_app"]


class GeneratedAlert(BaseModel):
    id: Optional[int] = None
    rule_id: int
    signal_id: Optional[int] = None
    opportunity_id: Optional[int] = None
    title: str
    message: str
    severity: str = Field(pattern="^(info|warning|high)$")
    status: str = Field(pattern="^(new|seen|sent|failed)$")
    notification_channel: str
    dedupe_key: str
    created_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    error_message: Optional[str] = None


class AlertPayload(BaseModel):
    rule_id: int
    signal_id: Optional[int] = None
    opportunity_id: Optional[int] = None
    title: str
    message: str
    severity: str
    notification_channels: list[str]
    dedupe_key: str
    beneficiary_company: Optional[str] = None
    beneficiary_ticker: Optional[str] = None
    primary_company: Optional[str] = None
    primary_ticker: Optional[str] = None
    event_type: Optional[str] = None
    sector: Optional[str] = None
    sentiment: Optional[str] = None
    confidence_score: Optional[float] = None
    overall_score: Optional[float] = None
    timing_label: Optional[str] = None
    headline: Optional[str] = None
    source: Optional[str] = None


class PaperTradeCreate(BaseModel):
    symbol: str
    entry_price: Optional[float] = None
    quantity: Optional[float] = None
    capital: Optional[float] = None
    source_type: str = Field(default="manual", pattern="^(manual|alert|opportunity)$")
    source_id: Optional[int] = None
    source_label: Optional[str] = None
    entry_date: Optional[datetime] = None
    notes: str = ""
    trade_reason: str = ""

    @field_validator("symbol", mode="before")
    @classmethod
    def _normalize_symbol(cls, value):
        return str(value or "").strip().upper()


class PaperTradeClose(BaseModel):
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    notes: str = ""


class PaperTrade(BaseModel):
    id: Optional[int] = None
    symbol: str
    source_type: str = Field(pattern="^(manual|alert|opportunity)$")
    source_id: Optional[int] = None
    source_label: Optional[str] = None
    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    quantity: Optional[float] = None
    capital: Optional[float] = None
    status: str = Field(pattern="^(open|closed)$")
    notes: str = ""
    trade_reason: str = ""
    current_price: Optional[float] = None
    current_pnl: Optional[float] = None
    current_return_pct: Optional[float] = None
    realized_pnl: Optional[float] = None
    realized_return_pct: Optional[float] = None
    holding_days: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PaperTradeAnalytics(BaseModel):
    total_trades: int = 0
    open_trades: int = 0
    closed_trades: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    total_pnl: float = 0.0
    avg_holding_days: float = 0.0

