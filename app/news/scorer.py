from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from app.news.models import AnalyzedNewsItem, BeneficiaryCompany, NewsAnalysisResult, NewsArticle


@dataclass
class NewsScorer:
    impact_weight: float = 0.35
    relation_weight: float = 0.25
    freshness_weight: float = 0.20
    price_weight: float = 0.20
    timeout_sec: float = 20.0
    price_reaction_max_abs_pct: float = 3.5
    early_move_threshold_pct: float = 1.5
    late_move_threshold_pct: float = 4.0

    def compute_freshness_score(self, published_at: datetime | None) -> float:
        if published_at is None:
            return 0.40
        now = datetime.now(timezone.utc)
        dt = published_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
        if age_hours <= 1:
            return 1.0
        if age_hours <= 6:
            return 0.90
        if age_hours <= 12:
            return 0.80
        if age_hours <= 24:
            return 0.70
        if age_hours <= 48:
            return 0.55
        if age_hours <= 72:
            return 0.40
        return 0.25

    def freshness_score(self, published_at: datetime | None) -> float:
        return self.compute_freshness_score(published_at)

    def classify_timing_label(self, day_change_pct: float | None) -> str:
        if day_change_pct is None:
            return "moderate"
        abs_change = abs(float(day_change_pct))
        early_threshold = max(0.1, float(self.early_move_threshold_pct))
        late_threshold = max(early_threshold, float(self.late_move_threshold_pct))
        if abs_change <= early_threshold:
            return "early"
        if abs_change <= late_threshold:
            return "moderate"
        return "late"

    def compute_price_opportunity_score(self, day_change_pct: float | None) -> float:
        if day_change_pct is None:
            return 0.50
        abs_change = abs(float(day_change_pct))
        timing = self.classify_timing_label(day_change_pct)
        if timing == "early":
            # 0-1.5% maps to ~1.0 down to 0.7
            return max(0.70, 1.0 - (abs_change / max(0.1, float(self.early_move_threshold_pct))) * 0.30)
        if timing == "moderate":
            span = max(0.1, float(self.late_move_threshold_pct) - float(self.early_move_threshold_pct))
            frac = (abs_change - float(self.early_move_threshold_pct)) / span
            # 1.5-4% maps from 0.7 to 0.35
            return max(0.35, 0.70 - max(0.0, min(1.0, frac)) * 0.35)
        # >4% considered late and penalized.
        return max(0.05, 0.35 - min((abs_change - float(self.late_move_threshold_pct)) / 8.0, 1.0) * 0.30)

    def price_opportunity_score(self, ticker: str | None) -> tuple[float, dict[str, Any]]:
        if not ticker:
            return 0.50, {
                "reason": "missing_ticker",
                "moved_too_much": False,
                "price_reaction_ok": True,
                "change_pct": None,
                "timing_label": "moderate",
            }
        try:
            hist = yf.Ticker(ticker).history(period="6d", interval="1d", auto_adjust=False)
        except Exception as exc:
            return 0.50, {
                "reason": f"yfinance_error:{type(exc).__name__}",
                "moved_too_much": False,
                "price_reaction_ok": True,
                "change_pct": None,
                "timing_label": "moderate",
            }
        if hist is None or hist.empty:
            return 0.50, {
                "reason": "no_price_data",
                "moved_too_much": False,
                "price_reaction_ok": True,
                "change_pct": None,
                "timing_label": "moderate",
            }

        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if len(close) < 2:
            return 0.50, {
                "reason": "insufficient_history",
                "moved_too_much": False,
                "price_reaction_ok": True,
                "change_pct": None,
                "timing_label": "moderate",
            }

        latest = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        if prev == 0:
            return 0.50, {
                "reason": "zero_previous_close",
                "moved_too_much": False,
                "price_reaction_ok": True,
                "change_pct": None,
                "timing_label": "moderate",
            }

        change_pct = ((latest - prev) / prev) * 100.0
        abs_change = abs(change_pct)
        threshold = max(0.1, float(self.late_move_threshold_pct))
        moved_too_much = abs_change >= threshold
        score = self.compute_price_opportunity_score(change_pct)
        score = max(0.0, min(1.0, score))
        return score, {
            "reason": "ok",
            "change_pct": change_pct,
            "latest_close": latest,
            "moved_too_much": moved_too_much,
            "price_reaction_ok": not moved_too_much,
            "threshold_pct": threshold,
            "timing_label": self.classify_timing_label(change_pct),
        }

    def relation_strength(self, beneficiaries: list[BeneficiaryCompany]) -> float:
        if not beneficiaries:
            return 0.0
        strengths = []
        for item in beneficiaries:
            if item.relation_strength is None:
                strengths.append(float(item.benefit_score))
            else:
                strengths.append(float(item.relation_strength))
        return max(strengths) if strengths else 0.0

    def compute_overall_opportunity_score(self, impact_score: float, relation_strength: float, freshness_score: float, price_opportunity_score: float) -> float:
        score = (
            self.impact_weight * float(impact_score)
            + self.relation_weight * float(relation_strength)
            + self.freshness_weight * float(freshness_score)
            + self.price_weight * float(price_opportunity_score)
        )
        return max(0.0, min(1.0, score))

    def overall_score(self, analysis: NewsAnalysisResult, freshness: float, price_score: float) -> float:
        relation_strength = self.relation_strength(analysis.beneficiary_companies)
        return self.compute_overall_opportunity_score(analysis.impact_score, relation_strength, freshness, price_score)

    def rank_beneficiaries(
        self,
        analysis: NewsAnalysisResult,
        freshness: float,
        price_scores: dict[str, float] | None = None,
        price_meta: dict[str, dict[str, Any]] | None = None,
    ) -> list[BeneficiaryCompany]:
        price_scores = price_scores or {}
        price_meta = price_meta or {}
        ranked: list[BeneficiaryCompany] = []
        for item in analysis.beneficiary_companies:
            base_price = price_scores.get(item.company, 0.50)
            base_relation = item.relation_strength if item.relation_strength is not None else item.benefit_score
            meta = price_meta.get(item.company, {})
            combined = (
                self.impact_weight * analysis.impact_score
                + self.relation_weight * float(base_relation)
                + self.freshness_weight * freshness
                + self.price_weight * base_price
            )
            ranked.append(
                BeneficiaryCompany(
                    company=item.company,
                    relation=item.relation,
                    reason=item.reason,
                    benefit_score=max(0.0, min(1.0, combined)),
                    ticker=item.ticker,
                    relation_strength=max(0.0, min(1.0, float(base_relation))),
                    price_change_pct_1d=meta.get("change_pct"),
                    price_reaction_ok=meta.get("price_reaction_ok"),
                )
            )
        ranked.sort(key=lambda x: x.benefit_score, reverse=True)
        return ranked

    def score_item(
        self,
        article: NewsArticle,
        analysis: NewsAnalysisResult,
        primary_ticker: str | None = None,
        beneficiary_scores: dict[str, float] | None = None,
        beneficiary_price_meta: dict[str, dict[str, Any]] | None = None,
    ) -> AnalyzedNewsItem:
        freshness = self.freshness_score(article.published_at)
        primary_price_score, _primary_price_meta = self.price_opportunity_score(primary_ticker)
        beneficiary_scores = beneficiary_scores or {}
        if beneficiary_scores:
            price_score = max(beneficiary_scores.values())
        else:
            price_score = primary_price_score
        overall = self.overall_score(analysis, freshness, price_score)
        ranked_beneficiaries = self.rank_beneficiaries(
            analysis,
            freshness,
            beneficiary_scores,
            price_meta=beneficiary_price_meta,
        )
        return AnalyzedNewsItem(
            article=article,
            analysis=analysis,
            freshness_score=freshness,
            relation_strength=self.relation_strength(analysis.beneficiary_companies),
            price_opportunity_score=price_score,
            overall_score=overall,
            ranked_beneficiaries=ranked_beneficiaries,
        )
