from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from app.news.models import BeneficiaryCompany, NewsAnalysisResult, NewsArticle
from app.news.relations import CompanyRelations

logger = logging.getLogger(__name__)

EVENT_KEYWORDS: dict[str, list[str]] = {
    "earnings_beat": ["beat estimates", "beats estimates", "beats earnings", "earnings beat", "record profit", "higher profit"],
    "earnings_miss": ["miss estimates", "misses estimates", "earnings miss", "lower profit", "profit drops"],
    "buyback": ["buyback", "share repurchase", "repurchase program"],
    "large_order": ["large order", "order win", "wins order", "big order", "contract win"],
    "merger": ["merger", "merge"],
    "acquisition": ["acquire", "acquisition", "buyout", "takes over"],
    "partnership": ["partnership", "partners with", "collaboration", "strategic tie-up"],
    "capacity_expansion": ["capacity expansion", "expand capacity", "expansion", "new plant", "add capacity"],
    "regulatory_approval": ["approval", "regulatory", "license", "clearance", "nod"],
    "management_change": ["ceo", "cfo", "management change", "resigns", "appoints"],
    "sector_tailwind": ["tailwind", "boosts sector", "sector gains", "industry up"],
    "commodity_price_change": ["crude", "oil price", "commodity prices", "copper prices", "steel prices"],
    "litigation": ["lawsuit", "litigation", "court", "legal"],
    "plant_shutdown": ["shutdown", "plant closed", "halt production", "strike"],
    "guidance_upgrade": ["raises guidance", "guidance upgrade", "upgrades outlook", "raises outlook"],
    "guidance_cut": ["cuts guidance", "lowers guidance", "weak outlook", "guidance cut"],
    "dividend": ["dividend", "payout", "interim dividend", "special dividend"],
}

POSITIVE_WORDS = ["surge", "beat", "profit", "growth", "upgrade", "approved", "wins", "strong", "expansion", "record", "bullish"]
NEGATIVE_WORDS = ["miss", "loss", "drop", "falls", "weak", "lawsuit", "shutdown", "cut", "lower", "delay", "probe", "decline"]

PROMPT_TEMPLATE = """You are a financial news extraction engine.
Given a news article, return strict JSON only with:
- primary_company
- sector
- event_type
- sentiment_label
- sentiment_score
- impact_score
- is_actionable
- confidence_score
- summary
- beneficiary_companies: list of {company, relation, reason, benefit_score}

Rules:
- Return JSON only, no markdown.
- Do not invent facts.
- If uncertain, lower confidence_score.
- beneficiary_companies should contain only plausible indirect beneficiaries.
- sentiment_score must be between -1 and 1.
- impact_score and confidence_score must be between 0 and 1.
"""


class NewsLLMProtocol:
    def generate(self, article: NewsArticle, candidates: list[dict[str, Any]]) -> str:
        raise NotImplementedError


@dataclass
class HTTPNewsLLMProvider(NewsLLMProtocol):
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: float = 20.0

    @retry(wait=wait_exponential(min=1, max=16), stop=stop_after_attempt(3), reraise=True)
    def generate(self, article: NewsArticle, candidates: list[dict[str, Any]]) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": PROMPT_TEMPLATE},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "title": article.title,
                            "summary": article.summary,
                            "raw_text": article.raw_text,
                            "link": article.link,
                            "published_at": article.published_at.isoformat() if article.published_at else None,
                            "candidate_companies": candidates,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content") or "")


@dataclass
class NewsAnalyzer:
    relations: CompanyRelations
    provider_name: str = "rule"
    llm_provider: NewsLLMProtocol | None = None

    def _text(self, article: NewsArticle) -> str:
        return f"{article.title}. {article.summary}. {article.raw_text}".lower()

    @staticmethod
    def _match_keywords(text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def classify_event_type(self, article: NewsArticle) -> str:
        text = self._text(article)
        for event_type, keywords in EVENT_KEYWORDS.items():
            if self._match_keywords(text, keywords):
                return event_type
        return "other"

    def classify_sentiment(self, article: NewsArticle) -> tuple[str, float, float]:
        text = self._text(article)
        positive_hits = sum(1 for word in POSITIVE_WORDS if word in text)
        negative_hits = sum(1 for word in NEGATIVE_WORDS if word in text)
        score = max(-1.0, min(1.0, (positive_hits - negative_hits) / 4.0))
        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"
        impact = min(1.0, 0.25 + abs(score) * 0.75 + (0.1 if label != "neutral" else 0.0))
        return label, score, impact

    def identify_primary_company(self, article: NewsArticle) -> tuple[str | None, dict[str, Any] | None]:
        text = f"{article.title} {article.summary} {article.raw_text}"
        company = self.relations.detect_primary_company(text)
        if not company:
            return None, None
        return company, self.relations.company_metadata(company)

    def _rule_based_summary(self, article: NewsArticle, event_type: str, sentiment_label: str) -> str:
        headline = article.title.strip().rstrip(".")
        return f"{headline} appears to be a {sentiment_label} {event_type.replace('_', ' ')} signal."

    def _rule_based_beneficiaries(self, primary_company: str | None, analysis: NewsAnalysisResult) -> list[BeneficiaryCompany]:
        if not primary_company or analysis.sentiment_label != "positive" or not analysis.is_actionable:
            return []

        candidates: list[BeneficiaryCompany] = []
        primary_relations = self.relations.related_companies(primary_company)
        for relation in primary_relations:
            strength = float(relation.get("strength") or 0.0)
            benefit_score = min(1.0, max(0.0, 0.45 + analysis.impact_score * 0.35 + strength * 0.40))
            company_name = str(relation.get("company") or "").strip()
            ticker = str(relation.get("ticker") or "").strip() or self.relations.resolve_ticker(company_name)
            candidates.append(
                BeneficiaryCompany(
                    company=company_name,
                    relation=str(relation.get("relation") or "related"),
                    reason=f"Indirectly linked to {primary_company} via {relation.get('relation') or 'market relationship'}.",
                    benefit_score=benefit_score,
                    ticker=ticker,
                    relation_strength=max(0.0, min(1.0, strength)),
                )
            )
        return candidates

    def _build_rule_based_analysis(self, article: NewsArticle) -> NewsAnalysisResult:
        sentiment_label, sentiment_score, impact_score = self.classify_sentiment(article)
        event_type = self.classify_event_type(article)
        primary_company, metadata = self.identify_primary_company(article)
        sector = str(metadata.get("sector") or "").strip() if metadata else None
        is_actionable = sentiment_label != "neutral" and event_type != "other"
        confidence = min(1.0, 0.45 + abs(sentiment_score) * 0.3 + (0.2 if primary_company else 0.0) + (0.1 if event_type != "other" else 0.0))
        analysis = NewsAnalysisResult(
            primary_company=primary_company,
            sector=sector,
            event_type=event_type,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            impact_score=impact_score,
            is_actionable=is_actionable,
            confidence_score=confidence,
            summary=self._rule_based_summary(article, event_type, sentiment_label),
            beneficiary_companies=[],
        )
        analysis.beneficiary_companies = self._rule_based_beneficiaries(primary_company, analysis)
        return analysis

    def _parse_llm_output(self, raw_text: str) -> NewsAnalysisResult:
        payload = json.loads(raw_text)
        return NewsAnalysisResult.model_validate(payload)

    def analyze(self, article: NewsArticle) -> NewsAnalysisResult:
        if self.llm_provider is None or self.provider_name == "rule":
            return self._build_rule_based_analysis(article)

        primary_company, metadata = self.identify_primary_company(article)
        candidates = self.relations.related_companies(primary_company) if primary_company else []
        rule_based = self._build_rule_based_analysis(article)
        try:
            raw_text = self.llm_provider.generate(article, candidates)
            analysis = self._parse_llm_output(raw_text)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            logger.warning("news_llm_parse_failed", extra={"error": str(exc), "symbol": primary_company or ""})
            try:
                raw_text = self.llm_provider.generate(article, candidates)
                analysis = self._parse_llm_output(raw_text)
            except Exception:
                analysis = rule_based
        except Exception as exc:
            logger.warning("news_llm_failed", extra={"error": str(exc), "symbol": primary_company or ""})
            analysis = rule_based

        if not analysis.summary:
            analysis.summary = self._rule_based_summary(article, analysis.event_type, analysis.sentiment_label)
        if not analysis.primary_company:
            analysis.primary_company = primary_company
        if not analysis.sector and metadata:
            analysis.sector = str(metadata.get("sector") or "").strip() or None
        rule_beneficiaries = self._rule_based_beneficiaries(primary_company, analysis)
        if analysis.beneficiary_companies:
            analysis.beneficiary_companies = self.relations.merge_beneficiary_suggestions(
                analysis.beneficiary_companies,
                rule_beneficiaries,
            )
        else:
            analysis.beneficiary_companies = rule_beneficiaries
        return analysis
