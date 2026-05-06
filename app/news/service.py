from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.news.analyzer import HTTPNewsLLMProvider, NewsAnalyzer
from app.news.fetcher import RSSFetcher, load_rss_feeds
from app.news.models import AnalyzedNewsItem, NewsImpactScanResult, NewsArticle, NewsImpactSignal
from app.news.relations import CompanyRelations
from app.news.scorer import NewsScorer
from src.data.metadata_store import MetadataStore
from src.data.storage import append_dedup_by_keys, read_parquet_if_exists, write_parquet

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_alert_matcher_service = None


def _get_alert_matcher_service():
    global _alert_matcher_service
    if _alert_matcher_service is None:
        from app.services.alert_matcher_service import alert_matcher_service
        _alert_matcher_service = alert_matcher_service
    return _alert_matcher_service


@dataclass
class NewsImpactService:
    settings: object

    def __post_init__(self) -> None:
        self.cache_dir = self.settings.output_dir / "news_impact"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feed_cache_path = self.cache_dir / "rss_cache.parquet"
        self.analysis_cache_path = self.cache_dir / "analysis_cache.parquet"
        self.meta_cache_path = self.cache_dir / "refresh_meta.json"
        self.relations = CompanyRelations(
            self.settings.news_company_relations_path,
            self.settings.news_effective_ticker_map_path,
        )
        self.fetcher = RSSFetcher(timeout_sec=self.settings.request_timeout_sec)
        self.scorer = NewsScorer(
            impact_weight=self.settings.news_score_impact_weight,
            relation_weight=self.settings.news_score_relation_weight,
            freshness_weight=self.settings.news_score_freshness_weight,
            price_weight=self.settings.news_score_price_weight,
            price_reaction_max_abs_pct=self.settings.news_price_reaction_max_abs_pct,
            early_move_threshold_pct=self.settings.news_price_move_early_threshold_pct,
            late_move_threshold_pct=self.settings.news_price_move_late_threshold_pct,
            timeout_sec=self.settings.request_timeout_sec,
        )
        self.store = MetadataStore(self.settings.database_url)
        llm_provider = None
        if self.settings.news_analysis_provider != "rule" and self.settings.news_llm_api_key_effective:
            llm_provider = HTTPNewsLLMProvider(
                api_key=self.settings.news_llm_api_key_effective,
                model=self.settings.news_llm_model_name,
                base_url=self.settings.news_llm_base_url,
                timeout_sec=self.settings.request_timeout_sec,
            )
        self.analyzer = NewsAnalyzer(
            relations=self.relations,
            provider_name=self.settings.news_analysis_provider,
            llm_provider=llm_provider,
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _as_naive_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def _load_refresh_meta(self) -> dict:
        if not self.meta_cache_path.exists():
            return {}
        try:
            return json.loads(self.meta_cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_refresh_meta(self, payload: dict) -> None:
        self.meta_cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _feeds(self) -> list[dict[str, str]]:
        feeds = load_rss_feeds(self.settings.news_rss_feeds_path)
        if feeds:
            return feeds
        return self.settings.news_rss_feeds_default

    def fetch_articles(self, force_refresh: bool = False) -> list[NewsArticle]:
        meta = self._load_refresh_meta()
        last_refresh = pd.to_datetime(meta.get("fetched_at"), errors="coerce", utc=True)
        ttl = int(self.settings.news_cache_ttl_seconds)
        if not force_refresh and pd.notna(last_refresh):
            age_seconds = (pd.Timestamp.now(tz="UTC") - last_refresh).total_seconds()
            if age_seconds < ttl and self.feed_cache_path.exists():
                cached = read_parquet_if_exists(self.feed_cache_path)
                if not cached.empty:
                    return [NewsArticle.model_validate(row) for row in cached.to_dict(orient="records")]

        feeds = self._feeds()[: max(1, int(self.settings.news_max_feeds_per_refresh))]
        articles = self.fetcher.fetch_many(feeds, limit_per_feed=int(self.settings.news_max_articles_per_refresh))
        if articles:
            df = pd.DataFrame([article.model_dump() for article in articles])
            write_parquet(df, self.feed_cache_path)
            self._save_refresh_meta({"fetched_at": pd.Timestamp.utcnow().isoformat(), "feed_count": len(feeds), "article_count": len(articles)})
        return articles

    def _load_analysis_cache(self) -> pd.DataFrame:
        return read_parquet_if_exists(self.analysis_cache_path)

    def _save_analysis_cache(self, df: pd.DataFrame) -> None:
        write_parquet(df, self.analysis_cache_path)

    @staticmethod
    def _serialize_record(record: dict) -> dict:
        payload = dict(record)
        for key in ["article", "analysis", "ranked_beneficiaries"]:
            value = payload.get(key)
            if value is not None and not isinstance(value, str):
                payload[key] = json.dumps(value, default=str, ensure_ascii=False)
        return payload

    @staticmethod
    def _deserialize_record(record: dict) -> dict:
        payload = dict(record)
        for key in ["article", "analysis", "ranked_beneficiaries"]:
            value = payload.get(key)
            if isinstance(value, str):
                try:
                    payload[key] = json.loads(value)
                except Exception:
                    payload[key] = value
        return payload

    def _signal_history_from_db(self, limit: int = 200) -> list[NewsImpactSignal]:
        rows = self.store.read_beneficiary_opportunity_history(limit=limit)
        history: list[NewsImpactSignal] = []
        for row in rows:
            try:
                history.append(
                    NewsImpactSignal(
                        generated_at=row.get("created_at") or self._utc_now(),
                        article_hash=str(row.get("article_hash") or ""),
                        headline=str(row.get("title") or ""),
                        source=str(row.get("source") or ""),
                        published_at=row.get("published_at"),
                        primary_company=row.get("primary_company"),
                        primary_ticker=row.get("primary_ticker"),
                        beneficiary_company=str(row.get("company") or ""),
                        beneficiary_ticker=row.get("ticker"),
                        relation=str(row.get("relation") or "related"),
                        relation_strength=float(row.get("relation_strength") or 0.0),
                        sentiment_label=str(row.get("sentiment_label") or "neutral"),
                        event_type=str(row.get("event_type") or "other"),
                        impact_score=float(row.get("impact_score") or 0.0),
                        freshness_score=float(row.get("freshness_score") or 0.0),
                        price_change_pct_1d=row.get("price_change_pct"),
                        price_reaction_ok=str(row.get("timing_label") or "moderate") != "late",
                        timing_label=str(row.get("timing_label") or "moderate"),
                        price_opportunity_score=float(row.get("price_opportunity_score") or 0.5),
                        signal_score=float(row.get("overall_score") or 0.0),
                        overall_score=float(row.get("overall_score") or 0.0),
                        is_early_opportunity=str(row.get("timing_label") or "moderate") == "early",
                        reason=str(row.get("reason") or ""),
                    )
                )
            except Exception:
                continue
        return history

    def _build_signals(
        self,
        generated_at: datetime,
        article: NewsArticle,
        analysis_item: AnalyzedNewsItem,
        primary_ticker: str | None,
        beneficiary_price_meta: dict[str, dict[str, Any]],
    ) -> list[NewsImpactSignal]:
        signals: list[NewsImpactSignal] = []
        for beneficiary in analysis_item.ranked_beneficiaries:
            meta = beneficiary_price_meta.get(beneficiary.company, {})
            relation_strength = float(beneficiary.relation_strength if beneficiary.relation_strength is not None else beneficiary.benefit_score)
            price_reaction_ok = bool(meta.get("price_reaction_ok", True))
            signal_score = max(0.0, min(1.0, float(beneficiary.benefit_score)))
            timing_label = str(meta.get("timing_label") or self.scorer.classify_timing_label(meta.get("change_pct")))
            is_early = bool(
                analysis_item.analysis.is_actionable
                and timing_label != "late"
                and signal_score >= float(self.settings.news_min_signal_score)
            )
            reason = f"{beneficiary.company} linked via {beneficiary.relation}; price reaction {'acceptable' if price_reaction_ok else 'already extended'}."
            signals.append(
                NewsImpactSignal(
                    generated_at=generated_at,
                    article_hash=article.article_hash,
                    headline=article.title,
                    source=article.source,
                    published_at=article.published_at,
                    primary_company=analysis_item.analysis.primary_company,
                    primary_ticker=primary_ticker,
                    beneficiary_company=beneficiary.company,
                    beneficiary_ticker=beneficiary.ticker,
                    relation=beneficiary.relation,
                    relation_strength=relation_strength,
                    sentiment_label=analysis_item.analysis.sentiment_label,
                    event_type=analysis_item.analysis.event_type,
                    impact_score=float(analysis_item.analysis.impact_score),
                    freshness_score=float(analysis_item.freshness_score),
                    signal_price=meta.get("latest_close"),
                    price_change_pct_1d=meta.get("change_pct"),
                    price_reaction_ok=timing_label != "late",
                    timing_label=timing_label,
                    price_opportunity_score=float(meta.get("score", analysis_item.price_opportunity_score)),
                    signal_score=signal_score,
                    overall_score=signal_score,
                    is_early_opportunity=is_early,
                    reason=reason,
                )
            )
        return signals

    def _persist_rows(self, generated_at: datetime, item_rows: list[AnalyzedNewsItem], signals: list[NewsImpactSignal]) -> None:
        if not bool(self.settings.news_persist_signals_enabled):
            return
        news_rows = []
        id_by_hash: dict[str, int] = {}
        for item in item_rows:
            row = {
                "article_hash": item.article.article_hash,
                "title": item.article.title,
                "link": item.article.link,
                "source": item.article.source,
                "published_at": self._as_naive_utc(item.article.published_at),
                "primary_company": item.analysis.primary_company,
                "primary_ticker": self.relations.resolve_ticker(item.analysis.primary_company),
                "sector": item.analysis.sector,
                "event_type": item.analysis.event_type,
                "sentiment_label": item.analysis.sentiment_label,
                "sentiment_score": float(item.analysis.sentiment_score),
                "impact_score": float(item.analysis.impact_score),
                "confidence_score": float(item.analysis.confidence_score),
                "is_actionable": bool(item.analysis.is_actionable),
                "summary": item.analysis.summary,
                "created_at": self._as_naive_utc(generated_at),
            }
            try:
                signal_id = self.store.upsert_analyzed_news_signal(row)
            except Exception as exc:
                logger.warning("news_signal_persist_failed", extra={"error": str(exc), "article_hash": item.article.article_hash})
                signal_id = None
            if signal_id is not None:
                id_by_hash[item.article.article_hash] = signal_id

            news_rows.append(
                {
                    "generated_at": self._as_naive_utc(generated_at),
                    "article_hash": item.article.article_hash,
                    "source": item.article.source,
                    "title": item.article.title,
                    "link": item.article.link,
                    "published_at": self._as_naive_utc(item.article.published_at),
                    "primary_company": item.analysis.primary_company,
                    "primary_ticker": self.relations.resolve_ticker(item.analysis.primary_company),
                    "event_type": item.analysis.event_type,
                    "sentiment_label": item.analysis.sentiment_label,
                    "impact_score": float(item.analysis.impact_score),
                    "confidence_score": float(item.analysis.confidence_score),
                    "is_actionable": bool(item.analysis.is_actionable),
                    "freshness_score": float(item.freshness_score),
                    "relation_strength": float(item.relation_strength),
                    "price_opportunity_score": float(item.price_opportunity_score),
                    "overall_score": float(item.overall_score),
                    "payload": item.model_dump(mode="json"),
                }
            )

        try:
            self.store.upsert_news_impact_items(news_rows)
        except Exception as exc:
            logger.warning("news_impact_cache_persist_failed", extra={"error": str(exc)})

        signal_rows = []
        opportunity_rows = []
        for signal in signals:
            payload = signal.model_dump(mode="json")
            signal_rows.append(
                {
                    "generated_at": self._as_naive_utc(signal.generated_at),
                    "article_hash": signal.article_hash,
                    "headline": signal.headline,
                    "source": signal.source,
                    "published_at": self._as_naive_utc(signal.published_at),
                    "primary_company": signal.primary_company,
                    "primary_ticker": signal.primary_ticker,
                    "beneficiary_company": signal.beneficiary_company,
                    "beneficiary_ticker": signal.beneficiary_ticker,
                    "relation": signal.relation,
                    "relation_strength": float(signal.relation_strength),
                    "sentiment_label": signal.sentiment_label,
                    "event_type": signal.event_type,
                    "impact_score": float(signal.impact_score),
                    "freshness_score": float(signal.freshness_score),
                    "price_change_pct_1d": signal.price_change_pct_1d,
                    "price_reaction_ok": bool(signal.price_reaction_ok),
                    "price_opportunity_score": float(signal.price_opportunity_score),
                    "signal_score": float(signal.signal_score),
                    "is_early_opportunity": bool(signal.is_early_opportunity),
                    "reason": signal.reason,
                    "payload": payload,
                }
            )
            signal_id = id_by_hash.get(signal.article_hash)
            if signal_id is not None:
                opportunity_rows.append(
                    {
                        "signal_id": signal_id,
                        "company": signal.beneficiary_company,
                        "ticker": signal.beneficiary_ticker,
                        "relation": signal.relation,
                        "relation_strength": float(signal.relation_strength),
                        "benefit_score": float(signal.signal_score),
                        "freshness_score": float(signal.freshness_score),
                        "price_change_pct": signal.price_change_pct_1d,
                        "price_opportunity_score": float(signal.price_opportunity_score),
                        "overall_score": float(signal.overall_score),
                        "timing_label": signal.timing_label,
                        "reason": signal.reason,
                        "signal_price": signal.signal_price,
                        "signal_timestamp": self._as_naive_utc(signal.generated_at),
                        "price_source": "yfinance_close",
                        "created_at": self._as_naive_utc(signal.generated_at),
                    }
                )
        try:
            self.store.write_news_impact_signals(signal_rows)
        except Exception as exc:
            logger.warning("news_impact_signal_history_persist_failed", extra={"error": str(exc)})
        try:
            self.store.write_beneficiary_opportunities(opportunity_rows)
        except Exception as exc:
            logger.warning("beneficiary_opportunity_persist_failed", extra={"error": str(exc)})
        
        # Evaluate alerts for newly created opportunities
        try:
            alert_matcher = _get_alert_matcher_service()
            for opp_row in opportunity_rows:
                opp_data = {
                    **opp_row,
                    "signal_id": opp_row.get("signal_id"),
                    "created_at": opp_row.get("created_at"),
                }
                alert_matcher.evaluate_opportunity(opp_data)
        except Exception as exc:
            logger.warning("alert_evaluation_failed", extra={"error": str(exc)})

    def _top_opportunities(self, signals: list[NewsImpactSignal]) -> list[NewsImpactSignal]:
        filtered = [
            signal
            for signal in signals
            if signal.is_early_opportunity and signal.timing_label in {"early", "moderate"}
        ]
        filtered.sort(key=lambda s: (float(s.overall_score), float(s.signal_score)), reverse=True)
        return filtered[: max(1, int(self.settings.news_top_opportunities_limit))]

    def refresh(self, force_refresh: bool = False) -> NewsImpactScanResult:
        articles = self.fetch_articles(force_refresh=force_refresh)
        generated_at = self._utc_now()
        if not articles:
            cached = self.from_cache()
            if cached.articles:
                return cached
            return NewsImpactScanResult(
                generated_at=generated_at,
                articles=[],
                top_opportunities=[],
                signal_history=self._signal_history_from_db() if bool(self.settings.news_persist_signals_enabled) else [],
            )

        cache = self._load_analysis_cache()
        analyzed_rows: list[dict] = []
        new_items: list[AnalyzedNewsItem] = []
        new_signals: list[NewsImpactSignal] = []
        cached_hashes = set(cache["article_hash"].astype(str).tolist()) if not cache.empty and "article_hash" in cache.columns else set()

        for article in articles:
            if article.article_hash in cached_hashes:
                continue
            analysis = self.analyzer.analyze(article)
            primary_ticker = self.relations.resolve_ticker(analysis.primary_company)
            beneficiary_scores: dict[str, float] = {}
            beneficiary_meta: dict[str, dict[str, Any]] = {}
            for beneficiary in analysis.beneficiary_companies:
                ticker = beneficiary.ticker or self.relations.resolve_ticker(beneficiary.company)
                if ticker and not beneficiary.ticker:
                    beneficiary.ticker = ticker
                score, meta = self.scorer.price_opportunity_score(ticker)
                beneficiary_scores[beneficiary.company] = score
                beneficiary_meta[beneficiary.company] = {**meta, "score": score, "ticker": ticker}
            scored = self.scorer.score_item(
                article,
                analysis,
                primary_ticker=primary_ticker,
                beneficiary_scores=beneficiary_scores,
                beneficiary_price_meta=beneficiary_meta,
            )
            signals = self._build_signals(generated_at, article, scored, primary_ticker, beneficiary_meta)
            analyzed_rows.append(
                self._serialize_record(
                {
                    "article_hash": article.article_hash,
                    "article": article.model_dump(),
                    "analysis": analysis.model_dump(),
                    "freshness_score": scored.freshness_score,
                    "relation_strength": scored.relation_strength,
                    "price_opportunity_score": scored.price_opportunity_score,
                    "overall_score": scored.overall_score,
                    "ranked_beneficiaries": [item.model_dump() for item in scored.ranked_beneficiaries],
                }
                )
            )
            new_items.append(scored)
            new_signals.extend(signals)

        if analyzed_rows:
            new_df = pd.DataFrame(analyzed_rows)
            cache = append_dedup_by_keys(cache, new_df, keys=["article_hash"])
            self._save_analysis_cache(cache)
            self._persist_rows(generated_at=generated_at, item_rows=new_items, signals=new_signals)

        if cache.empty:
            return NewsImpactScanResult(
                generated_at=generated_at,
                articles=[],
                top_opportunities=[],
                signal_history=self._signal_history_from_db() if bool(self.settings.news_persist_signals_enabled) else [],
            )

        rows = cache.sort_values(["overall_score", "article_hash"], ascending=[False, False]).head(100).to_dict(orient="records")
        items: list[AnalyzedNewsItem] = []
        for row in rows:
            try:
                items.append(AnalyzedNewsItem.model_validate(self._deserialize_record(row)))
            except Exception:
                continue

        signal_history = self._signal_history_from_db() if bool(self.settings.news_persist_signals_enabled) else []
        top_opportunities = self._top_opportunities(signal_history if signal_history else new_signals)
        return NewsImpactScanResult(
            generated_at=generated_at,
            articles=items,
            top_opportunities=top_opportunities,
            signal_history=signal_history,
        )

    def from_cache(self) -> NewsImpactScanResult:
        cache = self._load_analysis_cache()
        generated_at = self._utc_now()
        if cache.empty:
            return NewsImpactScanResult(
                generated_at=generated_at,
                articles=[],
                top_opportunities=[],
                signal_history=self._signal_history_from_db() if bool(self.settings.news_persist_signals_enabled) else [],
            )
        rows = cache.sort_values(["overall_score", "article_hash"], ascending=[False, False]).to_dict(orient="records")
        items: list[AnalyzedNewsItem] = []
        for row in rows:
            try:
                items.append(AnalyzedNewsItem.model_validate(self._deserialize_record(row)))
            except Exception:
                continue
        signal_history = self._signal_history_from_db() if bool(self.settings.news_persist_signals_enabled) else []
        top_opportunities = self._top_opportunities(signal_history)
        return NewsImpactScanResult(
            generated_at=generated_at,
            articles=items,
            top_opportunities=top_opportunities,
            signal_history=signal_history,
        )


news_impact_service = NewsImpactService(get_settings())
