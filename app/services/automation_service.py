from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from app.core.config import get_settings
from app.news.service import news_impact_service
from app.services.data_service import data_service
from app.services.news_service import news_service
from app.services.prediction_service import prediction_service
from app.services.training_service import training_service
from src.data.metadata_store import metadata_store

logger = logging.getLogger(__name__)


class AutomationService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.output_dir = self.settings.output_dir / "automation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _run_with_timeout(self, fn, *args, **kwargs):
        timeout_sec = max(1, int(self.settings.automation_step_timeout_sec))
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            logger.warning("automation_step_timeout", extra={"function": getattr(fn, "__name__", str(fn)), "timeout_sec": timeout_sec})
            future.cancel()
            return None
        except Exception as exc:
            logger.warning("automation_step_failed", extra={"function": getattr(fn, "__name__", str(fn)), "error": str(exc)})
            return None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _news_opportunity_symbols(self) -> list[str]:
        symbols: list[str] = []
        scan = self._run_with_timeout(
            news_impact_service.refresh,
            force_refresh=bool(self.settings.automation_force_news_refresh),
        )
        if scan is not None:
            for item in scan.top_opportunities:
                ticker = str(item.beneficiary_ticker or item.primary_ticker or "").strip().upper()
                if ticker:
                    symbols.append(ticker)

        try:
            recent = metadata_store.read_beneficiary_opportunities_with_signal(limit=500)
            for row in recent:
                ticker = str(row.get("ticker") or row.get("primary_ticker") or "").strip().upper()
                if ticker:
                    symbols.append(ticker)
        except Exception as exc:
            logger.warning("automation_opportunity_scan_failed", extra={"error": str(exc)})

        return symbols

    @staticmethod
    def _dedupe(symbols: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in symbols:
            symbol = str(raw).strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
        return out

    @staticmethod
    def _enrich_indian_symbols(symbols: list[str]) -> list[str]:
        """Enrich Indian symbols with NSE/BSE suffixes for proper yfinance resolution.
        
        Indian stock symbols without suffixes need explicit exchange tags
        to resolve properly, otherwise yfinance returns 'no timezone found' errors.
        """
        out: list[str] = []
        seen: set[str] = set()

        # Known major Indian stocks that should prioritize NSE resolution
        major_indian_stocks = {
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN",
            "WIPRO", "MARUTI", "BAJAJFINSV", "LT", "ASIANPAINT",
            "ULTRACEMCO", "POWERGRID", "DRREDDY", "SUNPHARMA",
        }

        for raw in symbols:
            symbol = str(raw).strip().upper()
            if not symbol or symbol in seen:
                continue

            base = symbol.split(".")[0]  # Extract base symbol if suffix exists

            # If it already has a suffix, keep as-is
            if symbol.endswith((".NS", ".BO", ".NSE", ".BSE")):
                if symbol not in seen:
                    out.append(symbol)
                    seen.add(symbol)
                continue

            # For all Indian-looking symbols without suffix, add .NS variant
            # Most Indian stocks are on NSE, so .NS is safer default than .BO
            nse_variant = f"{base}.NS"
            if nse_variant not in seen:
                out.append(nse_variant)
                seen.add(nse_variant)

            # Also add the original base symbol for fallback
            if symbol not in seen:
                out.append(symbol)
                seen.add(symbol)

        return out

    def build_universe(self) -> tuple[list[str], dict]:
        defaults = [str(sym).strip().upper() for sym in self.settings.default_symbols if str(sym).strip()]
        globals_ = [str(sym).strip().upper() for sym in self.settings.automation_global_symbols if str(sym).strip()]
        news = self._news_opportunity_symbols()

        sources = {
            "defaults": defaults,
            "global": globals_,
            "news": news,
        }

        symbol_sources: dict[str, list[str]] = {}
        for source_name, symbols in sources.items():
            for raw in symbols:
                symbol = str(raw).strip().upper()
                if not symbol:
                    continue
                symbol_sources.setdefault(symbol, [])
                if source_name not in symbol_sources[symbol]:
                    symbol_sources[symbol].append(source_name)

        combined = []
        for values in sources.values():
            combined.extend(values)
        universe = self._dedupe(combined)
        universe = self._enrich_indian_symbols(universe)
        final_universe = universe[: max(1, int(self.settings.automation_max_symbols))]

        breakdown = {
            "defaults_count": len(defaults),
            "global_count": len(globals_),
            "news_count": len(news),
            "combined_unique_count": len(universe),
            "final_selected_count": len(final_universe),
            "symbol_sources": {symbol: symbol_sources.get(symbol, []) for symbol in final_universe},
        }
        return final_universe, breakdown

    def _resolve_prediction_model(self, model_override: str | None = None) -> str:
        override = str(model_override or "").strip()
        if override:
            return override
        configured = str(self.settings.automation_prediction_model or "auto").strip()
        if configured and configured.lower() != "auto":
            return configured
        if bool(self.settings.openai_predict_enabled):
            return "openai_stock_llm"
        return "xgboost_classifier"

    @staticmethod
    def _opportunity_score_map(rows: list[dict]) -> dict[str, float]:
        best: dict[str, float] = {}
        for row in rows:
            ticker = str(row.get("ticker") or row.get("primary_ticker") or "").strip().upper()
            if not ticker:
                continue
            score = row.get("overall_score")
            try:
                score_value = float(score)
            except Exception:
                score_value = 0.0
            if ticker not in best or score_value > best[ticker]:
                best[ticker] = score_value
        return best

    def _rank_suggestions(self, predictions: list[dict]) -> list[dict]:
        recent = metadata_store.read_beneficiary_opportunities_with_signal(limit=1000)
        opp_map = self._opportunity_score_map(recent)

        ranked = []
        for row in predictions:
            symbol = str(row.get("symbol") or "").strip().upper()
            confidence = float(row.get("confidence") or 0.0)
            predicted_return = float(row.get("predicted_return") or 0.0)
            news_score = float(opp_map.get(symbol, 0.0))
            blended = (0.65 * confidence) + (0.25 * news_score) + (0.10 * max(predicted_return, 0.0))
            payload = {
                **row,
                "symbol": symbol,
                "news_opportunity_score": news_score,
                "blended_score": blended,
            }
            ranked.append(payload)

        ranked.sort(key=lambda item: float(item.get("blended_score") or 0.0), reverse=True)
        return ranked[: max(1, int(self.settings.automation_top_suggestions))]

    def _write_outputs(self, payload: dict) -> None:
        latest_path = self.output_dir / "auto_suggestions_latest.json"
        latest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        ts = self._utc_now().strftime("%Y%m%d%H%M%S")
        history_path = self.output_dir / f"auto_suggestions_{ts}.json"
        history_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _write_ai_summary(self, payload: dict) -> None:
        latest_path = self.output_dir / "ai_summary_latest.json"
        latest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        ts = self._utc_now().strftime("%Y%m%d%H%M%S")
        history_path = self.output_dir / f"ai_summary_{ts}.json"
        history_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def latest_ai_summary(self) -> dict:
        latest_path = self.output_dir / "ai_summary_latest.json"
        if not latest_path.exists():
            return {
                "status": "empty",
                "summary": "",
            }
        try:
            return json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "status": "corrupt",
                "summary": "",
            }

    def generate_ai_summary(self, predictions_limit: int = 20, news_limit: int = 25) -> dict:
        latest = self.latest()
        predictions = list(latest.get("top_suggestions") or [])[: max(1, int(predictions_limit))]
        news_rows = self.get_latest_news_impact(limit=max(1, int(news_limit)))

        if not predictions and not news_rows:
            payload = {
                "status": "empty",
                "generated_at": self._utc_now(),
                "model": None,
                "summary": "No predictions or news are available yet. Run automation first.",
                "highlights": [],
                "risks": [],
                "actions": [],
                "predictions_count": 0,
                "news_count": 0,
            }
            self._write_ai_summary(payload)
            return payload

        api_key = self.settings.openai_predict_api_key_effective
        if not api_key:
            payload = {
                "status": "missing_api_key",
                "generated_at": self._utc_now(),
                "model": None,
                "summary": "OpenAI API key is missing. Set OPENAI_API_KEY (or OPENAI_PREDICT_API_KEY) in .env.",
                "highlights": [],
                "risks": [],
                "actions": [],
                "predictions_count": len(predictions),
                "news_count": len(news_rows),
            }
            self._write_ai_summary(payload)
            return payload

        model_name = str(self.settings.openai_predict_model_name or self.settings.news_llm_model_name or "gpt-4o-mini")
        base_url = str(self.settings.openai_predict_base_url or self.settings.news_llm_base_url or "https://api.openai.com/v1").rstrip("/")
        url = base_url + "/chat/completions"

        compact_predictions = []
        for row in predictions:
            compact_predictions.append(
                {
                    "symbol": str(row.get("symbol") or "").upper(),
                    "decision": str(row.get("decision") or ""),
                    "confidence": float(row.get("confidence") or 0.0),
                    "predicted_return": float(row.get("predicted_return") or 0.0),
                    "blended_score": float(row.get("blended_score") or 0.0),
                    "news_opportunity_score": float(row.get("news_opportunity_score") or 0.0),
                    "current_price": row.get("current_price"),
                    "target_price": row.get("target_price"),
                    "stop_loss_price": row.get("stop_loss_price"),
                }
            )

        compact_news = []
        for row in news_rows:
            compact_news.append(
                {
                    "ticker": str(row.get("ticker") or "").upper(),
                    "title": str(row.get("news_title") or "")[:180],
                    "overall_score": float(row.get("overall_score") or 0.0),
                    "sentiment_score": float(row.get("sentiment_score") or 0.0),
                    "impact_type": str(row.get("impact_type") or "unknown"),
                    "published_at": str(row.get("published_at") or ""),
                }
            )

        system_prompt = (
            "You are a cautious equity market analyst. Summarize prediction and news signals into actionable insights. "
            "Return strict JSON only with keys: summary, highlights, risks, actions. "
            "summary must be a concise markdown paragraph. highlights/risks/actions must be arrays of short strings."
        )
        user_payload = {
            "generated_at": self._utc_now().isoformat(),
            "latest_run_status": str(latest.get("status") or "unknown"),
            "prediction_model": str(latest.get("prediction_model") or "unknown"),
            "predictions": compact_predictions,
            "news_impacts": compact_news,
        }

        request_payload = {
            "model": model_name,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            timeout_sec = float(self.settings.openai_predict_timeout_sec or self.settings.request_timeout_sec)
            with httpx.Client(timeout=timeout_sec) as client:
                resp = client.post(url, headers=headers, json=request_payload)
                resp.raise_for_status()
                data = resp.json()

            content = str(data.get("choices", [{}])[0].get("message", {}).get("content") or "")
            parsed = json.loads(content) if content else {}

            payload = {
                "status": "ok",
                "generated_at": self._utc_now(),
                "model": model_name,
                "summary": str(parsed.get("summary") or "No summary generated."),
                "highlights": list(parsed.get("highlights") or []),
                "risks": list(parsed.get("risks") or []),
                "actions": list(parsed.get("actions") or []),
                "predictions_count": len(compact_predictions),
                "news_count": len(compact_news),
            }
            self._write_ai_summary(payload)
            return payload
        except Exception as exc:
            logger.warning("automation_ai_summary_failed", extra={"error": str(exc)})
            payload = {
                "status": "error",
                "generated_at": self._utc_now(),
                "model": model_name,
                "summary": "Unable to generate AI summary right now.",
                "highlights": [],
                "risks": [],
                "actions": [],
                "predictions_count": len(compact_predictions),
                "news_count": len(compact_news),
                "error": str(exc),
            }
            self._write_ai_summary(payload)
            return payload

    def run_cycle(
        self,
        model_override: str | None = None,
        interval_minutes_override: int | None = None,
        progress_cb: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict:
        def _emit(progress: int, message: str) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(int(progress), str(message))
            except Exception:
                pass

        def _is_canceled() -> bool:
            if cancel_check is None:
                return False
            try:
                return bool(cancel_check())
            except Exception:
                return False

        def _canceled_result(started: datetime, interval_minutes: int, source_breakdown: dict) -> dict:
            return {
                "started_at": started,
                "completed_at": self._utc_now(),
                "status": "canceled",
                "interval_minutes": interval_minutes,
                "symbol_count": 0,
                "predictions_count": 0,
                "source_breakdown": source_breakdown,
                "top_suggestions": [],
            }

        started_at = self._utc_now()
        _emit(5, "Starting automation cycle")
        _emit(12, "Searching internet for market news")
        symbols, source_breakdown = self.build_universe()
        _emit(22, "Building top news-based share picks")
        interval_minutes = int(interval_minutes_override) if interval_minutes_override is not None else int(self.settings.automation_interval_minutes)
        if _is_canceled():
            _emit(100, "Automation canceled")
            result = _canceled_result(started_at, interval_minutes, source_breakdown)
            self._write_outputs(result)
            return result

        historical_summary: dict[str, int] = {}
        historical_source_breakdown: dict[str, int] = {}
        if not symbols:
            result = {
                "started_at": started_at,
                "completed_at": self._utc_now(),
                "status": "no_symbols",
                "interval_minutes": interval_minutes,
                "symbol_count": 0,
                "predictions_count": 0,
                "source_breakdown": source_breakdown,
                "top_suggestions": [],
            }
            _emit(100, "No eligible symbols found")
            self._write_outputs(result)
            return result

        try:
            _emit(35, "Checking historical data")
            historical_result = self._run_with_timeout(
                data_service.ingest_historical,
                symbols,
                self.settings.historical_interval,
                self.settings.historical_lookback_days,
            )
            historical_summary = historical_result if isinstance(historical_result, dict) else {}
            historical_source_breakdown = dict(getattr(data_service.loader, "last_source_counts", {}) or {})
        except Exception as exc:
            logger.warning("automation_historical_ingest_failed", extra={"error": str(exc)})

        if _is_canceled():
            _emit(100, "Automation canceled")
            result = _canceled_result(started_at, interval_minutes, source_breakdown)
            self._write_outputs(result)
            return result

        try:
            _emit(55, "Refreshing current news context")
            self._run_with_timeout(news_service.ingest_news, symbols, int(self.settings.max_news_items_per_symbol))
        except Exception as exc:
            logger.warning("automation_news_ingest_failed", extra={"error": str(exc)})

        if _is_canceled():
            _emit(100, "Automation canceled")
            result = _canceled_result(started_at, interval_minutes, source_breakdown)
            self._write_outputs(result)
            return result

        _emit(78, "Predicting share price from current situation and news")
        prediction_model = self._resolve_prediction_model(model_override=model_override)
        prediction_result = prediction_service.predict(
            symbols=symbols,
            model_name=prediction_model,
            horizon_days=int(self.settings.automation_horizon_days),
            atr_multiplier=1.0,
            include_live_quote=bool(self.settings.automation_include_live_quote),
        )
        predictions = prediction_result.get("predictions", [])
        top_suggestions = self._rank_suggestions(predictions)

        result = {
            "started_at": started_at,
            "completed_at": self._utc_now(),
            "status": "ok",
            "interval_minutes": interval_minutes,
            "prediction_model": prediction_model,
            "symbol_count": len(symbols),
            "predictions_count": len(predictions),
            "source_breakdown": source_breakdown,
            "historical_ingest_summary": {
                "symbols_with_rows": sum(1 for _, rows in historical_summary.items() if int(rows) > 0),
                "symbols_without_rows": sum(1 for _, rows in historical_summary.items() if int(rows) <= 0),
            },
            "historical_source_breakdown": historical_source_breakdown,
            "symbols": symbols,
            "top_suggestions": top_suggestions,
        }
        self._write_outputs(result)
        _emit(100, "Automation completed")
        return result

    def get_latest_news_impact(self, limit: int = 20) -> list[dict]:
        """Fetch latest news-driven market opportunities with impact scores."""
        opportunities: list[dict] = []
        try:
            recent = metadata_store.read_beneficiary_opportunities_with_signal(limit=limit * 3)
            for item in recent[:limit]:
                ticker = str(item.get("ticker") or item.get("primary_ticker") or "").strip().upper()
                if not ticker:
                    continue
                opportunities.append({
                    "ticker": ticker,
                    "news_title": str(item.get("title") or "")[:100],
                    "sentiment_score": float(item.get("sentiment_score") or 0.0),
                    "overall_score": float(item.get("overall_score") or 0.0),
                    "published_at": str(item.get("published_at") or ""),
                    "impact_type": str(item.get("impact_category") or "unknown"),
                })
        except Exception as exc:
            logger.warning("get_latest_news_impact_failed", extra={"error": str(exc)})
        return opportunities

    def get_market_summary(self) -> dict:
        """Get current market statistics and coverage summary."""
        try:
            symbols, breakdown = self.build_universe()
            total_news_items = 0
            try:
                all_news = metadata_store.read_news_records(limit=10000)
                total_news_items = len(all_news) if all_news else 0
            except Exception:
                pass

            return {
                "total_universe_symbols": len(symbols),
                "sources_active": sum(1 for v in breakdown.values() if isinstance(v, int) and v > 0),
                "defaults_count": breakdown.get("defaults_count", 0),
                "global_symbols_count": breakdown.get("global_count", 0),
                "news_driven_symbols": breakdown.get("news_count", 0),
                "total_news_records": total_news_items,
                "last_refresh": self._utc_now().isoformat(),
            }
        except Exception as exc:
            logger.warning("get_market_summary_failed", extra={"error": str(exc)})
            return {}

    def train_on_expanded_universe(self, horizon_days: int = 1, task_type: str = "classification") -> dict:
        """Train model on expanded universe (defaults + global + news symbols)."""
        try:
            symbols, breakdown = self.build_universe()
            if not symbols:
                return {
                    "status": "error",
                    "error": "No symbols available for training",
                    "symbols_count": 0,
                }

            logger.info(
                "training_on_expanded_universe_started",
                extra={"symbols_count": len(symbols), "horizon_days": horizon_days, "task_type": task_type}
            )

            result = self._run_with_timeout(
                training_service.train,
                symbols=symbols,
                horizon_days=horizon_days,
                task_type=task_type,
            )

            if result is None:
                return {
                    "status": "timeout",
                    "error": f"Training timed out after {self.settings.automation_step_timeout_sec}s",
                    "symbols_count": len(symbols),
                }

            logger.info(
                "training_on_expanded_universe_completed",
                extra={"symbols_count": len(symbols), "result_status": result.get("status"), "breakdown": breakdown}
            )

            return {
                "status": "ok",
                "symbols_count": len(symbols),
                "source_breakdown": breakdown,
                "training_result": result,
            }
        except Exception as exc:
            logger.error("training_on_expanded_universe_failed", extra={"error": str(exc)})
            return {
                "status": "error",
                "error": str(exc),
                "symbols_count": 0,
            }

    def latest(self) -> dict:
        latest_path = self.output_dir / "auto_suggestions_latest.json"
        if not latest_path.exists():
            return {
                "status": "empty",
                "top_suggestions": [],
            }
        try:
            return json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "status": "corrupt",
                "top_suggestions": [],
            }


automation_service = AutomationService()
