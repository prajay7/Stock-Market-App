from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Optional

from app.core.config import get_settings
from app.core.constants import OPENAI_STOCK_MODEL_ALIASES_SET
from src.inference.predict import predict_for_symbols

logger = logging.getLogger(__name__)


class PredictionService:
    @staticmethod
    def _prediction_model_available(settings, model_name: str) -> bool:
        name = str(model_name or "").strip().lower()
        if not name:
            return False
        if name in OPENAI_STOCK_MODEL_ALIASES_SET:
            return bool(settings.openai_predict_enabled)
        if name in {"movement", "movement_model"}:
            return bool(settings.movement_model_path.exists())

        latest_path = settings.model_dir / f"{name}_latest.joblib"
        if latest_path.exists():
            return True
        return any(settings.model_dir.glob(f"{name}_*.joblib"))

    @classmethod
    def _resolve_prediction_model_with_fallback(cls, requested_model: str, settings) -> tuple[str | None, str | None]:
        requested = str(requested_model or "").strip()
        if cls._prediction_model_available(settings, requested):
            return requested, None

        candidates = ["movement_model", "xgboost_classifier", "openai_stock_llm_fast", "openai_stock_llm"]
        seen: set[str] = {requested.lower()}
        for candidate in candidates:
            if candidate.lower() in seen:
                continue
            seen.add(candidate.lower())
            if cls._prediction_model_available(settings, candidate):
                return candidate, f"Requested model '{requested}' was unavailable; using '{candidate}' fallback."

        return None, f"No prediction model artifacts available for '{requested}'."

    def predict(
        self,
        symbols: Optional[list[str]] = None,
        model_name: str = "xgboost_classifier",
        horizon_days: int = 1,
        atr_multiplier: float = 1.0,
        include_live_quote: bool = False,
        use_trending: bool = False,
    ):
        settings = get_settings()

        # Resolve symbols: use trending if requested, fallback to provided or default
        if use_trending:
            try:
                from app.services.trending_symbols_service import trending_symbols_service
                resolved_symbols, source = trending_symbols_service.get_trending_symbols(limit=25)
                logger.info("prediction_using_trending", extra={"source": source, "count": len(resolved_symbols)})
            except Exception as exc:
                logger.warning("trending_symbols_failed, falling back to defaults", extra={"error": str(exc)})
                if symbols:
                    resolved_symbols = symbols
                else:
                    resolved_symbols = [str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()]
        elif symbols:
            resolved_symbols = symbols
        else:
            resolved_symbols = [str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()]

        effective_model_name, fallback_reason = self._resolve_prediction_model_with_fallback(model_name, settings)
        if effective_model_name is None:
            raise FileNotFoundError(fallback_reason or f"No artifacts found for model {model_name}")
        if fallback_reason:
            logger.warning(
                "prediction_model_fallback",
                extra={"requested_model_name": str(model_name), "effective_model_name": str(effective_model_name)},
            )

        logger.info(
            "prediction_started",
            extra={
                "symbols_count": len(resolved_symbols),
                "model_name": str(effective_model_name),
                "requested_model_name": str(model_name),
                "horizon_days": int(horizon_days),
                "include_live_quote": bool(include_live_quote),
                "use_trending": bool(use_trending),
            },
        )
        df = predict_for_symbols(
            symbols=resolved_symbols,
            model_name=effective_model_name,
            horizon_days=horizon_days,
            atr_multiplier=atr_multiplier,
            include_live_quote=include_live_quote,
        )
        logger.info(
            "prediction_completed",
            extra={
                "symbols_count": len(resolved_symbols),
                "rows": int(len(df)),
                "model_name": str(effective_model_name),
                "requested_model_name": str(model_name),
            },
        )
        return {
            "generated_at": datetime.now(timezone.utc),
            "predictions": df.to_dict(orient="records"),
            "symbols_used": resolved_symbols,
            "is_trending": use_trending,
            "requested_model_name": str(model_name),
            "effective_model_name": str(effective_model_name),
            "model_fallback_reason": fallback_reason,
        }


prediction_service = PredictionService()
