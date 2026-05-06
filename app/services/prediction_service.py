from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Optional

from src.inference.predict import predict_for_symbols

logger = logging.getLogger(__name__)


class PredictionService:
    def predict(
        self,
        symbols: Optional[list[str]] = None,
        model_name: str = "xgboost_classifier",
        horizon_days: int = 1,
        atr_multiplier: float = 1.0,
        include_live_quote: bool = False,
        use_trending: bool = False,
    ):
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
                    from app.core.config import get_settings
                    settings = get_settings()
                    resolved_symbols = [str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()]
        elif symbols:
            resolved_symbols = symbols
        else:
            from app.core.config import get_settings
            settings = get_settings()
            resolved_symbols = [str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()]

        logger.info(
            "prediction_started",
            extra={
                "symbols_count": len(resolved_symbols),
                "model_name": str(model_name),
                "horizon_days": int(horizon_days),
                "include_live_quote": bool(include_live_quote),
                "use_trending": bool(use_trending),
            },
        )
        df = predict_for_symbols(
            symbols=resolved_symbols,
            model_name=model_name,
            horizon_days=horizon_days,
            atr_multiplier=atr_multiplier,
            include_live_quote=include_live_quote,
        )
        logger.info(
            "prediction_completed",
            extra={"symbols_count": len(resolved_symbols), "rows": int(len(df)), "model_name": str(model_name)},
        )
        return {
            "generated_at": datetime.now(timezone.utc),
            "predictions": df.to_dict(orient="records"),
            "symbols_used": resolved_symbols,
            "is_trending": use_trending,
        }


prediction_service = PredictionService()

