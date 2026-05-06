from fastapi import APIRouter, HTTPException

from app.api.error_utils import build_error_envelope
from app.models.schemas import PredictRequest, PredictResponse
from app.services.prediction_service import prediction_service

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = prediction_service.predict(
            symbols=payload.symbols,
            model_name=payload.model_name,
            horizon_days=payload.horizon_days,
            atr_multiplier=payload.atr_multiplier,
            include_live_quote=payload.include_live_quote,
            use_trending=payload.use_trending,
        )
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=build_error_envelope(
                code="MODEL_NOT_FOUND",
                message=str(exc),
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=build_error_envelope(
                code="PREDICTION_ERROR",
                message="Failed to generate predictions",
                details={"reason": str(exc)},
            ),
        ) from exc
