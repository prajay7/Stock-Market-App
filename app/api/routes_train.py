from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
import json

from app.api.error_utils import build_error_envelope
from app.models.schemas import TrainRequest, TrainResponse
from app.services.training_service import training_service

router = APIRouter(tags=["training"])


@router.post("/train", response_model=TrainResponse)
def train(payload: TrainRequest) -> TrainResponse:
    try:
        result = training_service.train(payload.symbols, payload.horizon_days, payload.task_type)
        return TrainResponse(
            trained_at=datetime.now(timezone.utc),
            task_type=str(result.get("task_type") or payload.task_type),
            target=str(result.get("target") or ""),
            best=result.get("best") or {},
            models=result.get("models") or {},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=build_error_envelope(
                code="TRAINING_ERROR",
                message="Failed to train models",
                details={"reason": str(exc)},
            ),
        ) from exc


@router.get("/models")
def list_models() -> dict:
    from app.core.config import get_settings
    from src.training.model_registry import ModelRegistry

    settings = get_settings()
    registry = ModelRegistry(settings.model_dir)
    models = registry.list_models()
    if settings.movement_model_path.exists() and "movement_model" not in models:
        models.append("movement_model")
    return {"models": sorted(models)}


@router.get("/model-info")
def model_info(model_name: str = "xgboost_classifier") -> dict:
    from app.core.config import get_settings

    settings = get_settings()
    model_dir = settings.model_dir
    if model_name == "movement_model":
        metadata_path = settings.movement_model_path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            raise HTTPException(
                status_code=404,
                detail=build_error_envelope(
                    code="MODEL_METADATA_NOT_FOUND",
                    message=f"No metadata found for model '{model_name}'",
                ),
            )
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=build_error_envelope(
                    code="MODEL_METADATA_READ_ERROR",
                    message="Failed to read model metadata",
                    details={"reason": str(exc)},
                ),
            ) from exc
        return {
            "model_name": model_name,
            "latest_metadata_file": metadata_path.name,
            "latest": payload,
            "feature_importance_file": None,
            "feature_importance_top": [],
            "best_by_task": {},
        }

    candidates = sorted(model_dir.glob(f"{model_name}_*.metadata.json"))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=build_error_envelope(
                code="MODEL_METADATA_NOT_FOUND",
                message=f"No metadata found for model '{model_name}'",
            ),
        )

    latest_path = candidates[-1]
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=build_error_envelope(
                code="MODEL_METADATA_READ_ERROR",
                message="Failed to read model metadata",
                details={"reason": str(exc)},
            ),
        ) from exc

    feature_importance_path = latest_path.with_name(latest_path.name.replace(".metadata.json", ".feature_importance.json"))
    feature_importance = []
    if feature_importance_path.exists():
        try:
            fi_payload = json.loads(feature_importance_path.read_text(encoding="utf-8"))
            feature_importance = fi_payload.get("feature_importance") or []
        except Exception:
            feature_importance = []

    best_files = sorted(model_dir.glob("best_model_*.json"))
    best_by_task: dict[str, dict] = {}
    for path in best_files:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            task_key = str(item.get("task_type") or path.stem.replace("best_model_", ""))
            best_by_task[task_key] = item
        except Exception:
            continue

    return {
        "model_name": model_name,
        "latest_metadata_file": latest_path.name,
        "latest": payload,
        "feature_importance_file": feature_importance_path.name if feature_importance_path.exists() else None,
        "feature_importance_top": feature_importance[:20],
        "best_by_task": best_by_task,
    }


@router.get("/model-feature-importance")
def model_feature_importance(model_name: str = "xgboost_classifier", top_k: int = 30) -> dict:
    from app.core.config import get_settings

    settings = get_settings()
    model_dir = settings.model_dir
    if model_name == "movement_model":
        raise HTTPException(
            status_code=404,
            detail=build_error_envelope(
                code="FEATURE_IMPORTANCE_NOT_FOUND",
                message=f"No feature importance artifacts found for model '{model_name}'",
            ),
        )
    files = sorted(model_dir.glob(f"{model_name}_*.feature_importance.json"))
    if not files:
        raise HTTPException(
            status_code=404,
            detail=build_error_envelope(
                code="FEATURE_IMPORTANCE_NOT_FOUND",
                message=f"No feature importance artifacts found for model '{model_name}'",
            ),
        )

    path = files[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=build_error_envelope(
                code="FEATURE_IMPORTANCE_READ_ERROR",
                message="Failed to read feature importance artifact",
                details={"reason": str(exc)},
            ),
        ) from exc

    rows = payload.get("feature_importance") or []
    k = max(1, min(int(top_k), 500))
    return {
        "model_name": model_name,
        "artifact": path.name,
        "top_k": k,
        "feature_importance": rows[:k],
    }
