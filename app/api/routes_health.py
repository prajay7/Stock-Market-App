from fastapi import APIRouter

from app.core.constants import HEALTH_OK

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": HEALTH_OK}
