from contextlib import asynccontextmanager
import logging
import time
import uuid

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes_alerts import router as alerts_router
from app.api.routes_automation import router as automation_router
from app.api.routes_backtest import router as backtest_router
from app.api.routes_data import router as data_router
from app.api.routes_health import router as health_router
from app.api.routes_news_backtest import router as news_backtest_router
from app.api.routes_predict import router as predict_router
from app.api.routes_trades import router as trades_router
from app.api.routes_train import router as train_router
from app.api.routes_webhooks import router as webhooks_router
from app.api.error_utils import build_error_envelope
from app.core.config import get_settings
from app.core.logger import setup_logging
from app.services.scheduler_service import scheduler_service

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.scheduler_enabled:
        logger.info("scheduler_starting")
        scheduler_service.start()
    try:
        logger.info("app_startup_completed", extra={"scheduler_enabled": bool(settings.scheduler_enabled)})
        yield
    finally:
        logger.info("app_shutdown_started")
        scheduler_service.shutdown()
        logger.info("app_shutdown_completed")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    description="Educational stock prediction and backtesting platform.",
)


@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    path = request.url.path
    method = request.method
    query = str(request.url.query or "")

    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            "query": query,
            "client": request.client.host if request.client else None,
        },
    )

    try:
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": int(response.status_code),
                "duration_ms": elapsed_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "duration_ms": elapsed_ms,
                "error": str(exc),
            },
        )
        raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        "validation_error",
        extra={
            "method": request.method,
            "path": request.url.path,
            "errors": exc.errors(),
        },
    )
    return JSONResponse(
        status_code=422,
        content=build_error_envelope(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors()},
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        "http_exception",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": int(exc.status_code),
            "detail": str(exc.detail),
        },
    )
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        payload = detail
    else:
        payload = build_error_envelope(
            code="HTTP_ERROR",
            message=str(detail),
            details={"status_code": exc.status_code},
        )
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled_exception",
        extra={
            "method": request.method,
            "path": request.url.path,
            "error": str(exc),
        },
    )
    return JSONResponse(
        status_code=500,
        content=build_error_envelope(
            code="INTERNAL_SERVER_ERROR",
            message="Unexpected server error",
            details={"error": str(exc)},
        ),
    )

app.include_router(health_router)
app.include_router(data_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(backtest_router)
app.include_router(news_backtest_router)
app.include_router(alerts_router)
app.include_router(trades_router)
app.include_router(automation_router)
app.include_router(webhooks_router)
