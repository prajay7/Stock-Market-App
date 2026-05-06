from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.core.config import get_settings
from src.data.db import SQLiteDataStore
from src.data.historical_loader import HistoricalLoader
from src.training.train import train_models
from src.utils.symbols import load_symbols_from_csv, normalize_symbols


TaskType = Literal["classification", "regression_return", "regression_close", "movement"]


def _resolve_symbols(
    symbols: list[str] | None = None,
    symbols_csv: str | None = None,
    symbol_column: str | None = None,
    market: str = "us",
    series_filter: str | None = None,
    max_symbols: int | None = None,
) -> list[str]:
    settings = get_settings()
    if symbols_csv:
        resolved = load_symbols_from_csv(
            symbols_csv,
            symbol_column=symbol_column,
            market=market,
            series_filter=series_filter,
            max_symbols=max_symbols,
        )
        return normalize_symbols(resolved)
    if symbols:
        return normalize_symbols(symbols)
    return normalize_symbols(settings.default_symbols)


def run_model_train_pipeline(
    symbols: list[str] | None = None,
    symbols_csv: str | None = None,
    symbol_column: str | None = None,
    market: str = "us",
    series_filter: str | None = None,
    max_symbols: int | None = None,
    horizon_days: int = 1,
    task_type: TaskType = "classification",
    ingest_first: bool = False,
    lookback_days: int | None = None,
    interval: str | None = None,
) -> dict:
    settings = get_settings()
    run_started_at = datetime.now(timezone.utc)
    resolved_symbols = _resolve_symbols(
        symbols=symbols,
        symbols_csv=symbols_csv,
        symbol_column=symbol_column,
        market=market,
        series_filter=series_filter,
        max_symbols=max_symbols,
    )
    if not resolved_symbols:
        raise ValueError("No symbols resolved for training")

    ingest_summary: dict[str, int] | None = None
    refresh_prices = not ingest_first
    effective_interval = str(interval or settings.historical_interval)
    effective_lookback = int(lookback_days or settings.historical_lookback_days)

    if ingest_first:
        loader = HistoricalLoader(
            settings.raw_data_dir,
            alpha_vantage_api_key=settings.alpha_vantage_api_key,
            polygon_api_key=settings.polygon_api_key,
            stooq_api_key=settings.stooq_api_key,
            data_provider=settings.data_provider,
        )
        ingest_summary = loader.ingest(
            resolved_symbols,
            interval=effective_interval,
            lookback_days=effective_lookback,
        )

    training_result = train_models(
        symbols=resolved_symbols,
        horizon_days=int(horizon_days),
        refresh_prices=refresh_prices,
        task_type=task_type,
    )

    run_completed_at = datetime.now(timezone.utc)
    run_version = str(training_result.get("version") or run_completed_at.strftime("%Y%m%d%H%M%S"))
    run_payload = {
        "model_version": run_version,
        "started_at": run_started_at.isoformat(),
        "completed_at": run_completed_at.isoformat(),
        "task_type": task_type,
        "horizon_days": int(horizon_days),
        "symbols": resolved_symbols,
        "ingest_first": bool(ingest_first),
        "ingest_summary": ingest_summary,
        "metrics": training_result.get("best") or {},
        "artifact_path": str(settings.model_dir),
        "model_type": (training_result.get("best") or {}).get("best_model_name"),
        "payload": training_result,
    }
    SQLiteDataStore(settings.db_path).write_training_run(run_payload)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = settings.output_dir / f"training_pipeline_{run_version}.json"
    summary = {
        "version": run_version,
        "started_at": run_started_at.isoformat(),
        "completed_at": run_completed_at.isoformat(),
        "symbols_count": len(resolved_symbols),
        "task_type": task_type,
        "horizon_days": int(horizon_days),
        "ingest_first": bool(ingest_first),
        "ingest_summary": ingest_summary,
        "result": training_result,
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary