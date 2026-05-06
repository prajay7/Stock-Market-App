from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from app.core.config import get_settings
from src.data.db import SQLiteDataStore

logger = logging.getLogger(__name__)


class ValidationService:
    def validate_pending_predictions(self, interval: str | None = None) -> dict:
        settings = get_settings()
        store = SQLiteDataStore(settings.db_path)
        frame = store.read_predictions(limit=5000)
        if frame.empty:
            return {"status": "empty", "rows": 0, "metrics": {}}

        frame["validated"] = pd.to_numeric(frame.get("validated"), errors="coerce").fillna(0).astype(int)
        pending = frame[frame["validated"] == 0].copy()
        if pending.empty:
            return {"status": "empty", "rows": 0, "metrics": {}}

        candles = store.read_candles(pending["symbol"].astype(str).tolist(), interval or settings.historical_interval)
        if candles.empty:
            return {"status": "no_data", "rows": 0, "metrics": {}}

        candles["date"] = pd.to_datetime(candles["date"], errors="coerce")
        pending["prediction_time"] = pd.to_datetime(pending.get("prediction_time"), errors="coerce", utc=True)
        pending["prediction_time"] = pending["prediction_time"].dt.tz_convert(None)
        pending["current_price"] = pd.to_numeric(pending.get("current_price"), errors="coerce")

        outcomes: list[dict] = []
        actual_returns: list[float] = []
        y_true: list[int] = []
        y_pred: list[int] = []

        for _, row in pending.iterrows():
            symbol = str(row.get("symbol") or "").strip().upper()
            prediction_time = row.get("prediction_time")
            current_price = pd.to_numeric(row.get("current_price"), errors="coerce")
            if pd.isna(prediction_time) or pd.isna(current_price) or current_price <= 0:
                continue

            symbol_candles = candles[candles["symbol"].astype(str).str.upper() == symbol].copy()
            symbol_candles = symbol_candles[symbol_candles["date"] > prediction_time]
            if symbol_candles.empty:
                continue

            actual_close = pd.to_numeric(symbol_candles.iloc[0].get("close"), errors="coerce")
            if pd.isna(actual_close) or actual_close <= 0:
                continue

            actual_return = float(actual_close / current_price - 1.0)
            predicted_signal = str(row.get("signal") or row.get("decision") or "").lower()
            predicted_positive = int(predicted_signal in {"bullish", "buy_candidate"})
            actual_positive = int(actual_return > 0.01)

            actual_returns.append(actual_return)
            y_true.append(actual_positive)
            y_pred.append(predicted_positive)
            outcomes.append(
                {
                    "id": int(row.get("id")),
                    "validated": 1,
                    "validated_at": datetime.now(timezone.utc).isoformat(),
                    "actual_return": actual_return,
                    "outcome": "win" if actual_return > 0 else "loss",
                }
            )

        if not outcomes:
            return {"status": "no_valid_rows", "rows": 0, "metrics": {}}

        with store.connect() as conn:
            for item in outcomes:
                conn.execute(
                    """
                    UPDATE predictions
                    SET validated = ?, validated_at = ?, actual_return = ?, outcome = ?
                    WHERE id = ?
                    """,
                    (item["validated"], item["validated_at"], item["actual_return"], item["outcome"], item["id"]),
                )

        returns = np.asarray(actual_returns, dtype=float)
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        profit_factor = float(wins.sum() / abs(losses.sum())) if losses.size and abs(losses.sum()) > 0 else float("inf") if wins.size else 0.0
        cumulative = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / running_max) - 1.0
        metrics = {
            "rows": int(len(outcomes)),
            "win_rate": float((returns > 0).mean()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "profit_factor": profit_factor,
            "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        }
        store.write_validation_result(
            {
                "model_version": str(pending.iloc[-1].get("model_version") or "unknown"),
                "interval": str(interval or settings.historical_interval),
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "rows": outcomes,
            }
        )
        logger.info("validation_completed", extra={"rows": len(outcomes), "metrics": metrics})
        return {"status": "ok", "rows": len(outcomes), "metrics": metrics}


validation_service = ValidationService()
