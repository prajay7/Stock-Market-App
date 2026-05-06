from __future__ import annotations

import logging

from src.training.train import train_models

logger = logging.getLogger(__name__)


class TrainingService:
    def train(self, symbols: list[str], horizon_days: int, task_type: str = "classification") -> dict:
        logger.info(
            "training_started",
            extra={"symbols_count": len(symbols), "horizon_days": int(horizon_days), "task_type": str(task_type)},
        )
        result = train_models(symbols=symbols, horizon_days=horizon_days, task_type=task_type)
        logger.info("training_completed", extra={"symbols_count": len(symbols), "result": result})
        return result


training_service = TrainingService()
