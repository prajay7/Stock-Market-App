from __future__ import annotations

import logging

from src.backtesting.engine import run_backtest

logger = logging.getLogger(__name__)


class BacktestService:
    def backtest(
        self,
        symbols: list[str],
        start: str,
        end: str,
        horizon_days: int,
        top_n: int,
        model_name: str,
        mode: str = "static",
        retrain_every_days: int = 20,
        min_train_rows: int = 300,
        train_lookback_days: int | None = None,
    ) -> dict:
        logger.info(
            "backtest_started",
            extra={
                "symbols_count": len(symbols),
                "start": str(start),
                "end": str(end),
                "horizon_days": int(horizon_days),
                "top_n": int(top_n),
                "model_name": str(model_name),
                "mode": str(mode),
            },
        )
        result = run_backtest(
            symbols=symbols,
            start=start,
            end=end,
            horizon_days=horizon_days,
            top_n=top_n,
            model_name=model_name,
            mode=mode,
            retrain_every_days=retrain_every_days,
            min_train_rows=min_train_rows,
            train_lookback_days=train_lookback_days,
        )
        logger.info(
            "backtest_completed",
            extra={"symbols_count": len(symbols), "rows": int(len(result.daily)), "metrics": result.metrics},
        )
        return {"metrics": result.metrics, "rows": len(result.daily)}


backtest_service = BacktestService()
