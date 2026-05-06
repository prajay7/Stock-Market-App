from __future__ import annotations

import logging
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import get_settings
from app.news.backtest_service import news_signal_backtest_service
from app.services.alert_matcher_service import alert_matcher_service
from app.services.automation_service import automation_service
from app.services.data_service import data_service
from app.services.news_service import news_service
from app.services.paper_trading_service import paper_trading_service
from app.services.prediction_service import prediction_service
from app.services.training_service import training_service
from app.services.validation_service import validation_service

logger = logging.getLogger(__name__)


class SchedulerService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.scheduler = BackgroundScheduler(timezone=ZoneInfo(self.settings.market_timezone))
        self._started = False
        self._last_post_close_run: str | None = None

    def _market_now(self) -> datetime:
        return datetime.now(ZoneInfo(self.settings.market_timezone))

    def _market_hours_open(self) -> bool:
        now = self._market_now()
        if now.weekday() >= 5:
            return False
        open_time = dt_time.fromisoformat(self.settings.market_open_time)
        close_time = dt_time.fromisoformat(self.settings.market_close_time)
        return open_time <= now.time() <= close_time

    def _run_market_cycle(self) -> dict:
        if not self._market_hours_open():
            return {"status": "skipped", "reason": "outside_market_hours"}
        symbols = self.settings.default_symbols
        data_result = data_service.ingest_historical(symbols, self.settings.historical_interval, self.settings.historical_lookback_days)
        prediction_result = prediction_service.predict(symbols, "movement_model", 1, include_live_quote=False)
        return {"status": "ok", "data": data_result, "predictions": len(prediction_result.get("predictions") or [])}

    def _run_post_close_cycle(self) -> dict:
        now = self._market_now().date().isoformat()
        if self._last_post_close_run == now:
            return {"status": "skipped", "reason": "already_ran_today"}
        if self._market_hours_open():
            return {"status": "skipped", "reason": "market_still_open"}
        symbols = self.settings.default_symbols
        validation_result = validation_service.validate_pending_predictions(self.settings.historical_interval)
        training_result = training_service.train(symbols, 1, task_type="movement")
        self._last_post_close_run = now
        return {"status": "ok", "validation": validation_result, "training": training_result}

    def start(self) -> None:
        if self._started:
            logger.info("scheduler_start_skipped", extra={"reason": "already_started"})
            return
        symbols = self.settings.default_symbols

        logger.info(
            "scheduler_registering_jobs",
            extra={
                "symbols_count": len(symbols),
                "scheduler_enabled": True,
                "automation_enabled": bool(self.settings.automation_enabled),
            },
        )

        def _run_job(job_name: str, fn):
            logger.info("scheduler_job_started", extra={"job_name": str(job_name)})
            try:
                result = fn()
                logger.info("scheduler_job_completed", extra={"job_name": str(job_name), "result": result})
                return result
            except Exception as exc:
                logger.exception("scheduler_job_failed", extra={"job_name": str(job_name), "error": str(exc)})
                raise

        self.scheduler.add_job(
            lambda: _run_job("market_cycle", self._run_market_cycle),
            trigger="interval",
            minutes=15,
            id="market_cycle",
            replace_existing=True,
        )
        self.scheduler.add_job(
            lambda: _run_job("news_refresh", lambda: news_service.ingest_news(symbols, self.settings.max_news_items_per_symbol)),
            trigger="interval",
            minutes=self.settings.news_refresh_minutes,
            id="news_refresh",
            replace_existing=True,
        )
        self.scheduler.add_job(
            lambda: _run_job("post_close_cycle", self._run_post_close_cycle),
            trigger="cron",
            hour=int(self.settings.market_close_time.split(":", 1)[0]),
            minute=min(59, int(self.settings.market_close_time.split(":", 1)[1]) + 5),
            id="post_close_cycle",
            replace_existing=True,
        )

        if self.settings.backtest_enabled and self.settings.backtest_scheduler_enabled:
            self.scheduler.add_job(
                lambda: _run_job("news_signal_backtest_evaluator", lambda: news_signal_backtest_service.evaluate_pending_outcomes()),
                trigger="interval",
                minutes=max(1, int(self.settings.backtest_scheduler_interval_minutes)),
                id="news_signal_backtest_evaluator",
                replace_existing=True,
                max_instances=1,
            )

        if self.settings.alerts_enabled and self.settings.alert_scan_enabled:
            self.scheduler.add_job(
                lambda: _run_job("alert_rule_scanner", lambda: alert_matcher_service.scan_recent_opportunities()),
                trigger="interval",
                minutes=max(1, int(self.settings.alert_scan_interval_minutes)),
                id="alert_rule_scanner",
                replace_existing=True,
                max_instances=1,
            )

        if self.settings.paper_trading_enabled and self.settings.paper_trading_price_refresh_enabled:
            self.scheduler.add_job(
                lambda: _run_job(
                    "paper_trade_price_refresh",
                    lambda: paper_trading_service.refresh_open_trades(limit=self.settings.paper_trading_price_refresh_limit),
                ),
                trigger="interval",
                minutes=max(1, int(self.settings.paper_trading_price_refresh_interval_minutes)),
                id="paper_trade_price_refresh",
                replace_existing=True,
                max_instances=1,
            )

        if self.settings.automation_enabled:
            self.scheduler.add_job(
                lambda: _run_job("automation_cycle_runner", lambda: automation_service.run_cycle()),
                trigger="interval",
                minutes=max(1, int(self.settings.automation_interval_minutes)),
                id="automation_cycle_runner",
                replace_existing=True,
                max_instances=1,
            )

        self.scheduler.start()
        self._started = True
        logger.info("scheduler_started", extra={"jobs_count": len(self.scheduler.get_jobs())})

    def shutdown(self) -> None:
        if self._started:
            logger.info("scheduler_shutdown_started")
            self.scheduler.shutdown(wait=False)
            self._started = False
            logger.info("scheduler_shutdown_completed")


scheduler_service = SchedulerService()
