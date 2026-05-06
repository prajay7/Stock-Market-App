from __future__ import annotations

import logging

from app.core.config import get_settings
from src.data.historical_loader import HistoricalLoader

logger = logging.getLogger(__name__)


class DataService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.loader = HistoricalLoader(
            self.settings.raw_data_dir,
            alpha_vantage_api_key=self.settings.alpha_vantage_api_key,
            polygon_api_key=self.settings.polygon_api_key,
            stooq_api_key=self.settings.stooq_api_key,
            data_provider=self.settings.data_provider,
            db_path=self.settings.db_path,
        )

    def ingest_historical(self, symbols: list[str], interval: str, lookback_days: int) -> dict[str, int]:
        logger.info(
            "data_ingest_started",
            extra={"symbols_count": len(symbols), "interval": str(interval), "lookback_days": int(lookback_days)},
        )
        result = self.loader.ingest(
            symbols=symbols,
            interval=interval,
            lookback_days=lookback_days,
            force_refresh=bool(self.settings.force_refresh),
        )
        logger.info("data_ingest_completed", extra={"symbols_count": len(symbols), "result": result})
        return result


data_service = DataService()
