from __future__ import annotations

from dataclasses import dataclass

from src.data.db import SQLiteDataStore


@dataclass
class SourceManager:
    store: SQLiteDataStore
    failure_skip_threshold: int = 3

    def source_order(self, interval: str, preferred: str = "") -> list[str]:
        interval_value = str(interval or "1d").strip().lower()
        order = ["nse", "stooq", "scraper"] if interval_value == "1d" else ["scraper"]
        preferred_value = str(preferred or "").strip().lower()
        if preferred_value in order:
            order.remove(preferred_value)
            order.insert(0, preferred_value)
        return order

    def should_skip(self, source: str) -> bool:
        source_name = str(source or "").strip().lower()
        if not source_name:
            return False
        if source_name == "scraper":
            # Keep the public web scraper as a last-resort fallback even if past runs failed.
            return False
        for item in self.store.read_source_health(limit=100):
            if str(item.get("source") or "").strip().lower() != source_name:
                continue
            return int(item.get("failure_count") or 0) >= self.failure_skip_threshold
        return False

    def record_success(self, source: str, latest_available_time: str | None = None) -> None:
        self.store.upsert_source_health(source, success=True, latest_available_time=latest_available_time)

    def record_failure(self, source: str, error: str | None = None) -> None:
        self.store.upsert_source_health(source, success=False, error=error)
