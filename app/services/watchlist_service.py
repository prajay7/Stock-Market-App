"""
Watchlist service for managing user-defined watchlists and items.
"""
from __future__ import annotations

import logging
from datetime import datetime

from app.core.config import get_settings
from app.news.models import Watchlist, WatchlistItem
from src.data.metadata_store import metadata_store

logger = logging.getLogger(__name__)


class WatchlistService:
    """Service for managing watchlists and their items."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = metadata_store

    def create_watchlist(self, name: str, description: str = "", is_active: bool = True) -> Watchlist:
        """Create a new watchlist."""
        logger.info("watchlist_create_started", extra={"name": str(name), "is_active": bool(is_active)})
        try:
            watchlist_id = self.store.create_watchlist(name, description, is_active)
            result = self._read_watchlist_by_id(watchlist_id)
            logger.info("watchlist_create_completed", extra={"watchlist_id": int(watchlist_id), "name": str(name)})
            return result
        except Exception as e:
            logger.exception("watchlist_create_failed", extra={"name": str(name), "error": str(e)})
            raise

    def read_watchlists(self, active_only: bool = False) -> list[Watchlist]:
        """Read all watchlists with their items."""
        logger.info("watchlist_read_all_started", extra={"active_only": bool(active_only)})
        try:
            rows = self.store.read_watchlists(active_only=active_only)
            watchlists = []
            for row in rows:
                watchlist = Watchlist(**row)
                watchlist.items = self.read_watchlist_items(row["id"])
                watchlists.append(watchlist)
            logger.info("watchlist_read_all_completed", extra={"count": len(watchlists), "active_only": bool(active_only)})
            return watchlists
        except Exception as e:
            logger.exception("watchlist_read_all_failed", extra={"active_only": bool(active_only), "error": str(e)})
            return []

    def read_watchlist(self, watchlist_id: int) -> Watchlist | None:
        """Read a single watchlist with its items."""
        logger.info("watchlist_read_one_started", extra={"watchlist_id": int(watchlist_id)})
        try:
            result = self._read_watchlist_by_id(watchlist_id)
            logger.info("watchlist_read_one_completed", extra={"watchlist_id": int(watchlist_id), "found": bool(result)})
            return result
        except Exception as e:
            logger.exception("watchlist_read_one_failed", extra={"watchlist_id": int(watchlist_id), "error": str(e)})
            return None

    def _read_watchlist_by_id(self, watchlist_id: int) -> Watchlist | None:
        """Internal method to read a single watchlist with items."""
        row = self.store.read_watchlist(watchlist_id)
        if not row:
            return None
        watchlist = Watchlist(**row)
        watchlist.items = self.read_watchlist_items(watchlist_id)
        return watchlist

    def update_watchlist(
        self, watchlist_id: int, name: str = None, description: str = None, is_active: bool = None
    ) -> Watchlist | None:
        """Update a watchlist."""
        logger.info("watchlist_update_started", extra={"watchlist_id": int(watchlist_id)})
        try:
            self.store.update_watchlist(watchlist_id, name, description, is_active)
            result = self._read_watchlist_by_id(watchlist_id)
            logger.info("watchlist_update_completed", extra={"watchlist_id": int(watchlist_id), "updated": bool(result)})
            return result
        except Exception as e:
            logger.exception("watchlist_update_failed", extra={"watchlist_id": int(watchlist_id), "error": str(e)})
            return None

    def delete_watchlist(self, watchlist_id: int) -> bool:
        """Delete a watchlist and all its items."""
        logger.info("watchlist_delete_started", extra={"watchlist_id": int(watchlist_id)})
        try:
            self.store.delete_watchlist(watchlist_id)
            logger.info("watchlist_delete_completed", extra={"watchlist_id": int(watchlist_id)})
            return True
        except Exception as e:
            logger.exception("watchlist_delete_failed", extra={"watchlist_id": int(watchlist_id), "error": str(e)})
            return False

    def create_watchlist_item(
        self, watchlist_id: int, item_type: str, item_value: str
    ) -> WatchlistItem | None:
        """
        Create a watchlist item.
        Supported item types: company, ticker, sector, event_type
        """
        logger.info(
            "watchlist_item_create_started",
            extra={"watchlist_id": int(watchlist_id), "item_type": str(item_type), "item_value": str(item_value)},
        )
        try:
            normalized = self._normalize_item_value(item_type, item_value)
            item_id = self.store.create_watchlist_item(watchlist_id, item_type, item_value, normalized)
            result = WatchlistItem(
                id=item_id,
                watchlist_id=watchlist_id,
                item_type=item_type,
                item_value=item_value,
                normalized_value=normalized,
                created_at=datetime.utcnow(),
            )
            logger.info("watchlist_item_create_completed", extra={"item_id": int(item_id), "watchlist_id": int(watchlist_id)})
            return result
        except Exception as e:
            logger.exception(
                "watchlist_item_create_failed",
                extra={"watchlist_id": int(watchlist_id), "item_type": str(item_type), "item_value": str(item_value), "error": str(e)},
            )
            return None

    def read_watchlist_items(self, watchlist_id: int) -> list[WatchlistItem]:
        """Read all items in a watchlist."""
        logger.info("watchlist_items_read_started", extra={"watchlist_id": int(watchlist_id)})
        try:
            rows = self.store.read_watchlist_items(watchlist_id)
            result = [WatchlistItem(**row) for row in rows]
            logger.info("watchlist_items_read_completed", extra={"watchlist_id": int(watchlist_id), "count": len(result)})
            return result
        except Exception as e:
            logger.exception("watchlist_items_read_failed", extra={"watchlist_id": int(watchlist_id), "error": str(e)})
            return []

    def delete_watchlist_item(self, item_id: int) -> bool:
        """Delete a watchlist item."""
        logger.info("watchlist_item_delete_started", extra={"item_id": int(item_id)})
        try:
            self.store.delete_watchlist_item(item_id)
            logger.info("watchlist_item_delete_completed", extra={"item_id": int(item_id)})
            return True
        except Exception as e:
            logger.exception("watchlist_item_delete_failed", extra={"item_id": int(item_id), "error": str(e)})
            return False

    @staticmethod
    def _normalize_item_value(item_type: str, item_value: str) -> str:
        """Normalize item values for consistent matching."""
        if item_type in ("ticker", "company"):
            return str(item_value).upper().strip()
        if item_type == "sector":
            return str(item_value).upper().strip()
        if item_type == "event_type":
            return str(item_value).lower().strip().replace(" ", "_")
        return str(item_value).upper().strip()

    def get_items_by_type(self, watchlist_id: int, item_type: str) -> list[str]:
        """Get normalized values of a specific item type from a watchlist."""
        items = self.read_watchlist_items(watchlist_id)
        return [item.normalized_value for item in items if item.item_type == item_type]


watchlist_service = WatchlistService()
