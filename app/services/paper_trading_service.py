from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from app.core.config import get_settings
from app.news.models import PaperTradeCreate, PaperTradeClose
from src.data.metadata_store import metadata_store

logger = logging.getLogger(__name__)


class PaperTradingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = metadata_store

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _as_naive(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return str(symbol or "").strip().upper()

    @staticmethod
    def _latest_close(symbol: str) -> float | None:
        ticker = self_ticker = yf.Ticker(symbol)
        try:
            history = self_ticker.history(period="7d", interval="1d", auto_adjust=False)
        except Exception:
            return None
        if history is None or history.empty or "Close" not in history.columns:
            return None
        close_series = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if close_series.empty:
            return None
        return float(close_series.iloc[-1])

    def _get_market_price(self, symbol: str) -> float | None:
        try:
            return self._latest_close(self._normalize_symbol(symbol))
        except Exception as exc:
            logger.debug("paper_trade_price_lookup_failed", extra={"symbol": symbol, "error": str(exc)})
            return None

    @staticmethod
    def _coerce_float(value: float | int | str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _build_position_sizes(
        self,
        entry_price: float,
        quantity: float | None,
        capital: float | None,
    ) -> tuple[float, float]:
        entry_price = float(entry_price)
        quantity_value = self._coerce_float(quantity)
        capital_value = self._coerce_float(capital)

        if quantity_value is not None and quantity_value <= 0:
            quantity_value = None
        if capital_value is not None and capital_value <= 0:
            capital_value = None

        if quantity_value is not None and capital_value is not None:
            return float(quantity_value), float(capital_value)
        if quantity_value is not None:
            return float(quantity_value), float(quantity_value) * entry_price
        if capital_value is not None:
            return float(capital_value) / entry_price, float(capital_value)
        raise ValueError("Provide either quantity or capital")

    def _snapshot_open_metrics(self, trade: dict) -> dict[str, Any]:
        current_price = self._get_market_price(str(trade.get("symbol", "")))
        entry_price = self._coerce_float(trade.get("entry_price")) or 0.0
        quantity = self._coerce_float(trade.get("quantity"))
        capital = self._coerce_float(trade.get("capital"))
        entry_date = pd.to_datetime(trade.get("entry_date"), errors="coerce")
        holding_days = None
        if pd.notna(entry_date):
            holding_days = max(0, (datetime.utcnow().date() - pd.Timestamp(entry_date).to_pydatetime().date()).days)

        current_pnl = None
        current_return_pct = None
        if current_price is not None and quantity is not None:
            current_pnl = (current_price - entry_price) * quantity
            if capital:
                current_return_pct = (current_pnl / capital) * 100.0
            elif entry_price:
                current_return_pct = ((current_price / entry_price) - 1.0) * 100.0

        return {
            "current_price": current_price,
            "current_pnl": current_pnl,
            "current_return_pct": current_return_pct,
            "holding_days": holding_days,
        }

    def create_trade(self, payload: PaperTradeCreate) -> int:
        symbol = self._normalize_symbol(payload.symbol)
        logger.info("paper_trade_create_started", extra={"symbol": symbol, "source_type": str(payload.source_type)})
        entry_price = self._coerce_float(payload.entry_price)
        if entry_price is None:
            entry_price = self._get_market_price(symbol)
        if entry_price is None or entry_price <= 0:
            raise ValueError(f"Could not determine entry price for {symbol}")

        quantity, capital = self._build_position_sizes(entry_price, payload.quantity, payload.capital)
        entry_date = self._as_naive(payload.entry_date) or self._utc_now()
        trade_data = {
            "symbol": symbol,
            "source_type": payload.source_type,
            "source_id": payload.source_id,
            "source_label": payload.source_label,
            "entry_date": entry_date,
            "entry_price": float(entry_price),
            "quantity": float(quantity),
            "capital": float(capital),
            "status": "open",
            "notes": payload.notes,
            "trade_reason": payload.trade_reason,
        }
        trade_data.update(self._snapshot_open_metrics(trade_data))
        trade_id = self.store.create_paper_trade(trade_data)
        logger.info("paper_trade_create_completed", extra={"trade_id": int(trade_id), "symbol": symbol})
        return trade_id

    def create_trade_from_alert(
        self,
        alert_id: int,
        entry_price: float | None = None,
        quantity: float | None = None,
        capital: float | None = None,
        notes: str = "",
        trade_reason: str = "",
    ) -> int:
        alert = self.store.read_alert_with_context(alert_id)
        if not alert:
            raise ValueError("Alert not found")
        symbol = self._normalize_symbol(alert.get("ticker") or alert.get("primary_ticker") or "")
        if not symbol:
            raise ValueError("Alert does not include a tradable ticker")
        source_label = str(alert.get("title") or alert.get("message") or f"Alert {alert_id}")
        payload = PaperTradeCreate(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            capital=capital,
            source_type="alert",
            source_id=alert_id,
            source_label=source_label,
            notes=notes,
            trade_reason=trade_reason or str(alert.get("message") or source_label),
        )
        return self.create_trade(payload)

    def create_trade_from_opportunity(
        self,
        opportunity_id: int,
        entry_price: float | None = None,
        quantity: float | None = None,
        capital: float | None = None,
        notes: str = "",
        trade_reason: str = "",
    ) -> int:
        opportunity = self.store.read_beneficiary_opportunity(opportunity_id)
        if not opportunity:
            raise ValueError("Opportunity not found")
        symbol = self._normalize_symbol(opportunity.get("ticker") or opportunity.get("beneficiary_ticker") or opportunity.get("primary_ticker") or "")
        if not symbol:
            raise ValueError("Opportunity does not include a tradable ticker")
        source_label = str(opportunity.get("title") or opportunity.get("company") or f"Opportunity {opportunity_id}")
        payload = PaperTradeCreate(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            capital=capital,
            source_type="opportunity",
            source_id=opportunity_id,
            source_label=source_label,
            notes=notes,
            trade_reason=trade_reason or str(opportunity.get("reason") or source_label),
        )
        return self.create_trade(payload)

    def refresh_open_trades(self, limit: int = 500) -> int:
        logger.info("paper_trade_refresh_started", extra={"limit": int(limit)})
        updated = 0
        for trade in self.store.read_open_paper_trades(limit=limit):
            snapshot = self._snapshot_open_metrics(trade)
            if snapshot["current_price"] is None:
                continue
            self.store.update_paper_trade(trade["id"], snapshot)
            updated += 1
        logger.info("paper_trade_refresh_completed", extra={"updated": int(updated), "limit": int(limit)})
        return updated

    def close_trade(self, trade_id: int, close_payload: PaperTradeClose | None = None, exit_price: float | None = None) -> None:
        logger.info("paper_trade_close_started", extra={"trade_id": int(trade_id)})
        trade = self.store.read_paper_trade(trade_id)
        if not trade:
            raise ValueError("Trade not found")
        if str(trade.get("status")) != "open":
            raise ValueError("Trade is already closed")

        payload = close_payload or PaperTradeClose()
        trade_exit_price = self._coerce_float(exit_price) or self._coerce_float(payload.exit_price)
        if trade_exit_price is None:
            trade_exit_price = self._get_market_price(str(trade.get("symbol", "")))
        if trade_exit_price is None or trade_exit_price <= 0:
            raise ValueError("Could not determine exit price")

        entry_price = self._coerce_float(trade.get("entry_price")) or 0.0
        quantity = self._coerce_float(trade.get("quantity")) or 0.0
        capital = self._coerce_float(trade.get("capital")) or 0.0
        realized_pnl = (trade_exit_price - entry_price) * quantity
        realized_return_pct = (realized_pnl / capital) * 100.0 if capital else ((trade_exit_price / entry_price) - 1.0) * 100.0 if entry_price else None

        entry_date = pd.to_datetime(trade.get("entry_date"), errors="coerce")
        exit_date = self._as_naive(payload.exit_date) or self._utc_now()
        holding_days = None
        if pd.notna(entry_date):
            holding_days = max(0, (pd.Timestamp(exit_date).to_pydatetime().date() - pd.Timestamp(entry_date).to_pydatetime().date()).days)

        updates = {
            "exit_date": exit_date,
            "exit_price": float(trade_exit_price),
            "realized_pnl": float(realized_pnl),
            "realized_return_pct": float(realized_return_pct) if realized_return_pct is not None else None,
            "holding_days": holding_days,
            "notes": payload.notes or trade.get("notes") or "",
        }
        self.store.close_paper_trade(trade_id, updates)
        logger.info("paper_trade_close_completed", extra={"trade_id": int(trade_id), "symbol": str(trade.get("symbol") or "")})

    def get_analytics(self, limit: int = 5000) -> dict[str, Any]:
        logger.info("paper_trade_analytics_started", extra={"limit": int(limit)})
        trades = self.store.read_paper_trades(limit=limit)
        if not trades:
            result = {
                "total_trades": 0,
                "open_trades": 0,
                "closed_trades": 0,
                "win_rate": 0.0,
                "avg_return_pct": 0.0,
                "total_pnl": 0.0,
                "avg_holding_days": 0.0,
            }
            logger.info("paper_trade_analytics_completed", extra=result)
            return result

        closed = [trade for trade in trades if str(trade.get("status")) == "closed"]
        open_trades = [trade for trade in trades if str(trade.get("status")) == "open"]
        returns = [self._coerce_float(trade.get("realized_return_pct")) for trade in closed if self._coerce_float(trade.get("realized_return_pct")) is not None]
        pnls = [self._coerce_float(trade.get("realized_pnl")) for trade in closed if self._coerce_float(trade.get("realized_pnl")) is not None]
        holding_days = [self._coerce_float(trade.get("holding_days")) for trade in closed if self._coerce_float(trade.get("holding_days")) is not None]
        wins = [trade for trade in closed if (self._coerce_float(trade.get("realized_pnl")) or 0.0) > 0]
        closed_count = len(closed)

        result = {
            "total_trades": len(trades),
            "open_trades": len(open_trades),
            "closed_trades": closed_count,
            "win_rate": (len(wins) / closed_count * 100.0) if closed_count else 0.0,
            "avg_return_pct": float(sum(returns) / len(returns)) if returns else 0.0,
            "total_pnl": float(sum(pnls)) if pnls else 0.0,
            "avg_holding_days": float(sum(holding_days) / len(holding_days)) if holding_days else 0.0,
        }
        logger.info("paper_trade_analytics_completed", extra=result)
        return result


paper_trading_service = PaperTradingService()
