"""Finnhub webhook endpoint for real-time events."""

import logging
import hmac
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])

settings = get_settings()


class FinnhubWebhookEvent(BaseModel):
    """Finnhub webhook event payload."""
    type: str
    symbol: str | None = None
    timestamp: int | None = None
    data: dict | Any = None


def _verify_finnhub_signature(signature: str | None) -> bool:
    """Verify Finnhub webhook secret header."""
    if not settings.finnhub_webhook_secret:
        logger.warning("Finnhub webhook secret not configured")
        return False
    
    if not signature:
        logger.warning("No X-Finnhub-Secret header provided in webhook")
        return False
    
    # Finnhub sends the shared secret directly in the X-Finnhub-Secret header.
    # Constant-time comparison prevents timing attacks.
    is_valid = hmac.compare_digest(signature.strip(), settings.finnhub_webhook_secret.strip())
    
    if not is_valid:
        logger.warning(f"Invalid Finnhub webhook signature: {signature[:10]}...")
    
    return is_valid


@router.post("/finnhub/events")
async def handle_finnhub_webhook(request: Request) -> dict:
    """
    Handle Finnhub webhook events with signature verification.
    
    Finnhub sends events for:
    - Company news updates
    - Market alerts
    - Price movements
    - Insider transactions
    
    Returns 2xx status immediately to acknowledge receipt.
    """
    # Extract and verify signature
    signature = request.headers.get("X-Finnhub-Secret")
    
    if not _verify_finnhub_signature(signature):
        logger.error("Finnhub webhook signature verification failed")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")
    
    # Acknowledge receipt BEFORE processing (important!)
    acknowledgment = {"status": "received", "timestamp": datetime.utcnow().isoformat()}
    
    try:
        # Parse payload
        payload = await request.json()
        
        # Log webhook event
        logger.info(
            "finnhub_webhook_received",
            extra={
                "event_type": payload.get("type"),
                "symbol": payload.get("symbol"),
                "timestamp": payload.get("timestamp"),
            }
        )
        
        # Process event asynchronously (background task)
        event_type = payload.get("type", "unknown")
        
        if event_type == "company_news":
            await _process_company_news_event(payload)
        elif event_type == "market_alert":
            await _process_market_alert_event(payload)
        elif event_type == "price_target":
            await _process_price_target_event(payload)
        elif event_type == "insider_transaction":
            await _process_insider_transaction_event(payload)
        elif event_type == "earnings":
            await _process_earnings_event(payload)
        else:
            logger.warning(f"Unknown Finnhub event type: {event_type}")
        
        return acknowledgment
    
    except Exception as e:
        logger.exception(f"Error processing Finnhub webhook: {e}")
        # Still return 2xx to prevent endpoint disabling
        return acknowledgment


async def _process_company_news_event(payload: dict) -> None:
    """Process company news webhook event."""
    try:
        symbol = payload.get("symbol")
        news_items = payload.get("data", [])
        
        if not isinstance(news_items, list):
            news_items = [news_items]
        
        for news in news_items:
            logger.info(
                "finnhub_news_event_processed",
                extra={
                    "symbol": symbol,
                    "headline": news.get("headline", "")[:100],
                    "source": news.get("source", "Unknown"),
                    "url": news.get("url", ""),
                    "datetime": news.get("datetime"),
                }
            )
        
        # TODO: Store news in database, trigger sentiment analysis, etc.
        
    except Exception as e:
        logger.exception(f"Error processing company news event: {e}")


async def _process_market_alert_event(payload: dict) -> None:
    """Process market alert webhook event."""
    try:
        symbol = payload.get("symbol")
        alert_data = payload.get("data", {})
        
        logger.info(
            "finnhub_market_alert_processed",
            extra={
                "symbol": symbol,
                "alert_type": alert_data.get("type"),
                "price": alert_data.get("price"),
                "change_percent": alert_data.get("changePercent"),
            }
        )
        
        # TODO: Trigger trading alerts, notifications, etc.
        
    except Exception as e:
        logger.exception(f"Error processing market alert event: {e}")


async def _process_price_target_event(payload: dict) -> None:
    """Process price target webhook event."""
    try:
        symbol = payload.get("symbol")
        target_data = payload.get("data", {})
        
        logger.info(
            "finnhub_price_target_processed",
            extra={
                "symbol": symbol,
                "target_price": target_data.get("targetPrice"),
                "analyst": target_data.get("analystName"),
                "company": target_data.get("company"),
                "rating": target_data.get("rating"),
            }
        )
        
        # TODO: Store price targets, update forecasts, etc.
        
    except Exception as e:
        logger.exception(f"Error processing price target event: {e}")


async def _process_insider_transaction_event(payload: dict) -> None:
    """Process insider transaction webhook event."""
    try:
        symbol = payload.get("symbol")
        transaction = payload.get("data", {})
        
        logger.info(
            "finnhub_insider_transaction_processed",
            extra={
                "symbol": symbol,
                "person_name": transaction.get("personName"),
                "transaction_type": transaction.get("transactionType"),
                "share_count": transaction.get("shareCount"),
                "share_price": transaction.get("sharePrice"),
                "filing_date": transaction.get("filingDate"),
            }
        )
        
        # TODO: Store insider transactions, trigger alerts for significant transactions, etc.
        
    except Exception as e:
        logger.exception(f"Error processing insider transaction event: {e}")


async def _process_earnings_event(payload: dict) -> None:
    """Process earnings announcement webhook event."""
    try:
        symbol = payload.get("symbol")
        earnings_data = payload.get("data", {})
        
        logger.info(
            "finnhub_earnings_processed",
            extra={
                "symbol": symbol,
                "eps_estimate": earnings_data.get("epsEstimate"),
                "eps_actual": earnings_data.get("epsActual"),
                "revenue_estimate": earnings_data.get("revenueEstimate"),
                "revenue_actual": earnings_data.get("revenueActual"),
                "announcement_date": earnings_data.get("announcementDate"),
            }
        )
        
        # TODO: Store earnings data, update forecasts, trigger alerts, etc.
        
    except Exception as e:
        logger.exception(f"Error processing earnings event: {e}")


@router.get("/finnhub/health")
async def webhook_health() -> dict:
    """Health check endpoint for webhook configuration."""
    return {
        "status": "active",
        "webhook_configured": bool(settings.finnhub_webhook_secret),
        "timestamp": datetime.utcnow().isoformat(),
    }
