"""
FastAPI routes for watchlists, alert rules, and generated alerts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.config import get_settings
from app.news.models import AlertRule, Watchlist, WatchlistItem
from app.services.alert_matcher_service import alert_matcher_service
from app.services.watchlist_service import watchlist_service
from src.data.metadata_store import metadata_store

router = APIRouter(prefix="/alerts", tags=["alerts"])


# Watchlist endpoints
@router.post("/watchlists")
def create_watchlist(name: str, description: str = "", is_active: bool = True):
    """Create a new watchlist."""
    try:
        watchlist = watchlist_service.create_watchlist(name, description, is_active)
        return watchlist
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/watchlists")
def list_watchlists(active_only: bool = False):
    """Get all watchlists."""
    watchlists = watchlist_service.read_watchlists(active_only=active_only)
    return {"watchlists": watchlists}


@router.get("/watchlists/{watchlist_id}")
def get_watchlist(watchlist_id: int):
    """Get a specific watchlist."""
    watchlist = watchlist_service.read_watchlist(watchlist_id)
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    return watchlist


@router.put("/watchlists/{watchlist_id}")
def update_watchlist(watchlist_id: int, name: str = None, description: str = None, is_active: bool = None):
    """Update a watchlist."""
    try:
        watchlist = watchlist_service.update_watchlist(watchlist_id, name, description, is_active)
        if not watchlist:
            raise HTTPException(status_code=404, detail="Watchlist not found")
        return watchlist
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/watchlists/{watchlist_id}")
def delete_watchlist(watchlist_id: int):
    """Delete a watchlist."""
    success = watchlist_service.delete_watchlist(watchlist_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to delete watchlist")
    return {"status": "deleted"}


# Watchlist item endpoints
@router.post("/watchlists/{watchlist_id}/items")
def add_watchlist_item(watchlist_id: int, item_type: str, item_value: str):
    """Add an item to a watchlist."""
    try:
        item = watchlist_service.create_watchlist_item(watchlist_id, item_type, item_value)
        if not item:
            raise HTTPException(status_code=400, detail="Failed to create watchlist item")
        return item
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/watchlists/{watchlist_id}/items")
def list_watchlist_items(watchlist_id: int):
    """Get items in a watchlist."""
    items = watchlist_service.read_watchlist_items(watchlist_id)
    return {"items": items}


@router.delete("/watchlist-items/{item_id}")
def delete_watchlist_item(item_id: int):
    """Delete a watchlist item."""
    success = watchlist_service.delete_watchlist_item(item_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to delete item")
    return {"status": "deleted"}


# Alert rule endpoints
@router.post("/rules")
def create_alert_rule(rule_data: dict):
    """Create a new alert rule."""
    try:
        # Serialize list fields to JSON strings
        if "timing_labels" in rule_data:
            import json
            rule_data["timing_labels"] = json.dumps(rule_data["timing_labels"])
        if "event_types" in rule_data:
            import json
            rule_data["event_types"] = json.dumps(rule_data["event_types"])
        if "sectors" in rule_data:
            import json
            rule_data["sectors"] = json.dumps(rule_data["sectors"])
        if "notification_channels" in rule_data:
            import json
            rule_data["notification_channels"] = json.dumps(rule_data["notification_channels"])
        
        rule_id = metadata_store.create_alert_rule(rule_data)
        return {"id": rule_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/rules")
def list_alert_rules(active_only: bool = False):
    """Get all alert rules."""
    rules = metadata_store.read_alert_rules(active_only=active_only)
    return {"rules": rules}


@router.get("/rules/{rule_id}")
def get_alert_rule(rule_id: int):
    """Get a specific alert rule."""
    rule = metadata_store.read_alert_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return rule


@router.put("/rules/{rule_id}")
def update_alert_rule(rule_id: int, rule_data: dict):
    """Update an alert rule."""
    try:
        # Serialize list fields to JSON strings
        if "timing_labels" in rule_data:
            import json
            rule_data["timing_labels"] = json.dumps(rule_data["timing_labels"])
        if "event_types" in rule_data:
            import json
            rule_data["event_types"] = json.dumps(rule_data["event_types"])
        if "sectors" in rule_data:
            import json
            rule_data["sectors"] = json.dumps(rule_data["sectors"])
        if "notification_channels" in rule_data:
            import json
            rule_data["notification_channels"] = json.dumps(rule_data["notification_channels"])
        
        metadata_store.update_alert_rule(rule_id, rule_data)
        return {"status": "updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/rules/{rule_id}")
def delete_alert_rule(rule_id: int):
    """Delete an alert rule."""
    try:
        metadata_store.delete_alert_rule(rule_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Generated alert endpoints
@router.get("/")
def list_alerts(
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    severity: Optional[str] = None,
    hours: Optional[int] = None,
):
    """Get generated alerts."""
    if hours is not None:
        alerts = metadata_store.read_recent_alerts(hours=hours, limit=limit)
    else:
        alerts = metadata_store.read_alerts(limit=limit, status=status, severity=severity)
    return {"alerts": alerts}


@router.put("/{alert_id}/status")
def update_alert(alert_id: int, status: str):
    """Update alert status."""
    valid_statuses = {"new", "seen", "sent", "failed"}
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    try:
        metadata_store.update_alert_status(alert_id, status)
        return {"status": "updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scan")
def trigger_alert_scan():
    """Manually trigger alert scanning."""
    try:
        alert_ids = alert_matcher_service.scan_recent_opportunities()
        return {"scanned": True, "alerts_created": len(alert_ids)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
