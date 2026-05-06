"""
Tests for watchlists, alert rules, and alert matching logic.
"""
import json
from datetime import datetime

import pytest

from app.core.config import get_settings
from app.news.models import AlertPayload, AlertRule
from app.services.alert_matcher_service import alert_matcher_service
from app.services.alert_notifier import NotifierFactory
from app.services.watchlist_service import watchlist_service
from src.data.metadata_store import metadata_store


@pytest.fixture(scope="session")
def setup_metadata_store():
    """Setup metadata store for tests."""
    yield metadata_store


class TestWatchlistService:
    """Tests for watchlist service."""

    def test_create_watchlist(self, setup_metadata_store):
        """Test creating a watchlist."""
        watchlist = watchlist_service.create_watchlist("test_list", "A test watchlist")
        assert watchlist is not None
        assert watchlist.name == "test_list"
        assert watchlist.description == "A test watchlist"
        assert watchlist.is_active is True

    def test_read_watchlists(self, setup_metadata_store):
        """Test reading watchlists."""
        watchlist_service.create_watchlist("list1", "First")
        watchlist_service.create_watchlist("list2", "Second")
        
        lists = watchlist_service.read_watchlists()
        assert len(lists) >= 2

    def test_create_watchlist_item(self, setup_metadata_store):
        """Test creating watchlist items."""
        watchlist = watchlist_service.create_watchlist("test_items")
        item = watchlist_service.create_watchlist_item(watchlist.id, "ticker", "AAPL.NS")
        
        assert item is not None
        assert item.item_type == "ticker"
        assert item.normalized_value == "AAPL.NS"

    def test_normalize_item_value(self):
        """Test item value normalization."""
        assert watchlist_service._normalize_item_value("ticker", "aapl.ns") == "AAPL.NS"
        assert watchlist_service._normalize_item_value("sector", "Information Technology") == "INFORMATION TECHNOLOGY"
        assert watchlist_service._normalize_item_value("event_type", "Earnings Beat") == "earnings_beat"
        assert watchlist_service._normalize_item_value("company", "microsoft") == "MICROSOFT"

    def test_get_items_by_type(self, setup_metadata_store):
        """Test getting items by type."""
        watchlist = watchlist_service.create_watchlist("filter_test")
        watchlist_service.create_watchlist_item(watchlist.id, "ticker", "AAPL")
        watchlist_service.create_watchlist_item(watchlist.id, "sector", "Tech")
        
        tickers = watchlist_service.get_items_by_type(watchlist.id, "ticker")
        sectors = watchlist_service.get_items_by_type(watchlist.id, "sector")
        
        assert "AAPL" in tickers
        assert "TECH" in sectors

    def test_delete_watchlist(self, setup_metadata_store):
        """Test deleting a watchlist."""
        watchlist = watchlist_service.create_watchlist("to_delete")
        success = watchlist_service.delete_watchlist(watchlist.id)
        assert success is True


class TestAlertRules:
    """Tests for alert rules."""

    def test_create_alert_rule(self, setup_metadata_store):
        """Test creating an alert rule."""
        rule_data = {
            "name": "test_rule",
            "is_active": True,
            "sentiment_filter": "positive",
            "min_confidence_score": 0.7,
            "notification_channels": json.dumps(["in_app"]),
        }
        rule_id = metadata_store.create_alert_rule(rule_data)
        assert rule_id is not None

    def test_read_alert_rules(self, setup_metadata_store):
        """Test reading alert rules."""
        rule_data = {
            "name": f"rule_{datetime.utcnow().timestamp()}",
            "is_active": True,
            "notification_channels": json.dumps(["in_app"]),
        }
        metadata_store.create_alert_rule(rule_data)
        
        rules = metadata_store.read_alert_rules()
        assert len(rules) > 0

    def test_update_alert_rule(self, setup_metadata_store):
        """Test updating an alert rule."""
        rule_data = {
            "name": f"update_test_{datetime.utcnow().timestamp()}",
            "is_active": True,
            "notification_channels": json.dumps(["in_app"]),
        }
        rule_id = metadata_store.create_alert_rule(rule_data)
        
        metadata_store.update_alert_rule(rule_id, {"is_active": False})
        updated = metadata_store.read_alert_rule(rule_id)
        assert updated["is_active"] is False

    def test_delete_alert_rule(self, setup_metadata_store):
        """Test deleting an alert rule."""
        rule_data = {
            "name": f"delete_test_{datetime.utcnow().timestamp()}",
            "is_active": True,
            "notification_channels": json.dumps(["in_app"]),
        }
        rule_id = metadata_store.create_alert_rule(rule_data)
        
        metadata_store.delete_alert_rule(rule_id)
        deleted = metadata_store.read_alert_rule(rule_id)
        assert deleted is None


class TestAlertMatching:
    """Tests for alert matching logic."""

    def test_matches_watchlist_by_ticker(self, setup_metadata_store):
        """Test watchlist matching by ticker."""
        watchlist = watchlist_service.create_watchlist("ticker_match")
        watchlist_service.create_watchlist_item(watchlist.id, "ticker", "AAPL")
        
        opportunity = {
            "ticker": "AAPL",
            "company": "Apple",
            "overall_score": 0.8,
        }
        
        matches = alert_matcher_service._matches_watchlist(opportunity, watchlist.id)
        assert matches is True

    def test_matches_watchlist_by_sector(self, setup_metadata_store):
        """Test watchlist matching by sector."""
        watchlist = watchlist_service.create_watchlist("sector_match")
        watchlist_service.create_watchlist_item(watchlist.id, "sector", "Technology")
        
        opportunity = {
            "sector": "Technology",
            "overall_score": 0.8,
        }
        
        matches = alert_matcher_service._matches_watchlist(opportunity, watchlist.id)
        assert matches is True

    def test_rule_matching_sentiment_filter(self, setup_metadata_store):
        """Test rule matching with sentiment filter."""
        rule_data = {
            "name": f"sentiment_test_{datetime.utcnow().timestamp()}",
            "is_active": True,
            "sentiment_filter": "positive",
            "notification_channels": json.dumps(["in_app"]),
        }
        metadata_store.create_alert_rule(rule_data)
        rule = AlertRule(**rule_data)
        
        opp_positive = {"sentiment_label": "positive", "overall_score": 0.7}
        opp_negative = {"sentiment_label": "negative", "overall_score": 0.7}
        
        assert alert_matcher_service._matches_rule(opp_positive, rule) is True
        assert alert_matcher_service._matches_rule(opp_negative, rule) is False

    def test_rule_matching_confidence_threshold(self, setup_metadata_store):
        """Test rule matching with confidence threshold."""
        rule_data = {
            "name": f"confidence_test_{datetime.utcnow().timestamp()}",
            "is_active": True,
            "min_confidence_score": 0.7,
            "notification_channels": json.dumps(["in_app"]),
        }
        metadata_store.create_alert_rule(rule_data)
        rule = AlertRule(**rule_data)
        
        opp_high = {"confidence_score": 0.8, "overall_score": 0.7}
        opp_low = {"confidence_score": 0.5, "overall_score": 0.7}
        
        assert alert_matcher_service._matches_rule(opp_high, rule) is True
        assert alert_matcher_service._matches_rule(opp_low, rule) is False

    def test_rule_matching_overall_score(self, setup_metadata_store):
        """Test rule matching with overall score threshold."""
        rule_data = {
            "name": f"score_test_{datetime.utcnow().timestamp()}",
            "is_active": True,
            "min_overall_score": 0.75,
            "notification_channels": json.dumps(["in_app"]),
        }
        metadata_store.create_alert_rule(rule_data)
        rule = AlertRule(**rule_data)
        
        opp_high = {"overall_score": 0.8}
        opp_low = {"overall_score": 0.6}
        
        assert alert_matcher_service._matches_rule(opp_high, rule) is True
        assert alert_matcher_service._matches_rule(opp_low, rule) is False


class TestAlertCreation:
    """Tests for alert creation."""

    def test_create_alert(self, setup_metadata_store):
        """Test creating an alert."""
        alert_data = {
            "rule_id": 1,
            "title": "Test Alert",
            "message": "Test message",
            "severity": "info",
            "status": "new",
            "notification_channel": "in_app",
            "dedupe_key": "test_key_123",
        }
        alert_id = metadata_store.create_alert(alert_data)
        assert alert_id is not None

    def test_read_alerts(self, setup_metadata_store):
        """Test reading alerts."""
        alerts = metadata_store.read_alerts(limit=10)
        assert isinstance(alerts, list)

    def test_read_recent_alerts(self, setup_metadata_store):
        """Test reading recent alerts."""
        alerts = metadata_store.read_recent_alerts(hours=24, limit=10)
        assert isinstance(alerts, list)

    def test_update_alert_status(self, setup_metadata_store):
        """Test updating alert status."""
        alert_data = {
            "rule_id": 1,
            "title": "Status Test",
            "message": "Test",
            "severity": "info",
            "status": "new",
            "notification_channel": "in_app",
            "dedupe_key": f"key_{datetime.utcnow().timestamp()}",
        }
        alert_id = metadata_store.create_alert(alert_data)
        
        metadata_store.update_alert_status(alert_id, "seen")
        updated = metadata_store.read_alerts(limit=1000)
        updated_alert = next((a for a in updated if a["id"] == alert_id), None)
        assert updated_alert is not None
        assert updated_alert["status"] == "seen"


class TestAlertCooldown:
    """Tests for cooldown logic."""

    def test_check_alert_cooldown_suppress(self, setup_metadata_store):
        """Test that cooldown suppresses duplicate alerts."""
        rule_id = 999
        dedupe_key = f"cooldown_test_{datetime.utcnow().timestamp()}"
        
        # Create first alert
        alert_data = {
            "rule_id": rule_id,
            "title": "First",
            "message": "Test",
            "severity": "info",
            "status": "new",
            "notification_channel": "in_app",
            "dedupe_key": dedupe_key,
        }
        metadata_store.create_alert(alert_data)
        
        # Check cooldown
        on_cooldown = metadata_store.check_alert_cooldown(rule_id, dedupe_key, 60)
        assert on_cooldown is True

    def test_check_alert_cooldown_allow(self, setup_metadata_store):
        """Test that cooldown expires."""
        rule_id = 998
        dedupe_key = f"expired_cooldown_{datetime.utcnow().timestamp()}"
        
        # No alert created, so no cooldown
        on_cooldown = metadata_store.check_alert_cooldown(rule_id, dedupe_key, 60)
        assert on_cooldown is False


class TestSeverityCalculation:
    """Tests for severity calculation."""

    def test_severity_high(self):
        """Test high severity calculation."""
        opp = {
            "overall_score": 0.85,
            "confidence_score": 0.80,
            "timing_label": "early",
        }
        severity = alert_matcher_service._calculate_severity(opp)
        assert severity == "high"

    def test_severity_warning(self):
        """Test warning severity calculation."""
        opp = {
            "overall_score": 0.70,
            "confidence_score": 0.65,
            "timing_label": "moderate",
        }
        severity = alert_matcher_service._calculate_severity(opp)
        assert severity == "warning"

    def test_severity_info(self):
        """Test info severity calculation."""
        opp = {
            "overall_score": 0.55,
            "confidence_score": 0.50,
            "timing_label": "late",
        }
        severity = alert_matcher_service._calculate_severity(opp)
        assert severity == "info"


class TestNotifiers:
    """Tests for notifier functionality."""

    def test_in_app_notifier_configured(self):
        """Test in-app notifier is always configured."""
        notifier = NotifierFactory.get_notifier("in_app")
        assert notifier.is_configured() is True

    def test_in_app_notifier_send(self):
        """Test in-app notifier always succeeds."""
        payload = AlertPayload(
            rule_id=1,
            title="Test",
            message="Test message",
            severity="info",
            notification_channels=["in_app"],
            dedupe_key="test",
        )
        success, error = NotifierFactory.send_alert("in_app", payload)
        assert success is True
        assert error is None

    def test_webhook_notifier_not_configured(self):
        """Test webhook notifier handles missing config gracefully."""
        # By default, webhook should not be configured in test environment
        notifier = NotifierFactory.get_notifier("webhook")
        # Just verify it doesn't crash
        assert notifier is not None

    def test_alert_payload_model(self):
        """Test AlertPayload Pydantic model."""
        payload = AlertPayload(
            rule_id=1,
            title="Test Alert",
            message="This is a test",
            severity="high",
            notification_channels=["in_app"],
            dedupe_key="test_key",
            confidence_score=0.85,
            overall_score=0.80,
        )
        assert payload.rule_id == 1
        assert payload.severity == "high"
        assert payload.confidence_score == 0.85


class TestAlertIntegration:
    """Integration tests for alert system."""

    def test_end_to_end_alert_flow(self, setup_metadata_store):
        """Test complete alert flow from watchlist to alert creation."""
        # Create watchlist
        watchlist = watchlist_service.create_watchlist("integration_test")
        watchlist_service.create_watchlist_item(watchlist.id, "ticker", "TEST")
        
        # Create rule
        rule_data = {
            "name": f"integration_rule_{datetime.utcnow().timestamp()}",
            "is_active": True,
            "watchlist_id": watchlist.id,
            "min_overall_score": 0.6,
            "notification_channels": json.dumps(["in_app"]),
        }
        rule_id = metadata_store.create_alert_rule(rule_data)
        
        # Create matching opportunity
        opportunity = {
            "id": 999,
            "signal_id": 1,
            "ticker": "TEST",
            "company": "Test Company",
            "overall_score": 0.75,
            "confidence_score": 0.70,
            "impact_score": 0.65,
            "timing_label": "early",
            "sentiment_label": "positive",
            "event_type": "earnings_beat",
        }
        
        # Evaluate
        alert_ids = alert_matcher_service.evaluate_opportunity(opportunity)
        
        # Verify alert was created
        assert len(alert_ids) > 0
        
        # Read alert
        alerts = metadata_store.read_alerts(limit=100)
        assert len(alerts) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
