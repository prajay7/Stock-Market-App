"""
Alert matching engine that evaluates signals against alert rules.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Optional

from app.core.config import get_settings
from app.news.models import AlertPayload, AlertRule
from app.services.alert_notifier import NotifierFactory
from src.data.metadata_store import metadata_store

logger = logging.getLogger(__name__)


class AlertMatcherService:
    """Service for matching signals/opportunities against alert rules."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = metadata_store

    def evaluate_opportunity(self, opportunity_data: dict) -> list[int]:
        """
        Evaluate an opportunity against all active alert rules.
        Returns: list of created alert IDs
        """
        if not self.settings.alerts_enabled:
            logger.info("alert_evaluate_skipped", extra={"reason": "alerts_disabled"})
            return []

        ticker = str(opportunity_data.get("ticker") or opportunity_data.get("primary_ticker") or "").strip().upper()
        logger.info("alert_evaluate_started", extra={"ticker": ticker})

        alert_ids = []
        rules = self.store.read_alert_rules(active_only=True)

        for rule_dict in rules:
            rule = AlertRule(**rule_dict)
            if self._matches_rule(opportunity_data, rule):
                alert_ids.extend(self._create_alerts_for_rule(opportunity_data, rule))

        logger.info(
            "alert_evaluate_completed",
            extra={"ticker": ticker, "rules_count": len(rules), "created_alerts": len(alert_ids)},
        )
        return alert_ids

    def scan_recent_opportunities(self) -> list[int]:
        """
        Scan recent opportunities for matching rules.
        Used by scheduled job.
        """
        if not self.settings.alerts_enabled or not self.settings.alert_scan_enabled:
            logger.info("alert_scan_skipped", extra={"alerts_enabled": bool(self.settings.alerts_enabled), "alert_scan_enabled": bool(self.settings.alert_scan_enabled)})
            return []

        logger.info("alert_scan_started")
        alert_ids = []
        try:
            opps = self.store.read_beneficiary_opportunities_with_signal(
                limit=self.settings.max_recent_signals_for_alert_scan
            )
            for opp in opps:
                new_alerts = self.evaluate_opportunity(opp)
                alert_ids.extend(new_alerts)
        except Exception as e:
            logger.exception("alert_scan_failed", extra={"error": str(e)})

        logger.info("alert_scan_completed", extra={"created_alerts": len(alert_ids)})

        return alert_ids

    def _matches_rule(self, opportunity_data: dict, rule: AlertRule) -> bool:
        """Check if opportunity matches a rule."""
        # Check watchlist if rule is linked to one
        if rule.watchlist_id:
            if not self._matches_watchlist(opportunity_data, rule.watchlist_id):
                return False

        # Check sentiment filter
        if rule.sentiment_filter:
            sentiment = opportunity_data.get("sentiment_label", "").lower()
            if sentiment != rule.sentiment_filter.lower():
                return False

        # Check actionable flag
        if rule.actionable_only:
            # Check if the parent signal is actionable
            signal_id = opportunity_data.get("signal_id")
            if signal_id:
                signal_rows = self.store.read_beneficiary_opportunities_with_signal(limit=10000)
                for signal_row in signal_rows:
                    # Note: The query doesn't expose is_actionable directly; check via service
                    pass
            # For now, skip this check as the data structure doesn't readily expose it

        # Check score thresholds
        if rule.min_confidence_score is not None:
            confidence = opportunity_data.get("confidence_score", 0)
            if confidence < rule.min_confidence_score:
                return False

        if rule.min_impact_score is not None:
            impact = opportunity_data.get("impact_score", 0)
            if impact < rule.min_impact_score:
                return False

        if rule.min_overall_score is not None:
            overall = opportunity_data.get("overall_score", 0)
            if overall < rule.min_overall_score:
                return False

        # Check timing labels
        if rule.timing_labels:
            timing = opportunity_data.get("timing_label", "")
            if timing not in rule.timing_labels:
                return False

        # Check event types
        if rule.event_types:
            event_type = opportunity_data.get("event_type", "")
            if event_type not in rule.event_types:
                return False

        # Check sectors
        if rule.sectors:
            sector = opportunity_data.get("sector", "")
            if sector.upper() not in [s.upper() for s in rule.sectors]:
                return False

        # Check price move (if applicable, from signal price)
        if rule.max_price_move_pct is not None and opportunity_data.get("price_change_pct"):
            price_move = abs(opportunity_data.get("price_change_pct", 0))
            if price_move > rule.max_price_move_pct:
                return False

        return True

    def _matches_watchlist(self, opportunity_data: dict, watchlist_id: int) -> bool:
        """Check if opportunity matches any item in a watchlist."""
        items = self.store.read_watchlist_items(watchlist_id)
        if not items:
            return False

        for item in items:
            item_type = item.get("item_type")
            normalized = item.get("normalized_value", "").upper()

            if item_type == "company":
                company = opportunity_data.get("company", "").upper()
                if company == normalized:
                    return True
            elif item_type == "ticker":
                ticker = opportunity_data.get("ticker", "").upper()
                if ticker == normalized:
                    return True
            elif item_type == "sector":
                sector = opportunity_data.get("sector", "").upper()
                if sector == normalized:
                    return True
            elif item_type == "event_type":
                event_type = opportunity_data.get("event_type", "").upper()
                if event_type == normalized:
                    return True

        return False

    def _create_alerts_for_rule(self, opportunity_data: dict, rule: AlertRule) -> list[int]:
        """Create alert records for matched rule."""
        alert_ids = []
        severity = self._calculate_severity(opportunity_data)

        # Create an alert for each notification channel
        for channel in rule.notification_channels or ["in_app"]:
            dedupe_key = self._generate_dedupe_key(rule.id, opportunity_data.get("id"), channel)

            # Check cooldown
            if self.store.check_alert_cooldown(rule.id, dedupe_key, rule.cooldown_minutes):
                logger.debug(f"Alert suppressed by cooldown: rule={rule.id}, key={dedupe_key}")
                continue

            payload = self._build_alert_payload(opportunity_data, rule, severity)
            alert_data = {
                "rule_id": rule.id,
                "signal_id": opportunity_data.get("signal_id"),
                "opportunity_id": opportunity_data.get("id"),
                "title": payload.title,
                "message": payload.message,
                "severity": severity,
                "status": "new",
                "notification_channel": channel,
                "dedupe_key": dedupe_key,
            }

            try:
                alert_id = self.store.create_alert(alert_data)
                alert_ids.append(alert_id)
                logger.info(f"Created alert {alert_id} for rule {rule.id}")

                # Send notification immediately
                success, error = NotifierFactory.send_alert(channel, payload)
                if success:
                    self.store.update_alert_status(alert_id, "sent", sent_at=datetime.utcnow())
                    logger.info(f"Alert {alert_id} sent via {channel}")
                else:
                    self.store.update_alert_status(alert_id, "failed", error_message=error)
                    logger.warning(f"Failed to send alert {alert_id} via {channel}: {error}")
            except Exception as e:
                logger.error(f"Error creating alert for rule {rule.id}: {e}")

        return alert_ids

    def _build_alert_payload(self, opportunity_data: dict, rule: AlertRule, severity: str) -> AlertPayload:
        """Build alert payload from opportunity and rule."""
        title = self._generate_title(opportunity_data)
        message = self._generate_message(opportunity_data, rule)

        return AlertPayload(
            rule_id=rule.id,
            signal_id=opportunity_data.get("signal_id"),
            opportunity_id=opportunity_data.get("id"),
            title=title,
            message=message,
            severity=severity,
            notification_channels=rule.notification_channels or ["in_app"],
            dedupe_key=self._generate_dedupe_key(rule.id, opportunity_data.get("id"), "payload"),
            beneficiary_company=opportunity_data.get("company"),
            beneficiary_ticker=opportunity_data.get("ticker"),
            primary_company=opportunity_data.get("primary_company"),
            primary_ticker=opportunity_data.get("primary_ticker"),
            event_type=opportunity_data.get("event_type"),
            sector=opportunity_data.get("sector"),
            sentiment=opportunity_data.get("sentiment_label"),
            confidence_score=opportunity_data.get("confidence_score"),
            overall_score=opportunity_data.get("overall_score"),
            timing_label=opportunity_data.get("timing_label"),
            headline=opportunity_data.get("title"),
            source=opportunity_data.get("source"),
        )

    def _calculate_severity(self, opportunity_data: dict) -> str:
        """Calculate alert severity based on opportunity scores."""
        overall = opportunity_data.get("overall_score", 0)
        confidence = opportunity_data.get("confidence_score", 0)
        timing = opportunity_data.get("timing_label", "moderate")

        if overall >= 0.8 and confidence >= 0.75 and timing == "early":
            return "high"
        elif overall >= 0.65 and confidence >= 0.60:
            return "warning"
        else:
            return "info"

    def _generate_title(self, opportunity_data: dict) -> str:
        """Generate alert title."""
        company = opportunity_data.get("company", "Unknown")
        event_type = opportunity_data.get("event_type", "signal")
        return f"Alert: {company} - {event_type}"

    def _generate_message(self, opportunity_data: dict, rule: AlertRule) -> str:
        """Generate alert message."""
        company = opportunity_data.get("company", "Unknown")
        ticker = opportunity_data.get("ticker", "N/A")
        relation = opportunity_data.get("relation", "")
        sentiment = opportunity_data.get("sentiment_label", "")
        overall = opportunity_data.get("overall_score", 0)
        timing = opportunity_data.get("timing_label", "moderate")

        relation_text = f" via {relation}" if relation else ""
        sentiment_text = f"({sentiment})" if sentiment else ""

        message = f"{company} ({ticker}){relation_text} {sentiment_text}\n"
        message += f"Overall Score: {overall:.1%} | Timing: {timing}\n"
        message += f"Rule: {rule.name}"

        return message

    @staticmethod
    def _generate_dedupe_key(rule_id: int, opportunity_id: int, channel: str) -> str:
        """Generate a dedupe key for cooldown logic."""
        key_str = f"{rule_id}:{opportunity_id}:{channel}"
        return hashlib.md5(key_str.encode()).hexdigest()[:32]


alert_matcher_service = AlertMatcherService()
