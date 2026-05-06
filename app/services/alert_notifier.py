"""
Pluggable notifier interface for sending alerts through various channels.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

import requests

from app.core.config import get_settings
from app.news.models import AlertPayload

logger = logging.getLogger(__name__)


class BaseNotifier(ABC):
    """Base class for all notifiers."""

    @abstractmethod
    def send(self, payload: AlertPayload) -> tuple[bool, Optional[str]]:
        """
        Send an alert.
        Returns: (success, error_message)
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if notifier is properly configured."""
        pass


class InAppNotifier(BaseNotifier):
    """In-app notification (always stored in DB, always available)."""

    def __init__(self):
        self.settings = get_settings()

    def send(self, payload: AlertPayload) -> tuple[bool, Optional[str]]:
        """In-app alerts are always successful (stored in DB by alert service)."""
        return True, None

    def is_configured(self) -> bool:
        return True


class WebhookNotifier(BaseNotifier):
    """Send alerts via HTTP webhook."""

    def __init__(self):
        self.settings = get_settings()

    def send(self, payload: AlertPayload) -> tuple[bool, Optional[str]]:
        if not self.is_configured():
            return False, "Webhook URL not configured"

        try:
            response = requests.post(
                self.settings.webhook_url,
                json=payload.model_dump(mode="json"),
                timeout=self.settings.webhook_timeout_sec,
            )
            response.raise_for_status()
            return True, None
        except requests.exceptions.Timeout:
            return False, "Webhook request timeout"
        except requests.exceptions.ConnectionError as e:
            return False, f"Webhook connection failed: {str(e)}"
        except requests.exceptions.HTTPError as e:
            return False, f"Webhook HTTP error: {e.response.status_code}"
        except Exception as e:
            return False, f"Webhook error: {str(e)}"

    def is_configured(self) -> bool:
        return bool(self.settings.webhook_url and self.settings.webhook_url.strip())


class EmailNotifier(BaseNotifier):
    """Send alerts via email (optional, requires SMTP config)."""

    def __init__(self):
        self.settings = get_settings()

    def send(self, payload: AlertPayload) -> tuple[bool, Optional[str]]:
        if not self.is_configured():
            return False, "Email not configured"

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart("alternative")
            msg["Subject"] = payload.title
            msg["From"] = self.settings.alert_email_from
            msg["To"] = self.settings.alert_email_to

            text = payload.message
            html = f"""
            <html>
              <body>
                <h3>{payload.title}</h3>
                <p>{payload.message}</p>
                <p><strong>Severity:</strong> {payload.severity}</p>
                <p><strong>Confidence:</strong> {payload.confidence_score}</p>
              </body>
            </html>
            """

            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port) as server:
                server.starttls()
                server.login(self.settings.smtp_username, self.settings.smtp_password)
                server.sendmail(self.settings.alert_email_from, self.settings.alert_email_to, msg.as_string())
            return True, None
        except Exception as e:
            return False, f"Email send failed: {str(e)}"

    def is_configured(self) -> bool:
        return (
            self.settings.smtp_enabled
            and bool(self.settings.smtp_host)
            and bool(self.settings.alert_email_from)
            and bool(self.settings.alert_email_to)
        )


class TelegramNotifier(BaseNotifier):
    """Send alerts via Telegram (optional)."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.telegram.org"

    def send(self, payload: AlertPayload) -> tuple[bool, Optional[str]]:
        if not self.is_configured():
            return False, "Telegram not configured"

        try:
            message = f"""
🚨 <b>{payload.title}</b>

{payload.message}

<b>Severity:</b> {payload.severity}
<b>Confidence:</b> {payload.confidence_score:.1%}
<b>Overall Score:</b> {payload.overall_score:.1%}
"""
            url = f"{self.base_url}/bot{self.settings.telegram_bot_token}/sendMessage"
            response = requests.post(
                url,
                json={"chat_id": self.settings.telegram_chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10.0,
            )
            response.raise_for_status()
            return True, None
        except Exception as e:
            return False, f"Telegram send failed: {str(e)}"

    def is_configured(self) -> bool:
        return bool(self.settings.telegram_bot_token) and bool(self.settings.telegram_chat_id)


class SlackNotifier(BaseNotifier):
    """Send alerts via Slack (optional)."""

    def __init__(self):
        self.settings = get_settings()

    def send(self, payload: AlertPayload) -> tuple[bool, Optional[str]]:
        if not self.is_configured():
            return False, "Slack webhook URL not configured"

        try:
            severity_emoji = {"high": "🔴", "warning": "🟡", "info": "🔵"}.get(payload.severity, "ℹ️")
            color = {"high": "FF0000", "warning": "FFA500", "info": "0099FF"}.get(payload.severity, "808080")

            payload_dict = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{severity_emoji} {payload.title}",
                        "text": payload.message,
                        "fields": [
                            {"title": "Severity", "value": payload.severity, "short": True},
                            {"title": "Confidence", "value": f"{payload.confidence_score:.1%}", "short": True},
                            {"title": "Overall Score", "value": f"{payload.overall_score:.1%}", "short": True},
                            {"title": "Channel", "value": payload.notification_channels[0] if payload.notification_channels else "N/A", "short": True},
                        ],
                    }
                ]
            }
            response = requests.post(
                self.settings.slack_webhook_url,
                json=payload_dict,
                timeout=10.0,
            )
            response.raise_for_status()
            return True, None
        except Exception as e:
            return False, f"Slack send failed: {str(e)}"

    def is_configured(self) -> bool:
        return bool(self.settings.slack_webhook_url)


class NotifierFactory:
    """Factory for creating notifiers."""

    _notifiers: dict[str, BaseNotifier] = {}

    @classmethod
    def get_notifier(cls, channel: str) -> BaseNotifier:
        """Get or create a notifier for the specified channel."""
        if channel not in cls._notifiers:
            if channel == "in_app":
                cls._notifiers[channel] = InAppNotifier()
            elif channel == "webhook":
                cls._notifiers[channel] = WebhookNotifier()
            elif channel == "email":
                cls._notifiers[channel] = EmailNotifier()
            elif channel == "telegram":
                cls._notifiers[channel] = TelegramNotifier()
            elif channel == "slack":
                cls._notifiers[channel] = SlackNotifier()
            else:
                logger.warning(f"Unknown notifier channel: {channel}")
                return InAppNotifier()  # fallback
        return cls._notifiers[channel]

    @classmethod
    def send_alert(cls, channel: str, payload: AlertPayload) -> tuple[bool, Optional[str]]:
        """Send an alert via the specified channel."""
        try:
            logger.info(
                "alert_notification_send_started",
                extra={"channel": str(channel), "rule_id": payload.rule_id, "opportunity_id": payload.opportunity_id},
            )
            notifier = cls.get_notifier(channel)
            if not notifier.is_configured():
                logger.warning("alert_notification_send_skipped", extra={"channel": str(channel), "reason": "not_configured"})
                return False, f"{channel} not configured"
            success, error = notifier.send(payload)
            if success:
                logger.info(
                    "alert_notification_send_completed",
                    extra={"channel": str(channel), "rule_id": payload.rule_id, "opportunity_id": payload.opportunity_id},
                )
            else:
                logger.warning(
                    "alert_notification_send_failed",
                    extra={"channel": str(channel), "rule_id": payload.rule_id, "opportunity_id": payload.opportunity_id, "error": str(error)},
                )
            return success, error
        except Exception as e:
            logger.exception("alert_notification_send_exception", extra={"channel": str(channel), "error": str(e)})
            return False, str(e)
