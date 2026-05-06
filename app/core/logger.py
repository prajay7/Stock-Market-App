import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class SQLiteLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            from app.core.config import get_settings
            from src.data.db import SQLiteDataStore

            settings = get_settings()
            store = SQLiteDataStore(settings.db_path)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            extra = {
                key: value
                for key, value in record.__dict__.items()
                if key
                not in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                }
                and not key.startswith("_")
            }
            if extra:
                payload.update(extra)
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            store.write_app_log(payload)
        except Exception:
            return


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in reserved and not key.startswith("_")
        }
        if extra:
            for key, value in extra.items():
                try:
                    json.dumps(value)
                    payload[key] = value
                except Exception:
                    payload[key] = str(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    try:
        from app.core.config import get_settings

        settings = get_settings()
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(settings.log_dir) / "app.log")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
        root.addHandler(SQLiteLogHandler())
    except Exception:
        return
