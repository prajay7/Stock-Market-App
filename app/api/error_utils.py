from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def build_error_envelope(
    code: str,
    message: str,
    details: dict | None = None,
    trace_id: str | None = None,
) -> dict:
    return {
        "error": {
            "code": str(code),
            "message": str(message),
            "details": details or {},
            "trace_id": trace_id or str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }
