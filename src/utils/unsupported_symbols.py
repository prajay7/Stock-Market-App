from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_state() -> dict[str, Any]:
    return {"symbols": {}, "updated_at": _now_iso()}


def load_unsupported_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _default_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        symbols = payload.get("symbols", {}) if isinstance(payload, dict) else {}
        if not isinstance(symbols, dict):
            symbols = {}
        return {"symbols": symbols, "updated_at": payload.get("updated_at", _now_iso())}
    except Exception:
        return _default_state()


def save_unsupported_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbols": state.get("symbols", {}),
        "updated_at": _now_iso(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_unsupported_symbols(path: Path) -> set[str]:
    state = load_unsupported_state(path)
    unsupported: set[str] = set()
    for symbol, meta in state.get("symbols", {}).items():
        if not isinstance(meta, dict):
            continue
        if bool(meta.get("unsupported", False)):
            unsupported.add(str(symbol))
    return unsupported


def mark_failure(path: Path, symbol: str, reason: str, failure_threshold: int = 2) -> dict[str, Any]:
    with _LOCK:
        state = load_unsupported_state(path)
        symbols = state.setdefault("symbols", {})
        entry = symbols.setdefault(symbol, {})
        failure_count = int(entry.get("failure_count", 0)) + 1
        unsupported = failure_count >= max(1, int(failure_threshold))
        symbols[symbol] = {
            "failure_count": failure_count,
            "unsupported": unsupported,
            "last_failure_reason": str(reason),
            "last_failed_at": _now_iso(),
            "last_succeeded_at": entry.get("last_succeeded_at"),
        }
        save_unsupported_state(path, state)
        return symbols[symbol]


def mark_success(path: Path, symbol: str) -> dict[str, Any]:
    with _LOCK:
        state = load_unsupported_state(path)
        symbols = state.setdefault("symbols", {})
        entry = symbols.setdefault(symbol, {})
        symbols[symbol] = {
            "failure_count": 0,
            "unsupported": False,
            "last_failure_reason": "",
            "last_failed_at": entry.get("last_failed_at"),
            "last_succeeded_at": _now_iso(),
        }
        save_unsupported_state(path, state)
        return symbols[symbol]