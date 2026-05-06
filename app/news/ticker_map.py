from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def normalize_company_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


@dataclass
class CompanyTickerMap:
    path: Path

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        # Support both formats:
        # 1) {"Company": {"ticker": "XYZ", "aliases": [...]}}
        # 2) {"companies": [{"company": "Company", "ticker": "XYZ", "aliases": [...]}]}
        if "companies" in payload and isinstance(payload.get("companies"), list):
            out: dict[str, dict[str, Any]] = {}
            for item in payload.get("companies", []):
                if not isinstance(item, dict):
                    continue
                company = str(item.get("company") or "").strip()
                if not company:
                    continue
                out[company] = {
                    "ticker": str(item.get("ticker") or "").strip() or None,
                    "aliases": [str(a) for a in (item.get("aliases") or []) if str(a).strip()],
                }
            return out

        normalized: dict[str, dict[str, Any]] = {}
        for company, entry in payload.items():
            if not isinstance(company, str) or not isinstance(entry, dict):
                continue
            normalized[company] = {
                "ticker": str(entry.get("ticker") or "").strip() or None,
                "aliases": [str(a) for a in (entry.get("aliases") or []) if str(a).strip()],
            }
        return normalized

    def alias_index(self) -> dict[str, dict[str, str | None]]:
        index: dict[str, dict[str, str | None]] = {}
        for company, entry in self.load().items():
            ticker = str(entry.get("ticker") or "").strip() or None
            aliases = [company]
            aliases.extend(entry.get("aliases") or [])
            if ticker:
                aliases.append(ticker)
                aliases.append(ticker.split(".")[0])
            for alias in aliases:
                key = normalize_company_name(alias)
                if not key:
                    continue
                index[key] = {"company": company, "ticker": ticker}
        return index

    def resolve_company(self, name: str | None) -> str | None:
        value = str(name or "").strip()
        if not value:
            return None
        item = self.alias_index().get(normalize_company_name(value))
        if item:
            return str(item.get("company") or "").strip() or None
        return None

    def resolve_ticker(self, name: str | None) -> str | None:
        value = str(name or "").strip()
        if not value:
            return None
        item = self.alias_index().get(normalize_company_name(value))
        if item:
            ticker = str(item.get("ticker") or "").strip()
            if ticker:
                return ticker
        if re.match(r"^[A-Za-z0-9_.-]{1,20}$", value):
            return value.upper()
        return None


def resolve_ticker(name: str, map_path: Path) -> str | None:
    return CompanyTickerMap(map_path).resolve_ticker(name)
