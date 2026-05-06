from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.news.models import BeneficiaryCompany
from app.news.ticker_map import CompanyTickerMap, normalize_company_name


def _normalize_text(value: str) -> str:
    return normalize_company_name(value)


@dataclass
class CompanyRelations:
    path: Path
    ticker_alias_path: Path | None = None

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for company, value in payload.items():
            if not isinstance(company, str):
                continue
            entry: dict[str, Any]
            if isinstance(value, list):
                entry = {"relations": value}
            elif isinstance(value, dict):
                entry = dict(value)
                if "relations" not in entry and isinstance(entry.get("related"), list):
                    entry["relations"] = entry["related"]
            else:
                continue
            entry.setdefault("aliases", [])
            entry.setdefault("relations", [])
            normalized[company] = entry
        return normalized

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_ticker_aliases(self) -> dict[str, Any]:
        if self.ticker_alias_path is None or not self.ticker_alias_path.exists():
            return {}
        try:
            payload = json.loads(self.ticker_alias_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _alias_index(self) -> dict[str, dict[str, str | None]]:
        index: dict[str, dict[str, str | None]] = {}

        ticker_map = CompanyTickerMap(self.ticker_alias_path) if self.ticker_alias_path else None
        if ticker_map:
            index.update(ticker_map.alias_index())

        graph = self.load()
        for company, entry in graph.items():
            ticker = str(entry.get("ticker") or "").strip() or None
            if not ticker and ticker_map:
                ticker = ticker_map.resolve_ticker(company)
            aliases = self.aliases_for_company(company, entry)
            for alias in aliases:
                index[alias] = {
                    "company": company,
                    "ticker": ticker,
                }
        return index

    @staticmethod
    def aliases_for_company(company: str, entry: dict[str, Any] | None = None) -> list[str]:
        aliases = {_normalize_text(company)}
        if entry:
            ticker = str(entry.get("ticker") or "").strip()
            if ticker:
                aliases.add(_normalize_text(ticker))
                aliases.add(_normalize_text(ticker.split(".")[0]))
            extra_aliases = entry.get("aliases") or []
            if isinstance(extra_aliases, list):
                for alias in extra_aliases:
                    aliases.add(_normalize_text(alias))
        return [alias for alias in aliases if alias]

    @staticmethod
    def _text_contains_company(text: str, company: str, entry: dict[str, Any] | None = None) -> bool:
        normalized_text = _normalize_text(text)
        for alias in CompanyRelations.aliases_for_company(company, entry):
            if alias and alias in normalized_text:
                return True
        return False

    def detect_primary_company(self, text: str) -> str | None:
        normalized_text = _normalize_text(text)
        if not normalized_text:
            return None

        candidates: list[tuple[int, str]] = []
        alias_index = self._alias_index()
        if not alias_index:
            return None
        for alias, resolved in alias_index.items():
            if alias and alias in normalized_text:
                company = str(resolved.get("company") or "").strip()
                if company:
                    candidates.append((len(alias), company))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][1]

    def resolve_company(self, company_or_alias: str | None) -> str | None:
        value = str(company_or_alias or "").strip()
        if not value:
            return None
        alias = _normalize_text(value)
        alias_index = self._alias_index()
        resolved = alias_index.get(alias)
        if resolved:
            company = str(resolved.get("company") or "").strip()
            return company or None
        graph = self.load()
        if value in graph:
            return value
        return None

    def resolve_ticker(self, company_or_alias: str | None) -> str | None:
        value = str(company_or_alias or "").strip()
        if not value:
            return None

        ticker_map = CompanyTickerMap(self.ticker_alias_path) if self.ticker_alias_path else None
        if ticker_map:
            mapped = ticker_map.resolve_ticker(value)
            if mapped:
                return mapped

        alias = _normalize_text(value)
        alias_index = self._alias_index()
        resolved = alias_index.get(alias)
        if resolved:
            ticker = str(resolved.get("ticker") or "").strip()
            if ticker:
                return ticker

        company = self.resolve_company(value)
        if company:
            ticker = str(self.company_metadata(company).get("ticker") or "").strip()
            if ticker:
                return ticker

        # If user already passed a ticker-like string, use it directly.
        if ticker_map:
            mapped = ticker_map.resolve_ticker(value)
            if mapped:
                return mapped
        return None

    def company_metadata(self, company: str) -> dict[str, Any]:
        return self.load().get(company, {})

    def related_companies(self, company: str) -> list[dict[str, Any]]:
        canonical_company = self.resolve_company(company) or company
        entry = self.company_metadata(canonical_company)
        relations = entry.get("relations") or []
        out: list[dict[str, Any]] = []
        for item in relations:
            if not isinstance(item, dict):
                continue
            related_company = str(item.get("company") or "").strip()
            if not related_company:
                continue
            relation_ticker = str(item.get("ticker") or "").strip() or None
            if not relation_ticker:
                relation_ticker = self.resolve_ticker(related_company)
            out.append(
                {
                    "company": related_company,
                    "relation": str(item.get("relation") or "related"),
                    "strength": float(item.get("strength") or 0.0),
                    "ticker": relation_ticker,
                }
            )
        return out

    @staticmethod
    def merge_beneficiary_suggestions(*groups: list[BeneficiaryCompany]) -> list[BeneficiaryCompany]:
        merged: dict[str, BeneficiaryCompany] = {}
        for group in groups:
            for item in group:
                key = item.company.lower().strip()
                existing = merged.get(key)
                if existing is None or item.benefit_score > existing.benefit_score:
                    merged[key] = item
        return sorted(merged.values(), key=lambda x: x.benefit_score, reverse=True)
