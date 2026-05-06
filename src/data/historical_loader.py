from __future__ import annotations

import logging
import re
import time
from io import StringIO
from dataclasses import dataclass, field
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import httpx
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from src.data.db import SQLiteDataStore
from src.data.cache import symbol_price_path
from src.data.storage import append_dedup_by_keys, read_parquet_if_exists, write_parquet
from src.data.source_manager import SourceManager
from src.utils.symbols import normalize_symbols
from src.utils.validation import validate_ohlcv_frame

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.ERROR)


@dataclass
class HistoricalLoader:
    raw_data_dir: Path
    alpha_vantage_api_key: str = ""
    polygon_api_key: str = ""
    stooq_api_key: str = ""
    data_provider: str = "alphavantage"
    db_path: Path | None = None
    last_diagnostics: dict[str, str] = field(default_factory=dict, init=False)
    last_source_by_symbol: dict[str, str] = field(default_factory=dict, init=False)
    last_source_counts: dict[str, int] = field(default_factory=dict, init=False)
    store: SQLiteDataStore = field(init=False)
    source_manager: SourceManager = field(init=False)

    def __post_init__(self) -> None:
        settings = get_settings()
        db_path = self.db_path or settings.db_path
        self.store = SQLiteDataStore(db_path)
        self.source_manager = SourceManager(self.store)

    def _stooq_url(self, ticker: str, masked: bool = False) -> str:
        base = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
        if not self.stooq_api_key:
            return base
        if masked:
            return f"{base}&apikey=***"
        return f"{base}&apikey={self.stooq_api_key}"

    def _snapshot_dir(self) -> Path:
        return self.raw_data_dir.parent / "outputs" / "debug" / "ingest_failures"

    def _save_failure_snapshot(
        self,
        symbol: str,
        interval: str,
        lookback_days: int,
        diagnostics: list[str],
        debug_records: list[dict[str, str]],
    ) -> Path:
        out_dir = self._snapshot_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
        safe_symbol = str(symbol).replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{safe_symbol}_{ts}.json"
        payload = {
            "symbol": symbol,
            "interval": interval,
            "lookback_days": int(lookback_days),
            "saved_at": pd.Timestamp.utcnow().isoformat(),
            "diagnostics": diagnostics,
            "attempts": debug_records,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    @staticmethod
    def _stooq_requires_apikey(debug_records: list[dict[str, str]]) -> bool:
        for record in debug_records:
            if str(record.get("source", "")).lower() != "stooq":
                continue
            preview = str(record.get("preview", "")).lower()
            if "get_apikey" in preview or "append the <apikey>" in preview:
                return True
        return False

    @retry(wait=wait_exponential(min=1, max=16), stop=stop_after_attempt(3), reraise=True)
    def _download(self, symbol: str, start: date, interval: str) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.isoformat(), interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        df = df.reset_index().rename(
            columns={
                "Date": "date",
                "Datetime": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["symbol"] = symbol
        return df.sort_values("date").reset_index(drop=True)

    def _stooq_symbol_candidates(self, symbol: str) -> list[str]:
        cleaned = str(symbol).strip().upper()
        base = cleaned.split(".")[0].lower()
        suffix = cleaned.split(".")[-1] if "." in cleaned else ""

        out: list[str] = []
        seen: set[str] = set()

        def _add(v: str) -> None:
            if v and v not in seen:
                out.append(v)
                seen.add(v)

        _add(cleaned.lower())
        _add(base)

        # Indian symbols are commonly represented as .in on Stooq.
        if suffix in {"NS", "NSE", "BO", "BSE"} or cleaned.endswith(".NS") or cleaned.endswith(".BO"):
            _add(f"{base}.in")

        # Keep .us as broad fallback for US/default symbols.
        _add(f"{base}.us")
        return out

    def _download_stooq_daily(self, symbol: str, start: date) -> pd.DataFrame:
        # Fallback for provider throttling. Try multiple Stooq ticker namespaces.
        with httpx.Client(timeout=20.0) as client:
            for ticker in self._stooq_symbol_candidates(symbol):
                url = self._stooq_url(ticker, masked=False)
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                except Exception:
                    continue

                csv_text = resp.text.strip()
                if not csv_text or "Date,Open,High,Low,Close,Volume" not in csv_text:
                    continue

                raw = pd.read_csv(StringIO(csv_text))
                raw = raw.rename(
                    columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
                raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
                raw = raw[raw["date"].notna()]
                raw = raw[raw["date"] >= pd.Timestamp(start)]
                if raw.empty:
                    continue
                raw["symbol"] = symbol
                return raw[["date", "open", "high", "low", "close", "volume", "symbol"]].sort_values("date").reset_index(drop=True)

        return pd.DataFrame()

    def _download_stooq_daily_with_debug(self, symbol: str, start: date) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        debug: list[dict[str, str]] = []
        with httpx.Client(timeout=20.0) as client:
            for ticker in self._stooq_symbol_candidates(symbol):
                url = self._stooq_url(ticker, masked=False)
                safe_url = self._stooq_url(ticker, masked=True)
                try:
                    resp = client.get(url)
                    status = str(resp.status_code)
                    preview = (resp.text or "")[:1200]
                    debug.append({"source": "stooq", "ticker": ticker, "url": safe_url, "status": status, "preview": preview})
                    resp.raise_for_status()
                except Exception as exc:
                    debug.append({"source": "stooq", "ticker": ticker, "url": safe_url, "status": "error", "preview": str(exc)[:300]})
                    continue

                csv_text = resp.text.strip()
                if not csv_text or "Date,Open,High,Low,Close,Volume" not in csv_text:
                    continue

                raw = pd.read_csv(StringIO(csv_text))
                raw = raw.rename(
                    columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
                raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
                raw = raw[raw["date"].notna()]
                raw = raw[raw["date"] >= pd.Timestamp(start)]
                if raw.empty:
                    continue
                raw["symbol"] = symbol
                return raw[["date", "open", "high", "low", "close", "volume", "symbol"]].sort_values("date").reset_index(drop=True), debug

        return pd.DataFrame(), debug

    def _download_nse_daily_with_debug(self, symbol: str, start: date) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        debug: list[dict[str, str]] = []
        base_symbol = str(symbol).upper().split(".")[0]
        symbol_candidates = [base_symbol, str(symbol).upper()]
        from_str = pd.Timestamp(start).strftime("%d-%m-%Y")
        to_str = pd.Timestamp(date.today()).strftime("%d-%m-%Y")
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        }

        with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
            try:
                client.get("https://www.nseindia.com/")
            except Exception as exc:
                debug.append({"source": "nse", "ticker": base_symbol, "url": "https://www.nseindia.com/", "status": "error", "preview": str(exc)[:300]})

            for candidate in symbol_candidates:
                for series in ["EQ", "BE", "SM"]:
                    url = "https://www.nseindia.com/api/historical/cm/equity"
                    params = {
                        "symbol": candidate,
                        "series": f'["{series}"]',
                        "from": from_str,
                        "to": to_str,
                    }
                    try:
                        resp = client.get(url, params=params)
                        status = str(resp.status_code)
                        text_preview = (resp.text or "")[:1200]
                        debug.append(
                            {
                                "source": "nse",
                                "ticker": candidate,
                                "series": series,
                                "url": str(resp.url),
                                "status": status,
                                "preview": text_preview,
                            }
                        )
                        resp.raise_for_status()
                        payload = resp.json()
                    except Exception as exc:
                        debug.append(
                            {
                                "source": "nse",
                                "ticker": candidate,
                                "series": series,
                                "url": url,
                                "status": "error",
                                "preview": str(exc)[:300],
                            }
                        )
                        continue

                    data = payload.get("data") or []
                    if not data:
                        continue

                    rows = []
                    for item in data:
                        dt = pd.to_datetime(item.get("CH_TIMESTAMP"), errors="coerce")
                        if pd.isna(dt) or dt < pd.Timestamp(start):
                            continue
                        rows.append(
                            {
                                "date": dt,
                                "open": pd.to_numeric(item.get("CH_OPENING_PRICE"), errors="coerce"),
                                "high": pd.to_numeric(item.get("CH_TRADE_HIGH_PRICE"), errors="coerce"),
                                "low": pd.to_numeric(item.get("CH_TRADE_LOW_PRICE"), errors="coerce"),
                                "close": pd.to_numeric(item.get("CH_CLOSING_PRICE"), errors="coerce"),
                                "volume": pd.to_numeric(item.get("CH_TOT_TRADED_QTY"), errors="coerce"),
                                "symbol": symbol,
                            }
                        )

                    if not rows:
                        continue

                    out = pd.DataFrame(rows).dropna(subset=["date", "open", "high", "low", "close", "volume"])
                    if out.empty:
                        continue
                    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
                    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
                    return out, debug

        return pd.DataFrame(), debug

    def _download_alpha_vantage_daily(self, symbol: str, start: date) -> pd.DataFrame:
        if not self.alpha_vantage_api_key:
            return pd.DataFrame()

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": self.alpha_vantage_api_key,
        }
        time.sleep(1.1)
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        series = payload.get("Time Series (Daily)")
        if not series:
            return pd.DataFrame()

        records = []
        for ds, values in series.items():
            dt = pd.to_datetime(ds, errors="coerce")
            if pd.isna(dt) or dt < pd.Timestamp(start):
                continue
            records.append(
                {
                    "date": dt,
                    "open": pd.to_numeric(values.get("1. open"), errors="coerce"),
                    "high": pd.to_numeric(values.get("2. high"), errors="coerce"),
                    "low": pd.to_numeric(values.get("3. low"), errors="coerce"),
                    "close": pd.to_numeric(values.get("4. close"), errors="coerce"),
                    "volume": pd.to_numeric(values.get("5. volume"), errors="coerce"),
                    "symbol": symbol,
                }
            )

        if not records:
            return pd.DataFrame()

        out = pd.DataFrame(records).dropna(subset=["date", "open", "high", "low", "close", "volume"])
        return out.sort_values("date").reset_index(drop=True)

    def _download_polygon_daily(self, symbol: str, start: date) -> pd.DataFrame:
        if not self.polygon_api_key:
            return pd.DataFrame()

        from_date = pd.Timestamp(start).strftime("%Y-%m-%d")
        to_date = pd.Timestamp(date.today()).strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
            "apiKey": self.polygon_api_key,
        }

        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        results = payload.get("results") or []
        if not results:
            return pd.DataFrame()

        records = []
        for item in results:
            ts = item.get("t")
            dt = pd.to_datetime(ts, unit="ms", errors="coerce") if ts is not None else pd.NaT
            if pd.isna(dt) or dt < pd.Timestamp(start):
                continue
            records.append(
                {
                    "date": dt,
                    "open": pd.to_numeric(item.get("o"), errors="coerce"),
                    "high": pd.to_numeric(item.get("h"), errors="coerce"),
                    "low": pd.to_numeric(item.get("l"), errors="coerce"),
                    "close": pd.to_numeric(item.get("c"), errors="coerce"),
                    "volume": pd.to_numeric(item.get("v"), errors="coerce"),
                    "symbol": symbol,
                }
            )

        if not records:
            return pd.DataFrame()

        out = pd.DataFrame(records).dropna(subset=["date", "open", "high", "low", "close", "volume"])
        return out.sort_values("date").reset_index(drop=True)

    def _alpha_symbol_candidates(self, symbol: str) -> list[str]:
        base = symbol.upper().split(".")[0]
        if str(symbol).upper().endswith((".NS", ".BO", ".NSE", ".BSE")):
            candidates = [symbol.upper(), base]
        else:
            candidates = [f"{base}.NS", f"{base}.BO", f"{base}.NSE", f"{base}.BSE", symbol.upper(), base]
        seen = set()
        out = []
        for c in candidates:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def _internet_symbol_candidates(self, symbol: str) -> list[str]:
        base = symbol.upper().split(".")[0]
        if str(symbol).upper().endswith((".NS", ".BO", ".NSE", ".BSE")):
            candidates = [symbol.upper(), base]
        else:
            candidates = [f"{base}.NS", f"{base}.BO", f"{base}.NSE", f"{base}.BSE", symbol.upper(), base]
        seen = set()
        out = []
        for c in candidates:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def _google_finance_symbol_candidates(self, symbol: str) -> list[str]:
        cleaned = str(symbol).strip().upper()
        base = cleaned.split(".")[0]
        if cleaned.endswith(".NS"):
            candidates = [f"{base}:NSE", f"{base}:BOM"]
        elif cleaned.endswith((".BO", ".BSE")):
            candidates = [f"{base}:BOM", f"{base}:BSE", f"{base}:NSE"]
        elif cleaned.endswith((".NSE",)):
            candidates = [f"{base}:NSE", f"{base}:BOM"]
        else:
            candidates = [f"{base}:NSE", f"{base}:BOM", f"{base}:NASDAQ", f"{base}:NYSE", f"{base}:NYS"]

        seen: set[str] = set()
        out: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                out.append(candidate)
                seen.add(candidate)
        return out

    @staticmethod
    def _parse_google_price_token(value: str | None) -> float | None:
        if not value:
            return None
        cleaned = re.sub(r"[^\d.\-]", "", str(value))
        parsed = pd.to_numeric(cleaned, errors="coerce")
        if pd.isna(parsed):
            return None
        return float(parsed)

    def _parse_google_finance_snapshot(self, raw_text: str) -> dict[str, object]:
        text = re.sub(r"\s+", " ", str(raw_text or "")).strip()
        if not text:
            return {}

        price_match = re.search(r'data-last-price="(?P<price>[\d.]+)"', text, re.IGNORECASE)
        if not price_match:
            price_match = re.search(r'[₹$€£]\s?[\d,]+(?:\.\d+)?', text)
        current_price = self._parse_google_price_token(price_match.group("price") if price_match and price_match.lastindex else price_match.group(0) if price_match else None)
        if current_price is None:
            return {}

        previous_close = None
        prev_match = re.search(r"Previous close.*?([₹$€£]\s?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
        if prev_match:
            previous_close = self._parse_google_price_token(prev_match.group(1))

        day_low = None
        day_high = None
        range_match = re.search(
            r"Day range.*?([₹$€£]\s?[\d,]+(?:\.\d+)?)\s*[-–]\s*([₹$€£]\s?[\d,]+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if range_match:
            day_low = self._parse_google_price_token(range_match.group(1))
            day_high = self._parse_google_price_token(range_match.group(2))

        as_of_date = date.today()
        dt_match = re.search(
            r"(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2}),\s+(?P<time>\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)?)\s*(?P<tz>(?:UTC|GMT)[+\-]\d{1,2}:\d{2})",
            text,
        )
        if dt_match:
            parsed_dt = pd.to_datetime(
                f"{dt_match.group('month')} {dt_match.group('day')} {date.today().year} {dt_match.group('time')}",
                errors="coerce",
            )
            if pd.notna(parsed_dt):
                if parsed_dt.date() > date.today() + timedelta(days=3):
                    parsed_dt = parsed_dt - pd.DateOffset(years=1)
                as_of_date = parsed_dt.date()

        open_price = previous_close if previous_close is not None else current_price
        low_price = day_low if day_low is not None else min(open_price, current_price)
        high_price = day_high if day_high is not None else max(open_price, current_price)

        return {
            "date": pd.Timestamp(as_of_date),
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(current_price),
            "volume": 0.0,
        }

    def _download_google_finance_daily_with_debug(self, symbol: str, start: date) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        debug: list[dict[str, str]] = []
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; StockAI/1.0)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
            for candidate in self._google_finance_symbol_candidates(symbol):
                quote_id = quote(candidate, safe=":")
                url = f"https://www.google.com/finance/quote/{quote_id}?hl=en"
                try:
                    resp = client.get(url)
                    debug.append(
                        {
                            "source": "google_finance",
                            "ticker": candidate,
                            "url": url,
                            "status": str(resp.status_code),
                            "preview": (resp.text or "")[:1200],
                        }
                    )
                    resp.raise_for_status()
                except Exception as exc:
                    debug.append(
                        {
                            "source": "google_finance",
                            "ticker": candidate,
                            "url": url,
                            "status": "error",
                            "preview": str(exc)[:300],
                        }
                    )
                    continue

                snapshot = self._parse_google_finance_snapshot(resp.text)
                if not snapshot:
                    continue
                if pd.Timestamp(snapshot["date"]).date() < start:
                    continue

                out = pd.DataFrame([{**snapshot, "symbol": symbol}])
                out = out[["date", "open", "high", "low", "close", "volume", "symbol"]].sort_values("date").reset_index(drop=True)
                return out, debug

        return pd.DataFrame(), debug

    def _download_best_effort_with_variants(self, symbol: str, start: date, interval: str) -> pd.DataFrame:
        # Last resort internet scrape across common exchange symbol variants.
        # Keep this Yahoo-free to avoid rate-limit noise and slow retries.
        for candidate in self._internet_symbol_candidates(symbol):
            if interval == "1d":
                try:
                    df = self._download_stooq_daily(candidate, start=start)
                    if not df.empty:
                        df["symbol"] = symbol
                        return df
                except Exception:
                    pass

        return pd.DataFrame()

    def _download_yahoo_best_effort(self, symbol: str, start: date, interval: str) -> pd.DataFrame:
        # Try Yahoo across common exchange suffixes before giving up.
        for candidate in self._internet_symbol_candidates(symbol):
            try:
                df = self._download(candidate, start=start, interval=interval)
                if not df.empty:
                    df["symbol"] = symbol
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def _next_fetch_start(self, last_seen: pd.Timestamp, interval: str) -> date:
        if pd.isna(last_seen):
            return date.today()
        interval_value = str(interval or "1d").strip().lower()
        if interval_value.endswith("m") and interval_value[:-1].isdigit():
            delta = pd.Timedelta(minutes=max(1, int(interval_value[:-1])))
        elif interval_value.endswith("h") and interval_value[:-1].isdigit():
            delta = pd.Timedelta(hours=max(1, int(interval_value[:-1])))
        else:
            delta = pd.Timedelta(days=1)
        return (pd.Timestamp(last_seen).tz_localize(None) + delta).date()

    def _fetch_from_source(self, source: str, symbol: str, start: date, interval: str) -> tuple[pd.DataFrame, list[dict[str, str]], str]:
        source_name = str(source or "").strip().lower()
        debug: list[dict[str, str]] = []
        try:
            if source_name == "nse" and interval == "1d":
                df, debug = self._download_nse_daily_with_debug(symbol, start=start)
                return df, debug, "nse"
            if source_name == "stooq" and interval == "1d":
                df, debug = self._download_stooq_daily_with_debug(symbol, start=start)
                return df, debug, "stooq"
            if source_name == "yahoo":
                return self._download(symbol, start=start, interval=interval), debug, "yahoo"
            if source_name == "scraper":
                if interval == "1d":
                    df, debug = self._download_google_finance_daily_with_debug(symbol, start=start)
                    if not df.empty:
                        return df, debug, "scraper"
                return self._download_best_effort_with_variants(symbol, start=start, interval=interval), debug, "scraper"
            if source_name == "google_finance" and interval == "1d":
                df, debug = self._download_google_finance_daily_with_debug(symbol, start=start)
                return df, debug, "google_finance"
        except Exception as exc:
            return pd.DataFrame(), debug, f"error:{type(exc).__name__}:{exc}"
        return pd.DataFrame(), debug, source_name or "unknown"

    def _download_alpha_vantage_best_effort(self, symbol: str, start: date) -> pd.DataFrame:
        for candidate in self._alpha_symbol_candidates(symbol):
            try:
                df = self._download_alpha_vantage_daily(candidate, start)
                if not df.empty:
                    # Keep original symbol namespace in downstream storage.
                    df["symbol"] = symbol
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def _download_polygon_best_effort(self, symbol: str, start: date) -> pd.DataFrame:
        for candidate in self._internet_symbol_candidates(symbol):
            try:
                df = self._download_polygon_daily(candidate, start)
                if not df.empty:
                    df["symbol"] = symbol
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def ingest(
        self,
        symbols: list[str],
        interval: str = "1d",
        lookback_days: int = 3650,
        direct_internet_scrape: bool = False,
        force_refresh: bool = False,
        save_failure_snapshot: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        symbols = normalize_symbols(symbols)
        self.store.upsert_symbols(symbols)
        summary: dict[str, int] = {}
        self.last_diagnostics = {}
        self.last_source_by_symbol = {}
        self.last_source_counts = {}
        provider = str(self.data_provider or "").strip().lower()
        force_direct_scrape = bool(direct_internet_scrape or provider in {"google", "internet", "scrape"})

        total_symbols = len(symbols)
        for idx, symbol in enumerate(symbols, start=1):
            if progress_callback is not None:
                try:
                    progress_callback(f"historical [{idx}/{total_symbols}] start {symbol}")
                except Exception:
                    pass
            diagnostics: list[str] = []
            debug_records: list[dict[str, str]] = []
            path = symbol_price_path(self.raw_data_dir, symbol, interval)
            existing = self.store.read_candles([symbol], interval)
            if existing.empty:
                existing = read_parquet_if_exists(path)
                if not existing.empty:
                    try:
                        self.store.write_candles(existing, interval=interval, source="cache")
                    except Exception:
                        pass
            if not existing.empty:
                existing.columns = [str(c).strip().lower() for c in existing.columns]
                if "datetime" in existing.columns and "date" not in existing.columns:
                    existing = existing.rename(columns={"datetime": "date"})
                if not {"date", "symbol"}.issubset(set(existing.columns)):
                    existing = pd.DataFrame()

            if existing.empty or force_refresh:
                start = date.today() - timedelta(days=lookback_days)
            else:
                existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
                if getattr(existing["date"].dt, "tz", None) is not None:
                    existing["date"] = existing["date"].dt.tz_localize(None)
                start = self._next_fetch_start(existing["date"].max(), interval)

            fresh = pd.DataFrame()
            source_used = ""

            source_candidates = self.source_manager.source_order(interval, preferred="scraper" if force_direct_scrape else provider)
            if not force_direct_scrape and provider == "polygon":
                source_candidates = ["stooq", "scraper"]
            elif not force_direct_scrape and provider == "alphavantage":
                source_candidates = ["nse", "stooq", "scraper"]

            for source_name in source_candidates:
                if self.source_manager.should_skip(source_name):
                    continue
                fresh, source_debug, source_label = self._fetch_from_source(source_name, symbol, start=start, interval=interval)
                debug_records.extend(source_debug)
                if not fresh.empty:
                    source_used = source_label
                    latest_available_time = pd.to_datetime(fresh["date"], errors="coerce").max()
                    self.source_manager.record_success(source_label, latest_available_time=str(latest_available_time) if pd.notna(latest_available_time) else None)
                    break
                diagnostics.append(f"{source_name}:no_rows")
                self.source_manager.record_failure(source_name, error="no_rows")

            if fresh.empty:
                if not existing.empty:
                    # Preserve cached rows when provider calls are temporarily rate-limited.
                    existing = existing.sort_values(["symbol", "date"]).reset_index(drop=True)
                    validate_ohlcv_frame(existing)
                    write_parquet(existing, path)
                    self.store.write_candles(existing, interval=interval, source="cache")
                    summary[symbol] = len(existing)
                    logger.info("historical_cached_used", extra={"symbol": symbol, "rows": len(existing)})
                    self.last_diagnostics[symbol] = "cached_rows_used"
                    self.last_source_by_symbol[symbol] = "cache"
                    if progress_callback is not None:
                        try:
                            progress_callback(f"historical [{idx}/{total_symbols}] done {symbol} rows={len(existing)} source=cache")
                        except Exception:
                            pass
                    continue

                logger.warning(
                    "historical_all_providers_failed",
                    extra={
                        "symbol": symbol,
                        "diagnostics": diagnostics,
                    },
                )
                logger.info("no_historical_data", extra={"symbol": symbol})
                summary[symbol] = 0
                if save_failure_snapshot:
                    snap = self._save_failure_snapshot(symbol, interval, lookback_days, diagnostics, debug_records)
                    diagnostics.append(f"snapshot:{snap}")
                self.last_diagnostics[symbol] = "; ".join(diagnostics) if diagnostics else "no_provider_rows"
                self.last_source_by_symbol[symbol] = "none"
                if progress_callback is not None:
                    try:
                        progress_callback(f"historical [{idx}/{total_symbols}] done {symbol} rows=0 source=none")
                    except Exception:
                        pass
                continue

            fresh.columns = [str(c).strip().lower() for c in fresh.columns]
            if "datetime" in fresh.columns and "date" not in fresh.columns:
                fresh = fresh.rename(columns={"datetime": "date"})
            if not {"date", "symbol"}.issubset(set(fresh.columns)):
                logger.warning("historical_payload_missing_keys", extra={"symbol": symbol, "columns": list(fresh.columns)})
                summary[symbol] = 0
                self.last_diagnostics[symbol] = "payload_missing_keys"
                self.last_source_by_symbol[symbol] = source_used or "invalid_payload"
                if progress_callback is not None:
                    try:
                        progress_callback(f"historical [{idx}/{total_symbols}] done {symbol} rows=0 source={self.last_source_by_symbol[symbol]}")
                    except Exception:
                        pass
                continue

            combined = fresh.copy() if force_direct_scrape or existing.empty or force_refresh else append_dedup_by_keys(existing, fresh, keys=["symbol", "date"])
            combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
            validate_ohlcv_frame(combined)
            write_parquet(combined, path)
            self.store.write_candles(combined, interval=interval, source=source_used or "unknown")
            summary[symbol] = len(combined)
            self.last_diagnostics[symbol] = f"ok_rows:{len(combined)}"
            self.last_source_by_symbol[symbol] = source_used or "unknown"
            logger.info("historical_ingested", extra={"symbol": symbol, "rows": len(combined)})
            if progress_callback is not None:
                try:
                    progress_callback(f"historical [{idx}/{total_symbols}] done {symbol} rows={len(combined)} source={self.last_source_by_symbol[symbol]}")
                except Exception:
                    pass

        counts: dict[str, int] = {}
        for source in self.last_source_by_symbol.values():
            key = str(source or "unknown").strip().lower() or "unknown"
            counts[key] = int(counts.get(key, 0)) + 1
        self.last_source_counts = counts

        return summary
