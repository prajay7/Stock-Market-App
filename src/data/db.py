from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=True)
    except Exception:
        return json.dumps(str(value), ensure_ascii=True)


class SQLiteDataStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS symbols (
                    symbol TEXT PRIMARY KEY,
                    exchange TEXT,
                    asset_type TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ohlcv_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, trade_date)
                );

                CREATE TABLE IF NOT EXISTS ohlcv_intraday (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    candle_time TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, candle_time, interval)
                );

                CREATE TABLE IF NOT EXISTS source_health (
                    source TEXT PRIMARY KEY,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    last_success_time TEXT,
                    last_error TEXT,
                    latest_available_time TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    prediction_time TEXT NOT NULL,
                    probability REAL,
                    signal TEXT,
                    reason TEXT,
                    risk_level TEXT,
                    model_version TEXT,
                    payload TEXT,
                    created_at TEXT NOT NULL,
                    validated INTEGER NOT NULL DEFAULT 0,
                    validated_at TEXT,
                    actual_return REAL,
                    outcome TEXT
                );

                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    interval TEXT,
                    validated_at TEXT NOT NULL,
                    metrics_json TEXT,
                    payload TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT UNIQUE,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    task_type TEXT,
                    symbols_json TEXT,
                    horizon_days INTEGER,
                    metrics_json TEXT,
                    artifact_path TEXT,
                    model_type TEXT,
                    payload TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    logger TEXT,
                    message TEXT NOT NULL,
                    payload TEXT
                );
                """
            )

    def upsert_symbols(self, symbols: Iterable[str]) -> None:
        payload = []
        for symbol in symbols:
            sym = str(symbol or "").strip().upper()
            if not sym:
                continue
            payload.append((sym, _now_iso(), _now_iso()))
        if not payload:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO symbols(symbol, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET updated_at=excluded.updated_at
                """,
                payload,
            )

    def upsert_source_health(
        self,
        source: str,
        *,
        success: bool,
        latest_available_time: str | None = None,
        error: str | None = None,
    ) -> None:
        source_name = str(source or "").strip().lower()
        if not source_name:
            return
        with self.connect() as conn:
            row = conn.execute("SELECT failure_count FROM source_health WHERE source = ?", (source_name,)).fetchone()
            failure_count = int(row[0]) if row and row[0] is not None else 0
            if success:
                failure_count = 0
            else:
                failure_count += 1
            conn.execute(
                """
                INSERT INTO source_health(source, failure_count, last_success_time, last_error, latest_available_time, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    failure_count=excluded.failure_count,
                    last_success_time=excluded.last_success_time,
                    last_error=excluded.last_error,
                    latest_available_time=excluded.latest_available_time,
                    updated_at=excluded.updated_at
                """,
                (
                    source_name,
                    failure_count,
                    _now_iso() if success else None,
                    None if success else str(error or "source_failure"),
                    latest_available_time,
                    _now_iso(),
                ),
            )

    def write_candles(self, df: pd.DataFrame, interval: str, source: str = "") -> None:
        if df.empty:
            return
        frame = df.copy()
        frame.columns = [str(col).strip().lower() for col in frame.columns]
        if "datetime" in frame.columns and "date" not in frame.columns:
            frame = frame.rename(columns={"datetime": "date"})
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        interval_value = str(interval or "1d").strip().lower()
        if interval_value == "1d":
            target_table = "ohlcv_daily"
            time_column = "trade_date"
        else:
            target_table = "ohlcv_intraday"
            time_column = "candle_time"

        if "date" not in frame.columns:
            raise ValueError("candle frame is missing a date column")

        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[frame["date"].notna()].copy()
        if frame.empty:
            return

        if getattr(frame["date"].dt, "tz", None) is not None:
            normalized_dates = frame["date"].dt.tz_localize(None)
        else:
            normalized_dates = frame["date"]
        frame[time_column] = normalized_dates.dt.strftime("%Y-%m-%d %H:%M:%S")
        if target_table == "ohlcv_daily":
            frame[time_column] = normalized_dates.dt.strftime("%Y-%m-%d")

        now = _now_iso()
        rows = []
        for row in frame.itertuples(index=False):
            row_data = row._asdict()
            rows.append(
                (
                    row_data.get("symbol"),
                    row_data.get(time_column),
                    row_data.get("open") if pd.notna(row_data.get("open")) else None,
                    row_data.get("high") if pd.notna(row_data.get("high")) else None,
                    row_data.get("low") if pd.notna(row_data.get("low")) else None,
                    row_data.get("close") if pd.notna(row_data.get("close")) else None,
                    row_data.get("volume") if pd.notna(row_data.get("volume")) else None,
                    source or row_data.get("source", ""),
                    now,
                    now,
                )
            )
        with self.connect() as conn:
            conn.executemany(
                f"""
                INSERT INTO {target_table}(symbol, {time_column}, open, high, low, close, volume, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, {time_column}{', interval' if target_table == 'ohlcv_intraday' else ''}) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                rows,
            )

    def latest_candle_time(self, symbol: str, interval: str) -> str | None:
        symbol = str(symbol or "").strip().upper()
        interval_value = str(interval or "1d").strip().lower()
        table = "ohlcv_daily" if interval_value == "1d" else "ohlcv_intraday"
        column = "trade_date" if table == "ohlcv_daily" else "candle_time"
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT MAX({column}) AS latest_value FROM {table} WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            return row[0] if row and row[0] else None

    def read_candles(self, symbols: Iterable[str], interval: str) -> pd.DataFrame:
        cleaned = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
        if not cleaned:
            return pd.DataFrame()
        interval_value = str(interval or "1d").strip().lower()
        table = "ohlcv_daily" if interval_value == "1d" else "ohlcv_intraday"
        column = "trade_date" if table == "ohlcv_daily" else "candle_time"
        query = f"SELECT symbol, {column} AS date, open, high, low, close, volume, source FROM {table} WHERE symbol IN ({','.join('?' for _ in cleaned)}) ORDER BY symbol, {column}"
        with self.connect() as conn:
            rows = conn.execute(query, cleaned).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(row) for row in rows])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def write_predictions(self, rows: Iterable[dict], model_version: str, interval: str = "1d") -> None:
        payload = []
        now = _now_iso()
        for row in rows:
            payload.append(
                (
                    str(row.get("symbol") or "").strip().upper(),
                    str(interval or row.get("interval") or "1d"),
                    str(row.get("prediction_time") or now),
                    row.get("probability") if row.get("probability") is not None else row.get("prob_up"),
                    row.get("signal") or row.get("decision"),
                    row.get("reason") or row.get("news_reason"),
                    row.get("risk_level") or row.get("confidence"),
                    str(model_version),
                    _json_dumps(row),
                    now,
                )
            )
        if not payload:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO predictions(symbol, interval, prediction_time, probability, signal, reason, risk_level, model_version, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def write_training_run(self, row: dict) -> None:
        payload = dict(row)
        model_version = str(payload.get("model_version") or payload.get("version") or "")
        now = _now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO training_runs(model_version, started_at, completed_at, task_type, symbols_json, horizon_days, metrics_json, artifact_path, model_type, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_version) DO UPDATE SET
                    completed_at=excluded.completed_at,
                    task_type=excluded.task_type,
                    symbols_json=excluded.symbols_json,
                    horizon_days=excluded.horizon_days,
                    metrics_json=excluded.metrics_json,
                    artifact_path=excluded.artifact_path,
                    model_type=excluded.model_type,
                    payload=excluded.payload
                """,
                (
                    model_version,
                    payload.get("started_at") or now,
                    payload.get("completed_at"),
                    payload.get("task_type"),
                    _json_dumps(payload.get("symbols") or payload.get("symbols_json") or []),
                    payload.get("horizon_days"),
                    _json_dumps(payload.get("metrics") or payload.get("metrics_json") or {}),
                    payload.get("artifact_path"),
                    payload.get("model_type"),
                    _json_dumps(payload),
                    now,
                ),
            )

    def write_validation_result(self, row: dict) -> None:
        payload = dict(row)
        now = _now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO validation_results(model_version, interval, validated_at, metrics_json, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("model_version"),
                    payload.get("interval") or "1d",
                    payload.get("validated_at") or now,
                    _json_dumps(payload.get("metrics") or payload.get("metrics_json") or {}),
                    _json_dumps(payload),
                    now,
                ),
            )

    def write_app_log(self, row: dict) -> None:
        payload = dict(row)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO app_logs(timestamp, level, logger, message, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload.get("timestamp") or _now_iso(),
                    str(payload.get("level") or "INFO").upper(),
                    payload.get("logger"),
                    str(payload.get("message") or ""),
                    _json_dumps(payload),
                ),
            )

    def read_source_health(self, limit: int = 200) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT source, failure_count, last_success_time, last_error, latest_available_time, updated_at FROM source_health ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def read_predictions(self, limit: int = 500) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(row) for row in rows])

    def read_training_runs(self, limit: int = 200) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(row) for row in rows])

    def read_validation_results(self, limit: int = 200) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM validation_results ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(row) for row in rows])


def get_default_store() -> SQLiteDataStore:
    from app.core.config import get_settings

    return SQLiteDataStore(get_settings().db_path)
