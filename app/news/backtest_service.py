from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from app.core.config import get_settings
from app.news.models import SignalBacktestSummary
from src.data.metadata_store import MetadataStore


@dataclass
class NewsSignalBacktestService:
    settings: object

    def __post_init__(self) -> None:
        self.store = MetadataStore(self.settings.database_url)

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _as_naive_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def compute_forward_return(entry_price: float | None, exit_price: float | None) -> dict[str, float | bool | None]:
        if entry_price is None or exit_price is None:
            return {"absolute_return": None, "percent_return": None, "is_positive": None}
        if float(entry_price) == 0.0:
            return {"absolute_return": None, "percent_return": None, "is_positive": None}

        absolute = float(exit_price) - float(entry_price)
        percent = (absolute / float(entry_price)) * 100.0
        return {
            "absolute_return": absolute,
            "percent_return": percent,
            "is_positive": percent > 0.0,
        }

    @staticmethod
    def score_bucket(value: float | None) -> str:
        if value is None:
            return "unknown"
        score = float(value)
        if score < 0.40:
            return "0.00-0.39"
        if score < 0.60:
            return "0.40-0.59"
        if score < 0.80:
            return "0.60-0.79"
        return "0.80-1.00"

    def _history_window(self, start: datetime, end: datetime, ticker: str) -> pd.DataFrame:
        try:
            hist = yf.Ticker(ticker).history(
                start=start.date().isoformat(),
                end=(end.date() + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
            )
        except Exception:
            return pd.DataFrame()
        if hist is None or hist.empty:
            return pd.DataFrame()
        frame = hist.copy()
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
        frame = frame.dropna(subset=["Close"])
        return frame

    def get_price_on_or_after(self, ticker: str, target_date: datetime) -> tuple[float | None, datetime | None]:
        if not ticker:
            return None, None
        start = target_date - timedelta(days=3)
        end = target_date + timedelta(days=8)
        frame = self._history_window(start, end, ticker)
        if frame.empty:
            return None, None

        target_ts = pd.Timestamp(target_date)
        if target_ts.tzinfo is not None:
            target_ts = target_ts.tz_convert(None)

        candidates = frame[frame.index >= target_ts]
        if not candidates.empty:
            ts = candidates.index[0]
            return float(candidates.iloc[0]["Close"]), ts.to_pydatetime()

        # Fallback if no future bar available in window.
        if str(self.settings.trading_day_fallback_mode) == "latest_available_close":
            ts = frame.index[-1]
            return float(frame.iloc[-1]["Close"]), ts.to_pydatetime()

        return None, None

    def _benchmark_return(self, target_date: datetime, horizon: int) -> float | None:
        benchmark = str(self.settings.benchmark_ticker or "").strip()
        if not benchmark:
            return None
        entry, _ = self.get_price_on_or_after(benchmark, target_date)
        exit_price, _ = self.get_price_on_or_after(benchmark, target_date + timedelta(days=horizon))
        payload = self.compute_forward_return(entry, exit_price)
        return payload["percent_return"] if payload["percent_return"] is not None else None

    def evaluate_opportunity_outcomes(self, opportunity_id: int) -> list[dict[str, Any]]:
        all_rows = self.store.read_beneficiary_opportunities_with_signal(limit=200000)
        row = next((r for r in all_rows if int(r.get("id", -1)) == int(opportunity_id)), None)
        if row is None:
            return []

        ticker = str(row.get("ticker") or "").strip()
        signal_price = row.get("signal_price")
        signal_ts_raw = row.get("signal_timestamp") or row.get("created_at")
        signal_ts = pd.to_datetime(signal_ts_raw, errors="coerce")
        if pd.isna(signal_ts):
            signal_ts = pd.Timestamp(self._utc_now())
        signal_dt = signal_ts.to_pydatetime()
        if signal_dt.tzinfo is None:
            signal_dt = signal_dt.replace(tzinfo=timezone.utc)
        else:
            signal_dt = signal_dt.astimezone(timezone.utc)
        now = self._utc_now()

        if signal_price is None and ticker:
            inferred_entry, _entry_date = self.get_price_on_or_after(ticker, signal_dt)
            signal_price = inferred_entry

        outcomes: list[dict[str, Any]] = []
        for horizon in self.settings.backtest_horizons:
            target_date = signal_dt + timedelta(days=int(horizon))
            created_at = self._as_naive_utc(now)
            base = {
                "opportunity_id": int(opportunity_id),
                "ticker": ticker,
                "evaluation_horizon_days": int(horizon),
                "target_date": self._as_naive_utc(target_date),
                "entry_price": float(signal_price) if signal_price is not None else None,
                "created_at": created_at,
            }

            if now < target_date:
                payload = {
                    **base,
                    "exit_price": None,
                    "absolute_return": None,
                    "percent_return": None,
                    "benchmark_return": None,
                    "alpha_return": None,
                    "is_positive": None,
                    "evaluation_status": "pending",
                    "evaluated_at": None,
                }
                self.store.upsert_signal_outcome(payload)
                outcomes.append(payload)
                continue

            if not ticker or signal_price is None:
                payload = {
                    **base,
                    "exit_price": None,
                    "absolute_return": None,
                    "percent_return": None,
                    "benchmark_return": None,
                    "alpha_return": None,
                    "is_positive": None,
                    "evaluation_status": "failed",
                    "evaluated_at": created_at,
                }
                self.store.upsert_signal_outcome(payload)
                outcomes.append(payload)
                continue

            exit_price, _price_date = self.get_price_on_or_after(ticker, target_date)
            if exit_price is None:
                payload = {
                    **base,
                    "exit_price": None,
                    "absolute_return": None,
                    "percent_return": None,
                    "benchmark_return": None,
                    "alpha_return": None,
                    "is_positive": None,
                    "evaluation_status": "failed",
                    "evaluated_at": created_at,
                }
                self.store.upsert_signal_outcome(payload)
                outcomes.append(payload)
                continue

            perf = self.compute_forward_return(float(signal_price), float(exit_price))
            benchmark_return = self._benchmark_return(signal_dt, int(horizon))
            alpha_return = None
            if benchmark_return is not None and perf["percent_return"] is not None:
                alpha_return = float(perf["percent_return"]) - float(benchmark_return)

            payload = {
                **base,
                "exit_price": float(exit_price),
                "absolute_return": perf["absolute_return"],
                "percent_return": perf["percent_return"],
                "benchmark_return": benchmark_return,
                "alpha_return": alpha_return,
                "is_positive": perf["is_positive"],
                "evaluation_status": "completed",
                "evaluated_at": created_at,
            }
            self.store.upsert_signal_outcome(payload)
            outcomes.append(payload)

        return outcomes

    def evaluate_pending_outcomes(self) -> dict[str, Any]:
        opportunities = self.store.read_beneficiary_opportunities_with_signal(limit=200000)
        total = len(opportunities)
        evaluated = 0
        failed = 0
        pending = 0

        for row in opportunities:
            results = self.evaluate_opportunity_outcomes(int(row.get("id")))
            for item in results:
                status = str(item.get("evaluation_status") or "")
                if status == "completed":
                    evaluated += 1
                elif status == "failed":
                    failed += 1
                elif status == "pending":
                    pending += 1

        return {
            "total_opportunities": total,
            "completed_outcomes": evaluated,
            "failed_outcomes": failed,
            "pending_outcomes": pending,
        }

    def _group_metrics(self, frame: pd.DataFrame, key: str) -> list[dict[str, Any]]:
        if frame.empty or key not in frame.columns:
            return []
        data = []
        grouped = frame.groupby(key, dropna=False)
        for value, group in grouped:
            returns = pd.to_numeric(group["percent_return"], errors="coerce")
            valid = returns.dropna()
            hit_rate = float((valid > 0).mean()) if not valid.empty else 0.0
            data.append(
                {
                    key: "Unknown" if pd.isna(value) else value,
                    "count": int(len(group)),
                    "avg_return": float(valid.mean()) if not valid.empty else 0.0,
                    "hit_rate": hit_rate,
                }
            )
        data.sort(key=lambda row: (row["avg_return"], row["hit_rate"]), reverse=True)
        return data

    def _apply_filters(self, frame: pd.DataFrame, filters: dict[str, Any] | None = None) -> pd.DataFrame:
        if frame.empty:
            return frame
        filters = filters or {}
        out = frame.copy()

        if filters.get("start_date") is not None:
            start = pd.to_datetime(filters.get("start_date"), errors="coerce")
            out = out[out["signal_created_at"] >= start]
        if filters.get("end_date") is not None:
            end = pd.to_datetime(filters.get("end_date"), errors="coerce")
            out = out[out["signal_created_at"] <= end]
        for key in ["event_type", "sector", "sentiment_label", "timing_label", "relation"]:
            values = filters.get(key)
            if values:
                if not isinstance(values, list):
                    values = [values]
                out = out[out[key].isin(values)]
        min_conf = filters.get("min_confidence")
        if min_conf is not None:
            out = out[pd.to_numeric(out["confidence_score"], errors="coerce") >= float(min_conf)]
        min_score = filters.get("min_overall_score")
        if min_score is not None:
            out = out[pd.to_numeric(out["opportunity_overall_score"], errors="coerce") >= float(min_score)]
        horizons = filters.get("horizon_days")
        if horizons:
            if not isinstance(horizons, list):
                horizons = [horizons]
            out = out[out["evaluation_horizon_days"].isin([int(x) for x in horizons])]
        return out

    def summary(self, filters: dict[str, Any] | None = None) -> SignalBacktestSummary:
        rows = self.store.read_signal_outcomes_joined(limit=500000)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return SignalBacktestSummary(totals={"total_signals": 0, "evaluated_signals": 0, "pending_signals": 0, "hit_rate": 0.0, "avg_return": 0.0, "median_return": 0.0})

        frame["score_bucket"] = frame["opportunity_overall_score"].apply(self.score_bucket)
        frame = self._apply_filters(frame, filters=filters)

        total_signals = int(frame["opportunity_id"].nunique()) if "opportunity_id" in frame.columns else 0
        completed = frame[frame["evaluation_status"] == "completed"]
        pending = frame[frame["evaluation_status"] == "pending"]

        completed_returns = pd.to_numeric(completed["percent_return"], errors="coerce").dropna()
        hit_rate = float((completed_returns > 0).mean()) if not completed_returns.empty else 0.0

        totals = {
            "total_signals": total_signals,
            "evaluated_signals": int(completed["opportunity_id"].nunique()) if not completed.empty else 0,
            "pending_signals": int(pending["opportunity_id"].nunique()) if not pending.empty else 0,
            "hit_rate": hit_rate,
            "avg_return": float(completed_returns.mean()) if not completed_returns.empty else 0.0,
            "median_return": float(completed_returns.median()) if not completed_returns.empty else 0.0,
        }

        by_horizon = self._group_metrics(completed, "evaluation_horizon_days")
        by_event_type = self._group_metrics(completed, "event_type")
        by_sector = self._group_metrics(completed, "sector")
        by_relation_type = self._group_metrics(completed, "relation")
        by_timing_label = self._group_metrics(completed, "timing_label")
        by_score_bucket = self._group_metrics(completed, "score_bucket")

        return SignalBacktestSummary(
            totals=totals,
            by_horizon=by_horizon,
            by_event_type=by_event_type,
            by_sector=by_sector,
            by_relation_type=by_relation_type,
            by_timing_label=by_timing_label,
            by_score_bucket=by_score_bucket,
        )

    def signal_history(self, filters: dict[str, Any] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self.store.read_signal_outcomes_joined(limit=500000)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return []
        frame["score_bucket"] = frame["opportunity_overall_score"].apply(self.score_bucket)
        frame = self._apply_filters(frame, filters=filters)
        frame = frame.sort_values(["signal_created_at", "opportunity_id", "evaluation_horizon_days"], ascending=[False, False, True])
        max_rows = int(limit or self.settings.max_backtest_rows_in_ui)
        return frame.head(max_rows).to_dict(orient="records")


news_signal_backtest_service = NewsSignalBacktestService(get_settings())
