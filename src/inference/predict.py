from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx
import joblib
import numpy as np
import pandas as pd

from app.core.config import get_settings
from src.data.db import SQLiteDataStore
from src.data.cache import symbol_price_path
from src.data.historical_loader import HistoricalLoader
from src.inference.rank import rank_predictions
from src.training.dataset_builder import build_and_save_dataset


def _latest_model_path(model_dir: Path, model_name: str) -> Path:
    latest_path = model_dir / f"{model_name}_latest.joblib"
    if latest_path.exists():
        return latest_path
    candidates = sorted(model_dir.glob(f"{model_name}_*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No artifacts found for model {model_name}")
    return candidates[-1]


def _metadata_path_from_model(model_path: Path) -> Path:
    return model_path.with_suffix("").with_name(model_path.stem + ".metadata.json")


def _alpha_symbol_candidates(symbol: str) -> list[str]:
    base = str(symbol).upper().split(".")[0]
    if str(symbol).upper().endswith((".NS", ".BO", ".NSE", ".BSE")):
        candidates = [symbol.upper(), base]
    else:
        # Prefer Indian exchange variants first for suffix-less symbols.
        candidates = [f"{base}.NS", f"{base}.BO", f"{base}.NSE", f"{base}.BSE", symbol.upper(), base]
    seen = set()
    out = []
    for candidate in candidates:
        if candidate not in seen:
            out.append(candidate)
            seen.add(candidate)
    return out


def _fetch_live_quote_alpha_vantage(
    symbol: str,
    api_key: str,
    timeout_sec: float = 20.0,
) -> tuple[float | None, str | None, str | None, str | None]:
    if not api_key:
        return None, None, None, None

    url = "https://www.alphavantage.co/query"
    with httpx.Client(timeout=timeout_sec) as client:
        for candidate in _alpha_symbol_candidates(symbol):
            try:
                time.sleep(1.1)
                intraday_params = {
                    "function": "TIME_SERIES_INTRADAY",
                    "symbol": candidate,
                    "interval": "1min",
                    "outputsize": "compact",
                    "apikey": api_key,
                }
                intraday_payload = client.get(url, params=intraday_params).json()
                series_key = next((k for k in intraday_payload.keys() if k.startswith("Time Series")), None)
                if series_key and isinstance(intraday_payload.get(series_key), dict):
                    series = intraday_payload[series_key]
                    if series:
                        latest_key = max(series.keys())
                        latest_bar = series.get(latest_key, {})
                        px = pd.to_numeric(latest_bar.get("4. close"), errors="coerce")
                        ts = pd.to_datetime(latest_key, errors="coerce")
                        if pd.notna(px):
                            ts_date = ts.strftime("%Y-%m-%d") if pd.notna(ts) else None
                            ts_time = ts.strftime("%H:%M:%S") if pd.notna(ts) else None
                            return float(px), ts_date, ts_time, "alphavantage_intraday_1min"
            except Exception:
                pass

            try:
                time.sleep(1.1)
                quote_params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": candidate,
                    "apikey": api_key,
                }
                quote_payload = client.get(url, params=quote_params).json()
                quote = quote_payload.get("Global Quote", {})
                px = pd.to_numeric(quote.get("05. price"), errors="coerce")
                latest_day = quote.get("07. latest trading day")
                if pd.notna(px):
                    return float(px), str(latest_day) if latest_day else None, "00:00:00", "alphavantage_global_quote"
            except Exception:
                pass

    return None, None, None, None


def _fetch_live_quote_polygon(
    symbol: str,
    api_key: str,
    timeout_sec: float = 20.0,
) -> tuple[float | None, str | None, str | None, str | None]:
    if not api_key:
        return None, None, None, None

    base_symbol = str(symbol).upper().split(".")[0]
    url = f"https://api.polygon.io/v2/aggs/ticker/{base_symbol}/prev"
    params = {"adjusted": "true", "apiKey": api_key}
    with httpx.Client(timeout=timeout_sec) as client:
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return None, None, None, None

    results = payload.get("results") or []
    if not results:
        return None, None, None, None

    latest = results[0] if isinstance(results, list) else {}
    px = pd.to_numeric(latest.get("c"), errors="coerce")
    ts = pd.to_datetime(latest.get("t"), unit="ms", errors="coerce") if latest.get("t") is not None else pd.NaT
    if pd.isna(px):
        return None, None, None, None

    ts_date = ts.strftime("%Y-%m-%d") if pd.notna(ts) else None
    ts_time = ts.strftime("%H:%M:%S") if pd.notna(ts) else None
    return float(px), ts_date, ts_time, "polygon_prev"


def _google_finance_symbol_candidates(symbol: str) -> list[str]:
    base = str(symbol).upper().split(".")[0]
    if str(symbol).upper().endswith(".NS"):
        candidates = [f"{base}:NSE", f"{base}:BOM"]
    elif str(symbol).upper().endswith((".BO", ".BSE")):
        candidates = [f"{base}:BOM", f"{base}:BSE", f"{base}:NSE"]
    else:
        candidates = [f"{base}:NSE", f"{base}:BOM", f"{base}:NASDAQ", f"{base}:NYSE"]

    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            out.append(candidate)
            seen.add(candidate)
    return out


def _parse_google_finance_quote(raw_text: str) -> tuple[float | None, str | None, str | None, str | None]:
    text = str(raw_text or "")
    if not text:
        return None, None, None, None

    price_match = re.search(r'[₹$€£]\s?[\d,]+(?:\.\d+)?', text)
    if not price_match:
        price_match = re.search(r'data-last-price="(?P<price>[\d.]+)"', text, re.IGNORECASE)
    if not price_match:
        return None, None, None, None

    price_raw = str(price_match.group(0) if price_match.lastindex is None else price_match.group("price"))
    price_raw = price_raw.replace("₹", "").replace("$", "").replace("€", "").replace("£", "").replace(",", "").strip()
    price = pd.to_numeric(price_raw, errors="coerce")
    if pd.isna(price):
        return None, None, None, None

    time_match = re.search(
        r'Today\s+(?P<day>[A-Za-z]{3}\s+\d{1,2}),\s+(?P<time>\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)?)\s*(?P<tz>UTC[+\-]\d{1,2}:\d{2})',
        text,
        re.IGNORECASE,
    )
    if time_match:
        as_of_date = time_match.group("day").strip()
        as_of_time = f"{time_match.group('time').strip()} {time_match.group('tz').strip()}".strip()
        return float(price), as_of_date, as_of_time, "google_finance"

    date_match = re.search(r'data-last-trade-date="(?P<date>[^"]+)"', text, re.IGNORECASE)
    if date_match:
        return float(price), date_match.group("date").strip(), None, "google_finance"

    return float(price), None, None, "google_finance"


def _fetch_live_quote_google_finance(
    symbol: str,
    timeout_sec: float = 20.0,
) -> tuple[float | None, str | None, str | None, str | None]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StockAI/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    with httpx.Client(timeout=timeout_sec, follow_redirects=True, headers=headers) as client:
        for candidate in _google_finance_symbol_candidates(symbol):
            quote_id = quote(candidate, safe=":")
            url = f"https://www.google.com/finance/quote/{quote_id}"
            try:
                response = client.get(url)
                response.raise_for_status()
            except Exception:
                continue

            parsed = _parse_google_finance_quote(response.text)
            if parsed[0] is not None:
                return parsed

    return None, None, None, None


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        num = float(value)
    except Exception:
        return default
    if np.isnan(num) or np.isinf(num):
        return default
    return num


def _extract_json_object(raw_text: str) -> dict:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _llm_feature_snapshot(row: pd.Series) -> dict:
    keys = [
        "close",
        "open",
        "high",
        "low",
        "volume",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "rsi_14",
        "atr_14",
        "sma_20",
        "sma_50",
        "macd",
        "macd_signal",
        "sentiment_same_day",
        "sentiment_3d",
        "news_count",
        "news_impact_mean",
        "news_signal_score",
        "news_signal_3d",
    ]
    out: dict[str, float | None] = {}
    for key in keys:
        out[key] = _safe_float(row.get(key), None)
    return out


def _predict_symbol_with_openai(row: pd.Series, settings, horizon_days: int) -> dict:
    symbol = str(row.get("symbol") or "").upper().strip()
    sentiment = _safe_float(row.get("sentiment_same_day"), 0.0) or 0.0
    current_price = _safe_float(row.get("close"), 0.0) or 0.0

    api_key = settings.openai_predict_api_key_effective
    if not api_key:
        return {
            "symbol": symbol,
            "prob_up": 0.50,
            "predicted_return": 0.0,
            "confidence": 0.50,
            "latest_sentiment": sentiment,
            "decision": "HOLD",
            "llm_reason": "missing_api_key",
        }

    feature_snapshot = _llm_feature_snapshot(row)
    system_prompt = (
        "You are a conservative quant assistant for short-horizon stock movement forecasting. "
        "Return only one compact JSON object with keys: prob_up, predicted_return, confidence, decision, reason. "
        "Rules: prob_up in [0,1], predicted_return in [-0.2,0.2], confidence in [0,1], "
        "decision must be BUY_CANDIDATE or HOLD."
    )
    user_prompt = {
        "symbol": symbol,
        "horizon_days": int(horizon_days),
        "features": feature_snapshot,
        "current_price": current_price,
        "latest_sentiment": sentiment,
    }

    url = str(settings.openai_predict_base_url).rstrip("/") + "/chat/completions"
    request_payload = {
        "model": str(settings.openai_predict_model_name),
        "temperature": float(settings.openai_predict_temperature),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=float(settings.openai_predict_timeout_sec or settings.request_timeout_sec)) as client:
            resp = client.post(url, headers=headers, json=request_payload)
            resp.raise_for_status()
            payload = resp.json()
        content = str(payload.get("choices", [{}])[0].get("message", {}).get("content", ""))
        pred = _extract_json_object(content)
    except Exception:
        pred = {}

    prob_up = _safe_float(pred.get("prob_up"), 0.50)
    predicted_return = _safe_float(pred.get("predicted_return"), 0.0)
    confidence = _safe_float(pred.get("confidence"), prob_up)
    decision = str(pred.get("decision") or "HOLD").strip().upper()
    if decision not in {"BUY_CANDIDATE", "HOLD"}:
        decision = "BUY_CANDIDATE" if (prob_up or 0.50) >= 0.55 else "HOLD"

    prob_up = float(np.clip(prob_up if prob_up is not None else 0.50, 0.0, 1.0))
    predicted_return = float(np.clip(predicted_return if predicted_return is not None else 0.0, -0.20, 0.20))
    confidence = float(np.clip(confidence if confidence is not None else prob_up, 0.0, 1.0))

    return {
        "symbol": symbol,
        "prob_up": prob_up,
        "predicted_return": predicted_return,
        "confidence": confidence,
        "latest_sentiment": sentiment,
        "decision": decision,
        "llm_reason": str(pred.get("reason") or ""),
    }


def _series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.Series(pd.to_numeric(df[column], errors="coerce"), index=df.index).fillna(default).astype(float)


def _resolve_model_feature_cols(model, metadata: dict | None = None) -> list[str]:
    payload = metadata or {}
    from_metadata = payload.get("feature_list")
    if isinstance(from_metadata, list) and from_metadata:
        return [str(col) for col in from_metadata]

    try:
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            names = getattr(booster, "feature_names", None)
            if isinstance(names, list) and names:
                return [str(col) for col in names]
    except Exception:
        pass

    return []


def _prepare_inference_matrix(latest: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if not feature_cols:
        feature_cols = [c for c in latest.columns if c not in {"date", "symbol"} and pd.api.types.is_numeric_dtype(latest[c])]
    work = latest.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    return work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _add_news_adjusted_predictions(latest: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    sentiment_today = _series_or_default(out, "sentiment_same_day", 0.0)
    sentiment_3d = _series_or_default(out, "sentiment_3d", 0.0)
    impact = _series_or_default(out, "news_impact_3d", 0.0)
    if (impact == 0.0).all():
        impact = _series_or_default(out, "news_impact_mean", 0.0)
    signal = _series_or_default(out, "news_signal_3d", 0.0)
    if (signal == 0.0).all():
        signal = _series_or_default(out, "news_signal_score", 0.0)
    news_count = _series_or_default(out, "news_count_3d", 0.0)
    if (news_count == 0.0).all():
        news_count = _series_or_default(out, "news_count", 0.0)

    prob_up = _series_or_default(out, "prob_up", 0.5).clip(0.0, 1.0)
    predicted_return = _series_or_default(out, "predicted_return", 0.0).clip(-0.50, 0.50)
    confidence = _series_or_default(out, "confidence", 0.5).clip(0.0, 1.0)

    news_presence = np.minimum(1.0, np.log1p(news_count.clip(lower=0.0)) / np.log(8.0))
    news_prob_boost = (
        (0.12 * signal.clip(-1.0, 1.0))
        + (0.04 * sentiment_3d.clip(-1.0, 1.0))
        + (0.02 * np.sign(signal) * news_presence)
    ).clip(-0.18, 0.18)
    news_return_boost = ((0.025 * signal.clip(-1.0, 1.0)) + (0.010 * sentiment_today.clip(-1.0, 1.0))).clip(-0.05, 0.05)

    adjusted_prob = (prob_up + news_prob_boost).clip(0.0, 1.0)
    adjusted_return = (predicted_return + news_return_boost).clip(-0.50, 0.50)

    model_base = predicted_return.where(predicted_return != 0.0, prob_up - 0.5)
    model_direction = np.sign(model_base)
    news_direction = np.sign(signal)
    agrees = (news_direction == 0) | (model_direction == 0) | (news_direction == model_direction)
    confidence_delta = (0.12 * signal.abs().clip(0.0, 1.0) * news_presence) + (0.05 * impact.clip(0.0, 1.0))
    adjusted_confidence = np.where(agrees, confidence + confidence_delta, confidence - (confidence_delta * 0.5))

    out["latest_sentiment"] = sentiment_today
    out["news_count"] = news_count.round().astype(int)
    out["news_impact_score"] = impact.clip(0.0, 1.0)
    out["news_signal_score"] = signal.clip(-1.0, 1.0)
    out["news_probability_boost"] = news_prob_boost
    out["news_adjusted_prob_up"] = adjusted_prob
    out["news_adjusted_predicted_return"] = adjusted_return
    out["news_adjusted_confidence"] = pd.Series(adjusted_confidence, index=out.index).clip(0.0, 1.0)

    out["news_decision"] = "HOLD"
    out.loc[(out["news_adjusted_prob_up"] >= 0.57) & (out["news_adjusted_predicted_return"] > 0), "news_decision"] = "NEWS_BUY_CANDIDATE"
    out.loc[(out["news_adjusted_prob_up"] <= 0.43) | (out["news_adjusted_predicted_return"] < -0.01), "news_decision"] = "NEWS_RISK"
    out["news_reason"] = np.where(
        news_count <= 0,
        "no_recent_news",
        np.where(signal > 0.08, "positive_news_signal", np.where(signal < -0.08, "negative_news_signal", "mixed_or_neutral_news")),
    )
    return out


def _ensure_price_cache_for_symbols(settings, symbols: list[str], interval: str, horizon_days: int) -> None:
    normalized = [str(sym).strip().upper() for sym in symbols if str(sym).strip()]
    missing = [
        sym
        for sym in normalized
        if not symbol_price_path(settings.raw_data_dir, sym, interval).exists()
    ]
    if not missing:
        return

    loader = HistoricalLoader(
        settings.raw_data_dir,
        alpha_vantage_api_key=settings.alpha_vantage_api_key,
        polygon_api_key=settings.polygon_api_key,
        stooq_api_key=settings.stooq_api_key,
        data_provider=settings.data_provider,
    )
    loader.ingest(
        symbols=missing,
        interval=interval,
        lookback_days=max(int(settings.historical_lookback_days), int(horizon_days) + 40),
        direct_internet_scrape=False,
        save_failure_snapshot=False,
    )


def save_prediction_outputs(
    out: pd.DataFrame,
    settings=None,
    model_version: str | None = None,
) -> None:
    settings = settings or get_settings()
    resolved_model_version = str(model_version or "").strip()
    if not resolved_model_version and "model_version" in out.columns:
        model_versions = out["model_version"].dropna().astype(str)
        if not model_versions.empty:
            resolved_model_version = str(model_versions.iloc[0]).strip()
    if not resolved_model_version:
        resolved_model_version = "unknown"

    predictions_path = settings.output_dir / "latest_predictions.parquet"
    out.to_parquet(predictions_path, index=False)

    csv_path = settings.output_dir / "latest_predictions.csv"
    out.to_csv(csv_path, index=False)

    json_path = settings.output_dir / "latest_predictions.json"
    json_path.write_text(out.to_json(orient="records", indent=2), encoding="utf-8")
    SQLiteDataStore(settings.db_path).write_predictions(
        out.to_dict(orient="records"),
        model_version=resolved_model_version,
        interval=str(settings.historical_interval),
    )


def predict_for_symbols(
    symbols: list[str],
    model_name: str = "xgboost_classifier",
    horizon_days: int = 1,
    atr_multiplier: float = 1.0,
    include_live_quote: bool = False,
    persist_output: bool = True,
) -> pd.DataFrame:
    settings = get_settings()
    _ensure_price_cache_for_symbols(settings, symbols, settings.historical_interval, int(horizon_days))
    dataset_path = build_and_save_dataset(settings.raw_data_dir, settings.processed_data_dir, symbols, settings.historical_interval, horizon_days)
    df = pd.read_parquet(dataset_path).sort_values(["date", "symbol"]).reset_index(drop=True)
    prediction_time = datetime.now(timezone.utc)

    model_version = str(model_name)

    latest = df.groupby("symbol", as_index=False).tail(1).copy()

    normalized_model_name = str(model_name).strip().lower()
    if normalized_model_name in {"movement", "movement_model"}:
        model_path = settings.movement_model_path
        metadata_path = model_path.with_suffix(".metadata.json")
        if not model_path.exists():
            raise FileNotFoundError(f"No artifacts found for model {model_name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        model_version = str(metadata.get("model_version") or metadata.get("version") or model_path.stem)
        threshold = pd.to_numeric(metadata.get("classification_threshold", 0.5), errors="coerce")
        if pd.isna(threshold):
            threshold = 0.5
        threshold = float(min(max(float(threshold), 0.05), 0.95))

        model = joblib.load(model_path)
        feature_cols = _resolve_model_feature_cols(model, metadata=metadata)
        X = _prepare_inference_matrix(latest, feature_cols)
        prob_up = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        latest["prob_up"] = pd.Series(pd.to_numeric(prob_up, errors="coerce"), index=latest.index).fillna(0.5).clip(0.0, 1.0)
        latest["probability"] = latest["prob_up"]
        latest["predicted_return"] = latest["prob_up"] - threshold
        latest["confidence"] = (latest["prob_up"] - threshold).abs().mul(2.0).clip(0.0, 1.0)
        latest["signal"] = np.select(
            [latest["prob_up"] >= 0.65, latest["prob_up"] >= 0.55, latest["prob_up"] <= 0.35],
            ["bullish", "watch", "avoid"],
            default="bearish",
        )
        latest["reason"] = latest.apply(
            lambda row: f"probability={float(row.get('prob_up') or 0.0):.3f}; threshold={threshold:.2f}",
            axis=1,
        )
        latest["risk_level"] = np.where(latest["confidence"] > 0.7, "low", np.where(latest["confidence"] > 0.4, "medium", "high"))
        latest["decision"] = np.where(latest["signal"].isin(["bullish", "watch"]), "BUY_CANDIDATE", "HOLD")
        latest["prediction_time"] = prediction_time.isoformat() if prediction_time else None
        latest["interval"] = str(settings.historical_interval)
    elif normalized_model_name == "openai_stock_llm":
        if not settings.openai_predict_enabled:
            raise ValueError("OpenAI predictor is disabled. Set OPENAI_PREDICT_ENABLED=true.")
        llm_rows = [_predict_symbol_with_openai(row, settings, int(horizon_days)) for _, row in latest.iterrows()]
        llm_df = pd.DataFrame(llm_rows)
        latest = latest.merge(llm_df, on="symbol", how="left")
    else:
        model_path = _latest_model_path(settings.model_dir, model_name)
        metadata_path = _metadata_path_from_model(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_cols = [str(c) for c in metadata.get("feature_list", [])]
        task_type = str(metadata.get("task_type") or "classification").strip().lower()
        model = joblib.load(model_path)
        X = _prepare_inference_matrix(latest, feature_cols)
        model_version = metadata.get("version") or model_path.stem

        if task_type == "classification":
            prob_up = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
            threshold = pd.to_numeric(metadata.get("classification_threshold", 0.5), errors="coerce")
            if pd.isna(threshold):
                threshold = 0.5
            threshold = float(min(max(threshold, 0.05), 0.95))
            latest["prob_up"] = pd.Series(pd.to_numeric(prob_up, errors="coerce"), index=latest.index).fillna(0.5).clip(0.0, 1.0)
            latest["predicted_return"] = latest["prob_up"] - 0.5
            latest["confidence"] = (latest["prob_up"] - threshold).abs().mul(2.0).clip(0.0, 1.0)
            latest["decision"] = np.where(latest["prob_up"] >= threshold, "BUY_CANDIDATE", "HOLD")
            latest["signal"] = np.where(latest["prob_up"] >= threshold + 0.10, "bullish", np.where(latest["prob_up"] <= threshold - 0.10, "avoid", "watch"))
            latest["reason"] = latest.apply(lambda row: f"probability={float(row.get('prob_up') or 0.0):.3f}; threshold={threshold:.2f}", axis=1)
            latest["risk_level"] = np.where(latest["confidence"] > 0.7, "low", np.where(latest["confidence"] > 0.4, "medium", "high"))
        else:
            pred_values = pd.to_numeric(model.predict(X), errors="coerce")
            current_price = pd.to_numeric(latest.get("close"), errors="coerce")
            if task_type == "regression_close":
                latest["predicted_price"] = pred_values
                latest["predicted_return"] = ((pred_values / current_price) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                latest["predicted_return"] = pd.Series(pred_values, index=latest.index).fillna(0.0)
                latest["predicted_price"] = (current_price * (1.0 + latest["predicted_return"]))

            latest["prob_up"] = (1.0 / (1.0 + np.exp(-8.0 * latest["predicted_return"].clip(-0.25, 0.25)))).clip(0.0, 1.0)
            latest["confidence"] = (latest["predicted_return"].abs() / 0.05).clip(0.0, 1.0)
            latest["decision"] = np.where(latest["predicted_return"] >= 0, "BUY_CANDIDATE", "HOLD")
            latest["signal"] = np.where(latest["predicted_return"] > 0.01, "bullish", np.where(latest["predicted_return"] < -0.005, "avoid", "watch"))
            latest["reason"] = latest.apply(lambda row: f"predicted_return={float(row.get('predicted_return') or 0.0):.4f}", axis=1)
            latest["risk_level"] = np.where(latest["confidence"] > 0.7, "low", np.where(latest["confidence"] > 0.4, "medium", "high"))

        latest["latest_sentiment"] = latest.get("sentiment_same_day", 0.0)

    latest = _add_news_adjusted_predictions(latest)

    latest["current_price"] = pd.to_numeric(latest.get("close"), errors="coerce")
    latest["predicted_price"] = pd.to_numeric(latest.get("predicted_price"), errors="coerce")
    price_ts = pd.to_datetime(latest.get("date"), errors="coerce")
    latest["price_as_of"] = price_ts.dt.strftime("%Y-%m-%d")
    latest["price_as_of_time"] = price_ts.dt.strftime("%H:%M:%S")
    latest["live_price"] = np.nan
    latest["live_price_as_of"] = None
    latest["live_price_as_of_time"] = None
    latest["live_price_source"] = None
    if "interval" not in latest.columns:
        latest["interval"] = str(settings.historical_interval)
    if "prediction_time" not in latest.columns:
        latest["prediction_time"] = prediction_time.isoformat() if prediction_time else None
    if "probability" not in latest.columns:
        latest["probability"] = latest.get("prob_up")
    if "signal" not in latest.columns:
        latest["signal"] = np.where(latest["decision"].astype(str) == "BUY_CANDIDATE", "bullish", "watch")
    if "reason" not in latest.columns:
        latest["reason"] = ""
    if "risk_level" not in latest.columns:
        latest["risk_level"] = np.where(latest["confidence"] > 0.7, "low", np.where(latest["confidence"] > 0.4, "medium", "high"))
    latest["model_version"] = model_version

    if include_live_quote:
        polygon_map = {
            sym: _fetch_live_quote_polygon(
                sym,
                settings.polygon_api_key,
                settings.request_timeout_sec,
            )
            for sym in latest["symbol"].astype(str).tolist()
        }
        quote_map = {
            sym: _fetch_live_quote_alpha_vantage(
                sym,
                settings.alpha_vantage_api_key,
                settings.request_timeout_sec,
            )
            for sym in latest["symbol"].astype(str).tolist()
        }

        def _choose_live_price(symbol: str):
            poly = polygon_map.get(str(symbol), (None, None, None, None))
            if poly[0] is not None:
                return poly
            quote = quote_map.get(str(symbol), (None, None, None, None))
            if quote[0] is not None:
                return quote
            return _fetch_live_quote_google_finance(str(symbol), settings.request_timeout_sec)

        latest["live_price"] = latest["symbol"].map(lambda s: _choose_live_price(s)[0])
        latest["live_price_as_of"] = latest["symbol"].map(lambda s: _choose_live_price(s)[1])
        latest["live_price_as_of_time"] = latest["symbol"].map(lambda s: _choose_live_price(s)[2])
        latest["live_price_source"] = latest["symbol"].map(lambda s: _choose_live_price(s)[3])

    expected_move = pd.Series(pd.to_numeric(latest["predicted_return"], errors="coerce"), index=latest.index).fillna(0.0).clip(lower=-0.20, upper=0.20)
    latest["target_price"] = (latest["current_price"] * (1.0 + expected_move)).clip(lower=0.0)
    latest["predicted_price"] = latest["predicted_price"].fillna(latest["target_price"])
    news_expected_move = _series_or_default(latest, "news_adjusted_predicted_return", 0.0).clip(lower=-0.20, upper=0.20)
    latest["news_adjusted_target_price"] = (latest["current_price"] * (1.0 + news_expected_move)).clip(lower=0.0)

    atr = pd.to_numeric(latest.get("atr_14"), errors="coerce") if "atr_14" in latest.columns else pd.Series(np.nan, index=latest.index)
    risk_mult = max(0.25, float(atr_multiplier))
    fallback_stop_distance = (latest["current_price"] * np.maximum(expected_move.abs() * 0.5, 0.01) * risk_mult).clip(lower=0.01)
    stop_distance = np.where(np.isfinite(atr) & (atr > 0), atr * risk_mult, fallback_stop_distance)
    direction = np.where(expected_move >= 0, 1.0, -1.0)
    latest["stop_loss_price"] = (latest["current_price"] - direction * stop_distance).clip(lower=0.0)

    out = latest[
        [
            "symbol",
            "interval",
            "prediction_time",
            "price_as_of",
            "price_as_of_time",
            "current_price",
            "predicted_price",
            "live_price",
            "live_price_as_of",
            "live_price_as_of_time",
            "live_price_source",
            "target_price",
            "news_adjusted_target_price",
            "stop_loss_price",
            "probability",
            "prob_up",
            "news_adjusted_prob_up",
            "predicted_return",
            "news_adjusted_predicted_return",
            "confidence",
            "news_adjusted_confidence",
            "signal",
            "reason",
            "risk_level",
            "model_version",
            "latest_sentiment",
            "news_count",
            "news_impact_score",
            "news_signal_score",
            "news_probability_boost",
            "decision",
            "news_decision",
            "news_reason",
        ]
    ].sort_values(
        "confidence", ascending=False
    )

    out["model_version"] = model_version

    if persist_output:
        save_prediction_outputs(out, settings=settings, model_version=model_version)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=False)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="xgboost_classifier")
    parser.add_argument("--atr-multiplier", type=float, default=1.0)
    parser.add_argument("--include-live-quote", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    symbols = args.symbols or settings.default_symbols
    pred = predict_for_symbols(
        symbols=symbols,
        model_name=args.model_name,
        horizon_days=args.horizon_days,
        atr_multiplier=args.atr_multiplier,
        include_live_quote=args.include_live_quote,
    )
    ranked = rank_predictions(pred, top_n=args.top_n)
    print(ranked.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
