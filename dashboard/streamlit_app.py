from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st
import threading
import uuid
from datetime import datetime
import time

# Streamlit Cloud may execute this file with cwd at `dashboard/`, which can
# hide repo-root packages (`app/`, `src/`). Ensure repo root is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Global storage for background runs. Background threads must not access
# `st.session_state` directly (it's not thread-safe). Store mutable run
# records here and keep a reference in `st.session_state` for the UI.
GLOBAL_TRAIN_PRED_RUNS: dict = {}
GLOBAL_TRAIN_PRED_LOCK = threading.Lock()

from app.core.constants import OPENAI_STOCK_MODEL_ALIASES_SET
from app.core.config import get_settings
from app.services.data_service import data_service
from app.services.news_service import news_service
from app.services.prediction_service import prediction_service
from dashboard.dataset_creator import create_dataset
from dashboard.pipeline import run_full_pipeline
from src.data.db import SQLiteDataStore
from src.training.pipeline import run_model_train_pipeline
from src.utils.symbols import load_symbols_from_csv

OPENAI_PREDICTION_MODEL_OPTIONS = [
    "openai_stock_llm_fast",
    "openai_stock_llm",
    "openai_stock_llm_search",
    "openai_stock_llm_cheap",
]


def _parse_symbols(raw: str) -> list[str]:
    parts = [item.strip().upper() for item in str(raw or "").split(",")]
    return [item for item in parts if item]


def _load_all_symbols_from_csv(csv_path: str, symbol_column: str, market: str, series_filter: str | None) -> list[str]:
    if not Path(csv_path).exists():
        return []
    try:
        return load_symbols_from_csv(
            csv_path=csv_path,
            symbol_column=symbol_column,
            market=market,
            series_filter=series_filter,
        )
    except Exception:
        return []


def _filter_symbol_options(options: list[str], query: str, limit: int = 600) -> list[str]:
    all_options = [str(item).strip().upper() for item in options if str(item).strip()]
    if not all_options:
        return []
    q = str(query or "").strip().upper()
    if not q:
        return all_options[: int(limit)]

    starts_with = [sym for sym in all_options if sym.startswith(q)]
    contains = [sym for sym in all_options if q in sym and not sym.startswith(q)]
    merged = starts_with + contains
    return merged[: int(limit)]


def _latest_candle_snapshot(symbols: list[str], interval: str) -> list[dict[str, str]]:
    settings = get_settings()
    store = SQLiteDataStore(settings.db_path)
    rows: list[dict[str, str]] = []
    for sym in symbols:
        symbol = str(sym).strip().upper()
        if not symbol:
            continue
        latest = store.latest_candle_time(symbol=symbol, interval=interval)
        rows.append({"symbol": symbol, "latest_candle_time": str(latest or "")})
    return rows


def _prediction_model_available(model_name: str, settings) -> bool:
    name = str(model_name or "").strip().lower()
    if not name:
        return False
    if name in OPENAI_STOCK_MODEL_ALIASES_SET:
        return bool(settings.openai_predict_enabled)
    if name in {"movement", "movement_model"}:
        return bool(settings.movement_model_path.exists())

    latest_path = settings.model_dir / f"{name}_latest.joblib"
    if latest_path.exists():
        return True
    return any(settings.model_dir.glob(f"{name}_*.joblib"))


def _resolve_prediction_model_with_fallback(requested_model: str, settings) -> tuple[str | None, str | None]:
    requested = str(requested_model or "").strip()
    if _prediction_model_available(requested, settings):
        return requested, None

    candidates = ["movement_model", "xgboost_classifier", "openai_stock_llm_fast", "openai_stock_llm"]
    seen: set[str] = {requested.lower()}
    for candidate in candidates:
        if candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        if _prediction_model_available(candidate, settings):
            return candidate, f"Requested model '{requested}' was unavailable; using '{candidate}' fallback."

    return None, f"No prediction model artifacts available for '{requested}'."


def _resolve_prediction_symbols(
    use_trending: bool,
    use_all_symbols: bool,
    symbols_csv: str,
    symbol_column: str,
    market: str,
    series_filter: str | None,
    selected_symbols: list[str],
    symbols_text: str,
) -> tuple[list[str], bool, str]:
    if use_trending:
        try:
            from app.services.trending_symbols_service import trending_symbols_service

            symbols, source = trending_symbols_service.get_trending_symbols(limit=25, fallback_to_defaults=True)
            return symbols, True, f"trending ({source})"
        except Exception as exc:
            return [], True, f"trending fetch failed: {exc}"

    if use_all_symbols:
        symbols = _load_all_symbols_from_csv(
            csv_path=symbols_csv,
            symbol_column=symbol_column or "Symbol",
            market=market,
            series_filter=series_filter,
        )
        return symbols, False, "csv"

    symbols = selected_symbols or _parse_symbols(symbols_text)
    return symbols, False, "manual"


def render_app() -> None:
    st.set_page_config(page_title="Dataset Creator", layout="wide")
    st.title("Dataset Creator")
    st.caption("Scrape public RSS feeds and build a simple dataset.")

    symbol_limit = st.number_input("Limit symbols (0 = all)", min_value=0, value=0, step=1)
    feed_limit = st.number_input("Number of RSS feeds to use", min_value=1, max_value=10, value=3, step=1)
    items_per_feed = st.number_input("Items per feed to scan", min_value=1, max_value=500, value=50, step=1)
    download_history = st.checkbox("Download historical price data and store in DB", value=False)
    if download_history:
        lookback_days = st.number_input("Historical lookback days", min_value=1, value=365, step=1)
        interval = st.text_input("Price interval (e.g. 1d, 1h)", value="1d")
    else:
        lookback_days = 365
        interval = "1d"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create raw dataset"):
            with st.spinner("Scraping feeds and building dataset..."):
                limit = int(symbol_limit) if int(symbol_limit) > 0 else None
                res = create_dataset(symbol_limit=limit, feed_limit=int(feed_limit), items_per_feed=int(items_per_feed))
            st.success(f"Raw dataset saved: {res.get('path')} ({res.get('rows')} rows)")
            st.write(res.get('path'))

    with col2:
        if st.button("Run full pipeline"):
            status_box = st.empty()
            log_box = st.empty()
            live_logs: list[str] = []

            def _on_progress(message: str) -> None:
                live_logs.append(str(message))
                status_box.info(f"Current process: {message}")
                log_box.code("\n".join(live_logs[-30:]), language="text")

            with st.spinner("Running pipeline: scraping → processing..."):
                limit = int(symbol_limit) if int(symbol_limit) > 0 else None
                out = run_full_pipeline(
                    symbol_limit=limit,
                    feed_limit=int(feed_limit),
                    items_per_feed=int(items_per_feed),
                    download_history=bool(download_history),
                    lookback_days=int(lookback_days),
                    interval=str(interval),
                    progress_callback=_on_progress,
                )
            raw = out.get('raw') or {}
            proc = out.get('processed') or {}
            hist = out.get('historical') or {}
            logs = out.get('logs') or live_logs
            st.success("Pipeline completed")
            st.write("Raw CSV:", raw.get('path'))
            st.write("Raw rows:", raw.get('rows'))
            st.write("Processed CSV:", proc.get('csv_path'))
            st.write("Processed rows:", proc.get('rows'))
            if download_history:
                st.write("Historical ingest summary:")
                st.json(hist)
            st.write("Pipeline logs:")
            st.code("\n".join(str(line) for line in logs), language="text")

    st.divider()
    st.subheader("Smart Stock Pipeline")
    st.caption("Search and multi-select stocks, scrape broad news sentiment, refresh latest historical data, train, and predict in one run.")

    settings = get_settings()

    smart_left, smart_right = st.columns(2)
    with smart_left:
        smart_symbols_csv = st.text_input(
            "Symbols CSV (smart workflow)",
            value="sec_list.csv",
            key="smart_symbols_csv",
        ).strip()
        smart_symbol_column = st.text_input("Symbol column", value="Symbol", key="smart_symbol_column").strip()
        smart_market = st.selectbox("Market", options=["us", "india"], index=1, key="smart_market")
        smart_series_filter = st.text_input("Series filter (optional)", value="EQ", key="smart_series_filter").strip()
        smart_search_text = st.text_input(
            "Search symbol",
            value="",
            key="smart_symbol_search",
            placeholder="Type ticker/company symbol text to narrow options",
        ).strip()

        smart_all_options = _load_all_symbols_from_csv(
            csv_path=smart_symbols_csv,
            symbol_column=smart_symbol_column or "Symbol",
            market=smart_market,
            series_filter=smart_series_filter or None,
        )
        if not smart_all_options:
            smart_all_options = [str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()]

        if "smart_selected_symbols" not in st.session_state:
            st.session_state["smart_selected_symbols"] = smart_all_options[: min(10, len(smart_all_options))]
        else:
            st.session_state["smart_selected_symbols"] = [
                sym for sym in st.session_state["smart_selected_symbols"] if sym in smart_all_options
            ]

        smart_filtered_options = _filter_symbol_options(smart_all_options, smart_search_text, limit=600)
        smart_merged_options = sorted(set(smart_filtered_options + st.session_state.get("smart_selected_symbols", [])))

        st.caption(f"Matches: {len(smart_filtered_options)} | Total symbols loaded: {len(smart_all_options)}")
        action_col1, action_col2 = st.columns(2)
        if action_col1.button("Select all matches", key="smart_select_all_matches", use_container_width=True):
            st.session_state["smart_selected_symbols"] = smart_filtered_options
            st.rerun()
        if action_col2.button("Clear selected", key="smart_clear_selected", use_container_width=True):
            st.session_state["smart_selected_symbols"] = []
            st.rerun()

        smart_selected_symbols = st.multiselect(
            "Select stock(s)",
            options=smart_merged_options,
            key="smart_selected_symbols",
            help="This dropdown supports search. Use the search box above to quickly narrow a very large universe.",
        )
        st.caption(f"Selected stocks: {len(smart_selected_symbols)}")

    with smart_right:
        smart_news_limit = st.number_input(
            "News items per symbol",
            min_value=3,
            max_value=300,
            value=int(settings.max_news_items_per_symbol),
            step=5,
            key="smart_news_limit",
        )
        smart_interval = st.text_input("Historical interval", value=str(settings.historical_interval), key="smart_interval").strip()
        smart_lookback_days = st.number_input(
            "Historical lookback days",
            min_value=30,
            max_value=7300,
            value=int(settings.historical_lookback_days),
            step=30,
            key="smart_lookback_days",
        )
        smart_task_type = st.selectbox(
            "Training task",
            options=["classification", "regression_return", "regression_close", "movement"],
            index=0,
            key="smart_task_type",
        )
        smart_train_horizon = st.number_input(
            "Training horizon (days)",
            min_value=1,
            max_value=30,
            value=1,
            step=1,
            key="smart_train_horizon",
        )
        smart_pred_model = st.selectbox(
            "Prediction model",
            options=["xgboost_classifier", "movement_model"] + OPENAI_PREDICTION_MODEL_OPTIONS,
            index=0,
            key="smart_pred_model",
        )
        smart_pred_horizon = st.number_input(
            "Prediction horizon (days)",
            min_value=1,
            max_value=30,
            value=1,
            step=1,
            key="smart_pred_horizon",
        )
        smart_include_live_quote = st.checkbox("Include live quote in prediction", value=True, key="smart_include_live_quote")

    if st.button("Run Smart Pipeline (News -> Historical -> Train -> Predict)", type="primary", use_container_width=True):
        chosen_symbols = [str(sym).strip().upper() for sym in smart_selected_symbols if str(sym).strip()]
        if not chosen_symbols:
            st.error("Please select at least one symbol.")
            return

        if len(chosen_symbols) > 300:
            st.warning("Large batch selected. This may take a long time and can hit provider limits.")

        smart_effective_task = smart_task_type
        if smart_pred_model == "movement_model" and smart_task_type != "movement":
            smart_effective_task = "movement"
            st.info("Task auto-switched to 'movement' so training matches the selected prediction model.")

        status_box = st.empty()
        logs_box = st.empty()
        smart_logs: list[str] = []

        def _smart_log(message: str) -> None:
            smart_logs.append(str(message))
            status_box.info(str(message))
            logs_box.code("\n".join(smart_logs[-40:]), language="text")

        train_result: dict | None = None
        prediction_result: dict | None = None
        training_error: str | None = None
        prediction_error: str | None = None
        effective_prediction_model = str(smart_pred_model)

        with st.spinner("Running smart stock pipeline..."):
            _smart_log("Step 1/4: Scraping news for selected symbols")
            news_summary = news_service.ingest_news(chosen_symbols, int(smart_news_limit))

            _smart_log("Step 2/4: Refreshing latest historical price data")
            historical_summary = data_service.ingest_historical(
                symbols=chosen_symbols,
                interval=smart_interval or str(settings.historical_interval),
                lookback_days=int(smart_lookback_days),
            )
            latest_snapshot = _latest_candle_snapshot(
                symbols=chosen_symbols,
                interval=smart_interval or str(settings.historical_interval),
            )

            _smart_log("Step 3/4: Training model with sentiment + technical features")
            try:
                train_result = run_model_train_pipeline(
                    symbols=chosen_symbols,
                    symbols_csv=None,
                    symbol_column=None,
                    market=smart_market,
                    series_filter=smart_series_filter or None,
                    max_symbols=None,
                    horizon_days=int(smart_train_horizon),
                    task_type=smart_effective_task,
                    ingest_first=False,
                    lookback_days=int(smart_lookback_days),
                    interval=smart_interval or None,
                )
            except Exception as exc:
                training_error = str(exc)
                _smart_log(f"Step 3/4: Training warning - {training_error}")

            _smart_log("Step 4/4: Generating price predictions")
            resolved_model, fallback_reason = _resolve_prediction_model_with_fallback(smart_pred_model, settings)
            if fallback_reason:
                _smart_log(f"Step 4/4: {fallback_reason}")
            if resolved_model is None:
                prediction_error = fallback_reason or f"No artifacts found for model {smart_pred_model}"
                _smart_log(f"Step 4/4: Prediction failed - {prediction_error}")
            else:
                effective_prediction_model = str(resolved_model)
                try:
                    prediction_result = prediction_service.predict(
                        symbols=chosen_symbols,
                        model_name=effective_prediction_model,
                        horizon_days=int(smart_pred_horizon),
                        include_live_quote=bool(smart_include_live_quote),
                        use_trending=False,
                    )
                except Exception as exc:
                    prediction_error = str(exc)
                    _smart_log(f"Step 4/4: Prediction failed - {prediction_error}")

        predictions = (prediction_result or {}).get("predictions") or []
        if prediction_error:
            status_box.error("Smart pipeline finished with prediction error.")
        elif training_error:
            status_box.warning("Smart pipeline finished with training warning. Predictions used available model artifacts.")
        else:
            status_box.success("Smart pipeline completed successfully.")

        if training_error:
            st.warning(f"Training step warning: {training_error}")
            if "Not enough rows for time-series validation" in training_error:
                st.info(
                    "Not enough historical rows for time-series validation. "
                    "Try increasing lookback days, selecting more liquid symbols, or selecting multiple stocks."
                )
        if prediction_error:
            st.error(f"Prediction step failed: {prediction_error}")

        if predictions:
            import pandas as pd

            pred_df = pd.DataFrame(predictions)
            preferred_cols = [
                "symbol",
                "current_price",
                "predicted_price",
                "target_price",
                "stop_loss_price",
                "prob_up",
                "predicted_return",
                "confidence",
                "decision",
                "news_decision",
                "news_signal_score",
            ]
            show_cols = [c for c in preferred_cols if c in pred_df.columns]
            st.dataframe(pred_df[show_cols] if show_cols else pred_df, use_container_width=True, height=380)

        st.json(
            {
                "symbols_count": len(chosen_symbols),
                "symbols": chosen_symbols,
                "news_ingest_summary": news_summary,
                "historical_ingest_summary": historical_summary,
                "latest_historical_snapshot": latest_snapshot,
                "trained": {
                    "status": "ok" if train_result else ("warning" if training_error else "unknown"),
                    "task_type": str((train_result or {}).get("task_type") or smart_effective_task),
                    "version": str((train_result or {}).get("version") or ""),
                    "best": ((train_result or {}).get("result") or {}).get("best") or {},
                    "error": training_error,
                },
                "predicted": {
                    "status": "ok" if not prediction_error else "failed",
                    "model_name": effective_prediction_model,
                    "rows": len(predictions),
                    "symbols_used": (prediction_result or {}).get("symbols_used") or [],
                    "error": prediction_error,
                },
            }
        )

    st.divider()
    st.subheader("Model Train Pipeline")
    st.caption("Run end-to-end model training from Streamlit.")

    tp_col1, tp_col2 = st.columns(2)
    with tp_col1:
        use_all_symbols = st.checkbox(
            "Train all symbols from CSV",
            value=True,
            help="When enabled, symbols are loaded from the CSV below and manual symbol input is ignored.",
        )
        symbol_text = st.text_input(
            "Symbols (comma-separated)",
            value="RELIANCE,TCS,INFY",
            help="Used when Symbols CSV is empty.",
            disabled=bool(use_all_symbols),
        )
        symbols_csv = st.text_input(
            "Symbols CSV path",
            value="sec_list.csv",
            help="Example: sec_list.csv",
        ).strip()
        symbol_column = st.text_input("Symbol column", value="Symbol").strip()
        market = st.selectbox("Market", options=["us", "india"], index=1)
        series_filter = st.text_input("Series filter (optional)", value="EQ").strip()
        max_symbols = st.number_input("Max symbols (0 = no limit)", min_value=0, value=0, step=1)

    with tp_col2:
        task_type = st.selectbox(
            "Task type",
            options=["classification", "regression_return", "regression_close", "movement"],
            index=0,
        )
        horizon_days = st.number_input("Horizon days", min_value=1, value=1, step=1)
        ingest_first = st.checkbox("Ingest historical data before training", value=False)
        lookback_days = st.number_input("Lookback days", min_value=1, value=3650, step=1)
        interval = st.text_input("Interval", value="1d").strip()

    if st.button("Run Model Train Pipeline", use_container_width=True):
        symbols_csv_value = symbols_csv or None
        symbols_value = [] if use_all_symbols else _parse_symbols(symbol_text)

        if use_all_symbols and symbols_csv_value and not Path(symbols_csv_value).exists():
            st.error(f"Symbols CSV not found: {symbols_csv_value}")
            return

        with st.spinner("Running training pipeline..."):
            result = run_model_train_pipeline(
                symbols=symbols_value,
                symbols_csv=symbols_csv_value,
                symbol_column=symbol_column or None,
                market=market,
                series_filter=series_filter or None,
                max_symbols=(int(max_symbols) if int(max_symbols) > 0 else None),
                horizon_days=int(horizon_days),
                task_type=task_type,
                ingest_first=bool(ingest_first),
                lookback_days=int(lookback_days),
                interval=interval or None,
            )
        st.success("Training pipeline completed")
        st.json(result)

    st.divider()
    st.subheader("Model Predict Pipeline")
    st.caption("Generate trend and price predictions from trained models.")

    # Background run state: expose the module-global runs dict to session_state
    if "train_pred_runs" not in st.session_state:
        st.session_state["train_pred_runs"] = GLOBAL_TRAIN_PRED_RUNS
    if "train_pred_active" not in st.session_state:
        st.session_state["train_pred_active"] = None

    pp_col1, pp_col2 = st.columns(2)
    with pp_col1:
        pred_use_all_symbols = st.checkbox(
            "Predict all symbols from CSV",
            value=True,
            help="When enabled, symbols are loaded from CSV and manual symbol input is ignored.",
        )
        pred_symbols_text = st.text_input(
            "Prediction symbols (comma-separated)",
            value="RELIANCE,TCS,INFY",
            disabled=bool(pred_use_all_symbols),
        )
        pred_symbols_csv = st.text_input("Prediction symbols CSV path", value="sec_list.csv").strip()
        pred_symbol_column = st.text_input("Prediction symbol column", value="Symbol").strip()
        pred_market = st.selectbox("Prediction market", options=["us", "india"], index=1)
        pred_series_filter = st.text_input("Prediction series filter (optional)", value="EQ").strip()

        dropdown_options = _load_all_symbols_from_csv(
            csv_path=pred_symbols_csv,
            symbol_column=pred_symbol_column or "Symbol",
            market=pred_market,
            series_filter=pred_series_filter or None,
        )
        pred_selected_symbols = st.multiselect(
            "Select stock(s) from dropdown",
            options=dropdown_options,
            default=[] if pred_use_all_symbols else dropdown_options[:5],
            disabled=bool(pred_use_all_symbols),
            help="Used for prediction when 'Predict all symbols from CSV' is OFF.",
        )

    with pp_col2:
        pred_model_name = st.selectbox(
            "Prediction model",
            options=["movement_model", "xgboost_classifier"] + OPENAI_PREDICTION_MODEL_OPTIONS,
            index=0,
        )
        pred_horizon_days = st.number_input("Prediction horizon days", min_value=1, max_value=10, value=1, step=1)
        pred_include_live_quote = st.checkbox("Include live quote (slower)", value=True)
        pred_use_trending = st.checkbox("Use trending symbols from news", value=False)

    if st.button("Run Prediction Pipeline", use_container_width=True):
        pred_symbols, used_trending, symbol_source = _resolve_prediction_symbols(
            use_trending=bool(pred_use_trending),
            use_all_symbols=bool(pred_use_all_symbols),
            symbols_csv=pred_symbols_csv,
            symbol_column=pred_symbol_column,
            market=pred_market,
            series_filter=pred_series_filter or None,
            selected_symbols=pred_selected_symbols,
            symbols_text=pred_symbols_text,
        )
        if not pred_symbols:
            if used_trending:
                st.error(f"Unable to load trending symbols. Detail: {symbol_source}")
            elif symbol_source == "csv":
                st.error(f"Prediction symbols CSV not found or empty: {pred_symbols_csv}")
            else:
                st.error("Please provide at least one prediction symbol.")
            return

        with st.spinner("Running prediction pipeline..."):
            result = prediction_service.predict(
                symbols=pred_symbols,
                model_name=pred_model_name,
                horizon_days=int(pred_horizon_days),
                include_live_quote=bool(pred_include_live_quote),
                use_trending=bool(pred_use_trending),
            )

        st.success("Prediction pipeline completed")
        predictions = result.get("predictions") or []
        if predictions:
            import pandas as pd

            pred_df = pd.DataFrame(predictions)
            preferred_cols = [
                "symbol",
                "current_price",
                "predicted_price",
                "target_price",
                "stop_loss_price",
                "prob_up",
                "predicted_return",
                "confidence",
                "decision",
                "signal",
                "news_decision",
            ]
            show_cols = [c for c in preferred_cols if c in pred_df.columns]
            st.dataframe(pred_df[show_cols] if show_cols else pred_df, use_container_width=True, height=360)
        st.json(
            {
                "generated_at": str(result.get("generated_at") or ""),
                "symbols_used": result.get("symbols_used") or [],
                "is_trending": bool(result.get("is_trending")),
                "rows": len(predictions),
            }
        )

    def _start_background_train_predict(symbols: list[str], task_type: str, horizon: int, model_name: str, include_live: bool, ingest_first_flag: bool, lookback: int, interval_val: str) -> str:
        run_id = str(uuid.uuid4())[:8]
        # Create run entry in module-global storage (thread-safe via lock)
        with GLOBAL_TRAIN_PRED_LOCK:
            GLOBAL_TRAIN_PRED_RUNS[run_id] = {
                "run_id": run_id,
                "status": "queued",
                "started_at": None,
                "finished_at": None,
                "message": "Queued",
                "symbols": symbols,
                "train_result": None,
                "prediction_result": None,
                # per-symbol progress and aggregated predictions
                "per_symbol": {},
                "aggregated_predictions": [],
                "current_index": 0,
                "error": None,
            }

        def _worker():
            try:
                with GLOBAL_TRAIN_PRED_LOCK:
                    entry = GLOBAL_TRAIN_PRED_RUNS[run_id]
                    entry["status"] = "running"
                    entry["started_at"] = datetime.utcnow().isoformat()
                    entry["message"] = "Training symbols"

                # Process symbols one-by-one so UI can preview completed ones
                total = len(symbols)
                for idx, sym in enumerate(symbols, start=1):
                    with GLOBAL_TRAIN_PRED_LOCK:
                        GLOBAL_TRAIN_PRED_RUNS[run_id]["current_index"] = idx
                        GLOBAL_TRAIN_PRED_RUNS[run_id]["message"] = f"Processing {idx}/{total}: {sym}"

                    try:
                        # Train for single symbol (limits training to that symbol)
                        per_train = run_model_train_pipeline(
                            symbols=[sym],
                            symbols_csv=None,
                            symbol_column=None,
                            market="india",
                            series_filter=None,
                            max_symbols=1,
                            horizon_days=horizon,
                            task_type=task_type,
                            ingest_first=ingest_first_flag,
                            lookback_days=lookback,
                            interval=interval_val,
                        )

                        with GLOBAL_TRAIN_PRED_LOCK:
                            GLOBAL_TRAIN_PRED_RUNS[run_id]["per_symbol"][sym] = {"train_result": per_train, "status": "trained"}

                        # Predict for this symbol and append to aggregated list
                        per_pred = prediction_service.predict(
                            symbols=[sym],
                            model_name=model_name,
                            horizon_days=horizon,
                            include_live_quote=include_live,
                        )

                        preds = per_pred.get("predictions") or []
                        with GLOBAL_TRAIN_PRED_LOCK:
                            GLOBAL_TRAIN_PRED_RUNS[run_id]["per_symbol"][sym]["prediction_result"] = per_pred
                            GLOBAL_TRAIN_PRED_RUNS[run_id]["per_symbol"][sym]["status"] = "predicted"
                            # append rows with a run marker
                            for r in preds:
                                r["_run_id"] = run_id
                            GLOBAL_TRAIN_PRED_RUNS[run_id]["aggregated_predictions"].extend(preds)

                    except Exception as inner_exc:
                        with GLOBAL_TRAIN_PRED_LOCK:
                            GLOBAL_TRAIN_PRED_RUNS[run_id]["per_symbol"][sym] = {"status": "failed", "error": str(inner_exc)}

                with GLOBAL_TRAIN_PRED_LOCK:
                    GLOBAL_TRAIN_PRED_RUNS[run_id]["train_result"] = {"task_type": task_type, "symbols_processed": total}
                    GLOBAL_TRAIN_PRED_RUNS[run_id]["prediction_result"] = {"symbols_processed": total, "rows": len(GLOBAL_TRAIN_PRED_RUNS[run_id]["aggregated_predictions"])}
                    GLOBAL_TRAIN_PRED_RUNS[run_id]["status"] = "completed"
                    GLOBAL_TRAIN_PRED_RUNS[run_id]["message"] = "Completed"
                    GLOBAL_TRAIN_PRED_RUNS[run_id]["finished_at"] = datetime.utcnow().isoformat()
            except Exception as exc:
                with GLOBAL_TRAIN_PRED_LOCK:
                    entry = GLOBAL_TRAIN_PRED_RUNS.get(run_id, {})
                    entry["status"] = "failed"
                    entry["error"] = str(exc)
                    entry["message"] = "Failed: " + str(exc)
                    entry["finished_at"] = datetime.utcnow().isoformat()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        # expose active run id to session state so UI can highlight it
        st.session_state["train_pred_active"] = run_id
        return run_id

    st.markdown("---")
    st.subheader("Background Train + Predict")
    st.caption("Run training and prediction in background and monitor status.")
    bg_col1, bg_col2 = st.columns([3, 1])
    with bg_col1:
        bg_start = st.button("Start Background Train+Predict")
    with bg_col2:
        bg_refresh = st.button("Refresh Status")

    if bg_start:
        # resolve symbols same as sync flow
        sync_symbols, used_trending, symbol_source = _resolve_prediction_symbols(
            use_trending=bool(pred_use_trending),
            use_all_symbols=bool(pred_use_all_symbols),
            symbols_csv=pred_symbols_csv,
            symbol_column=pred_symbol_column,
            market=pred_market,
            series_filter=pred_series_filter or None,
            selected_symbols=pred_selected_symbols,
            symbols_text=pred_symbols_text,
        )
        if not sync_symbols:
            st.error("No symbols resolved for background run")
        else:
            run_id = _start_background_train_predict(
                symbols=sync_symbols,
                task_type=task_type,
                horizon=int(pred_horizon_days),
                model_name=pred_model_name,
                include_live=bool(pred_include_live_quote),
                ingest_first_flag=bool(ingest_first),
                lookback=int(lookback_days),
                interval_val=interval,
            )
            st.success(f"Background run started: {run_id}")

    # Status panel
    runs = st.session_state.get("train_pred_runs") or {}
    active = st.session_state.get("train_pred_active")
    if active and active in runs:
        info = runs[active]
        st.markdown(f"**Active run**: {active} — Status: {info.get('status')} | Message: {info.get('message')}")
        # progress
        cur = int(info.get("current_index") or 0)
        total = len(info.get("symbols") or [])
        if total:
            st.progress(min(cur / total, 1.0))
            st.caption(f"Processed {cur}/{total} symbols")

        # per-symbol status table
        per = info.get("per_symbol") or {}
        if per:
            import pandas as pd

            rows = []
            for s, v in per.items():
                rows.append({"symbol": s, "status": v.get("status"), "error": v.get("error")})
            st.markdown("**Per-symbol progress**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=180)

        # aggregated prediction preview
        ag = info.get("aggregated_predictions") or []
        if ag:
            import pandas as pd

            st.markdown("**Aggregated prediction preview (top rows)**")
            st.dataframe(pd.DataFrame(ag).head(10), use_container_width=True, height=280)
    elif runs:
        # show last 3 runs
        for rid, entry in list(runs.items())[-3:]:
            st.markdown(f"- {rid}: {entry.get('status')} — {entry.get('message')}")

    if st.button("Run Training + Prediction (Sync)", use_container_width=True, type="primary"):
        sync_symbols, used_trending, symbol_source = _resolve_prediction_symbols(
            use_trending=bool(pred_use_trending),
            use_all_symbols=bool(pred_use_all_symbols),
            symbols_csv=pred_symbols_csv,
            symbol_column=pred_symbol_column,
            market=pred_market,
            series_filter=pred_series_filter or None,
            selected_symbols=pred_selected_symbols,
            symbols_text=pred_symbols_text,
        )
        if not sync_symbols:
            if used_trending:
                st.error(f"Unable to load trending symbols. Detail: {symbol_source}")
            elif symbol_source == "csv":
                st.error(f"Symbols CSV not found or empty: {pred_symbols_csv}")
            else:
                st.error("Please provide at least one symbol for sync run.")
            return

        sync_status = st.empty()
        sync_task_type = task_type
        if pred_model_name == "movement_model" and task_type != "movement":
            sync_task_type = "movement"
            st.info("For sync run, task type switched to 'movement' to match selected prediction model 'movement_model'.")

        sync_status.info("Step 1/2: Training model...")
        with st.spinner("Training and prediction are running synchronously..."):
            train_result = run_model_train_pipeline(
                symbols=sync_symbols,
                symbols_csv=None,
                symbol_column=None,
                market=market,
                series_filter=series_filter or None,
                max_symbols=(int(max_symbols) if int(max_symbols) > 0 else None),
                horizon_days=int(horizon_days),
                task_type=sync_task_type,
                ingest_first=bool(ingest_first),
                lookback_days=int(lookback_days),
                interval=interval or None,
            )

            sync_status.info("Step 2/2: Generating predictions...")
            prediction_result = prediction_service.predict(
                symbols=sync_symbols,
                model_name=pred_model_name,
                horizon_days=int(pred_horizon_days),
                include_live_quote=bool(pred_include_live_quote),
                use_trending=False,
            )

        sync_status.success("Synchronous train + prediction run completed")
        predictions = prediction_result.get("predictions") or []
        if predictions:
            import pandas as pd

            pred_df = pd.DataFrame(predictions)
            preferred_cols = [
                "symbol",
                "current_price",
                "predicted_price",
                "target_price",
                "stop_loss_price",
                "prob_up",
                "predicted_return",
                "confidence",
                "decision",
                "signal",
                "news_decision",
            ]
            show_cols = [c for c in preferred_cols if c in pred_df.columns]
            st.dataframe(pred_df[show_cols] if show_cols else pred_df, use_container_width=True, height=360)

        st.json(
            {
                "symbols_source": symbol_source,
                "symbols_count": len(sync_symbols),
                "trained": {
                    "task_type": str(train_result.get("task_type") or sync_task_type),
                    "version": str(train_result.get("version") or ""),
                    "best": train_result.get("best") or {},
                },
                "predicted": {
                    "model_name": pred_model_name,
                    "rows": len(predictions),
                    "symbols_used": prediction_result.get("symbols_used") or [],
                },
            }
        )


if __name__ == "__main__":
    render_app()
