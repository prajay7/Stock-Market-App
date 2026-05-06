from __future__ import annotations

from pathlib import Path

import streamlit as st
import threading
import uuid
from datetime import datetime
import time

# Global storage for background runs. Background threads must not access
# `st.session_state` directly (it's not thread-safe). Store mutable run
# records here and keep a reference in `st.session_state` for the UI.
GLOBAL_TRAIN_PRED_RUNS: dict = {}
GLOBAL_TRAIN_PRED_LOCK = threading.Lock()

from app.services.prediction_service import prediction_service
from dashboard.dataset_creator import create_dataset
from dashboard.pipeline import run_full_pipeline
from src.training.pipeline import run_model_train_pipeline
from src.utils.symbols import load_symbols_from_csv


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
            options=["movement_model", "xgboost_classifier", "openai_stock_llm"],
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
                "error": None,
            }

        def _worker():
            try:
                with GLOBAL_TRAIN_PRED_LOCK:
                    entry = GLOBAL_TRAIN_PRED_RUNS[run_id]
                    entry["status"] = "running"
                    entry["started_at"] = datetime.utcnow().isoformat()
                    entry["message"] = "Training"

                # Run training (may take a long time)
                train_res = run_model_train_pipeline(
                    symbols=symbols,
                    symbols_csv=None,
                    symbol_column=None,
                    market="india",
                    series_filter=None,
                    max_symbols=None,
                    horizon_days=horizon,
                    task_type=task_type,
                    ingest_first=ingest_first_flag,
                    lookback_days=lookback,
                    interval=interval_val,
                )

                with GLOBAL_TRAIN_PRED_LOCK:
                    GLOBAL_TRAIN_PRED_RUNS[run_id]["train_result"] = train_res
                    GLOBAL_TRAIN_PRED_RUNS[run_id]["message"] = "Predicting"

                # Run prediction
                pred_res = prediction_service.predict(
                    symbols=symbols,
                    model_name=model_name,
                    horizon_days=horizon,
                    include_live_quote=include_live,
                )

                with GLOBAL_TRAIN_PRED_LOCK:
                    GLOBAL_TRAIN_PRED_RUNS[run_id]["prediction_result"] = pred_res
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
        if info.get("train_result"):
            st.markdown("**Train result (summary)**")
            tr = info.get("train_result")
            st.json({"task_type": tr.get("task_type"), "version": tr.get("version"), "best": tr.get("best")})
        if info.get("prediction_result"):
            st.markdown("**Prediction preview (top rows)**")
            import pandas as pd

            rows = info.get("prediction_result", {}).get("predictions") or []
            if rows:
                st.dataframe(pd.DataFrame(rows).head(10), use_container_width=True, height=280)
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
