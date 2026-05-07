from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.constants import OPENAI_STOCK_MODEL_ALIASES_SET
from src.inference.predict import predict_for_symbols, save_prediction_outputs
from src.utils.symbols import load_symbols_from_csv


PREDICTION_UI_LOCK = threading.Lock()
PREDICTION_UI_RUNS: dict[str, dict] = {}
PREDICTION_UI_ACTIVE_RUN_ID: str | None = None
OPENAI_PREDICTION_MODEL_OPTIONS = [
    "openai_stock_llm_fast",
    "openai_stock_llm",
    "openai_stock_llm_search",
    "openai_stock_llm_cheap",
]


def _color_diff(val):
    try:
        v = float(val)
    except Exception:
        return ""
    if pd.isna(v):
        return ""
    if v > 0:
        return "background-color: #e6f7ea; color: #0f5c2e;"
    if v < 0:
        return "background-color: #fdeaea; color: #8a1f1f;"
    return "background-color: #f3f4f6; color: #374151;"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _all_trained_symbols(model_dir, model_name: str) -> list[str]:
    symbols: set[str] = set()
    for path in sorted(model_dir.glob(f"{model_name}_*.metadata.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for symbol in payload.get("symbols", []):
            if symbol:
                symbols.add(str(symbol))
    return sorted(symbols)


def _trained_symbols_by_latest_training(model_dir, model_name: str) -> list[str]:
    latest_by_symbol: dict[str, pd.Timestamp] = {}
    for path in sorted(model_dir.glob(f"{model_name}_*.metadata.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        saved_at_raw = payload.get("saved_at")
        saved_at = pd.to_datetime(saved_at_raw, errors="coerce")
        if pd.isna(saved_at):
            saved_at = pd.Timestamp.min

        for symbol in payload.get("symbols", []):
            if not symbol:
                continue
            symbol_str = str(symbol)
            current = latest_by_symbol.get(symbol_str)
            if current is None or saved_at > current:
                latest_by_symbol[symbol_str] = saved_at

    ordered = sorted(latest_by_symbol.items(), key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in ordered]


def _load_prediction_symbols_from_csv(
    csv_path: str = "sec_list.csv",
    symbol_column: str = "Symbol",
    market: str = "india",
    series_filter: str | None = "EQ",
) -> list[str]:
    try:
        if not Path(csv_path).exists():
            return []
        return load_symbols_from_csv(
            csv_path=csv_path,
            symbol_column=symbol_column,
            market=market,
            series_filter=series_filter,
        )
    except Exception:
        return []


def _format_prediction_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    numeric_formats = {
        "Current Share Price": "{:.2f}",
        "Predicted Price": "{:.2f}",
        "News Predicted Price": "{:.2f}",
        "Target Price": "{:.2f}",
        "Stop Loss": "{:.2f}",
        "Live Price": "{:.2f}",
        "Prediction Return %": "{:+.2f}%",
        "News Return %": "{:+.2f}%",
        "Up Probability %": "{:.1f}%",
        "News Up Probability %": "{:.1f}%",
        "Confidence %": "{:.1f}%",
        "News Confidence %": "{:.1f}%",
        "News Signal": "{:+.3f}",
    }
    formats = {column: fmt for column, fmt in numeric_formats.items() if column in df.columns}
    return df.style.format(formats)


def _prediction_preview_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df

    preferred = [
        "symbol",
        "current_price",
        "predicted_price",
        "news_adjusted_target_price",
        "predicted_return",
        "news_adjusted_predicted_return",
        "prob_up",
        "news_adjusted_prob_up",
        "confidence",
        "news_adjusted_confidence",
        "decision",
        "news_decision",
    ]
    display_df = pred_df[[column for column in preferred if column in pred_df.columns]].copy()
    for column in [
        "predicted_return",
        "news_adjusted_predicted_return",
        "prob_up",
        "news_adjusted_prob_up",
        "confidence",
        "news_adjusted_confidence",
    ]:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce") * 100.0

    return display_df.rename(
        columns={
            "symbol": "Symbol",
            "current_price": "Current Share Price",
            "predicted_price": "Predicted Price",
            "news_adjusted_target_price": "News Predicted Price",
            "predicted_return": "Prediction Return %",
            "news_adjusted_predicted_return": "News Return %",
            "prob_up": "Up Probability %",
            "news_adjusted_prob_up": "News Up Probability %",
            "confidence": "Confidence %",
            "news_adjusted_confidence": "News Confidence %",
            "decision": "Decision",
            "news_decision": "News Decision",
        }
    )


def _combine_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    sort_col = "news_adjusted_confidence" if "news_adjusted_confidence" in combined.columns else "confidence"
    return combined.sort_values(sort_col, ascending=False).reset_index(drop=True)


def _active_prediction_run() -> dict | None:
    with PREDICTION_UI_LOCK:
        if not PREDICTION_UI_ACTIVE_RUN_ID:
            return None
        run = PREDICTION_UI_RUNS.get(PREDICTION_UI_ACTIVE_RUN_ID)
        return run.copy() if run else None


def _start_background_prediction(
    settings,
    symbols: list[str],
    model_name: str,
    horizon_days: int,
    atr_multiplier: float,
    include_live_quote: bool,
) -> tuple[str | None, bool]:
    global PREDICTION_UI_ACTIVE_RUN_ID

    with PREDICTION_UI_LOCK:
        if PREDICTION_UI_ACTIVE_RUN_ID:
            active = PREDICTION_UI_RUNS.get(PREDICTION_UI_ACTIVE_RUN_ID)
            if active and str(active.get("status")) == "running":
                return str(active.get("run_id")), False

        run_id = str(uuid.uuid4())[:8]
        PREDICTION_UI_ACTIVE_RUN_ID = run_id
        PREDICTION_UI_RUNS[run_id] = {
            "run_id": run_id,
            "status": "running",
            "progress": 0,
            "current_step": "Queued",
            "message": "Prediction run queued",
            "started_at": _now_iso(),
            "finished_at": None,
            "model_name": str(model_name),
            "horizon_days": int(horizon_days),
            "include_live_quote": bool(include_live_quote),
            "total_symbols": int(len(symbols)),
            "processed_symbols": 0,
            "current_symbol": None,
            "result_count": 0,
            "preview": [],
            "error": None,
        }

    def _update_run(**updates) -> None:
        with PREDICTION_UI_LOCK:
            run = PREDICTION_UI_RUNS.get(run_id)
            if not run:
                return
            run.update(updates)

    def _worker() -> None:
        try:
            if not symbols:
                _update_run(
                    status="completed",
                    progress=100,
                    current_step="Completed",
                    message="No symbols selected for prediction.",
                    finished_at=_now_iso(),
                )
                return

            frames: list[pd.DataFrame] = []
            for index, symbol in enumerate(symbols, start=1):
                total = max(len(symbols), 1)
                _update_run(
                    progress=int(((index - 1) / total) * 100),
                    current_step=f"Processing {symbol}",
                    message=f"Processing {symbol} ({index}/{total})",
                    current_symbol=str(symbol),
                    processed_symbols=index - 1,
                )
                frame = predict_for_symbols(
                    symbols=[symbol],
                    model_name=model_name,
                    horizon_days=int(horizon_days),
                    atr_multiplier=float(atr_multiplier),
                    include_live_quote=bool(include_live_quote),
                    persist_output=False,
                )
                if not frame.empty:
                    frames.append(frame)

                preview_df = _combine_prediction_frames(frames).head(20)
                _update_run(
                    progress=int((index / total) * 100),
                    current_step="Generating predictions",
                    message=f"Processed {symbol} ({index}/{total})",
                    current_symbol=str(symbol),
                    processed_symbols=index,
                    result_count=int(sum(len(item) for item in frames)),
                    preview=preview_df.to_dict(orient="records"),
                )

            pred_df = _combine_prediction_frames(frames)
            save_prediction_outputs(pred_df, settings=settings)

            _update_run(
                status="completed",
                progress=100,
                current_step="Completed",
                message=f"Generated predictions for {len(pred_df)} stock(s).",
                finished_at=_now_iso(),
                current_symbol=None,
                result_count=int(len(pred_df)),
                preview=pred_df.head(20).to_dict(orient="records"),
            )
        except Exception as exc:
            _update_run(
                status="failed",
                progress=100,
                current_step="Failed",
                message=str(exc),
                finished_at=_now_iso(),
                current_symbol=None,
                error=str(exc),
            )

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return run_id, True


@st.fragment
def _render_prediction_controls(settings) -> None:
    movement_model_name = "movement_model"

    model_name = st.selectbox(
        "Model",
        options=["xgboost_classifier", movement_model_name] + OPENAI_PREDICTION_MODEL_OPTIONS,
        key="pred_model_name",
    )
    horizon_days = st.number_input("Horizon days", min_value=1, max_value=10, value=1, step=1, key="pred_horizon_days")
    atr_multiplier = st.slider(
        "Stop-loss ATR multiplier",
        min_value=0.25,
        max_value=3.0,
        value=1.0,
        step=0.25,
        key="pred_atr_multiplier",
    )
    include_live_quote = st.checkbox(
        "Fetch live quote/time from Alpha Vantage (slower)",
        value=True,
        key="pred_include_live_quote",
    )

    if model_name == movement_model_name:
        metadata_path = settings.movement_model_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                trained_symbols = sorted({str(sym).strip().upper() for sym in payload.get("symbols", []) if str(sym).strip()})
            except Exception:
                trained_symbols = sorted({str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()})
        else:
            trained_symbols = sorted({str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()})
        st.caption(f"Movement model universe size: {len(trained_symbols)} symbol(s).")
    elif model_name in OPENAI_STOCK_MODEL_ALIASES_SET:
        alias_to_model = {
            "openai_stock_llm": str(settings.openai_predict_model_name or settings.openai_fast_model_name).strip(),
            "openai_stock_llm_fast": str(settings.openai_fast_model_name).strip(),
            "openai_stock_llm_search": str(settings.openai_search_model_name).strip(),
            "openai_stock_llm_cheap": str(settings.openai_cheap_model_name).strip(),
        }
        resolved_provider_model = alias_to_model.get(str(model_name).strip(), str(settings.openai_predict_model_name).strip())
        trained_symbols = sorted({str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()})
        st.caption(
            f"OpenRouter profile `{model_name}` -> `{resolved_provider_model}` | "
            f"universe size: {len(trained_symbols)} symbol(s) from DEFAULT_SYMBOLS."
        )
        if not settings.openai_predict_enabled:
            st.warning("OpenAI predictor is disabled. Set OPENAI_PREDICT_ENABLED=true in environment.")
    else:
        trained_symbols = _all_trained_symbols(settings.model_dir, model_name)
        trained_symbols_by_latest = _trained_symbols_by_latest_training(settings.model_dir, model_name)
        if not trained_symbols_by_latest:
            trained_symbols_by_latest = trained_symbols
        trained_symbols = trained_symbols_by_latest
        st.caption(f"Trained symbols available for {model_name}: {len(trained_symbols)}")

    st.markdown("### Symbol Universe")
    universe_source = st.radio(
        "Prediction symbols source",
        ["All symbols from sec_list.csv", "Configured defaults", "Trending from news", "Custom symbols"],
        horizontal=True,
        key="pred_universe_source",
        index=0,
    )

    selected_symbols: list[str] = []
    if universe_source == "All symbols from sec_list.csv":
        selected_symbols = _load_prediction_symbols_from_csv(
            csv_path="sec_list.csv",
            symbol_column="Symbol",
            market="india",
            series_filter="EQ",
        )
        if selected_symbols:
            st.info(f"Using {len(selected_symbols)} symbols from sec_list.csv")
        else:
            st.warning("sec_list.csv not found or empty. Falling back to configured defaults.")
            selected_symbols = [str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()]
    elif universe_source == "Configured defaults":
        selected_symbols = [str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()]
    elif universe_source == "Trending from news":
        try:
            from app.services.trending_symbols_service import trending_symbols_service

            with st.spinner("Fetching trending symbols from news..."):
                symbols, source = trending_symbols_service.get_trending_symbols(limit=25, fallback_to_defaults=True)
                selected_symbols = symbols
                st.info(f"Using {len(selected_symbols)} trending symbols (source: {source})")
        except Exception as exc:
            st.error(f"Failed to fetch trending symbols: {exc}")
            selected_symbols = [str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip()]
    else:
        raw_custom = st.text_area(
            "Custom symbols (comma or newline separated)",
            value="\n".join(settings.default_symbols[:6]),
            key="pred_custom_symbols",
            height=120,
        )
        parts = [part.strip().upper() for part in raw_custom.replace("\n", ",").split(",")]
        selected_symbols = [part for part in parts if part]

    selected_symbols = sorted(list(dict.fromkeys(selected_symbols)))
    st.caption(f"Selected symbols for prediction: {len(selected_symbols)}")

    if model_name not in OPENAI_STOCK_MODEL_ALIASES_SET:
        trained_set = set(trained_symbols)
        selected_symbols = [sym for sym in selected_symbols if sym in trained_set]
        st.caption(f"After trained-symbol filter: {len(selected_symbols)}")

    active_run = _active_prediction_run()
    prediction_running = bool(active_run and str(active_run.get("status")) == "running")
    if prediction_running:
        st.info("A prediction run is already in progress. Only the status panel below will keep refreshing.")

    auto_run = st.checkbox("Auto-generate on page load", value=False, key="pred_auto_run")
    max_symbols = st.number_input("Auto-run max symbols", min_value=1, max_value=200, value=10, step=1, key="pred_max_symbols")

    auto_run_key = (
        f"predictions-auto-run-done-{model_name}-{int(horizon_days)}-"
        f"{float(atr_multiplier):.2f}-{int(include_live_quote)}"
    )

    if auto_run and selected_symbols and not st.session_state.get(auto_run_key, False):
        target_symbols = selected_symbols[: int(max_symbols)]
        run_id, started = _start_background_prediction(
            settings=settings,
            symbols=target_symbols,
            model_name=model_name,
            horizon_days=int(horizon_days),
            atr_multiplier=float(atr_multiplier),
            include_live_quote=bool(include_live_quote),
        )
        st.session_state[auto_run_key] = True
        if started:
            st.success(f"Started prediction job {run_id} for {len(target_symbols)} stock(s).")
        else:
            st.info(f"Prediction job {run_id} is already running.")

    if not auto_run:
        st.session_state[auto_run_key] = False

    if st.button(
        "Generate Predictions For Selected Symbols",
        disabled=not selected_symbols or prediction_running,
        type="primary",
    ):
        run_id, started = _start_background_prediction(
            settings=settings,
            symbols=selected_symbols,
            model_name=model_name,
            horizon_days=int(horizon_days),
            atr_multiplier=float(atr_multiplier),
            include_live_quote=bool(include_live_quote),
        )
        if started:
            st.success(f"Started prediction job {run_id} for {len(selected_symbols)} stock(s).")
        else:
            st.info(f"Prediction job {run_id} is already running.")


@st.fragment(run_every="2s")
def _render_prediction_run_status() -> None:
    run = _active_prediction_run()
    if not run:
        return

    status = str(run.get("status") or "")
    progress = float(run.get("progress") or 0.0)
    processed = int(run.get("processed_symbols") or 0)
    total = int(run.get("total_symbols") or 0)
    current_symbol = str(run.get("current_symbol") or "").strip()

    st.markdown("### Prediction Run Status")
    st.progress(
        min(max(progress / 100.0, 0.0), 1.0),
        text=f"{run.get('current_step', 'Working')} - {progress:.0f}%",
    )
    details = f"Processed {processed}/{total} symbol(s)"
    if current_symbol and status == "running":
        details += f" | Current: {current_symbol}"
    details += f" | Model: {run.get('model_name', 'unknown')}"
    st.caption(details)

    if status == "failed":
        st.error(str(run.get("message") or "Prediction run failed"))
    elif status == "completed":
        st.success(str(run.get("message") or "Prediction run completed"))
    else:
        st.info(str(run.get("message") or "Prediction run is working"))

    preview = run.get("preview") or []
    if preview:
        st.dataframe(
            _format_prediction_table(_prediction_preview_frame(pd.DataFrame(preview))),
            use_container_width=True,
        )

    if status in {"completed", "failed"}:
        refresh_key = f"prediction-run-refresh-{run.get('run_id')}"
        if not st.session_state.get(refresh_key, False):
            st.session_state[refresh_key] = True
            st.rerun()


def _render_saved_predictions(settings) -> None:
    pred_path = settings.output_dir / "latest_predictions.parquet"
    if pred_path.exists():
        pred_df = pd.read_parquet(pred_path).sort_values("confidence", ascending=False)
        if "predicted_price" not in pred_df.columns and {"current_price", "predicted_return"}.issubset(pred_df.columns):
            pred_df["predicted_price"] = pd.to_numeric(pred_df["current_price"], errors="coerce") * (
                1.0 + pd.to_numeric(pred_df["predicted_return"], errors="coerce").fillna(0.0)
            )
        if {"current_price", "live_price"}.issubset(set(pred_df.columns)):
            model_px = pd.to_numeric(pred_df["current_price"], errors="coerce")
            live_px = pd.to_numeric(pred_df["live_price"], errors="coerce")
            pred_df["live_vs_model_abs"] = live_px - model_px
            pred_df["live_vs_model_pct"] = ((live_px - model_px) / model_px.replace(0, pd.NA)) * 100.0

        price_cols = st.columns(4)
        if not pred_df.empty:
            top_row = pred_df.sort_values(
                "news_adjusted_confidence" if "news_adjusted_confidence" in pred_df.columns else "confidence",
                ascending=False,
            ).iloc[0]
            price_cols[0].metric("Top Symbol", str(top_row.get("symbol") or "N/A"))
            price_cols[1].metric("Current Share Price", f"{pd.to_numeric(top_row.get('current_price'), errors='coerce'):.2f}")
            price_cols[2].metric("Predicted Price", f"{pd.to_numeric(top_row.get('predicted_price'), errors='coerce'):.2f}")
            news_target = pd.to_numeric(top_row.get("news_adjusted_target_price"), errors="coerce")
            price_cols[3].metric("News Predicted Price", f"{news_target:.2f}" if pd.notna(news_target) else "N/A")

        st.caption(f"Currently loaded predictions: {len(pred_df)} stock(s)")
        if "price_as_of" in pred_df.columns:
            latest_as_of = pred_df["price_as_of"].dropna().astype(str).max()
            if latest_as_of:
                st.caption(f"Price source: latest ingested daily close (not live tick). Latest as-of date: {latest_as_of}")
                st.caption("For daily interval (1d), time may appear as 00:00:00 from provider candle timestamp.")
        if "live_price" in pred_df.columns and pred_df["live_price"].notna().any():
            st.caption("Live quote/time columns are best-effort via Polygon, Alpha Vantage, and Google Finance, and may be delayed/unavailable for some symbols.")
        preferred = [
            "symbol",
            "current_price",
            "predicted_price",
            "news_adjusted_target_price",
            "target_price",
            "stop_loss_price",
            "predicted_return",
            "news_adjusted_predicted_return",
            "prob_up",
            "news_adjusted_prob_up",
            "confidence",
            "news_adjusted_confidence",
            "decision",
            "news_decision",
            "price_as_of",
            "price_as_of_time",
            "live_price",
            "live_vs_model_abs",
            "live_vs_model_pct",
            "live_price_as_of",
            "live_price_as_of_time",
            "live_price_source",
            "latest_sentiment",
            "news_count",
            "news_impact_score",
            "news_signal_score",
            "news_probability_boost",
            "news_reason",
        ]
        cols = [c for c in preferred if c in pred_df.columns] + [c for c in pred_df.columns if c not in preferred]
        display_df = pred_df[cols].copy()
        percent_cols = [
            "predicted_return",
            "news_adjusted_predicted_return",
            "prob_up",
            "news_adjusted_prob_up",
            "confidence",
            "news_adjusted_confidence",
        ]
        for column in percent_cols:
            if column in display_df.columns:
                display_df[column] = pd.to_numeric(display_df[column], errors="coerce") * 100.0
        display_df = display_df.rename(
            columns={
                "symbol": "Symbol",
                "current_price": "Current Share Price",
                "predicted_price": "Predicted Price",
                "news_adjusted_target_price": "News Predicted Price",
                "target_price": "Target Price",
                "stop_loss_price": "Stop Loss",
                "predicted_return": "Prediction Return %",
                "news_adjusted_predicted_return": "News Return %",
                "prob_up": "Up Probability %",
                "news_adjusted_prob_up": "News Up Probability %",
                "confidence": "Confidence %",
                "news_adjusted_confidence": "News Confidence %",
                "decision": "Decision",
                "news_decision": "News Decision",
                "price_as_of": "Price Date",
                "price_as_of_time": "Price Time",
                "live_price": "Live Price",
                "live_vs_model_abs": "Live Diff",
                "live_vs_model_pct": "Live Diff %",
                "live_price_as_of": "Live Date",
                "live_price_as_of_time": "Live Time",
                "live_price_source": "Live Source",
                "latest_sentiment": "Latest Sentiment",
                "news_count": "News Count",
                "news_impact_score": "News Impact",
                "news_signal_score": "News Signal",
                "news_probability_boost": "News Probability Boost",
                "news_reason": "News Reason",
            }
        )
        if {"Live Diff", "Live Diff %"}.issubset(set(display_df.columns)):
            show_only_large_mismatch = st.checkbox(
                "Show only large mismatch rows",
                value=False,
                key="pred_show_only_large_mismatch",
            )
            mismatch_threshold_pct = st.number_input(
                "Mismatch threshold (%)",
                min_value=0.1,
                max_value=50.0,
                value=1.0,
                step=0.1,
                key="pred_mismatch_threshold_pct",
            )
            if show_only_large_mismatch:
                pct_series = pd.to_numeric(display_df["Live Diff %"], errors="coerce").abs()
                display_df = display_df[pct_series >= float(mismatch_threshold_pct)]
                st.caption(f"Filtered rows: {len(display_df)} stock(s) with mismatch >= {float(mismatch_threshold_pct):.1f}%")
            styled = _format_prediction_table(display_df)
            styled = styled.map(_color_diff, subset=[c for c in ["Live Diff", "Live Diff %"] if c in display_df.columns])
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(_format_prediction_table(display_df), use_container_width=True)
    else:
        st.info("No predictions yet. Use the button above to generate predictions for all trained stocks.")


def render_predictions(settings) -> None:
    st.subheader("Top Predictions")
    _render_prediction_controls(settings)
    _render_prediction_run_status()
    _render_saved_predictions(settings)
