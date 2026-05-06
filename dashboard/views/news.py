from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
import uuid

import pandas as pd
import streamlit as st

from app.news.fetcher import load_rss_feeds
from src.data.cache import symbol_news_path
from src.data.metadata_store import MetadataStore
from src.features.news_scoring import enrich_news_scores
from src.data.web_news_scraper import WebsiteNewsScraper, build_common_news_sources
from src.inference.predict import predict_for_symbols
from src.training.train import train_models

SEC_LIST_PATH = Path("sec_list.csv")
PIPELINE_LOCK = threading.Lock()
PIPELINE_RUNS: dict[str, dict] = {}
ACTIVE_PIPELINE_RUN_ID: str | None = None


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalise_news_frame(df: pd.DataFrame, symbol: str, file_updated_at: pd.Timestamp | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "symbol" not in out.columns:
        out["symbol"] = symbol
    out["symbol"] = out["symbol"].fillna(symbol).astype(str).str.upper()

    for column in ["source", "title", "summary", "url", "scraped_from"]:
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str)

    if "published_at" not in out.columns:
        out["published_at"] = pd.NaT
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce")

    if "sentiment_score" in out.columns:
        out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce")
    if "relevance_score" in out.columns:
        out["relevance_score"] = pd.to_numeric(out["relevance_score"], errors="coerce")

    out = enrich_news_scores(out)
    out["file_updated_at"] = file_updated_at
    return out


@st.cache_data(ttl=20, show_spinner=False)
def _load_all_scraped_news(raw_data_dir: str) -> pd.DataFrame:
    news_dir = Path(raw_data_dir) / "news"
    if not news_dir.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(news_dir.glob("*.parquet")):
        try:
            updated_at = pd.to_datetime(path.stat().st_mtime, unit="s")
            frame = pd.read_parquet(path)
        except Exception:
            continue
        normalised = _normalise_news_frame(frame, path.stem.upper(), updated_at)
        if not normalised.empty:
            frames.append(normalised)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "url" in combined.columns:
        combined = combined.drop_duplicates(subset=["symbol", "url"], keep="last")
    combined = combined.sort_values(["published_at", "file_updated_at", "title"], ascending=[False, False, True], na_position="last")
    return combined.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_sec_list_symbols(sec_list_path: str) -> pd.DataFrame:
    path = Path(sec_list_path)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "series", "security_name"])

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["symbol", "series", "security_name"])

    rename_map = {
        "Symbol": "symbol",
        "Series": "series",
        "Security Name": "security_name",
    }
    df = df.rename(columns=rename_map)
    required = ["symbol", "series", "security_name"]
    for column in required:
        if column not in df.columns:
            df[column] = ""

    out = df[required].copy()
    out["symbol"] = out["symbol"].fillna("").astype(str).str.strip().str.upper()
    out["series"] = out["series"].fillna("").astype(str).str.strip().str.upper()
    out["security_name"] = out["security_name"].fillna("").astype(str).str.strip()
    out = out[out["symbol"] != ""].drop_duplicates(subset=["symbol"], keep="first")
    return out.sort_values(["series", "symbol"]).reset_index(drop=True)


def _symbol_options(settings, status_merged: pd.DataFrame, all_news: pd.DataFrame) -> list[str]:
    symbols = {str(symbol).strip().upper() for symbol in settings.default_symbols if str(symbol).strip()}
    sec_symbols = _load_sec_list_symbols(str(SEC_LIST_PATH))
    if not sec_symbols.empty:
        symbols.update(sec_symbols["symbol"].dropna().astype(str).str.upper().tolist())
    if not status_merged.empty and "train_symbol" in status_merged.columns:
        symbols.update(str(symbol).strip().upper() for symbol in status_merged["train_symbol"].dropna() if str(symbol).strip())
    if not all_news.empty and "symbol" in all_news.columns:
        symbols.update(str(symbol).strip().upper() for symbol in all_news["symbol"].dropna() if str(symbol).strip())
    return sorted(symbols)


def _scrape_symbol(settings, symbol: str, limit_per_source: int, query: str | None = None) -> pd.DataFrame:
    scraper = WebsiteNewsScraper(
        raw_data_dir=settings.raw_data_dir,
        database_url=settings.database_url,
        timeout_sec=settings.request_timeout_sec,
    )
    sources = build_common_news_sources(query or f"{symbol} stock news")
    return scraper.ingest_many(sources=sources, symbol=symbol, limit_per_source=limit_per_source)


def _scrape_all_market_news(settings, limit_per_source: int) -> pd.DataFrame:
    scraper = WebsiteNewsScraper(
        raw_data_dir=settings.raw_data_dir,
        database_url=settings.database_url,
        timeout_sec=settings.request_timeout_sec,
    )
    sources = load_rss_feeds(settings.news_rss_feeds_path) or settings.news_rss_feeds_default
    return scraper.ingest_many(sources=sources, symbol="MARKET", limit_per_source=limit_per_source)


def _scrape_market_and_stock_news(settings, symbols: list[dict[str, str]], limit_per_source: int, progress_callback=None) -> dict[str, int]:
    summary: dict[str, int] = {}
    market_df = _scrape_all_market_news(settings, limit_per_source=limit_per_source)
    summary["MARKET"] = int(len(market_df))
    if progress_callback is not None:
        progress_callback(0, max(1, len(symbols)), "MARKET", int(len(market_df)))

    for index, item in enumerate(symbols, start=1):
        symbol_clean = str(item.get("symbol") or "").strip().upper()
        if not symbol_clean or symbol_clean == "MARKET":
            continue
        company_name = str(item.get("security_name") or "").strip()
        query = f"{symbol_clean} {company_name} stock news".strip() if company_name else f"{symbol_clean} stock news"
        try:
            symbol_df = _scrape_symbol(settings, symbol_clean, limit_per_source=limit_per_source, query=query)
        except Exception:
            continue
        row_count = int(len(symbol_df))
        if row_count > 0:
            summary[symbol_clean] = row_count
        if progress_callback is not None:
            progress_callback(index, max(1, len(symbols)), symbol_clean, row_count)

    return summary


def _clear_all_saved_news(settings) -> dict[str, int]:
    news_dir = Path(settings.raw_data_dir) / "news"
    deleted_files = 0
    if news_dir.exists():
        for path in news_dir.glob("*.parquet"):
            path.unlink()
            deleted_files += 1

    deleted_db_rows = 0
    store = MetadataStore(settings.database_url)
    with store.engine.begin() as conn:
        result = conn.execute(store.news.delete())
        if result.rowcount is not None and result.rowcount > 0:
            deleted_db_rows = int(result.rowcount)

    return {"files": deleted_files, "db_rows": deleted_db_rows}


def _render_news_metrics(news: pd.DataFrame) -> None:
    cols = st.columns(4)
    cols[0].metric("Headlines", int(len(news)))
    cols[1].metric("Symbols", int(news["symbol"].nunique()) if "symbol" in news.columns and not news.empty else 0)
    cols[2].metric("Sources", int(news["source"].nunique()) if "source" in news.columns and not news.empty else 0)

    latest = None
    if not news.empty and "published_at" in news.columns:
        published = pd.to_datetime(news["published_at"], errors="coerce").dropna()
        if not published.empty:
            latest = published.max().strftime("%Y-%m-%d %H:%M")
    cols[3].metric("Latest Published", latest or "N/A")


def _symbols_with_price_cache(settings) -> set[str]:
    price_dir = Path(settings.raw_data_dir) / "prices" / str(settings.historical_interval)
    if not price_dir.exists():
        return set()
    return {path.stem.upper() for path in price_dir.glob("*.parquet") if path.stem.strip()}


def _eligible_news_prediction_symbols(settings, news: pd.DataFrame) -> list[str]:
    if news.empty or "symbol" not in news.columns:
        return []
    return sorted(
        {
            str(symbol).strip().upper()
            for symbol in news["symbol"].dropna().tolist()
            if str(symbol).strip() and str(symbol).strip().upper() != "MARKET"
        }
    )


def _cached_news_prediction_symbols(settings, news: pd.DataFrame) -> list[str]:
    if news.empty or "symbol" not in news.columns:
        return []
    news_symbols = {
        str(symbol).strip().upper()
        for symbol in news["symbol"].dropna().tolist()
        if str(symbol).strip() and str(symbol).strip().upper() != "MARKET"
    }
    price_symbols = _symbols_with_price_cache(settings)
    return sorted(news_symbols & price_symbols)


def _build_stock_items(
    sec_symbols: pd.DataFrame,
    symbol_options: list[str],
    selected_series: list[str],
    start_at_symbol: int,
    max_stock_symbols: int,
) -> list[dict[str, str]]:
    if sec_symbols.empty:
        return [
            {"symbol": str(symbol).strip().upper(), "security_name": ""}
            for symbol in symbol_options
            if str(symbol).strip().upper() != "MARKET"
        ][: int(max_stock_symbols)]

    work = sec_symbols.copy()
    if selected_series:
        work = work[work["series"].isin(selected_series)]
    work = work.iloc[int(start_at_symbol) :]
    work = work.head(int(max_stock_symbols))
    return work[["symbol", "security_name"]].to_dict(orient="records")


def _prediction_preview_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    preview_cols = [
        "symbol",
        "current_price",
        "predicted_price",
        "news_adjusted_target_price",
        "prob_up",
        "news_adjusted_prob_up",
        "predicted_return",
        "news_adjusted_predicted_return",
        "news_count",
        "news_signal_score",
        "news_decision",
    ]
    preview = pred_df[[column for column in preview_cols if column in pred_df.columns]].copy()
    for column in ["prob_up", "news_adjusted_prob_up", "predicted_return", "news_adjusted_predicted_return"]:
        if column in preview.columns:
            preview[column] = pd.to_numeric(preview[column], errors="coerce") * 100.0
    return preview.rename(
        columns={
            "symbol": "Symbol",
            "current_price": "Current Share Price",
            "predicted_price": "Predicted Price",
            "news_adjusted_target_price": "News Predicted Price",
            "prob_up": "Up Probability %",
            "news_adjusted_prob_up": "News Up Probability %",
            "predicted_return": "Prediction Return %",
            "news_adjusted_predicted_return": "News Return %",
            "news_count": "News Count",
            "news_signal_score": "News Signal",
            "news_decision": "News Decision",
        }
    )


def _pipeline_update(run_id: str, **updates) -> None:
    with PIPELINE_LOCK:
        run = PIPELINE_RUNS.get(run_id)
        if not run:
            return
        run.update(updates)
        run["updated_at"] = _now_iso()


def _pipeline_add_log(run_id: str, message: str) -> None:
    with PIPELINE_LOCK:
        run = PIPELINE_RUNS.get(run_id)
        if not run:
            return
        logs = list(run.get("logs") or [])
        logs.append({"time": _now_iso(), "message": str(message)})
        run["logs"] = logs[-100:]
        run["message"] = str(message)
        run["updated_at"] = _now_iso()


def _active_pipeline_run() -> dict | None:
    with PIPELINE_LOCK:
        if not ACTIVE_PIPELINE_RUN_ID:
            return None
        run = PIPELINE_RUNS.get(ACTIVE_PIPELINE_RUN_ID)
        return run.copy() if run else None


def _start_full_news_pipeline(
    settings,
    stock_items: list[dict[str, str]],
    limit_per_source: int,
    horizon_days: int,
    fallback_model_name: str,
    refresh_prices: bool,
) -> str:
    global ACTIVE_PIPELINE_RUN_ID
    active = _active_pipeline_run()
    if active and str(active.get("status")) == "running":
        return str(active.get("run_id"))

    run_id = str(uuid.uuid4())[:8]
    with PIPELINE_LOCK:
        ACTIVE_PIPELINE_RUN_ID = run_id
        PIPELINE_RUNS[run_id] = {
            "run_id": run_id,
            "status": "running",
            "progress": 0.0,
            "current_step": "Starting",
            "message": "Starting full news prediction pipeline",
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "finished_at": None,
            "stock_count": len(stock_items),
            "summary": {},
            "best": {},
            "predictions": [],
            "logs": [],
        }

    def _worker() -> None:
        try:
            _pipeline_add_log(run_id, f"Scraping market news and {len(stock_items)} stock symbol(s)")

            def _progress(index: int, total: int, symbol: str, rows: int) -> None:
                progress = 5.0 + (40.0 * min(max(index, 0), max(total, 1)) / max(total, 1))
                _pipeline_update(
                    run_id,
                    progress=round(progress, 1),
                    current_step="Scraping news",
                    message=f"Scraped {symbol}: {rows} row(s)",
                )

            summary = _scrape_market_and_stock_news(
                settings,
                stock_items,
                limit_per_source=int(limit_per_source),
                progress_callback=_progress,
            )
            hit_symbols = [symbol for symbol, rows in summary.items() if symbol != "MARKET" and int(rows) > 0]
            _pipeline_update(run_id, summary=summary, progress=45.0)
            _pipeline_add_log(run_id, f"News scrape complete. Stock hits: {len(hit_symbols)}/{len(stock_items)}")
            if not hit_symbols:
                raise ValueError("No stock-specific news found. Nothing to train or predict.")

            _pipeline_update(
                run_id,
                current_step="Training model",
                progress=55.0,
                message=f"Training news-aware model on {len(hit_symbols)} symbol(s)",
            )
            result = train_models(
                symbols=hit_symbols,
                horizon_days=int(horizon_days),
                refresh_prices=bool(refresh_prices),
                task_type="classification",
            )
            best = result.get("best") or {}
            best_model_name = str(best.get("best_model_name") or fallback_model_name or "xgboost_classifier")
            _pipeline_update(run_id, best=best, progress=78.0)
            _pipeline_add_log(run_id, f"Training complete. Best model: {best_model_name}")

            _pipeline_update(
                run_id,
                current_step="Generating predictions",
                progress=88.0,
                message=f"Generating predictions with {best_model_name}",
            )
            pred_df = predict_for_symbols(
                symbols=hit_symbols,
                model_name=best_model_name,
                horizon_days=int(horizon_days),
                include_live_quote=False,
            )
            sort_col = "news_adjusted_confidence" if "news_adjusted_confidence" in pred_df.columns else "confidence"
            pred_df = pred_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
            predictions = pred_df.head(50).to_dict(orient="records")
            _pipeline_update(
                run_id,
                status="completed",
                progress=100.0,
                current_step="Completed",
                message=f"Pipeline complete. Generated predictions for {len(pred_df)} stock(s).",
                predictions=predictions,
                finished_at=_now_iso(),
            )
            _pipeline_add_log(run_id, f"Pipeline complete. Generated predictions for {len(pred_df)} stock(s)")
        except Exception as exc:
            _pipeline_update(
                run_id,
                status="failed",
                progress=100.0,
                current_step="Failed",
                message=str(exc),
                finished_at=_now_iso(),
            )
            _pipeline_add_log(run_id, f"Pipeline failed: {exc}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return run_id


@st.fragment(run_every="5s")
def _render_pipeline_status() -> None:
    run = _active_pipeline_run()
    if not run:
        return

    status = str(run.get("status") or "")
    progress = float(run.get("progress") or 0.0)
    st.markdown("#### Full Pipeline Status")
    st.progress(min(max(progress / 100.0, 0.0), 1.0), text=f"{run.get('current_step', 'Working')} - {progress:.1f}%")
    if status == "failed":
        st.error(str(run.get("message") or "Pipeline failed"))
    elif status == "completed":
        st.success(str(run.get("message") or "Pipeline completed"))
    else:
        st.info(str(run.get("message") or "Pipeline running"))

    logs = run.get("logs") or []
    if logs:
        with st.expander("Pipeline log", expanded=False):
            st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)

    best = run.get("best") or {}
    if best:
        st.caption(f"Best trained model: {best.get('best_model_name', 'unknown')} | F1: {best.get('best_score', 'N/A')}")

    predictions = run.get("predictions") or []
    if predictions:
        st.markdown("#### Predicted Stocks")
        st.dataframe(_prediction_preview_frame(pd.DataFrame(predictions)), use_container_width=True, hide_index=True)


def _render_news_prediction_workflow(settings, news: pd.DataFrame, stock_items: list[dict[str, str]], limit_per_source: int) -> None:
    st.markdown("### News-Based Prediction Workflow")
    eligible_symbols = _eligible_news_prediction_symbols(settings, news)
    cached_symbols = set(_cached_news_prediction_symbols(settings, news))

    if not eligible_symbols:
        st.session_state["scraped_news_workflow_symbols"] = []
    elif "scraped_news_workflow_symbols" not in st.session_state:
        st.session_state["scraped_news_workflow_symbols"] = eligible_symbols[: min(10, len(eligible_symbols))]
    else:
        st.session_state["scraped_news_workflow_symbols"] = [
            symbol for symbol in st.session_state["scraped_news_workflow_symbols"] if symbol in eligible_symbols
        ]

    controls = st.columns([2.0, 0.8, 1.0, 1.0])
    selected_symbols = controls[0].multiselect(
        "Symbols for training / prediction",
        options=eligible_symbols,
        key="scraped_news_workflow_symbols",
    )
    horizon_days = controls[1].number_input(
        "Horizon days",
        min_value=1,
        max_value=30,
        value=1,
        step=1,
        key="scraped_news_workflow_horizon_days",
    )
    model_name = controls[2].selectbox(
        "Prediction model",
        options=["xgboost_classifier"],
        key="scraped_news_workflow_model_name",
    )
    refresh_prices = controls[3].checkbox(
        "Fetch missing prices",
        value=True,
        key="scraped_news_workflow_refresh_prices",
    )

    pipeline_run = _active_pipeline_run()
    pipeline_running = bool(pipeline_run and str(pipeline_run.get("status")) == "running")
    pipeline_cols = st.columns([1.2, 2.8])
    pipeline_clicked = pipeline_cols[0].button(
        "Run Full Pipeline",
        disabled=pipeline_running or not stock_items,
        type="primary",
        use_container_width=True,
        key="scraped_news_run_full_pipeline",
    )
    pipeline_cols[1].caption(f"One click: scrape -> train -> predict for up to {len(stock_items)} sec_list symbol(s).")

    action_cols = st.columns([1.0, 1.0, 2.0])
    train_clicked = action_cols[0].button(
        "Retrain News Model",
        disabled=pipeline_running or not selected_symbols,
        use_container_width=True,
        key="scraped_news_train_model",
    )
    predict_clicked = action_cols[1].button(
        "Generate Predictions",
        disabled=pipeline_running or not selected_symbols,
        use_container_width=True,
        key="scraped_news_generate_predictions",
    )
    action_cols[2].caption(
        f"Eligible news symbols: {len(eligible_symbols)}. Price cache ready: {len(cached_symbols)}. "
        "Missing prices are fetched with provider fallbacks, ending with Google Finance."
    )

    if pipeline_clicked:
        run_id = _start_full_news_pipeline(
            settings=settings,
            stock_items=stock_items,
            limit_per_source=int(limit_per_source),
            horizon_days=int(horizon_days),
            fallback_model_name=str(model_name),
            refresh_prices=bool(refresh_prices),
        )
        st.success(f"Started full pipeline job {run_id}.")

    _render_pipeline_status()

    if train_clicked:
        try:
            with st.spinner(f"Training news-aware models for {len(selected_symbols)} symbol(s)..."):
                result = train_models(
                    symbols=selected_symbols,
                    horizon_days=int(horizon_days),
                    refresh_prices=bool(refresh_prices),
                    task_type="classification",
                )
            best = result.get("best") or {}
            st.success(f"Training complete. Best model: {best.get('best_model_name', 'unknown')}.")
            st.json(best)
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    if predict_clicked:
        try:
            with st.spinner(f"Generating predictions for {len(selected_symbols)} symbol(s)..."):
                pred_df = predict_for_symbols(
                    symbols=selected_symbols,
                    model_name=str(model_name),
                    horizon_days=int(horizon_days),
                    include_live_quote=False,
                )
            st.success(f"Generated predictions for {len(pred_df)} symbol(s).")
            st.dataframe(
                _prediction_preview_frame(pred_df),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def _filter_news(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return news

    control_cols = st.columns([1.4, 1.4, 2.0, 1.0])
    symbol_options = sorted(news["symbol"].dropna().unique().tolist())
    selected_symbols = control_cols[0].multiselect(
        "Symbols",
        options=symbol_options,
        default=[],
        key="scraped_news_symbols",
    )

    source_options = sorted(news["source"].replace("", "Unknown").dropna().unique().tolist())
    selected_sources = control_cols[1].multiselect(
        "Sources",
        options=source_options,
        default=[],
        key="scraped_news_sources",
    )

    text_query = control_cols[2].text_input("Search headlines", key="scraped_news_search").strip().lower()
    max_rows = control_cols[3].number_input("Rows", min_value=10, max_value=1000, value=100, step=10, key="scraped_news_max_rows")

    filtered = news.copy()
    filtered["_source_label"] = filtered["source"].replace("", "Unknown")

    if selected_symbols:
        filtered = filtered[filtered["symbol"].isin(selected_symbols)]
    if selected_sources:
        filtered = filtered[filtered["_source_label"].isin(selected_sources)]
    if text_query:
        text = (
            filtered["title"].fillna("").astype(str)
            + " "
            + filtered["summary"].fillna("").astype(str)
            + " "
            + filtered["symbol"].fillna("").astype(str)
        ).str.lower()
        filtered = filtered[text.str.contains(text_query, na=False, regex=False)]

    return filtered.drop(columns=["_source_label"], errors="ignore").head(int(max_rows))


def render_news(settings, status_merged: pd.DataFrame, filter_universe, decorate_status) -> None:
    st.subheader("Scraped News")

    if st.session_state.pop("scraped_news_reset_clear_confirm", False):
        st.session_state["scraped_news_confirm_clear"] = False

    last_message = st.session_state.get("scraped_news_last_message")
    if last_message:
        st.success(str(last_message))
        last_summary = st.session_state.get("scraped_news_last_summary")
        if isinstance(last_summary, list) and last_summary:
            with st.expander("Last scrape summary", expanded=False):
                st.dataframe(pd.DataFrame(last_summary), use_container_width=True, hide_index=True)

    all_news = _load_all_scraped_news(str(settings.raw_data_dir))
    symbol_options = _symbol_options(settings, status_merged, all_news)
    sec_symbols = _load_sec_list_symbols(str(SEC_LIST_PATH))

    scrape_cols = st.columns([1.5, 0.8, 1.0, 1.0, 0.9])
    selected_symbol = scrape_cols[0].selectbox(
        "Symbol to scrape",
        options=symbol_options or settings.default_symbols,
        key="scraped_news_scrape_symbol",
    )
    limit_per_source = scrape_cols[1].number_input(
        "Per source",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        key="scraped_news_limit_per_source",
    )
    scrape_clicked = scrape_cols[2].button("Scrape Symbol", type="primary", use_container_width=True)
    scrape_all_clicked = scrape_cols[3].button("Scrape Market + Stocks", use_container_width=True)
    refresh_clicked = scrape_cols[4].button("Refresh Table", use_container_width=True)

    if sec_symbols.empty:
        st.warning("sec_list.csv was not found or could not be read. Market + Stocks will only use the current dashboard symbols.")
        selected_series: list[str] = []
        max_stock_symbols = len(symbol_options)
        start_at_symbol = 0
    else:
        universe_cols = st.columns([1.6, 1.0, 1.0])
        series_options = sorted(sec_symbols["series"].dropna().unique().tolist())
        default_series = ["EQ"] if "EQ" in series_options else series_options[:1]
        selected_series = universe_cols[0].multiselect(
            "sec_list series for all-stock scrape",
            options=series_options,
            default=default_series,
            key="scraped_news_sec_series",
        )
        eligible_count = int(len(sec_symbols[sec_symbols["series"].isin(selected_series)])) if selected_series else int(len(sec_symbols))
        max_stock_symbols = universe_cols[1].number_input(
            "Max stock symbols",
            min_value=1,
            max_value=max(1, int(len(sec_symbols))),
            value=max(1, min(250, eligible_count or int(len(sec_symbols)))),
            step=50,
            key="scraped_news_max_stock_symbols",
        )
        start_at_symbol = universe_cols[2].number_input(
            "Start at row",
            min_value=0,
            max_value=max(0, int(len(sec_symbols)) - 1),
            value=0,
            step=100,
            key="scraped_news_start_at_symbol",
        )
        st.caption(
            f"sec_list.csv symbols available: {len(sec_symbols)}. "
            f"Eligible after series filter: {eligible_count}. Empty-result symbols are skipped."
        )
    stock_items = _build_stock_items(
        sec_symbols=sec_symbols,
        symbol_options=symbol_options,
        selected_series=selected_series,
        start_at_symbol=int(start_at_symbol),
        max_stock_symbols=int(max_stock_symbols),
    )

    if refresh_clicked:
        _load_all_scraped_news.clear()
        st.rerun()

    if scrape_clicked:
        try:
            with st.spinner(f"Scraping news for {selected_symbol}..."):
                scraped = _scrape_symbol(settings, str(selected_symbol), int(limit_per_source))
            _load_all_scraped_news.clear()
            st.session_state["scraped_news_last_message"] = f"Saved {len(scraped)} headline(s) for {selected_symbol}."
            st.session_state["scraped_news_last_summary"] = [{"symbol": str(selected_symbol), "saved_rows": int(len(scraped))}]
            st.rerun()
        except Exception as exc:
            st.error(f"Scrape failed: {exc}")

    if scrape_all_clicked:
        try:
            with st.spinner(f"Scraping market news and {len(stock_items)} sec_list stock symbol(s)..."):
                summary = _scrape_market_and_stock_news(settings, stock_items, int(limit_per_source))
            _load_all_scraped_news.clear()
            total_rows = sum(summary.values())
            hit_symbols = max(0, len(summary) - (1 if "MARKET" in summary else 0))
            st.session_state["scraped_news_last_message"] = (
                f"Saved/updated {total_rows} headline row(s). "
                f"Stock hits: {hit_symbols}/{len(stock_items)}; no-news symbols skipped."
            )
            st.session_state["scraped_news_last_summary"] = [
                {"symbol": symbol, "saved_rows": rows} for symbol, rows in summary.items()
            ]
            st.rerun()
        except Exception as exc:
            st.error(f"All-news scrape failed: {exc}")

    clear_cols = st.columns([2.6, 1.0])
    confirm_clear = clear_cols[0].checkbox(
        "Confirm clear all saved scraped news",
        value=False,
        key="scraped_news_confirm_clear",
    )
    if clear_cols[1].button(
        "Clear All News",
        disabled=not confirm_clear,
        use_container_width=True,
        key="scraped_news_clear_all",
    ):
        try:
            result = _clear_all_saved_news(settings)
            _load_all_scraped_news.clear()
            st.session_state["scraped_news_last_message"] = (
                f"Cleared {result['files']} news file(s) and {result['db_rows']} database row(s)."
            )
            st.session_state["scraped_news_last_summary"] = []
            st.session_state["scraped_news_reset_clear_confirm"] = True
            st.rerun()
        except Exception as exc:
            st.error(f"Clear failed: {exc}")

    all_news = _load_all_scraped_news(str(settings.raw_data_dir))
    _render_news_metrics(all_news)

    selected_path = symbol_news_path(settings.raw_data_dir, str(selected_symbol))
    if selected_path.exists():
        updated_at = pd.to_datetime(selected_path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Selected symbol cache: {selected_path} | Updated: {updated_at}")
    else:
        st.caption(f"Selected symbol cache will be created at: {selected_path}")

    _render_news_prediction_workflow(settings, all_news, stock_items, int(limit_per_source))

    if all_news.empty:
        st.info("No scraped news is saved yet. Pick a symbol and click Scrape News.")
        return

    filtered = _filter_news(all_news)
    if filtered.empty:
        st.info("No saved headlines match the current filters.")
        return

    display_cols = [
        "published_at",
        "symbol",
        "source",
        "title",
        "sentiment_score",
        "news_sentiment_label",
        "news_impact_score",
        "news_signal_score",
        "relevance_score",
        "url",
        "scraped_from",
        "file_updated_at",
    ]
    display = filtered[[column for column in display_cols if column in filtered.columns]].copy()

    st.dataframe(
        display,
        use_container_width=True,
        height=520,
        column_config={
            "url": st.column_config.LinkColumn("Article", display_text="Open"),
            "scraped_from": st.column_config.LinkColumn("Scraped From", display_text="Source"),
            "published_at": st.column_config.DatetimeColumn("Published", format="YYYY-MM-DD HH:mm"),
            "file_updated_at": st.column_config.DatetimeColumn("Saved", format="YYYY-MM-DD HH:mm"),
        },
        hide_index=True,
    )

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered News CSV",
        data=csv,
        file_name="scraped_news.csv",
        mime="text/csv",
        key="download_scraped_news_csv",
    )
