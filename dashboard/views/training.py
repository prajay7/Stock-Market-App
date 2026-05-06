from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.config import get_settings
from app.services.data_service import data_service
from app.services.prediction_service import prediction_service
from app.services.training_service import training_service
from app.services.validation_service import validation_service


def _training_step(status: str) -> tuple[str, str]:
    status_norm = str(status or "").strip()
    if status_norm == "In Progress":
        return "Training", "Your job is queued or running now."
    if status_norm == "Trained":
        return "Complete", "Latest training finished successfully."
    if status_norm == "Failed":
        return "Review", "The last run failed. Check the failure reason below."
    return "Ready", "Pick a stock and start training."


def _training_live_label(status: str) -> str:
    status_norm = str(status or "").strip()
    if status_norm == "Never Trained":
        return "Waiting for selection"
    if status_norm == "queued":
        return "Waiting for queue"
    if status_norm == "running":
        return "Training in progress"
    if status_norm == "In Progress":
        return "Queued or training"
    if status_norm == "Trained":
        return "Completed successfully"
    if status_norm == "Failed":
        return "Completed with a failure"
    return "Status updating"


def _extract_snapshot_path(message: str) -> str | None:
    text = str(message or "")
    marker = "snapshot:"
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1].strip()
    if not tail:
        return None
    return tail.split(";", 1)[0].strip()


def _training_tracker_steps(status: str) -> list[tuple[str, str, str]]:
    status_norm = str(status or "").strip()
    if status_norm == "Trained":
        states = ["done", "done", "done", "current"]
    elif status_norm == "In Progress":
        states = ["done", "done", "current", "upcoming"]
    elif status_norm == "Failed":
        states = ["done", "done", "done", "current"]
    elif status_norm == "Never Trained":
        states = ["current", "upcoming", "upcoming", "upcoming"]
    else:
        states = ["done", "current", "upcoming", "upcoming"]
    return [
        ("Select", "Choose a stock", states[0]),
        ("Queue", "Send the job", states[1]),
        ("Train", "Process the model", states[2]),
        ("Done", "Finished or needs review", states[3]),
    ]


def _render_training_tracker(status: str) -> None:
    steps = _training_tracker_steps(status)
    cols = st.columns(len(steps))
    for col, (title, detail, state) in zip(cols, steps):
        if state == "current":
            label = "Now"
            color = "#7a5af8"
            background = "#f4f3ff"
            border = "#7a5af8"
            opacity = "1"
            shadow = "0 0 0 2px rgba(122,90,248,0.18)"
        elif state == "done":
            label = "Done"
            color = "#027a48"
            background = "#ecfdf3"
            border = "#abefc6"
            opacity = "0.92"
            shadow = "none"
        else:
            label = "Waiting"
            color = "#667085"
            background = "#f9fafb"
            border = "#eaecf0"
            opacity = "0.55"
            shadow = "none"
        col.markdown(
            f"""
            <div style="border:1px solid {border}; background:{background}; border-radius:14px; padding:0.8rem 0.9rem; min-height: 92px; opacity:{opacity}; box-shadow:{shadow};">
              <div style="font-size:0.82rem; font-weight:700; color:#344054; text-transform:uppercase; letter-spacing:0.04em;">{title}</div>
              <div style="font-size:1.05rem; font-weight:800; color:{color}; margin-top:0.35rem;">{label}</div>
              <div style="font-size:0.82rem; color:#475467; margin-top:0.35rem; line-height:1.35;">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_training(
    status_merged: pd.DataFrame,
    queue_jobs: list[dict],
    unsupported_symbols: set[str],
    plan_summary: dict,
    plan: dict | None,
    filter_universe,
    decorate_status,
    queue_status_for_symbol,
    status_badge,
    last_failure_reason,
    enqueue_training_job,
    bulk_enqueue_with_checkpoint,
    enqueue_bulk_training_jobs,
    resume_bulk_plan,
    load_bulk_plan,
    save_bulk_plan,
    cancel_jobs_by_symbols,
) -> None:
    filtered = decorate_status(filter_universe(status_merged, "training"))
    selection_options = filtered["train_symbol"].dropna().tolist()
    st.subheader("Training")
    st.caption("Monitor queue status and run manual data, training, prediction, or validation actions when needed.")

    if "train_selected_symbols_multi" not in st.session_state:
        st.session_state["train_selected_symbols_multi"] = []

    # Keep selected symbols valid when filters change.
    st.session_state["train_selected_symbols_multi"] = [
        s for s in st.session_state["train_selected_symbols_multi"] if s in selection_options
    ]

    c1, c2, c3 = st.columns([3, 1, 1])
    if c2.button("Select All Filtered"):
        st.session_state["train_selected_symbols_multi"] = selection_options.copy()
        st.rerun()
    if c3.button("Clear Selection"):
        st.session_state["train_selected_symbols_multi"] = []
        st.rerun()

    selected_symbols = c1.multiselect(
        "Select stock(s) to inspect",
        options=selection_options,
        key="train_selected_symbols_multi",
    )

    settings = get_settings()
    action_symbols = selected_symbols or settings.default_symbols

    st.markdown("#### Manual Actions")
    action_cols = st.columns(4)
    if action_cols[0].button("Update Missing Data", use_container_width=True):
        with st.spinner("Updating missing data..."):
            result = data_service.ingest_historical(action_symbols, settings.historical_interval, settings.historical_lookback_days)
        st.success("Missing data updated")
        st.json(result)
    if action_cols[1].button("Train Model", use_container_width=True):
        with st.spinner("Training movement model..."):
            result = training_service.train(action_symbols, 1, task_type="movement")
        st.success("Model training finished")
        st.json(result.get("best") or result)
    if action_cols[2].button("Run Prediction", use_container_width=True):
        with st.spinner("Running predictions..."):
            result = prediction_service.predict(action_symbols, "movement_model", 1, include_live_quote=False)
        st.success("Predictions generated")
        st.dataframe(pd.DataFrame(result.get("predictions") or []), use_container_width=True)
    if action_cols[3].button("Validate Predictions", use_container_width=True):
        with st.spinner("Validating predictions..."):
            result = validation_service.validate_pending_predictions(settings.historical_interval)
        st.success("Validation completed")
        st.json(result)

    status_cards = st.columns(4)
    status_cards[0].metric("Selected", len(selected_symbols))
    status_cards[1].metric("Unsupported", len(unsupported_symbols))
    status_cards[2].metric("Queue State", plan_summary["resume_state"])
    status_cards[3].metric("Queue Jobs", len(queue_jobs))

    failed_symbols = [
        symbol
        for symbol in selection_options
        if queue_status_for_symbol(symbol) == "Failed" and symbol not in unsupported_symbols
    ]
    st.caption(f"Failed symbols currently tracked: {len(failed_symbols)}")

    st.markdown("#### Recent Queue Activity")
    if not queue_jobs:
        st.info("No live queue jobs right now.")
    else:
        queue_df = pd.DataFrame(queue_jobs)
        for col in ["created_at", "started_at", "finished_at"]:
            if col in queue_df.columns:
                queue_df[col] = pd.to_datetime(queue_df[col], errors="coerce")
        queue_df["status_badge"] = queue_df["status"].apply(status_badge)
        queue_df = queue_df.sort_values(["created_at", "symbol"], ascending=[False, True])
        progress_total = len(queue_df)
        progress_done = int(queue_df["status"].isin(["completed", "failed"]).sum())
        st.progress(progress_done / progress_total if progress_total else 0.0, text=f"{progress_done}/{progress_total} jobs finished")
        st.dataframe(
            queue_df[["job_id", "symbol", "status_badge", "message", "created_at", "started_at", "finished_at"]],
            use_container_width=True,
            height=280,
        )

    # Keep failure details and snapshot download for the latest selected symbol.
    if selected_symbols:
        selected_train_symbol = str(selected_symbols[0])
        selected_failure = last_failure_reason(selected_train_symbol)
        if selected_failure:
            st.warning(selected_failure)
            snapshot_path = _extract_snapshot_path(selected_failure)
            if snapshot_path:
                snap_file = Path(snapshot_path)
                if snap_file.exists() and snap_file.is_file():
                    st.caption(f"Debug snapshot: {snapshot_path}")
                    st.download_button(
                        "Download ingest failure snapshot",
                        data=snap_file.read_text(encoding="utf-8"),
                        file_name=snap_file.name,
                        mime="application/json",
                        key=f"download_snapshot_{snap_file.name}",
                    )

    st.markdown("#### Training Status")
    status_view = status_merged[["symbol", "train_symbol", "series", "last_trained_at", "last_model_name", "last_model_version", "last_roc_auc"]].copy()
    status_view["last_trained_at"] = pd.to_datetime(status_view["last_trained_at"], errors="coerce")
    status_view["queue_status"] = status_view["train_symbol"].apply(queue_status_for_symbol)
    status_view["status_badge"] = status_view["queue_status"].apply(status_badge)
    status_view["last_failure_reason"] = status_view["train_symbol"].apply(last_failure_reason)
    status_view = status_view.sort_values(["last_trained_at", "symbol"], ascending=[False, True])
    status_display = status_view[["symbol", "train_symbol", "series", "status_badge", "last_failure_reason", "last_trained_at", "last_model_name", "last_model_version", "last_roc_auc"]].copy()
    st.dataframe(status_display, use_container_width=True, height=340)
