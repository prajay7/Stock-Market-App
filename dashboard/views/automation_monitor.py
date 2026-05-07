from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import threading
import time
import uuid

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.core.config import get_settings
from app.services.automation_service import automation_service


AUTOMATION_UI_LOCK = threading.Lock()
AUTOMATION_UI_RUNS: dict[str, dict] = {}
AUTOMATION_UI_ACTIVE_RUN_ID: str | None = None
RUN_HISTORY_PATH = Path("data/outputs/checkpoints/dashboard_automation_runs.json")
AUTOMATION_MODEL_OPTIONS = [
    "auto",
    "xgboost_classifier",
    "openai_stock_llm_fast",
    "openai_stock_llm",
    "openai_stock_llm_search",
    "openai_stock_llm_cheap",
]


def _persist_runs() -> None:
    payload = {
        "active_run_id": AUTOMATION_UI_ACTIVE_RUN_ID,
        "runs": AUTOMATION_UI_RUNS,
    }
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_HISTORY_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _restore_runs_once() -> None:
    global AUTOMATION_UI_ACTIVE_RUN_ID
    with AUTOMATION_UI_LOCK:
        if AUTOMATION_UI_RUNS:
            return
        if not RUN_HISTORY_PATH.exists():
            return
        try:
            payload = json.loads(RUN_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        runs = payload.get("runs", {})
        if isinstance(runs, dict):
            AUTOMATION_UI_RUNS.update(runs)
            # Any "running" run from disk means the prior Streamlit process died/restarted.
            for run in AUTOMATION_UI_RUNS.values():
                if str(run.get("status")) != "running":
                    continue
                run["status"] = "interrupted"
                run["current_step"] = "Interrupted (app restarted)"
                run["finished_at"] = datetime.now().isoformat(timespec="seconds")
                run["messages"] = list(run.get("messages") or [])
                run["messages"].append(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "message": "Run interrupted because Streamlit process restarted",
                    }
                )
                _finalize_current_step(run)
        active = payload.get("active_run_id")
        if isinstance(active, str) and active in AUTOMATION_UI_RUNS:
            AUTOMATION_UI_ACTIVE_RUN_ID = active
        _persist_runs()


def _touch_run_heartbeat(run: dict) -> None:
    run["heartbeat_at"] = datetime.now().isoformat(timespec="seconds")


def _active_run() -> dict | None:
    _restore_runs_once()
    with AUTOMATION_UI_LOCK:
        if not AUTOMATION_UI_ACTIVE_RUN_ID:
            return None
        run = AUTOMATION_UI_RUNS.get(AUTOMATION_UI_ACTIVE_RUN_ID)
        return run.copy() if run else None


def _latest_interrupted_run() -> dict | None:
    interrupted = _interrupted_runs()
    if not interrupted:
        return None
    return interrupted[0]


def _interrupted_runs() -> list[dict]:
    _restore_runs_once()
    with AUTOMATION_UI_LOCK:
        interrupted = [
            run.copy()
            for run in AUTOMATION_UI_RUNS.values()
            if str(run.get("status")) == "interrupted"
        ]
    interrupted.sort(key=lambda run: str(run.get("started_at") or ""), reverse=True)
    return interrupted


def _finalize_current_step(run: dict) -> None:
    step_logs = run.get("step_logs", [])
    if not isinstance(step_logs, list) or not step_logs:
        return
    last = step_logs[-1]
    if last.get("ended_at"):
        return
    ended_at = datetime.now().isoformat(timespec="seconds")
    started_at = pd.to_datetime(last.get("started_at"), errors="coerce")
    end_ts = pd.to_datetime(ended_at, errors="coerce")
    elapsed_sec = 0.0
    if pd.notna(started_at) and pd.notna(end_ts):
        elapsed_sec = max(0.0, float((end_ts - started_at).total_seconds()))
    last["ended_at"] = ended_at
    last["elapsed_sec"] = round(elapsed_sec, 2)


def _eta_seconds(run: dict) -> float | None:
    progress = float(run.get("progress") or 0.0)
    if progress <= 0.0 or progress >= 100.0:
        return 0.0 if progress >= 100.0 else None
    started = pd.to_datetime(run.get("started_at"), errors="coerce")
    now = pd.Timestamp.utcnow()
    if now.tzinfo is not None:
        now = now.tz_convert(None)
    if pd.isna(started):
        return None
    if started.tzinfo is not None:
        started = started.tz_convert(None)
    elapsed = max(0.0, float((now - started).total_seconds()))
    if elapsed <= 0:
        return None
    total_est = elapsed * (100.0 / progress)
    return max(0.0, total_est - elapsed)


def _start_background_automation(
    model_override: str | None,
    interval_override: int,
    resumed_from_run_id: str | None = None,
    resume_retry_count: int = 0,
    resume_retry_backoff_sec: int = 10,
) -> str | None:
    global AUTOMATION_UI_ACTIVE_RUN_ID
    _restore_runs_once()
    with AUTOMATION_UI_LOCK:
        if AUTOMATION_UI_ACTIVE_RUN_ID:
            active = AUTOMATION_UI_RUNS.get(AUTOMATION_UI_ACTIVE_RUN_ID)
            if active and str(active.get("status")) == "running":
                return None

        run_id = str(uuid.uuid4())[:8]
        AUTOMATION_UI_ACTIVE_RUN_ID = run_id
        AUTOMATION_UI_RUNS[run_id] = {
            "run_id": run_id,
            "status": "running",
            "progress": 0,
            "current_step": "Queued",
            "messages": [{"time": datetime.now().isoformat(timespec="seconds"), "message": "Run queued"}],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "result": None,
            "error": None,
            "cancel_requested": False,
            "heartbeat_at": datetime.now().isoformat(timespec="seconds"),
            "model_override": model_override,
            "interval_override": int(interval_override),
            "resumed_from": resumed_from_run_id,
            "resume_retry_count": int(max(0, resume_retry_count)),
            "resume_retry_backoff_sec": int(max(1, resume_retry_backoff_sec)),
            "resume_attempt": 0,
            "step_logs": [
                {
                    "step": "Queued",
                    "progress": 0,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "ended_at": None,
                    "elapsed_sec": None,
                }
            ],
        }
        _persist_runs()

    def _worker() -> None:
        def _progress_cb(progress: int, message: str) -> None:
            with AUTOMATION_UI_LOCK:
                run = AUTOMATION_UI_RUNS.get(run_id)
                if not run:
                    return
                _finalize_current_step(run)
                run["progress"] = max(0, min(100, int(progress)))
                run["current_step"] = str(message)
                _touch_run_heartbeat(run)
                run["messages"].append(
                    {"time": datetime.now().isoformat(timespec="seconds"), "message": str(message)}
                )
                run["step_logs"].append(
                    {
                        "step": str(message),
                        "progress": int(progress),
                        "started_at": datetime.now().isoformat(timespec="seconds"),
                        "ended_at": None,
                        "elapsed_sec": None,
                    }
                )
                _persist_runs()

        def _cancel_check() -> bool:
            with AUTOMATION_UI_LOCK:
                run = AUTOMATION_UI_RUNS.get(run_id)
                return bool(run and run.get("cancel_requested"))

        max_retries = int(max(0, resume_retry_count)) if resumed_from_run_id else 0
        backoff_base = int(max(1, resume_retry_backoff_sec))
        attempt = 0

        while True:
            attempt += 1
            with AUTOMATION_UI_LOCK:
                run = AUTOMATION_UI_RUNS.get(run_id)
                if run:
                    run["resume_attempt"] = attempt
                    _touch_run_heartbeat(run)
                    _persist_runs()

            try:
                result = automation_service.run_cycle(
                    model_override=model_override,
                    interval_minutes_override=int(interval_override),
                    progress_cb=_progress_cb,
                    cancel_check=_cancel_check,
                )

                status = str(result.get("status"))
                should_retry = status == "failed" and attempt <= max_retries
                if should_retry:
                    sleep_sec = backoff_base * (2 ** (attempt - 1))
                    with AUTOMATION_UI_LOCK:
                        run = AUTOMATION_UI_RUNS.get(run_id)
                        if run:
                            run["messages"].append(
                                {
                                    "time": datetime.now().isoformat(timespec="seconds"),
                                    "message": (
                                        f"Resume attempt {attempt} failed. Retrying in {sleep_sec}s "
                                        f"({attempt}/{max_retries + 1})."
                                    ),
                                }
                            )
                            run["current_step"] = f"Retry backoff {sleep_sec}s"
                            _touch_run_heartbeat(run)
                            _persist_runs()
                    time.sleep(sleep_sec)
                    continue

                with AUTOMATION_UI_LOCK:
                    run = AUTOMATION_UI_RUNS.get(run_id)
                    if run:
                        _finalize_current_step(run)
                        is_canceled = status == "canceled"
                        is_failed = status == "failed"
                        run["status"] = "canceled" if is_canceled else ("failed" if is_failed else "completed")
                        run["progress"] = int(run.get("progress", 0)) if (is_canceled or is_failed) else 100
                        run["current_step"] = "Canceled" if is_canceled else ("Failed" if is_failed else "Completed")
                        _touch_run_heartbeat(run)
                        run["result"] = result
                        run["error"] = result.get("error") if isinstance(result, dict) else None
                        run["finished_at"] = datetime.now().isoformat(timespec="seconds")
                        _persist_runs()
                break
            except Exception as exc:
                should_retry = attempt <= max_retries
                if should_retry:
                    sleep_sec = backoff_base * (2 ** (attempt - 1))
                    with AUTOMATION_UI_LOCK:
                        run = AUTOMATION_UI_RUNS.get(run_id)
                        if run:
                            run["messages"].append(
                                {
                                    "time": datetime.now().isoformat(timespec="seconds"),
                                    "message": (
                                        f"Resume attempt {attempt} crashed: {str(exc)}. "
                                        f"Retrying in {sleep_sec}s ({attempt}/{max_retries + 1})."
                                    ),
                                }
                            )
                            run["current_step"] = f"Retry backoff {sleep_sec}s"
                            _touch_run_heartbeat(run)
                            _persist_runs()
                    time.sleep(sleep_sec)
                    continue

                with AUTOMATION_UI_LOCK:
                    run = AUTOMATION_UI_RUNS.get(run_id)
                    if run:
                        _finalize_current_step(run)
                        run["status"] = "failed"
                        run["current_step"] = "Failed"
                        _touch_run_heartbeat(run)
                        run["error"] = str(exc)
                        run["finished_at"] = datetime.now().isoformat(timespec="seconds")
                        _persist_runs()
                break

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return run_id


def _to_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _format_dt(value) -> str:
    if value is None:
        return "N/A"
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def _source_breakdown_df(breakdown: dict) -> pd.DataFrame:
    if not isinstance(breakdown, dict):
        return pd.DataFrame()
    rows = [
        {"Source": "defaults", "Count": int(breakdown.get("defaults_count", 0))},
        {"Source": "global", "Count": int(breakdown.get("global_count", 0))},
        {"Source": "news", "Count": int(breakdown.get("news_count", 0))},
        {"Source": "combined_unique", "Count": int(breakdown.get("combined_unique_count", 0))},
        {"Source": "final_selected", "Count": int(breakdown.get("final_selected_count", 0))},
    ]
    return pd.DataFrame(rows)


def _historical_source_breakdown_df(breakdown: dict) -> pd.DataFrame:
    if not isinstance(breakdown, dict) or not breakdown:
        return pd.DataFrame()
    rows = [{"Source": str(source), "Count": int(count)} for source, count in breakdown.items()]
    out = pd.DataFrame(rows)
    return out.sort_values("Count", ascending=False).reset_index(drop=True)


def _upsert_env_values(env_path: Path, updates: dict[str, str]) -> bool:
    if not env_path.exists():
        return False

    lines = env_path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    out_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            out_lines.append(line)
            continue

        key = line.split("=", 1)[0].strip()
        if key in updates:
            out_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            out_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return True


def render_automation_monitor() -> None:
    _restore_runs_once()
    settings = get_settings()

    st.header("Automation Monitor")
    st.caption("Track last auto-run, inspect top suggestions, source mix, and trigger override runs.")

    st.subheader("Quick Switches")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        configured_model = str(settings.automation_prediction_model or "auto").strip()
        default_model = configured_model if configured_model in AUTOMATION_MODEL_OPTIONS else "auto"
        model_override = st.selectbox(
            "Prediction model",
            options=AUTOMATION_MODEL_OPTIONS,
            index=AUTOMATION_MODEL_OPTIONS.index(default_model),
            key="automation_model_override",
        )
    with col2:
        interval_override = st.number_input(
            "Interval (minutes)",
            min_value=1,
            max_value=1440,
            value=int(settings.automation_interval_minutes),
            step=1,
            key="automation_interval_override",
        )
    with col3:
        run_now = st.button("Start Automation", use_container_width=True, type="primary")

    col4, col5, col6 = st.columns([2, 1, 1])
    with col4:
        auto_resume_on_startup = st.checkbox(
            "Auto-resume interrupted run on startup",
            value=bool(settings.automation_auto_resume_on_startup),
            key="automation_auto_resume_on_startup",
        )
    with col5:
        resume_retry_count = st.number_input(
            "Resume retries",
            min_value=0,
            max_value=10,
            value=int(settings.automation_resume_retry_count),
            step=1,
            key="automation_resume_retry_count",
        )
    with col6:
        resume_retry_backoff_sec = st.number_input(
            "Retry backoff (sec)",
            min_value=1,
            max_value=600,
            value=int(settings.automation_resume_retry_backoff_sec),
            step=1,
            key="automation_resume_retry_backoff_sec",
        )

    interrupted_runs = _interrupted_runs()
    interrupted_map = {
        f"{run.get('run_id')} | {run.get('started_at', '-')}": run for run in interrupted_runs
    }
    interrupted_labels = list(interrupted_map.keys())
    selected_label = st.selectbox(
        "Interrupted run to resume",
        options=interrupted_labels if interrupted_labels else ["No interrupted runs available"],
        index=0,
        disabled=not interrupted_labels,
        key="automation_interrupted_run_picker",
    )
    resume_now = st.button(
        "Resume Selected Interrupted Run",
        use_container_width=True,
        disabled=not interrupted_labels,
    )

    if "automation_auto_resume_checked" not in st.session_state:
        st.session_state["automation_auto_resume_checked"] = False

    save_defaults = st.button("Save Defaults To .env", use_container_width=True)

    if save_defaults:
        env_path = Path(".env")
        ok = _upsert_env_values(
            env_path,
            {
                "AUTOMATION_PREDICTION_MODEL": str(model_override),
                "AUTOMATION_INTERVAL_MINUTES": str(int(interval_override)),
                "AUTOMATION_AUTO_RESUME_ON_STARTUP": "true" if bool(auto_resume_on_startup) else "false",
                "AUTOMATION_RESUME_RETRY_COUNT": str(int(resume_retry_count)),
                "AUTOMATION_RESUME_RETRY_BACKOFF_SEC": str(int(resume_retry_backoff_sec)),
                "OPENAI_PREDICT_ENABLED": "true",
            },
        )
        if ok:
            st.success("Saved defaults to .env. Restart FastAPI/Streamlit to apply scheduler interval updates.")
        else:
            st.error("Could not find .env file to update.")

    if run_now:
        run_id = _start_background_automation(
            model_override=model_override if model_override != "auto" else None,
            interval_override=int(interval_override),
            resume_retry_count=0,
            resume_retry_backoff_sec=int(resume_retry_backoff_sec),
        )
        if run_id is None:
            st.warning("Automation already running. Please wait for current run to finish.")
        else:
            st.success(f"Automation started in background. Run ID: {run_id}")

    if (
        bool(auto_resume_on_startup)
        and not st.session_state.get("automation_auto_resume_checked", False)
        and _active_run() is None
        and interrupted_runs
    ):
        auto_target = interrupted_runs[0]
        resumed_model = auto_target.get("model_override")
        if resumed_model in {"", "auto"}:
            resumed_model = None
        resumed_interval = int(auto_target.get("interval_override") or int(settings.automation_interval_minutes))

        auto_run_id = _start_background_automation(
            model_override=resumed_model,
            interval_override=resumed_interval,
            resumed_from_run_id=str(auto_target.get("run_id")),
            resume_retry_count=int(resume_retry_count),
            resume_retry_backoff_sec=int(resume_retry_backoff_sec),
        )
        st.session_state["automation_auto_resume_checked"] = True
        if auto_run_id is not None:
            st.info(
                f"Auto-resume started for interrupted run {auto_target.get('run_id')} as {auto_run_id}."
            )

    selected_interrupted_run = interrupted_map.get(selected_label) if interrupted_labels else None
    if resume_now and selected_interrupted_run is not None:
        resumed_model = selected_interrupted_run.get("model_override")
        if resumed_model in {"", "auto"}:
            resumed_model = None
        resumed_interval = int(
            selected_interrupted_run.get("interval_override") or int(settings.automation_interval_minutes)
        )

        run_id = _start_background_automation(
            model_override=resumed_model,
            interval_override=resumed_interval,
            resumed_from_run_id=str(selected_interrupted_run.get("run_id")),
            resume_retry_count=int(resume_retry_count),
            resume_retry_backoff_sec=int(resume_retry_backoff_sec),
        )
        if run_id is None:
            st.warning("Automation already running. Please wait for current run to finish.")
        else:
            st.success(
                f"Resumed interrupted run {selected_interrupted_run.get('run_id')} as new run {run_id}."
            )

    active_run = _active_run()
    if active_run:
        st.subheader("Live Automation Progress")
        st.progress(int(active_run.get("progress", 0)), text=str(active_run.get("current_step", "Running")))
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Run ID", str(active_run.get("run_id", "-")))
        c2.metric("Status", str(active_run.get("status", "unknown")).upper())
        c3.metric("Started", str(active_run.get("started_at", "-")))
        c4.metric("Finished", str(active_run.get("finished_at") or "-"))

        started = pd.to_datetime(active_run.get("started_at"), errors="coerce")
        elapsed_sec = 0.0
        if pd.notna(started):
            now = pd.Timestamp.utcnow()
            if now.tzinfo is not None:
                now = now.tz_convert(None)
            if started.tzinfo is not None:
                started = started.tz_convert(None)
            elapsed_sec = max(0.0, float((now - started).total_seconds()))
        eta = _eta_seconds(active_run)
        c5.metric("ETA", "-" if eta is None else f"{int(eta)}s")
        c6.metric("Heartbeat", str(active_run.get("heartbeat_at") or "-"))
        st.caption(f"Elapsed: {int(elapsed_sec)}s")

        run_status = str(active_run.get("status"))
        if run_status == "running":
            if st.button("Cancel Running Automation", type="secondary", use_container_width=True):
                with AUTOMATION_UI_LOCK:
                    run = AUTOMATION_UI_RUNS.get(str(active_run.get("run_id")))
                    if run:
                        run["cancel_requested"] = True
                        run["messages"].append(
                            {
                                "time": datetime.now().isoformat(timespec="seconds"),
                                "message": "Cancel requested by user",
                            }
                        )
                        _persist_runs()
                st.warning("Cancel requested. Current step will stop at the next safe checkpoint.")

        step_logs = active_run.get("step_logs", [])
        if isinstance(step_logs, list) and step_logs:
            step_df = pd.DataFrame(step_logs).tail(10)
            st.caption("Per-step timings")
            st.dataframe(step_df, use_container_width=True, hide_index=True)

        messages = active_run.get("messages", [])
        if isinstance(messages, list) and messages:
            msg_df = pd.DataFrame(messages).tail(20)
            st.dataframe(msg_df, use_container_width=True, hide_index=True)

        if str(active_run.get("status")) == "failed":
            st.error(f"Automation failed: {active_run.get('error')}")
        elif str(active_run.get("status")) == "canceled":
            st.warning("Automation was canceled")
        elif str(active_run.get("status")) == "interrupted":
            st.warning("Automation was interrupted (app restart/process stop). Last known progress has been restored.")

        if str(active_run.get("status")) == "running":
            components.html(
                """
                <script>
                    setTimeout(function() {
                        window.parent.location.reload();
                    }, 2500);
                </script>
                """,
                height=0,
                width=0,
            )

    st.subheader("Run History")
    with AUTOMATION_UI_LOCK:
        history_rows = list(AUTOMATION_UI_RUNS.values())
    history_rows.sort(key=lambda x: str(x.get("started_at") or ""), reverse=True)
    history_rows = history_rows[:30]
    if history_rows:
        hist_df = pd.DataFrame(
            [
                {
                    "run_id": row.get("run_id"),
                    "status": row.get("status"),
                    "progress": row.get("progress"),
                    "started_at": row.get("started_at"),
                    "finished_at": row.get("finished_at"),
                    "step": row.get("current_step"),
                    "resumed_from": row.get("resumed_from"),
                    "resume_attempt": row.get("resume_attempt"),
                }
                for row in history_rows
            ]
        )
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("No run history available yet.")

    latest = automation_service.latest()

    st.subheader("Last Run Status/Time")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", str(latest.get("status", "unknown")))
    m2.metric("Started", _format_dt(latest.get("started_at")))
    m3.metric("Completed", _format_dt(latest.get("completed_at")))
    m4.metric("Model", str(latest.get("prediction_model", "n/a")))

    st.caption(f"Configured/last interval: {latest.get('interval_minutes', 'n/a')} minute(s)")

    st.subheader("Source Breakdown")
    breakdown_df = _source_breakdown_df(latest.get("source_breakdown", {}))
    if breakdown_df.empty:
        st.info("No source breakdown available yet.")
    else:
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    st.subheader("Current Market Summary")
    market_summary = automation_service.get_market_summary()
    if market_summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Universe Symbols", market_summary.get("total_universe_symbols", 0))
        m2.metric("Default Symbols", market_summary.get("defaults_count", 0))
        m3.metric("Global Symbols", market_summary.get("global_symbols_count", 0))
        m4.metric("News-Driven", market_summary.get("news_driven_symbols", 0))
        
        n1, n2, n3 = st.columns(3)
        n1.metric("Total News Records", market_summary.get("total_news_records", 0))
        n2.metric("Active Sources", market_summary.get("sources_active", 0))
        n3.caption(f"Updated: {market_summary.get('last_refresh', 'N/A')[:19]}")
    else:
        st.info("Market summary not available yet.")

    st.subheader("Latest News-Driven Market Opportunities")
    news_impacts = automation_service.get_latest_news_impact(limit=15)
    if news_impacts:
        news_df = _to_dataframe(news_impacts)
        preferred_cols = ["ticker", "overall_score", "sentiment_score", "impact_type", "news_title", "published_at"]
        cols = [col for col in preferred_cols if col in news_df.columns]
        st.dataframe(news_df[cols], use_container_width=True, height=300)
    else:
        st.info("No recent news-driven opportunities. Run news refresh in automation cycle.")

    st.subheader("Historical Ingest Source Breakdown")
    historical_df = _historical_source_breakdown_df(latest.get("historical_source_breakdown", {}))
    if historical_df.empty:
        st.info("No historical source usage available yet.")
    else:
        st.dataframe(historical_df, use_container_width=True, hide_index=True)

    ingest_summary = latest.get("historical_ingest_summary", {})
    if isinstance(ingest_summary, dict) and ingest_summary:
        c1, c2 = st.columns(2)
        c1.metric("Symbols With Rows", int(ingest_summary.get("symbols_with_rows", 0)))
        c2.metric("Symbols Without Rows", int(ingest_summary.get("symbols_without_rows", 0)))

    st.subheader("Top Suggestions Table")
    top_df = _to_dataframe(latest.get("top_suggestions", []))
    if top_df.empty:
        st.info("No suggestions available yet. Run automation cycle from above.")
    else:
        preferred = [
            "symbol",
            "blended_score",
            "confidence",
            "predicted_return",
            "decision",
            "news_opportunity_score",
            "target_price",
            "stop_loss_price",
            "current_price",
        ]
        cols = [col for col in preferred if col in top_df.columns] + [col for col in top_df.columns if col not in preferred]
        st.dataframe(top_df[cols], use_container_width=True, height=360)

    st.subheader("AI Summary (Predictions + News)")
    st.caption("Use OpenAI to summarize all current stock predictions and latest market news impacts.")

    s1, s2, s3 = st.columns([1, 1, 2])
    with s1:
        summary_predictions_limit = st.number_input(
            "Predictions used",
            min_value=5,
            max_value=100,
            value=20,
            step=1,
            key="automation_summary_predictions_limit",
        )
    with s2:
        summary_news_limit = st.number_input(
            "News items used",
            min_value=5,
            max_value=200,
            value=25,
            step=1,
            key="automation_summary_news_limit",
        )
    with s3:
        generate_summary = st.button("Generate AI Summary", type="primary", use_container_width=True)

    if generate_summary:
        with st.spinner("Generating AI summary via OpenAI..."):
            ai_summary = automation_service.generate_ai_summary(
                predictions_limit=int(summary_predictions_limit),
                news_limit=int(summary_news_limit),
            )
            st.session_state["automation_ai_summary"] = ai_summary

    ai_summary_latest = st.session_state.get("automation_ai_summary") or automation_service.latest_ai_summary()
    if isinstance(ai_summary_latest, dict):
        summary_status = str(ai_summary_latest.get("status") or "unknown")
        st.caption(
            f"Status: {summary_status.upper()} | Model: {ai_summary_latest.get('model') or '-'} | "
            f"Generated: {ai_summary_latest.get('generated_at') or '-'}"
        )

        if summary_status == "missing_api_key":
            st.warning(str(ai_summary_latest.get("summary") or "OpenAI API key not configured."))
        elif summary_status in {"error", "corrupt"}:
            st.error(str(ai_summary_latest.get("error") or ai_summary_latest.get("summary") or "Summary generation failed."))
        elif summary_status == "empty":
            st.info(str(ai_summary_latest.get("summary") or "No data available for summary."))
        else:
            st.markdown(str(ai_summary_latest.get("summary") or "No summary text."))

            highlights = list(ai_summary_latest.get("highlights") or [])
            risks = list(ai_summary_latest.get("risks") or [])
            actions = list(ai_summary_latest.get("actions") or [])

            h1, h2, h3 = st.columns(3)
            with h1:
                st.caption("Highlights")
                if highlights:
                    for item in highlights[:8]:
                        st.write(f"- {item}")
                else:
                    st.write("- None")
            with h2:
                st.caption("Risks")
                if risks:
                    for item in risks[:8]:
                        st.write(f"- {item}")
                else:
                    st.write("- None")
            with h3:
                st.caption("Actions")
                if actions:
                    for item in actions[:8]:
                        st.write(f"- {item}")
                else:
                    st.write("- None")

    st.subheader("Model Training on Expanded Universe")
    st.caption("Train ML models on all symbols (defaults + global + news-driven) for more robust predictions.")
    
    train_col1, train_col2, train_col3 = st.columns([2, 2, 1])
    with train_col1:
        train_task = st.selectbox(
            "Task Type",
            options=["classification", "regression_return", "regression_close"],
            index=0,
            key="train_task_type",
        )
    with train_col2:
        train_horizon = st.number_input(
            "Horizon Days",
            min_value=1,
            max_value=30,
            value=int(settings.automation_horizon_days),
            step=1,
            key="train_horizon_days",
        )
    with train_col3:
        train_now = st.button("Train Model", use_container_width=True)

    if train_now:
        with st.spinner("Training on expanded universe (this may take several minutes)..."):
            train_result = automation_service.train_on_expanded_universe(
                horizon_days=int(train_horizon),
                task_type=train_task,
            )
        
        if train_result.get("status") == "ok":
            st.success(
                f"✅ Training complete | Symbols trained: {train_result.get('symbols_count', 0)} | "
                f"Defaults: {train_result.get('source_breakdown', {}).get('defaults_count', 0)}, "
                f"Global: {train_result.get('source_breakdown', {}).get('global_count', 0)}"
            )
            if train_result.get("training_result"):
                tr = train_result["training_result"]
                with st.expander("Training Metrics"):
                    st.json({
                        "model_name": tr.get("best_model_name"),
                        "task_type": tr.get("task_type"),
                        "metric_value": tr.get("best_cv_metric"),
                        "features_used": len(tr.get("feature_list", [])),
                    })
        else:
            error_msg = train_result.get("error", "Unknown error")
            st.error(f"❌ Training failed: {error_msg}")
