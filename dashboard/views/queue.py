from __future__ import annotations

import pandas as pd
import streamlit as st


def render_queue(
    queue_jobs: list[dict],
    queued_count: int,
    running_count: int,
    completed_count: int,
    failed_count: int,
    status_badge,
    get_job,
    reset_queue,
    bulk_plan_path,
) -> None:
    st.subheader("Training Queue")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Queued", queued_count)
    q2.metric("Running", running_count)
    q3.metric("Completed", completed_count)
    q4.metric("Failed", failed_count)

    tool1, tool2 = st.columns(2)
    if tool1.button("Refresh queue status"):
        st.cache_data.clear()
        st.rerun()
    if tool2.button("Clear Queue"):
        reset_queue()
        if bulk_plan_path.exists():
            bulk_plan_path.unlink()
        st.warning("Queue cleared.")
        st.cache_data.clear()
        st.rerun()

    if queue_jobs:
        total_jobs = len(queue_jobs)
        done_jobs = completed_count + failed_count
        progress_value = done_jobs / total_jobs if total_jobs else 0.0
        st.progress(progress_value, text=f"Queue progress: {done_jobs}/{total_jobs} jobs finished")

        queue_view = pd.DataFrame(queue_jobs)
        queue_view["status_badge"] = queue_view["status"].map(status_badge)
        st.dataframe(
            queue_view[["job_id", "symbol", "status_badge", "created_at", "started_at", "finished_at", "message"]].sort_values(
                ["created_at", "symbol"], ascending=[False, True]
            ),
            use_container_width=True,
            height=300,
        )

        selected_job_id = st.selectbox(
            "Inspect job",
            options=[job["job_id"] for job in queue_jobs],
            format_func=lambda job_id: f"{job_id} - {get_job(job_id)['symbol']}" if get_job(job_id) else job_id,
        )
        if selected_job_id:
            selected_job = get_job(selected_job_id)
            if selected_job:
                st.json(selected_job)
    else:
        st.info("No background training jobs yet.")
