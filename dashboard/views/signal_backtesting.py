from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.news.backtest_service import news_signal_backtest_service


def _as_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["signal_created_at", "target_date", "evaluated_at", "created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _best_label(rows: list[dict], key: str) -> str:
    if not rows:
        return "N/A"
    top = rows[0]
    return str(top.get(key) or "N/A")


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.caption(f"No data for {title}.")
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(df[x].astype(str).tolist(), pd.to_numeric(df[y], errors="coerce").fillna(0.0).tolist())
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def _performance_section(title: str, rows: list[dict], key_col: str) -> None:
    st.markdown(f"### {title}")
    if not rows:
        st.caption("No completed outcomes yet.")
        return
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, height=220)
    _plot_bar(frame, key_col, "hit_rate", f"Hit Rate by {title}")
    _plot_bar(frame, key_col, "avg_return", f"Average Return by {title}")


def _history_with_horizons(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df

    base_cols = [
        "signal_created_at",
        "primary_company",
        "company",
        "beneficiary_ticker",
        "event_type",
        "relation",
        "opportunity_overall_score",
        "timing_label",
        "evaluation_status",
    ]
    available_base = [c for c in base_cols if c in history_df.columns]

    returns = history_df.copy()
    returns["horizon_col"] = returns["evaluation_horizon_days"].apply(lambda v: f"{int(v)}d_ret" if pd.notna(v) else "ret")
    pivot = (
        returns.pivot_table(
            index=[c for c in available_base if c != "evaluation_status"],
            columns="horizon_col",
            values="percent_return",
            aggfunc="first",
        )
        .reset_index()
    )

    status_group = (
        returns.groupby([c for c in available_base if c != "evaluation_status"], dropna=False)["evaluation_status"]
        .agg(lambda s: "completed" if (s == "completed").any() else ("pending" if (s == "pending").any() else "failed"))
        .reset_index()
    )

    merged = pivot.merge(status_group, on=[c for c in available_base if c != "evaluation_status"], how="left")
    return merged.sort_values("signal_created_at", ascending=False)


def render_signal_backtesting(settings) -> None:
    st.subheader("Signal Backtesting")
    st.caption("Track whether news-driven beneficiary opportunities worked after signal generation.")

    if not settings.backtest_enabled:
        st.info("Signal backtesting is disabled in settings.")
        return

    controls = st.columns(4)
    if controls[0].button("Run Pending Evaluation", type="primary"):
        with st.spinner("Evaluating pending outcomes..."):
            summary = news_signal_backtest_service.evaluate_pending_outcomes()
        st.success(f"Evaluation completed: {summary}")

    max_rows = controls[1].number_input(
        "Max rows",
        min_value=50,
        max_value=5000,
        value=int(settings.max_backtest_rows_in_ui),
        step=50,
        key="signal_backtest_max_rows",
    )

    horizon_filter = controls[2].multiselect(
        "Horizons",
        options=[int(h) for h in settings.backtest_horizons],
        default=[int(h) for h in settings.backtest_horizons],
        key="signal_backtest_horizons",
    )

    min_score = controls[3].slider(
        "Min overall score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="signal_backtest_min_score",
    )

    date_controls = st.columns(3)
    default_start = (datetime.utcnow() - timedelta(days=180)).date()
    start_date = date_controls[0].date_input("Start date", value=default_start, key="signal_backtest_start_date")
    end_date = date_controls[1].date_input("End date", value=datetime.utcnow().date(), key="signal_backtest_end_date")
    min_conf = date_controls[2].slider("Min confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="signal_backtest_min_conf")

    rows = news_signal_backtest_service.signal_history(limit=int(max_rows))
    frame = _as_df(rows)

    event_options = sorted(frame["event_type"].dropna().astype(str).unique().tolist()) if not frame.empty and "event_type" in frame.columns else []
    sector_options = sorted(frame["sector"].dropna().astype(str).unique().tolist()) if not frame.empty and "sector" in frame.columns else []
    sentiment_options = sorted(frame["sentiment_label"].dropna().astype(str).unique().tolist()) if not frame.empty and "sentiment_label" in frame.columns else []
    timing_options = sorted(frame["timing_label"].dropna().astype(str).unique().tolist()) if not frame.empty and "timing_label" in frame.columns else []
    relation_options = sorted(frame["relation"].dropna().astype(str).unique().tolist()) if not frame.empty and "relation" in frame.columns else []

    filter_controls = st.columns(5)
    selected_events = filter_controls[0].multiselect("Event type", options=event_options, default=event_options, key="signal_backtest_events")
    selected_sectors = filter_controls[1].multiselect("Sector", options=sector_options, default=sector_options, key="signal_backtest_sector")
    selected_sentiment = filter_controls[2].multiselect("Sentiment", options=sentiment_options, default=sentiment_options, key="signal_backtest_sentiment")
    selected_timing = filter_controls[3].multiselect("Timing", options=timing_options, default=timing_options, key="signal_backtest_timing")
    selected_relation = filter_controls[4].multiselect("Relation type", options=relation_options, default=relation_options, key="signal_backtest_relation")

    filters = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "event_type": selected_events,
        "sector": selected_sectors,
        "sentiment_label": selected_sentiment,
        "timing_label": selected_timing,
        "relation": selected_relation,
        "min_confidence": float(min_conf),
        "min_overall_score": float(min_score),
        "horizon_days": [int(v) for v in horizon_filter] if horizon_filter else None,
    }

    summary = news_signal_backtest_service.summary(filters=filters)
    history = news_signal_backtest_service.signal_history(filters=filters, limit=int(max_rows))

    totals = summary.totals
    cards = st.columns(8)
    cards[0].metric("Total Signals", int(totals.get("total_signals", 0)))
    cards[1].metric("Evaluated", int(totals.get("evaluated_signals", 0)))
    cards[2].metric("Pending", int(totals.get("pending_signals", 0)))
    cards[3].metric("Hit Rate", f"{float(totals.get('hit_rate', 0.0)) * 100.0:.1f}%")
    cards[4].metric("Avg Return", f"{float(totals.get('avg_return', 0.0)):.2f}%")
    cards[5].metric("Best Event", _best_label(summary.by_event_type, "event_type"))
    cards[6].metric("Best Sector", _best_label(summary.by_sector, "sector"))
    cards[7].metric("Best Relation", _best_label(summary.by_relation_type, "relation"))

    _performance_section("Horizon", summary.by_horizon, "evaluation_horizon_days")
    _performance_section("Event Type", summary.by_event_type, "event_type")
    _performance_section("Sector", summary.by_sector, "sector")
    _performance_section("Relation Type", summary.by_relation_type, "relation")
    _performance_section("Timing Label", summary.by_timing_label, "timing_label")
    _performance_section("Score Bucket", summary.by_score_bucket, "score_bucket")

    st.markdown("### Signal History")
    history_df = _as_df(history)
    if history_df.empty:
        st.info("No signal outcomes found for selected filters.")
        return

    merged = _history_with_horizons(history_df)
    preferred_cols = [
        "signal_created_at",
        "primary_company",
        "company",
        "beneficiary_ticker",
        "event_type",
        "relation",
        "opportunity_overall_score",
        "timing_label",
        "1d_ret",
        "3d_ret",
        "5d_ret",
        "7d_ret",
        "evaluation_status",
    ]
    cols = [c for c in preferred_cols if c in merged.columns] + [c for c in merged.columns if c not in preferred_cols]
    st.dataframe(merged[cols], use_container_width=True, height=340)

    completed = history_df[history_df["evaluation_status"] == "completed"].copy()
    if not completed.empty and "signal_created_at" in completed.columns:
        daily = (
            completed.sort_values("signal_created_at")
            .groupby(completed["signal_created_at"].dt.date, as_index=False)["percent_return"]
            .mean()
        )
        daily["cum_return"] = (1.0 + (pd.to_numeric(daily["percent_return"], errors="coerce").fillna(0.0) / 100.0)).cumprod() - 1.0
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(daily["signal_created_at"], daily["cum_return"], marker="o")
        ax.set_title("Cumulative Simple Strategy Return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
