from __future__ import annotations

import json

import streamlit as st


def render_backtest(settings) -> None:
    st.subheader("Backtest Equity Curve")
    curve = settings.output_dir / "backtest_equity_curve.png"
    if curve.exists():
        st.image(str(curve), use_container_width=True)
    else:
        st.info("No backtest curve yet. Trigger /backtest or run backtest CLI.")

    st.subheader("Backtest Metrics")
    metrics_path = settings.output_dir / "backtest_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.json(metrics)
    else:
        st.info("No backtest metrics available.")
