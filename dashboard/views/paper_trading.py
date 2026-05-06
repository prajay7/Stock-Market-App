from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from app.core.config import get_settings
from app.news.models import PaperTradeClose, PaperTradeCreate
from app.services.paper_trading_service import paper_trading_service
from src.data.metadata_store import metadata_store

settings = get_settings()


def _format_dt(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%Y-%m-%d %H:%M")


def _trade_dataframe(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    frame = pd.DataFrame(trades)
    for column in ["entry_date", "exit_date", "created_at", "updated_at"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def render_paper_trading() -> None:
    st.header("📈 Paper Trading & Decision Journal")
    st.caption("Create, track, and review simulated trades from alerts, opportunities, or manual entries.")

    if "paper_trade_create_source" not in st.session_state:
        st.session_state["paper_trade_create_source"] = "manual"

    if st.button("🔄 Refresh open trade prices"):
        updated = paper_trading_service.refresh_open_trades(limit=settings.paper_trading_price_refresh_limit)
        st.success(f"Updated {updated} open trades.")
        st.rerun()

    tab_create, tab_open, tab_closed, tab_analytics = st.tabs(["Create Trade", "Open Trades", "Closed Trades", "Analytics"])

    with tab_create:
        _render_create_trade_tab()

    with tab_open:
        _render_open_trades_tab()

    with tab_closed:
        _render_closed_trades_tab()

    with tab_analytics:
        _render_analytics_tab()


def _render_create_trade_tab() -> None:
    st.subheader("Create Trade")
    source = st.radio(
        "Source",
        ["manual", "alert", "opportunity"],
        horizontal=True,
        key="paper_trade_create_source",
    )

    if source == "manual":
        _render_manual_trade_form()
    elif source == "alert":
        _render_alert_trade_form()
    else:
        _render_opportunity_trade_form()


def _render_manual_trade_form() -> None:
    with st.form("paper_trade_manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Symbol", value="AAPL")
            entry_price = st.number_input("Entry price", min_value=0.01, value=100.0, step=0.5)
            quantity = st.number_input("Quantity", min_value=0.0, value=0.0, step=1.0)
        with col2:
            capital = st.number_input("Capital", min_value=0.0, value=float(settings.paper_trading_default_capital), step=100.0)
            entry_date = st.date_input("Entry date", value=datetime.now().date())
            trade_reason = st.text_input("Trade reason", value="")
        notes = st.text_area("Notes", value="")
        submitted = st.form_submit_button("Create Manual Trade")
        if submitted:
            try:
                payload = PaperTradeCreate(
                    symbol=symbol,
                    entry_price=float(entry_price),
                    quantity=float(quantity) if quantity > 0 else None,
                    capital=float(capital) if capital > 0 else None,
                    source_type="manual",
                    source_label="Manual trade",
                    entry_date=datetime.combine(entry_date, datetime.min.time()),
                    notes=notes,
                    trade_reason=trade_reason,
                )
                trade_id = paper_trading_service.create_trade(payload)
                st.success(f"Trade #{trade_id} created.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")


def _render_alert_trade_form() -> None:
    alerts = metadata_store.read_recent_alerts(hours=72, limit=200)
    if not alerts:
        st.info("No alerts available yet.")
        return

    alert_options: dict[str, dict] = {}
    for alert in alerts:
        ctx = metadata_store.read_alert_with_context(int(alert["id"]))
        if not ctx:
            continue
        label = f"#{alert['id']} | {ctx.get('ticker') or ctx.get('primary_ticker') or 'N/A'} | {alert.get('title', 'Alert')}"
        alert_options[label] = ctx

    if not alert_options:
        st.info("No alert has a tradable ticker yet.")
        return

    with st.form("paper_trade_alert_form"):
        selected_label = st.selectbox("Alert", list(alert_options.keys()))
        selected = alert_options[selected_label]
        col1, col2 = st.columns(2)
        with col1:
            entry_price = st.number_input("Entry price", min_value=0.01, value=1.0, step=0.5)
            quantity = st.number_input("Quantity", min_value=0.0, value=0.0, step=1.0)
        with col2:
            capital = st.number_input("Capital", min_value=0.0, value=float(settings.paper_trading_default_capital), step=100.0)
            trade_reason = st.text_input("Trade reason", value=str(selected.get("message") or selected.get("title") or ""))
        notes = st.text_area("Notes", value="")
        submitted = st.form_submit_button("Create Trade from Alert")
        if submitted:
            try:
                trade_id = paper_trading_service.create_trade_from_alert(
                    int(selected["id"]),
                    entry_price=float(entry_price) if entry_price > 0 else None,
                    quantity=float(quantity) if quantity > 0 else None,
                    capital=float(capital) if capital > 0 else None,
                    notes=notes,
                    trade_reason=trade_reason,
                )
                st.success(f"Trade #{trade_id} created from alert.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")


def _render_opportunity_trade_form() -> None:
    opportunities = metadata_store.read_beneficiary_opportunities_with_signal(limit=200)
    if not opportunities:
        st.info("No opportunities available yet.")
        return

    option_map: dict[str, dict] = {}
    for opportunity in opportunities:
        label = (
            f"#{opportunity['id']} | {opportunity.get('ticker') or opportunity.get('beneficiary_ticker') or 'N/A'} | "
            f"{opportunity.get('company') or opportunity.get('primary_company') or 'Opportunity'}"
        )
        option_map[label] = opportunity

    with st.form("paper_trade_opportunity_form"):
        selected_label = st.selectbox("Opportunity", list(option_map.keys()))
        selected = option_map[selected_label]
        col1, col2 = st.columns(2)
        with col1:
            entry_price = st.number_input("Entry price", min_value=0.01, value=1.0, step=0.5)
            quantity = st.number_input("Quantity", min_value=0.0, value=0.0, step=1.0)
        with col2:
            capital = st.number_input("Capital", min_value=0.0, value=float(settings.paper_trading_default_capital), step=100.0)
            trade_reason = st.text_input("Trade reason", value=str(selected.get("reason") or selected.get("title") or ""))
        notes = st.text_area("Notes", value="")
        submitted = st.form_submit_button("Create Trade from Opportunity")
        if submitted:
            try:
                trade_id = paper_trading_service.create_trade_from_opportunity(
                    int(selected["id"]),
                    entry_price=float(entry_price) if entry_price > 0 else None,
                    quantity=float(quantity) if quantity > 0 else None,
                    capital=float(capital) if capital > 0 else None,
                    notes=notes,
                    trade_reason=trade_reason,
                )
                st.success(f"Trade #{trade_id} created from opportunity.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")


def _render_open_trades_tab() -> None:
    st.subheader("Open Trades")
    trades = metadata_store.read_open_paper_trades(limit=settings.max_paper_trades_in_ui)
    if not trades:
        st.info("No open trades yet.")
        return

    df = _trade_dataframe(trades)
    cols = [c for c in ["id", "symbol", "entry_date", "entry_price", "quantity", "capital", "current_price", "current_pnl", "current_return_pct", "holding_days", "source_type", "source_label"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, height=260)

    for trade in trades:
        with st.expander(f"{trade.get('symbol')} | Trade #{trade.get('id')} | Open", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Entry:** {_format_dt(trade.get('entry_date'))}")
                st.write(f"**Entry price:** {trade.get('entry_price')}")
                st.write(f"**Quantity:** {trade.get('quantity')}")
                st.write(f"**Capital:** {trade.get('capital')}")
                st.write(f"**Current price:** {trade.get('current_price') or 'Refreshing...'}")
            with col2:
                st.write(f"**Current PnL:** {trade.get('current_pnl') or 0:.2f}" if trade.get("current_pnl") is not None else "**Current PnL:** N/A")
                st.write(f"**Return %:** {trade.get('current_return_pct') or 0:.2f}" if trade.get("current_return_pct") is not None else "**Return %:** N/A")
                st.write(f"**Holding days:** {trade.get('holding_days') or 0}")
                st.write(f"**Source:** {trade.get('source_type')} | {trade.get('source_label') or '—'}")
            st.write(f"**Reason:** {trade.get('trade_reason') or '—'}")
            st.write(f"**Notes:** {trade.get('notes') or '—'}")

            with st.form(f"close_trade_form_{trade['id']}"):
                close_price_default = float(trade.get("current_price") or trade.get("entry_price") or 0.0)
                exit_price = st.number_input("Exit price", min_value=0.0, value=close_price_default, step=0.5, key=f"exit_price_{trade['id']}")
                exit_notes = st.text_input("Exit notes", value="", key=f"exit_notes_{trade['id']}")
                submitted = st.form_submit_button("Close Trade")
                if submitted:
                    try:
                        paper_trading_service.close_trade(
                            int(trade["id"]),
                            PaperTradeClose(exit_price=float(exit_price) if exit_price > 0 else None, notes=exit_notes),
                        )
                        st.success("Trade closed.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")


def _render_closed_trades_tab() -> None:
    st.subheader("Closed Trades")
    trades = metadata_store.read_closed_paper_trades(limit=settings.max_paper_trades_in_ui)
    if not trades:
        st.info("No closed trades yet.")
        return

    df = _trade_dataframe(trades)
    cols = [c for c in ["id", "symbol", "entry_date", "exit_date", "entry_price", "exit_price", "quantity", "realized_pnl", "realized_return_pct", "holding_days", "source_type", "source_label"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, height=280)


def _render_analytics_tab() -> None:
    st.subheader("Analytics")
    analytics = paper_trading_service.get_analytics(limit=settings.max_paper_trades_in_ui)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Trades", analytics.get("total_trades", 0))
    m2.metric("Win Rate", f"{analytics.get('win_rate', 0.0):.1f}%")
    m3.metric("Avg Return", f"{analytics.get('avg_return_pct', 0.0):.2f}%")
    m4.metric("Total PnL", f"{analytics.get('total_pnl', 0.0):.2f}")

    trades = metadata_store.read_paper_trades(limit=settings.max_paper_trades_in_ui)
    if not trades:
        st.info("No trades available for analytics.")
        return

    frame = _trade_dataframe(trades)
    closed = frame[frame["status"] == "closed"].copy() if "status" in frame.columns else pd.DataFrame()
    if not closed.empty:
        if "updated_at" in closed.columns:
            closed = closed.sort_values("updated_at")
        closed["cumulative_pnl"] = closed["realized_pnl"].fillna(0).cumsum() if "realized_pnl" in closed.columns else 0
        pnl_chart = closed.set_index("updated_at")["cumulative_pnl"] if "updated_at" in closed.columns else closed.set_index("exit_date")["cumulative_pnl"]
        st.line_chart(pnl_chart)

        by_symbol = (
            closed.groupby("symbol", dropna=False)["realized_return_pct"].mean().sort_values(ascending=False)
            if "realized_return_pct" in closed.columns else pd.Series(dtype=float)
        )
        if not by_symbol.empty:
            st.bar_chart(by_symbol.head(15))
    else:
        st.info("Close some trades to see performance charts.")
