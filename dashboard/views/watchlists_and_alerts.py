"""
Streamlit page for watchlists, alert rules, and alerts feed.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from app.core.config import get_settings
from app.services.alert_matcher_service import alert_matcher_service
from app.services.paper_trading_service import paper_trading_service
from app.services.watchlist_service import watchlist_service
from src.data.metadata_store import metadata_store

settings = get_settings()


def render_watchlists_and_alerts() -> None:
    """Render the watchlists & alerts page."""
    
    st.header("📋 Watchlists & Alerts")
    
    tab_watchlists, tab_rules, tab_alerts = st.tabs(
        ["Watchlists", "Alert Rules", "Alerts Feed"]
    )

    with tab_watchlists:
        _render_watchlists_tab()

    with tab_rules:
        _render_rules_tab()

    with tab_alerts:
        _render_alerts_tab()


def _render_watchlists_tab() -> None:
    """Render the watchlists management tab."""
    st.subheader("Manage Watchlists")
    
    # Create new watchlist
    st.write("**Create New Watchlist**")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        new_name = st.text_input("Name", key="new_watchlist_name", placeholder="My Watchlist")
    with col2:
        new_desc = st.text_input("Description (optional)", key="new_watchlist_desc", placeholder="Description")
    with col3:
        if st.button("Create", use_container_width=True, key="btn_create_watchlist"):
            if new_name.strip():
                try:
                    watchlist_service.create_watchlist(new_name, new_desc or "")
                    st.success(f"✓ Watchlist '{new_name}' created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a watchlist name")
    
    st.divider()
    
    # List existing watchlists
    watchlists = watchlist_service.read_watchlists()
    
    if not watchlists:
        st.info("No watchlists yet. Create one above!")
        return
    
    for watchlist in watchlists:
        with st.expander(f"📌 {watchlist.name}", expanded=False):
            # Basic info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Description:** {watchlist.description or '(none)'}")
                st.write(f"**Created:** {watchlist.created_at}")
            with col2:
                # Update active status
                new_active = st.checkbox(
                    "Active",
                    value=watchlist.is_active,
                    key=f"wl_active_{watchlist.id}",
                )
                if new_active != watchlist.is_active:
                    watchlist_service.update_watchlist(watchlist.id, is_active=new_active)
                    st.rerun()
                
                # Delete button
                if st.button("🗑️ Delete Watchlist", key=f"del_wl_{watchlist.id}"):
                    watchlist_service.delete_watchlist(watchlist.id)
                    st.success("Watchlist deleted!")
                    st.rerun()
            
            st.divider()
            
            # Show items
            st.write("**Items**")
            items = watchlist.items or []
            if items:
                items_df = pd.DataFrame([
                    {
                        "Type": item.item_type,
                        "Value": item.item_value,
                        "Normalized": item.normalized_value,
                    }
                    for item in items
                ])
                st.dataframe(items_df, use_container_width=True, hide_index=True)
                
                # Delete item
                if len(items) > 0:
                    item_options = [f"{i.item_type}={i.item_value}" for i in items]
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        item_to_delete = st.selectbox(
                            "Delete item",
                            item_options,
                            key=f"del_item_{watchlist.id}",
                        )
                    with col2:
                        if st.button("Remove", key=f"remove_item_{watchlist.id}", use_container_width=True):
                            item_idx = item_options.index(item_to_delete)
                            watchlist_service.delete_watchlist_item(items[item_idx].id)
                            st.success("✓ Item removed!")
                            st.rerun()
            else:
                st.info("No items yet")
            
            st.divider()
            
            # Add new item
            st.write("**Add Item**")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                item_type = st.selectbox(
                    "Type",
                    ["company", "ticker", "sector", "event_type"],
                    key=f"item_type_{watchlist.id}",
                )
            with col2:
                item_value = st.text_input(
                    "Value",
                    key=f"item_value_{watchlist.id}",
                    placeholder="e.g., AAPL, Technology",
                )
            with col3:
                if st.button("Add", key=f"add_item_{watchlist.id}", use_container_width=True):
                    if item_value.strip():
                        try:
                            watchlist_service.create_watchlist_item(
                                watchlist.id, item_type, item_value
                            )
                            st.success(f"✓ Added {item_type}={item_value}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("Please enter a value")


def _render_rules_tab() -> None:
    """Render the alert rules management tab."""
    st.subheader("Manage Alert Rules")
    
    # Create new rule
    st.write("**Create New Rule**")
    col1, col2 = st.columns([3, 1])
    with col1:
        rule_name = st.text_input("Rule Name", key="rule_name", placeholder="Tech Alerts")
    with col2:
        if st.button("Create", use_container_width=True, key="btn_create_rule"):
            if rule_name.strip():
                _show_rule_form(rule_name, is_new=True)
            else:
                st.warning("Please enter a rule name")
    
    st.divider()
    
    # List existing rules
    rules_data = metadata_store.read_alert_rules()
    
    if not rules_data:
        st.info("No alert rules yet. Create one above!")
        return
    
    for rule_dict in rules_data:
        rule_status = "✓" if rule_dict.get("is_active") else "✗"
        with st.expander(f"🔔 {rule_dict.get('name')} [{rule_status}]", expanded=False):
            # Basic controls
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                new_active = st.checkbox(
                    "Active",
                    value=rule_dict.get("is_active", True),
                    key=f"rule_active_{rule_dict['id']}",
                )
                if new_active != rule_dict.get("is_active", True):
                    try:
                        metadata_store.update_alert_rule(rule_dict["id"], {"is_active": new_active})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col2:
                if st.button("✏️ Edit", key=f"edit_rule_{rule_dict['id']}", use_container_width=True):
                    _show_rule_form(rule_dict.get("name"), is_new=False, rule_id=rule_dict["id"])
            
            with col3:
                if st.button("🗑️ Delete", key=f"del_rule_{rule_dict['id']}", use_container_width=True):
                    try:
                        metadata_store.delete_alert_rule(rule_dict["id"])
                        st.success("Rule deleted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            st.divider()
            
            # Display rule details
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Watchlist:** {rule_dict.get('watchlist_id') or 'Global'}")
                st.write(f"**Sentiment:** {rule_dict.get('sentiment_filter') or 'Any'}")
                st.write(f"**Min Confidence:** {rule_dict.get('min_confidence_score') or '—'}")
            with col2:
                st.write(f"**Min Impact:** {rule_dict.get('min_impact_score') or '—'}")
                st.write(f"**Min Overall:** {rule_dict.get('min_overall_score') or '—'}")
                st.write(f"**Cooldown:** {rule_dict.get('cooldown_minutes') or '—'} min")


def _show_rule_form(rule_name: str, is_new: bool = True, rule_id: int = None) -> None:
    """Show form to create/edit a rule."""
    with st.form(key=f"rule_form_{rule_name}_{rule_id}"):
        name = st.text_input("Rule Name", value=rule_name if not is_new else "")
        
        # Watchlist selection
        watchlists = watchlist_service.read_watchlists()
        watchlist_options = {wl.name: wl.id for wl in watchlists}
        watchlist_options["Global (all companies)"] = None
        watchlist_id = st.selectbox("Watchlist", list(watchlist_options.keys()))
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            sentiment_filter = st.selectbox(
                "Sentiment",
                ["(none)", "positive", "neutral", "negative"],
            )
        with col2:
            actionable_only = st.checkbox("Actionable Only", value=False)
        
        # Score thresholds
        col1, col2, col3 = st.columns(3)
        with col1:
            min_confidence = st.slider("Min Confidence Score", 0.0, 1.0, 0.5, step=0.05)
        with col2:
            min_impact = st.slider("Min Impact Score", 0.0, 1.0, 0.5, step=0.05)
        with col3:
            min_overall = st.slider("Min Overall Score", 0.0, 1.0, 0.6, step=0.05)
        
        # Timing and event types
        col1, col2 = st.columns(2)
        with col1:
            timing_labels = st.multiselect(
                "Timing Labels",
                ["early", "moderate", "late"],
                default=["early", "moderate"],
            )
        with col2:
            event_types = st.multiselect(
                "Event Types",
                ["earnings_beat", "product_launch", "acquisition", "partnership", "other"],
                default=[],
            )
        
        # Notification channels
        channels = st.multiselect(
            "Notification Channels",
            ["in_app", "webhook", "email", "telegram", "slack"],
            default=["in_app"],
        )
        
        # Cooldown
        cooldown_minutes = st.number_input("Cooldown (minutes)", min_value=0, max_value=10080, value=60, step=5)
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Save Rule")
        with col2:
            st.form_submit_button("Cancel")
        
        if submitted:
            rule_data = {
                "name": name,
                "watchlist_id": watchlist_options.get(watchlist_id),
                "sentiment_filter": None if sentiment_filter == "(none)" else sentiment_filter,
                "actionable_only": actionable_only,
                "min_confidence_score": min_confidence if min_confidence > 0 else None,
                "min_impact_score": min_impact if min_impact > 0 else None,
                "min_overall_score": min_overall if min_overall > 0 else None,
                "timing_labels": json.dumps(timing_labels) if timing_labels else None,
                "event_types": json.dumps(event_types) if event_types else None,
                "notification_channels": json.dumps(channels),
                "cooldown_minutes": cooldown_minutes if cooldown_minutes > 0 else None,
            }
            
            try:
                if is_new:
                    metadata_store.create_alert_rule(rule_data)
                    st.success(f"Rule '{name}' created!")
                else:
                    metadata_store.update_alert_rule(rule_id, rule_data)
                    st.success(f"Rule '{name}' updated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


def _render_alerts_tab() -> None:
    """Render the alerts feed tab."""
    st.subheader("Alert Feed")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        severity_filter = st.selectbox(
            "Severity",
            ["(all)", "info", "warning", "high"],
        )
    with col2:
        status_filter = st.selectbox(
            "Status",
            ["(all)", "new", "seen", "sent", "failed"],
        )
    with col3:
        hours_filter = st.number_input("Hours", min_value=1, max_value=720, value=24, step=1)
    
    if st.button("🔄 Refresh"):
        st.rerun()
    
    if st.button("▶️ Trigger Scan"):
        with st.spinner("Scanning opportunities..."):
            alert_ids = alert_matcher_service.scan_recent_opportunities()
            st.success(f"Scan complete. Created {len(alert_ids)} alerts.")
            st.rerun()
    
    # Fetch alerts
    try:
        alerts_data = metadata_store.read_recent_alerts(hours=hours_filter, limit=settings.max_alerts_in_ui)
        
        if not alerts_data:
            st.info("No alerts yet.")
            return
        
        # Filter
        if severity_filter != "(all)":
            alerts_data = [a for a in alerts_data if a.get("severity") == severity_filter]
        if status_filter != "(all)":
            alerts_data = [a for a in alerts_data if a.get("status") == status_filter]
        
        # Summary
        col1, col2, col3, col4 = st.columns(4)
        total = len(alerts_data)
        new_count = len([a for a in alerts_data if a.get("status") == "new"])
        high_count = len([a for a in alerts_data if a.get("severity") == "high"])
        failed_count = len([a for a in alerts_data if a.get("status") == "failed"])
        
        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("New", new_count)
        with col3:
            st.metric("High Severity", high_count)
        with col4:
            st.metric("Failed", failed_count)
        
        # Top recent alerts
        st.subheader("Recent Alerts")
        
        for alert in alerts_data[:50]:  # Show top 50
            alert_context = metadata_store.read_alert_with_context(int(alert["id"]))
            severity_emoji = {"high": "🔴", "warning": "🟡", "info": "🔵"}.get(alert.get("severity"), "ℹ️")
            status_emoji = {"new": "✨", "seen": "👁️", "sent": "✅", "failed": "❌"}.get(alert.get("status"), "•")
            tradable_symbol = None
            if alert_context:
                tradable_symbol = alert_context.get("ticker") or alert_context.get("primary_ticker")
            
            with st.expander(
                f"{severity_emoji} {status_emoji} {alert.get('title', 'Alert')} ({alert.get('created_at', '')[:10]})",
                expanded=False,
            ):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(alert.get("message", ""))
                    st.caption(f"Rule ID: {alert.get('rule_id')} | Channel: {alert.get('notification_channel')}")
                    if tradable_symbol:
                        st.caption(f"Tradable symbol: {str(tradable_symbol).upper()}")
                    if alert.get("error_message"):
                        st.error(f"Error: {alert.get('error_message')}")
                
                with col2:
                    if alert.get("status") != "seen":
                        if st.button("Mark Seen", key=f"mark_seen_{alert['id']}"):
                            metadata_store.update_alert_status(alert["id"], "seen")
                            st.success("Marked as seen")
                            st.rerun()
                    if tradable_symbol and st.button("Create Trade", key=f"create_trade_alert_{alert['id']}"):
                        try:
                            trade_id = paper_trading_service.create_trade_from_alert(
                                int(alert["id"]),
                                capital=float(settings.paper_trading_default_capital),
                                trade_reason=str(alert.get("message") or alert.get("title") or ""),
                            )
                            st.success(f"Trade #{trade_id} created.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error: {exc}")
    
    except Exception as e:
        st.error(f"Error loading alerts: {e}")
