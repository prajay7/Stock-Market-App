from __future__ import annotations

import pandas as pd
import streamlit as st

from app.news.service import news_impact_service


def _format_dt(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%Y-%m-%d %H:%M")


def _render_beneficiaries(beneficiaries: list[dict]) -> None:
    if not beneficiaries:
        st.caption("No beneficiary companies identified.")
        return
    df = pd.DataFrame(beneficiaries)
    cols = [
        c
        for c in [
            "company",
            "ticker",
            "relation",
            "relation_strength",
            "benefit_score",
            "price_change_pct_1d",
            "price_reaction_ok",
            "reason",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols], use_container_width=True, height=220)


def render_news_impact(settings) -> None:
    st.subheader("News Impact Scanner")
    st.caption("RSS-based market news scanner with sentiment, event, primary company, and beneficiary ranking.")

    if "news_impact_refresh" not in st.session_state:
        st.session_state["news_impact_refresh"] = False

    controls_left, controls_right = st.columns([2, 1])
    refresh_clicked = controls_left.button("Refresh News", type="primary")
    force_refresh = refresh_clicked or bool(st.session_state.get("news_impact_refresh", False))
    if refresh_clicked:
        st.session_state["news_impact_refresh"] = True

    with controls_right:
        st.caption(f"Feeds: {settings.news_max_feeds_per_refresh} max")
        st.caption(f"Cache TTL: {settings.news_cache_ttl_seconds}s")

    try:
        scan = news_impact_service.refresh(force_refresh=force_refresh)
    except Exception as exc:
        st.warning(f"News scanner refresh failed, showing cached data if available. Detail: {exc}")
        scan = news_impact_service.from_cache()

    st.session_state["news_impact_refresh"] = False

    articles = scan.articles
    if not articles:
        st.info("No analyzed news available yet. Click Refresh News to scan RSS feeds.")
        return

    top_filters = st.columns(5)
    sentiment_filter = top_filters[0].multiselect("Sentiment", ["positive", "neutral", "negative"], default=["positive", "neutral", "negative"], key="news_impact_sentiment")
    event_options = sorted({item.analysis.event_type for item in articles})
    event_filter = top_filters[1].multiselect("Event type", event_options, default=event_options, key="news_impact_event")
    sector_options = sorted({(item.analysis.sector or "Unknown") for item in articles})
    sector_filter = top_filters[2].multiselect("Sector", sector_options, default=sector_options, key="news_impact_sector")
    actionable_only = top_filters[3].checkbox("Actionable only", value=False, key="news_impact_actionable")
    min_confidence = top_filters[4].slider("Min confidence", min_value=0.0, max_value=1.0, value=0.4, step=0.05, key="news_impact_min_confidence")

    filtered = []
    for item in articles:
        analysis = item.analysis
        if sentiment_filter and analysis.sentiment_label not in sentiment_filter:
            continue
        if event_filter and analysis.event_type not in event_filter:
            continue
        sector = analysis.sector or "Unknown"
        if sector_filter and sector not in sector_filter:
            continue
        if actionable_only and not analysis.is_actionable:
            continue
        if float(analysis.confidence_score) < float(min_confidence):
            continue
        filtered.append(item)

    st.markdown("### Top Early Opportunities")
    opp_filters = st.columns(5)
    price_move_cap = opp_filters[0].slider(
        "Max abs 1D price move (%)",
        min_value=0.5,
        max_value=10.0,
        value=float(settings.news_price_move_late_threshold_pct),
        step=0.5,
        key="news_impact_price_move_cap",
    )
    min_signal_score = opp_filters[1].slider(
        "Min overall score",
        min_value=0.0,
        max_value=1.0,
        value=float(settings.news_min_signal_score),
        step=0.05,
        key="news_impact_min_overall_score",
    )
    timing_filter = opp_filters[2].multiselect(
        "Timing",
        ["early", "moderate", "late"],
        default=["early", "moderate"],
        key="news_impact_timing_filter",
    )
    top_actionable_only = opp_filters[3].checkbox("Actionable only", value=True, key="news_impact_top_actionable_only")
    top_sentiment_filter = opp_filters[4].multiselect(
        "Top sentiment",
        ["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"],
        key="news_impact_top_sentiment_filter",
    )
    top_event_filter = st.multiselect(
        "Top event type",
        event_options,
        default=event_options,
        key="news_impact_top_event_filter",
    )

    top_items = []
    for signal in scan.top_opportunities:
        change_pct = signal.price_change_pct_1d
        abs_change = abs(float(change_pct)) if change_pct is not None else 0.0
        if abs_change > float(price_move_cap):
            continue
        if float(signal.overall_score) < float(min_signal_score):
            continue
        if timing_filter and signal.timing_label not in timing_filter:
            continue
        if top_actionable_only and signal.sentiment_label == "neutral":
            continue
        if top_sentiment_filter and signal.sentiment_label not in top_sentiment_filter:
            continue
        if top_event_filter and signal.event_type not in top_event_filter:
            continue
        top_items.append(signal)

    top_items = sorted(top_items, key=lambda x: (float(x.overall_score), float(x.signal_score)), reverse=True)

    if top_items:
        top_df = pd.DataFrame(
            [
                {
                    "generated_at": _format_dt(item.generated_at),
                    "headline": item.headline,
                    "primary_company": item.primary_company,
                    "beneficiary": item.beneficiary_company,
                    "beneficiary_ticker": item.beneficiary_ticker,
                    "relation": item.relation,
                    "event_type": item.event_type,
                    "sentiment": item.sentiment_label,
                    "timing_label": item.timing_label,
                    "impact": item.impact_score,
                    "price_change_1d_pct": item.price_change_pct_1d,
                    "price_reaction_ok": item.price_reaction_ok,
                    "overall_score": item.overall_score,
                    "reason": item.reason,
                }
                for item in top_items[:20]
            ]
        )
        st.dataframe(top_df, use_container_width=True, height=300)
    else:
        st.info("No early opportunities match the current price-reaction filter.")

    st.markdown("### Signal History")
    signal_history = scan.signal_history
    if signal_history:
        history_df = pd.DataFrame(
            [
                {
                    "generated_at": _format_dt(item.generated_at),
                    "headline": item.headline,
                    "primary_company": item.primary_company,
                    "beneficiary": item.beneficiary_company,
                    "beneficiary_ticker": item.beneficiary_ticker,
                    "relation": item.relation,
                    "event_type": item.event_type,
                    "sentiment": item.sentiment_label,
                    "timing_label": item.timing_label,
                    "price_change_1d_pct": item.price_change_pct_1d,
                    "price_reaction_ok": item.price_reaction_ok,
                    "overall_score": item.overall_score,
                    "is_early_opportunity": item.is_early_opportunity,
                }
                for item in signal_history[:200]
            ]
        )
        st.dataframe(history_df, use_container_width=True, height=280)
    else:
        st.caption("No persisted signals yet.")

    st.markdown("### Analyzed News")
    if not filtered:
        st.info("No articles match the current filters.")
        return

    for item in filtered:
        analysis = item.analysis
        with st.expander(f"{item.article.title}  |  {analysis.sentiment_label.upper()}  |  {analysis.event_type}", expanded=False):
            left, right = st.columns([2, 1])
            left.markdown(f"**Headline:** {item.article.title}")
            left.markdown(f"**Source:** {item.article.source}")
            left.markdown(f"**Published:** {_format_dt(item.article.published_at)}")
            left.markdown(f"**Primary company:** {analysis.primary_company or 'Unknown'}")
            primary_ticker = next((s.primary_ticker for s in signal_history if s.article_hash == item.article.article_hash and s.primary_ticker), None)
            left.markdown(f"**Primary ticker:** {primary_ticker or 'Unknown'}")
            left.markdown(f"**Event type:** {analysis.event_type}")
            left.markdown(f"**Sentiment:** {analysis.sentiment_label} ({analysis.sentiment_score:+.2f})")
            left.markdown(f"**Impact:** {analysis.impact_score:.2f}")
            left.markdown(f"**Confidence:** {analysis.confidence_score:.2f}")
            left.markdown(f"**Summary:** {analysis.summary}")
            left.markdown(f"**Link:** [{item.article.link}]({item.article.link})")
            right.metric("Overall score", f"{item.overall_score:.2f}")
            right.metric("Freshness", f"{item.freshness_score:.2f}")
            right.metric("Relation", f"{item.relation_strength:.2f}")
            right.metric("Price opportunity", f"{item.price_opportunity_score:.2f}")
            st.markdown("**Beneficiary companies**")
            _render_beneficiaries([benef.model_dump() for benef in item.ranked_beneficiaries])
