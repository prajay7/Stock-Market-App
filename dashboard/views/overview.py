from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

from src.data.cache import symbol_price_path
from src.data.storage import read_parquet_if_exists


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_live_nse_market_snapshot() -> tuple[pd.DataFrame, str | None, str | None]:
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20TOTAL%20MARKET"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        with httpx.Client(timeout=25.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        return pd.DataFrame(), None, f"Live NSE fetch failed: {exc}"

    rows = []
    for item in payload.get("data", []):
        # Skip index summary rows; keep instrument rows with actual stock series.
        if not item.get("series"):
            continue
        meta = item.get("meta") or {}
        rows.append(
            {
                "symbol": str(item.get("symbol") or "").upper(),
                "series": str(item.get("series") or ""),
                "security_name": str(meta.get("companyName") or ""),
                "last_price": pd.to_numeric(item.get("lastPrice"), errors="coerce"),
                "day_high": pd.to_numeric(item.get("dayHigh"), errors="coerce"),
                "day_low": pd.to_numeric(item.get("dayLow"), errors="coerce"),
                "day_change_pct": pd.to_numeric(item.get("pChange"), errors="coerce"),
                "volume": pd.to_numeric(item.get("totalTradedVolume"), errors="coerce"),
                "last_update_time": str(item.get("lastUpdateTime") or payload.get("timestamp") or ""),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), str(payload.get("timestamp") or ""), "Live NSE response had no stock rows"

    return df, str(payload.get("timestamp") or ""), None


def _latest_day_change(raw_data_dir, interval: str, symbol: str) -> tuple[pd.Timestamp | None, float | None, float | None, float | None]:
    path = symbol_price_path(raw_data_dir, symbol, interval)
    df = read_parquet_if_exists(path)
    if df.empty:
        return None, None, None, None

    cols = {str(c).strip().lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("datetime")
    close_col = cols.get("close")
    if date_col is None or close_col is None:
        return None, None, None, None

    tmp = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    ).dropna(subset=["date", "close"])
    if len(tmp) < 2:
        return None, None, None, None

    tmp = tmp.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if len(tmp) < 2:
        return None, None, None, None

    prev_close = float(tmp["close"].iloc[-2])
    curr_close = float(tmp["close"].iloc[-1])
    if prev_close == 0:
        return pd.Timestamp(tmp["date"].iloc[-1]), prev_close, curr_close, None
    day_change_pct = ((curr_close - prev_close) / prev_close) * 100.0
    return pd.Timestamp(tmp["date"].iloc[-1]), prev_close, curr_close, float(day_change_pct)


def _top_investment_signals(pred_path, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = read_parquet_if_exists(pred_path)
    if pred_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = pred_df.copy()
    work["prob_up"] = pd.to_numeric(work.get("prob_up"), errors="coerce")
    work["predicted_return"] = pd.to_numeric(work.get("predicted_return"), errors="coerce")
    work["confidence"] = pd.to_numeric(work.get("confidence"), errors="coerce")
    work = work.dropna(subset=["symbol", "prob_up"])
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Convert probability into directional indicator for easier action.
    work["indicator"] = "HOLD"
    work.loc[work["prob_up"] >= 0.55, "indicator"] = "BUY"
    work.loc[work["prob_up"] <= 0.45, "indicator"] = "SELL"
    work["signal_strength"] = (work["prob_up"] - 0.5).abs()

    buy_df = work[work["indicator"] == "BUY"].copy()
    sell_df = work[work["indicator"] == "SELL"].copy()

    buy_ranked = buy_df.sort_values(["signal_strength", "confidence"], ascending=[False, False]).head(int(top_n))
    sell_ranked = sell_df.sort_values(["signal_strength", "confidence"], ascending=[False, False]).head(int(top_n))
    return buy_ranked, sell_ranked


def render_overview(
    universe: pd.DataFrame,
    status_merged: pd.DataFrame,
    settings,
    filter_universe,
    decorate_status,
) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Universe Size", int(len(universe)))
    col2.metric("Benchmark", settings.benchmark_symbol)
    trained_count = int(status_merged["last_trained_at"].notna().sum()) if "last_trained_at" in status_merged.columns else 0
    col3.metric("Trained Symbols", trained_count)

    st.subheader("Top 10 Stock / Investment Signals")
    top_buy, top_sell = _top_investment_signals(settings.output_dir / "latest_predictions.parquet", top_n=10)
    if top_buy.empty and top_sell.empty:
        st.info("No actionable BUY/SELL signals found yet. Generate predictions first.")
    else:
        signal_cols = [
            "symbol",
            "indicator",
            "prob_up",
            "predicted_return",
            "confidence",
            "current_price",
            "target_price",
            "stop_loss_price",
            "price_as_of",
            "price_as_of_time",
            "live_price",
            "live_price_as_of",
            "live_price_as_of_time",
            "signal_strength",
        ]
        st.caption("Indicator rule: BUY if probability >= 55%, SELL if probability <= 45%.")

        bcol, scol = st.columns(2)

        with bcol:
            st.markdown("**Top 10 BUY**")
            if top_buy.empty:
                st.caption("No BUY signals currently.")
            else:
                buy_cols = [c for c in signal_cols if c in top_buy.columns]
                st.dataframe(
                    top_buy[buy_cols],
                    use_container_width=True,
                    height=300,
                )

        with scol:
            st.markdown("**Top 10 SELL**")
            if top_sell.empty:
                st.caption("No SELL signals currently.")
            else:
                sell_cols = [c for c in signal_cols if c in top_sell.columns]
                st.dataframe(
                    top_sell[sell_cols],
                    use_container_width=True,
                    height=300,
                )

    st.subheader("Stock Universe")
    filtered = decorate_status(filter_universe(status_merged, "overview"))
    display_cols = [
        "symbol",
        "train_symbol",
        "series",
        "security_name",
        "unsupported",
        "status_badge",
        "last_failure_reason",
        "last_trained_at",
        "last_model_name",
        "last_roc_auc",
    ]
    st.dataframe(
        filtered[display_cols].sort_values(["last_trained_at", "symbol"], ascending=[False, True]),
        use_container_width=True,
        height=500,
    )

    st.subheader("Upper Circuit / Lower Circuit")
    st.caption("Live source: NSE internet data (NIFTY TOTAL MARKET). Falls back to local data if unavailable.")
    circuit_threshold = st.number_input(
        "Circuit threshold (%)",
        min_value=1.0,
        max_value=25.0,
        value=10.0,
        step=0.5,
        key="overview_circuit_threshold",
    )
    strict_locked_only = st.checkbox(
        "Strict lock-only (day high equals day low)",
        value=False,
        key="overview_strict_locked_only",
    )

    live_df, live_timestamp, live_error = _fetch_live_nse_market_snapshot()

    if not live_df.empty:
        work = live_df.copy()
        if strict_locked_only:
            tolerance = 1e-9
            work = work[(work["day_high"] - work["day_low"]).abs() <= tolerance]

        upper_df = work[work["day_change_pct"] >= float(circuit_threshold)].copy()
        lower_df = work[work["day_change_pct"] <= -float(circuit_threshold)].copy()

        st.caption(f"Live market timestamp: {live_timestamp or 'N/A'}")

        c1, c2 = st.columns(2)
        c1.metric("Upper Circuit Stocks", len(upper_df))
        c2.metric("Lower Circuit Stocks", len(lower_df))

        upper_cols = ["symbol", "series", "last_update_time", "last_price", "day_change_pct", "volume", "security_name"]
        lower_cols = ["symbol", "series", "last_update_time", "last_price", "day_change_pct", "volume", "security_name"]

        st.markdown("**Upper Circuit List**")
        if upper_df.empty:
            st.caption("No stocks at or above the selected upper threshold in live data.")
        else:
            upper_export = upper_df[upper_cols].sort_values("day_change_pct", ascending=False)
            st.download_button(
                "Download Upper Circuit CSV",
                data=upper_export.to_csv(index=False).encode("utf-8"),
                file_name="upper_circuit_stocks_live.csv",
                mime="text/csv",
                key="overview_download_upper_circuit",
            )
            st.dataframe(
                upper_export,
                use_container_width=True,
                height=260,
            )

        st.markdown("**Lower Circuit List**")
        if lower_df.empty:
            st.caption("No stocks at or below the selected lower threshold in live data.")
        else:
            lower_export = lower_df[lower_cols].sort_values("day_change_pct", ascending=True)
            st.download_button(
                "Download Lower Circuit CSV",
                data=lower_export.to_csv(index=False).encode("utf-8"),
                file_name="lower_circuit_stocks_live.csv",
                mime="text/csv",
                key="overview_download_lower_circuit",
            )
            st.dataframe(
                lower_export,
                use_container_width=True,
                height=260,
            )

        return

    if live_error:
        st.warning(f"Live internet feed unavailable. Showing local fallback list. Detail: {live_error}")

    circuit_rows: list[dict] = []
    for _, row in filtered[["symbol", "train_symbol", "series", "security_name"]].dropna(subset=["train_symbol"]).iterrows():
        as_of, prev_close, curr_close, change_pct = _latest_day_change(
            settings.raw_data_dir,
            settings.historical_interval,
            str(row["train_symbol"]),
        )
        if change_pct is None:
            continue
        circuit_rows.append(
            {
                "symbol": row["symbol"],
                "train_symbol": row["train_symbol"],
                "series": row["series"],
                "security_name": row["security_name"],
                "as_of": as_of.date().isoformat() if pd.notna(as_of) else None,
                "prev_close": prev_close,
                "close": curr_close,
                "day_change_pct": change_pct,
            }
        )

    if not circuit_rows:
        st.info("No price data available yet to compute circuit lists.")
        return

    circuit_df = pd.DataFrame(circuit_rows).sort_values("day_change_pct", ascending=False)
    upper_df = circuit_df[circuit_df["day_change_pct"] >= float(circuit_threshold)].copy()
    lower_df = circuit_df[circuit_df["day_change_pct"] <= -float(circuit_threshold)].copy()

    c1, c2 = st.columns(2)
    c1.metric("Upper Circuit Stocks", len(upper_df))
    c2.metric("Lower Circuit Stocks", len(lower_df))

    upper_cols = ["symbol", "train_symbol", "series", "as_of", "close", "day_change_pct", "security_name"]
    lower_cols = ["symbol", "train_symbol", "series", "as_of", "close", "day_change_pct", "security_name"]

    st.markdown("**Upper Circuit List**")
    if upper_df.empty:
        st.caption("No stocks at or above the selected upper threshold.")
    else:
        upper_export = upper_df[upper_cols].sort_values("day_change_pct", ascending=False)
        st.download_button(
            "Download Upper Circuit CSV",
            data=upper_export.to_csv(index=False).encode("utf-8"),
            file_name="upper_circuit_stocks.csv",
            mime="text/csv",
            key="overview_download_upper_circuit",
        )
        st.dataframe(
            upper_export,
            use_container_width=True,
            height=260,
        )

    st.markdown("**Lower Circuit List**")
    if lower_df.empty:
        st.caption("No stocks at or below the selected lower threshold.")
    else:
        lower_export = lower_df[lower_cols].sort_values("day_change_pct", ascending=True)
        st.download_button(
            "Download Lower Circuit CSV",
            data=lower_export.to_csv(index=False).encode("utf-8"),
            file_name="lower_circuit_stocks.csv",
            mime="text/csv",
            key="overview_download_lower_circuit",
        )
        st.dataframe(
            lower_export,
            use_container_width=True,
            height=260,
        )
