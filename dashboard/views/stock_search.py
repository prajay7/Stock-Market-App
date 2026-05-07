from __future__ import annotations

import logging
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
import yfinance as yf
import finnhub
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.constants import OPENAI_STOCK_MODEL_ALIASES_SET
from src.inference.predict import predict_for_symbols

logger = logging.getLogger(__name__)

# Suppress yfinance warnings
logging.getLogger("yfinance").setLevel(logging.ERROR)

# Initialize Finnhub client
FINNHUB_API_KEY = os.getenv("FINNHUB")
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY) if FINNHUB_API_KEY else None


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
def _fetch_ticker_info(symbol: str) -> dict:
    """Fetch ticker info with retry logic."""
    ticker = yf.Ticker(symbol, session=None)
    info = ticker.info
    if not info or info.get("regularMarketPrice") is None:
        raise ValueError(f"No valid price data for {symbol}")
    return info


@st.cache_data(ttl=3600, show_spinner=False)
def _search_stock_info(symbol: str) -> dict | None:
    """Fetch comprehensive stock information from yfinance with Finnhub fallback."""
    try:
        return _fetch_ticker_info(symbol)
    except Exception as e:
        error_msg = str(e).lower()
        logger.warning(f"yfinance failed for {symbol}, trying Finnhub fallback: {e}")
        # Try Finnhub as fallback
        if finnhub_client:
            try:
                finnhub_info = _fetch_finnhub_stock_info(symbol)
                if finnhub_info:
                    return finnhub_info
            except Exception as fh_e:
                logger.warning(f"Finnhub also failed for {symbol}: {fh_e}")
        
        if "too many requests" in error_msg or "429" in error_msg or "rate" in error_msg:
            logger.warning(f"Rate limited for {symbol}: {e}")
            return None
        elif "no valid price data" in error_msg:
            logger.warning(f"No data for {symbol}")
            return None
        else:
            logger.warning(f"Error fetching stock info for {symbol}: {e}")
            return None


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
def _fetch_ticker_history(symbol: str, period: str) -> pd.DataFrame:
    """Fetch historical data with retry logic."""
    ticker = yf.Ticker(symbol, session=None)
    df = ticker.history(period=period, interval="1d")
    if df is None or df.empty:
        raise ValueError(f"No historical data for {symbol}")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_historical_data(symbol: str, period: str) -> pd.DataFrame:
    """Fetch historical OHLCV data from yfinance with rate-limit handling."""
    try:
        df = _fetch_ticker_history(symbol, period)
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        return df
    except Exception as e:
        error_msg = str(e).lower()
        if "too many requests" in error_msg or "429" in error_msg or "rate" in error_msg:
            logger.warning(f"Rate limited fetching history for {symbol}")
        else:
            logger.warning(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()


def _fetch_finnhub_stock_info(symbol: str) -> dict | None:
    """Fetch stock information from Finnhub as fallback."""
    if not finnhub_client:
        return None
    
    try:
        quote = finnhub_client.quote(symbol)
        company = finnhub_client.company_profile2(symbol=symbol)
        
        if not quote or quote.get("c") is None:
            raise ValueError(f"No valid price data from Finnhub for {symbol}")
        
        # Map Finnhub data to yfinance-like structure
        info = {
            "regularMarketPrice": quote.get("c", 0),
            "currentPrice": quote.get("c", 0),
            "regularMarketChangePercent": quote.get("dp", 0),
            "regularMarketChange": quote.get("d", 0),
            "fiftyTwoWeekHigh": quote.get("h", 0),
            "fiftyTwoWeekLow": quote.get("l", 0),
            "marketCap": company.get("marketCapitalization", 0) * 1e6 if company else 0,
            "longName": company.get("name", symbol) if company else symbol,
            "trailingPE": company.get("pe", None) if company else None,
            "dividendYield": 0,  # Finnhub doesn't provide this directly
            "averageVolume": quote.get("vo", 0),
            "beta": company.get("beta", None) if company else None,
            "trailingEps": company.get("eps", None) if company else None,
            "totalRevenue": company.get("revenue", 0) * 1e6 if company else 0,
            "profitMargins": company.get("profitMargin", 0) if company else 0,
            "returnOnEquity": company.get("roe", 0) if company else 0,
        }
        
        logger.info(f"Successfully fetched Finnhub stock info for {symbol}")
        return info
    except Exception as e:
        logger.info(f"Error fetching Finnhub stock info for {symbol}: {e}")
        return None


def _fetch_finnhub_news(symbol: str, max_results: int = 20) -> list[dict]:
    """Fetch news from Finnhub as fallback."""
    if not finnhub_client:
        return []
    
    try:
        # Finnhub news endpoint
        news_list = finnhub_client.company_news(symbol, _from=datetime.now() - timedelta(days=30), to=datetime.now())
        
        if not news_list:
            return []
        
        results = []
        for item in news_list[:max_results]:
            results.append({
                "title": item.get("headline", ""),
                "link": item.get("url", ""),
                "source": item.get("source", "Unknown"),
                "providerPublishTime": item.get("datetime", "")
            })
        
        logger.info(f"Successfully fetched {len(results)} Finnhub news articles for {symbol}")
        return results
    except Exception as e:
        logger.warning(f"Error fetching Finnhub news for {symbol}: {e}")
        return []


def _calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators using TA library."""
    if df.empty:
        return df
    
    try:
        # Ensure required columns exist
        if not all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            return df
        
        # Copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Moving Averages
        df['MA20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['MA50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        macd_line = ta.trend.macd(df['close'])
        if macd_line is not None:
            df['MACD'] = macd_line
            df['MACD_signal'] = ta.trend.macd_signal(df['close'])
        
        # Bollinger Bands
        bb = ta.volatility.bollinger_hband(df['close'], window=20)
        df['BB_high'] = bb
        df['BB_low'] = ta.volatility.bollinger_lband(df['close'], window=20)
        
        return df
    except Exception as e:
        logger.warning(f"Error calculating technical indicators: {e}")
        return df


def _create_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create interactive price chart with moving averages."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            visible=True
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if 'MA20' in df.columns and df['MA20'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['MA20'],
                mode='lines', name='MA20',
                line=dict(color='orange', width=1),
                visible=True
            ),
            row=1, col=1
        )
    
    if 'MA50' in df.columns and df['MA50'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['MA50'],
                mode='lines', name='MA50',
                line=dict(color='blue', width=1),
                visible=True
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'BB_high' in df.columns and df['BB_high'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['BB_high'],
                mode='lines', name='BB Upper',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['BB_low'],
                mode='lines', name='BB Lower',
                line=dict(color='rgba(0,0,0,0)'),
                fillcolor='rgba(0,100,80,0.2)',
                fill='tonexty',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if close < open_ else 'green' 
              for close, open_ in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(
            x=df['date'], y=df['volume'],
            name='Volume', marker=dict(color=colors),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_layout(
        title=f"{symbol} - Price & Volume",
        height=600,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def _create_rsi_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create RSI technical indicator chart."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['RSI'],
            mode='lines', name='RSI',
            line=dict(color='purple', width=2)
        )
    )
    
    # Overbought/Oversold levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig.update_layout(
        title=f"{symbol} - RSI (14)",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=350,
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def _create_macd_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create MACD technical indicator chart."""
    fig = go.Figure()
    
    if 'MACD' in df.columns and df['MACD'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['MACD'],
                mode='lines', name='MACD',
                line=dict(color='blue', width=2)
            )
        )
    
    if 'MACD_signal' in df.columns and df['MACD_signal'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['MACD_signal'],
                mode='lines', name='Signal',
                line=dict(color='red', width=2)
            )
        )
    
    fig.update_layout(
        title=f"{symbol} - MACD",
        xaxis_title="Date",
        yaxis_title="Value",
        height=350,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def _display_fundamentals(info: dict) -> None:
    """Display stock fundamentals and key statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"₹{info.get('currentPrice', info.get('regularMarketPrice', 0)):.2f}",
            delta=f"{info.get('regularMarketChangePercent', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            "Market Cap",
            f"₹{info.get('marketCap', 0) / 1e9:.2f}B"
        )
    
    with col3:
        st.metric(
            "P/E Ratio",
            f"{info.get('trailingPE', 'N/A')}"
        )
    
    with col4:
        st.metric(
            "Dividend Yield",
            f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A"
        )
    
    st.divider()
    
    # Additional details
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("52 Week High", f"₹{info.get('fiftyTwoWeekHigh', 0):.2f}")
    with col2:
        st.metric("52 Week Low", f"₹{info.get('fiftyTwoWeekLow', 0):.2f}")
    with col3:
        st.metric("Avg Volume (3m)", f"{info.get('averageVolume', 0) / 1e6:.2f}M")
    with col4:
        st.metric("Beta", f"{info.get('beta', 'N/A')}")
    
    st.divider()
    
    # Financial metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "EPS",
            f"₹{info.get('trailingEps', 'N/A')}"
        )
    
    with col2:
        st.metric(
            "Revenue",
            f"₹{info.get('totalRevenue', 0) / 1e9:.2f}B" if info.get('totalRevenue') else "N/A"
        )
    
    with col3:
        st.metric(
            "Profit Margin",
            f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else "N/A"
        )
    
    with col4:
        st.metric(
            "ROE",
            f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A"
        )


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
def _fetch_ticker_news(symbol: str) -> list:
    """Fetch news articles with retry logic."""
    ticker = yf.Ticker(symbol, session=None)
    news = ticker.news
    return news if news else []


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_news_for_symbol(symbol: str) -> pd.DataFrame:
    """Fetch news articles for a symbol with Finnhub fallback."""
    try:
        news = _fetch_ticker_news(symbol)
        if not news:
            # Try Finnhub as fallback
            if finnhub_client:
                finnhub_news = _fetch_finnhub_news(symbol)
                if finnhub_news:
                    rows = []
                    for item in finnhub_news[:20]:
                        rows.append({
                            'title': item.get('title', ''),
                            'link': item.get('link', ''),
                            'source': item.get('source', 'Unknown'),
                            'published': item.get('providerPublishTime', '')
                        })
                    return pd.DataFrame(rows)
            return pd.DataFrame()
        
        rows = []
        for item in news[:20]:  # Limit to 20 articles
            rows.append({
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'source': item.get('source', 'Unknown'),
                'published': item.get('providerPublishTime', '')
            })
        return pd.DataFrame(rows)
    except Exception as e:
        error_msg = str(e).lower()
        logger.warning(f"yfinance news fetch failed for {symbol}, trying Finnhub: {e}")
        # Try Finnhub as fallback
        if finnhub_client:
            try:
                finnhub_news = _fetch_finnhub_news(symbol)
                if finnhub_news:
                    rows = []
                    for item in finnhub_news[:20]:
                        rows.append({
                            'title': item.get('title', ''),
                            'link': item.get('link', ''),
                            'source': item.get('source', 'Unknown'),
                            'published': item.get('providerPublishTime', '')
                        })
                    return pd.DataFrame(rows)
            except Exception as fh_e:
                logger.warning(f"Finnhub news fetch also failed for {symbol}: {fh_e}")
        
        if "too many requests" in error_msg or "429" in error_msg or "rate" in error_msg:
            logger.warning(f"Rate limited fetching news for {symbol}")
        else:
            logger.warning(f"Error fetching news for {symbol}: {e}")
        return pd.DataFrame()


def _get_available_models(model_dir: Path) -> list[str]:
    """Get list of available trained models."""
    if not model_dir.exists():
        return []
    
    preferred = [
        "xgboost_classifier",
        "openai_stock_llm_fast",
        "openai_stock_llm",
        "openai_stock_llm_search",
        "openai_stock_llm_cheap",
    ]
    models = set()
    for model_name in preferred:
        candidates = list(model_dir.glob(f"{model_name}_*.joblib"))
        if candidates or model_name in OPENAI_STOCK_MODEL_ALIASES_SET:
            models.add(model_name)
    
    return [model_name for model_name in preferred if model_name in models] if models else ["xgboost_classifier"]


def _get_trained_symbols_for_model(model_dir: Path, model_name: str) -> list[str]:
    """Get symbols trained for a specific model."""
    if not model_dir.exists() or model_name in OPENAI_STOCK_MODEL_ALIASES_SET:
        return []
    
    symbols = set()
    for path in sorted(model_dir.glob(f"{model_name}_*.metadata.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for symbol in payload.get("symbols", []):
                if symbol:
                    symbols.add(str(symbol))
        except Exception:
            continue
    
    return sorted(symbols)


def _predict_stock_price(
    symbol: str,
    model_name: str,
    horizon_days: int,
    model_dir: Path,
    settings
) -> dict | None:
    """Get AI prediction for a stock symbol."""
    try:
        result = predict_for_symbols(
            symbols=[symbol],
            model_name=model_name,
            horizon_days=horizon_days,
            atr_multiplier=1.0,
            include_live_quote=False
        )
        
        if result.empty:
            return None
        
        row = result.iloc[0]
        return {
            "symbol": str(row.get("symbol", symbol)),
            "current_price": float(row.get("current_price") or 0),
            "predicted_price": float(row.get("predicted_price") or 0),
            "predicted_return": float(row.get("predicted_return") or 0),
            "confidence": float(row.get("confidence") or 0),
            "decision": str(row.get("decision", "HOLD")),
            "prob_up": float(row.get("prob_up") or 0),
            "sentiment": float(row.get("latest_sentiment") or 0),
        }
    except Exception as e:
        logger.warning(f"Error predicting for {symbol}: {e}")
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_symbol_suggestions(query: str, default_symbols: tuple[str, ...], max_results: int = 15) -> list[dict]:
    """Fetch stock suggestions for a partial query using Yahoo search with local fallback."""
    query_clean = str(query or "").strip()
    if not query_clean:
        return []

    suggestions: list[dict] = []
    seen: set[str] = set()

    try:
        search = yf.Search(query=query_clean, max_results=max_results)
        quotes = getattr(search, "quotes", []) or []
        for quote in quotes:
            symbol = str(quote.get("symbol") or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            name = str(quote.get("shortname") or quote.get("longname") or "").strip()
            exchange = str(quote.get("exchange") or quote.get("exchDisp") or "").strip()
            label = symbol
            if name:
                label = f"{symbol} - {name}"
            if exchange:
                label = f"{label} ({exchange})"
            suggestions.append({"symbol": symbol, "label": label})
            seen.add(symbol)
    except Exception as e:
        logger.warning(f"Suggestion lookup failed for query '{query_clean}': {e}")

    # Fallback to local configured symbols when internet results are empty or partial.
    q_upper = query_clean.upper()
    for symbol in default_symbols:
        sym = str(symbol).strip().upper()
        if not sym or sym in seen:
            continue
        if q_upper in sym:
            suggestions.append({"symbol": sym, "label": sym})
            seen.add(sym)
        if len(suggestions) >= max_results:
            break

    return suggestions[:max_results]


def render_stock_search(settings) -> None:
    """Render the stock search and details page."""
    st.subheader("📈 Stock Search & Details")
    
    # Rate limit info
    st.caption("💡 Tip: Results are cached for 1 hour to minimize rate limits. Data may be slightly delayed.")
    
    # Search input + autocomplete suggestions
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Search stock (type any letter)",
            placeholder="e.g., AAPL, RELIANCE, TCS, MSFT",
            key="stock_search_query",
        ).strip()

    default_symbols = tuple(str(sym).strip().upper() for sym in settings.default_symbols if str(sym).strip())
    suggestions = _fetch_symbol_suggestions(search_query, default_symbols)

    selected_symbol = ""
    if search_query and suggestions:
        suggestion_labels = [item["label"] for item in suggestions]
        label_to_symbol = {item["label"]: item["symbol"] for item in suggestions}
        selected_label = st.selectbox(
            "Suggestions",
            options=suggestion_labels,
            key="stock_search_suggestion_select",
            help="Select a matching stock from live search suggestions",
        )
        selected_symbol = label_to_symbol.get(selected_label, "")
    elif search_query:
        st.caption("No direct suggestion found. Press Search to try exact symbol.")
        selected_symbol = search_query.upper()
    
    with col2:
        search_btn = st.button("Search", use_container_width=True)
    
    if not search_query and not search_btn:
        st.info("👉 Enter a stock symbol to get started (e.g., AAPL, RELIANCE.NS)")
        return
    
    symbol = selected_symbol.strip().upper() if selected_symbol else None
    if not symbol:
        st.warning("Please enter a symbol")
        return
    
    # Fetch stock info
    with st.spinner(f"Loading {symbol}... (with automatic retries)"):
        info = _search_stock_info(symbol)
    
    if not info:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.error(f"❌ Stock '{symbol}' not found or data temporarily unavailable")
        with col2:
            st.markdown("**Suggestions:**")
        st.markdown("""
        - **Rate limited?** Results cache for 1 hour. Please wait or try a different symbol.
        - **Wrong symbol?** Check the ticker format (e.g., RELIANCE.NS for Indian stocks, AAPL for US)
        - **Market closed?** Some data is only available during market hours
        - **Try again in a few seconds** if you just tried another search
        """)
        return
    
    st.success(f"✅ Found: {info.get('longName', symbol)}")
    
    
    # Time period selector
    period_col, interval_col = st.columns(2)
    with period_col:
        time_period = st.selectbox(
            "Time Period",
            options=["1d", "5d", "1mo", "3mo", "1y"],
            format_func=lambda x: {"1d": "1 Day", "5d": "5 Days", "1mo": "1 Month", "3mo": "3 Months", "1y": "1 Year"}[x],
            key="stock_search_period"
        )
    
    with interval_col:
        st.markdown("**Chart Type**")
        st.markdown("Candlestick with Moving Averages & Technical Indicators")
    
    # Display fundamentals
    st.markdown("### 📊 Key Metrics")
    _display_fundamentals(info)
    
    # Fetch historical data
    with st.spinner("Loading historical data..."):
        df = _fetch_historical_data(symbol, time_period)
    
    if df.empty:
        st.warning(f"⏳ Historical chart data not available (cached for 1 hour). Basic metrics shown above.")
        st.info("Charts will load once data is available. Try another symbol or wait a moment.")
        
        # Still show news tab at least
        st.markdown("### 📰 Latest News")
        news_df = _fetch_news_for_symbol(symbol)
        
        if news_df.empty:
            st.info("No recent news available")
        else:
            for idx, row in news_df.iterrows():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**[{row['title']}]({row['link']})**")
                    st.caption(f"*{row['source']}*")
                with col2:
                    st.caption(row['published'])
                st.divider()
        return
    
    # Calculate technical indicators
    df = _calculate_technical_indicators(df)
    
    # Charts tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price & MA", "RSI", "MACD", "Volume", "News"])
    
    with tab1:
        try:
            st.plotly_chart(_create_price_chart(df, symbol), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render price chart: {e}")
    
    with tab2:
        try:
            if 'RSI' in df.columns and df['RSI'].notna().any():
                st.plotly_chart(_create_rsi_chart(df, symbol), use_container_width=True)
            else:
                st.info("RSI data not available for this period")
        except Exception as e:
            st.warning(f"Could not render RSI chart: {e}")
    
    with tab3:
        try:
            if 'MACD' in df.columns and df['MACD'].notna().any():
                st.plotly_chart(_create_macd_chart(df, symbol), use_container_width=True)
            else:
                st.info("MACD data not available for this period")
        except Exception as e:
            st.warning(f"Could not render MACD chart: {e}")
    
    with tab4:
        try:
            # Volume chart
            fig = go.Figure()
            colors = ['red' if close < open_ else 'green' 
                      for close, open_ in zip(df['close'], df['open'])]
            fig.add_trace(
                go.Bar(
                    x=df['date'], y=df['volume'],
                    marker=dict(color=colors),
                    name='Volume'
                )
            )
            fig.update_layout(
                title=f"{symbol} - Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render volume chart: {e}")
    
    with tab5:
        st.markdown("### 📰 Latest News")
        news_df = _fetch_news_for_symbol(symbol)
        
        if news_df.empty:
            st.info("No recent news available for this symbol")
        else:
            for idx, row in news_df.iterrows():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**[{row['title']}]({row['link']})**")
                    st.caption(f"*{row['source']}*")
                with col2:
                    st.caption(row['published'])
                st.divider()
    
    # AI Prediction section
    st.markdown("---")
    st.markdown("### 🤖 AI Price Prediction")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        available_models = _get_available_models(settings.model_dir)
        pred_model = st.selectbox(
            "Select Model",
            options=available_models,
            key="stock_search_pred_model",
            help="Different models trained on various patterns"
        )
    
    with pred_col2:
        pred_horizon = st.number_input(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            key="stock_search_pred_horizon"
        )
    
    with pred_col3:
        st.markdown("**Actions**")
        predict_btn = st.button("🔮 Predict Price", use_container_width=True, key="stock_search_predict_btn")
    
    if predict_btn:
        with st.spinner(f"🔄 Running {pred_model} prediction for {symbol}..."):
            prediction = _predict_stock_price(
                symbol=symbol,
                model_name=pred_model,
                horizon_days=pred_horizon,
                model_dir=settings.model_dir,
                settings=settings
            )
        
        if prediction:
            # Display prediction results
            st.success("✅ Prediction generated successfully!")
            
            # Key metrics
            pred_metric_col1, pred_metric_col2, pred_metric_col3, pred_metric_col4 = st.columns(4)
            
            current_price = prediction["current_price"]
            predicted_price = prediction["predicted_price"]
            predicted_return = prediction["predicted_return"]
            confidence = prediction["confidence"]
            
            with pred_metric_col1:
                st.metric(
                    "Current Price",
                    f"₹{current_price:.2f}"
                )
            
            with pred_metric_col2:
                st.metric(
                    f"Predicted Price ({pred_horizon}d)",
                    f"₹{predicted_price:.2f}",
                    delta=f"{predicted_return*100:.2f}%" if predicted_return else "N/A"
                )
            
            with pred_metric_col3:
                prob_color = "🟢" if prediction["prob_up"] >= 0.5 else "🔴"
                st.metric(
                    "Probability Up",
                    f"{prob_color} {prediction['prob_up']*100:.1f}%"
                )
            
            with pred_metric_col4:
                st.metric(
                    "Confidence",
                    f"{confidence*100:.1f}%"
                )
            
            # Decision and summary
            st.divider()
            
            decision = prediction["decision"]
            decision_color = "🟢" if decision == "BUY_CANDIDATE" else "🟡"
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"### {decision_color} Decision")
            with col2:
                st.markdown(f"### {decision}")
            
            # Summary details
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("**Prediction Summary:**")
                st.markdown(f"""
                - **Model:** {pred_model}
                - **Horizon:** {pred_horizon} days
                - **Current Price:** ₹{current_price:.2f}
                - **Predicted Price:** ₹{predicted_price:.2f}
                """)
            
            with summary_col2:
                st.markdown("**Technical Metrics:**")
                sentiment_indicator = "Positive" if prediction["sentiment"] > 0 else "Neutral" if prediction["sentiment"] == 0 else "Negative"
                st.markdown(f"""
                - **Expected Return:** {predicted_return*100:+.2f}%
                - **Win Probability:** {prediction['prob_up']*100:.1f}%
                - **Model Confidence:** {confidence*100:.1f}%
                - **Sentiment:** {sentiment_indicator}
                """)
            
            st.info("""
            💡 **Disclaimer:** These predictions are based on historical patterns and should not be used as the sole basis for investment decisions. 
            Always conduct your own research and consult with a financial advisor. Past performance does not guarantee future results.
            """)
        else:
            st.error(f"❌ Could not generate prediction for {symbol}")
            st.info("""
            **Possible reasons:**
            - Model hasn't been trained on this symbol yet
            - Symbol not found in training data
            - Insufficient historical data for this symbol
            
            Try selecting a different symbol or train the model first.
            """)
    
    # Company info
    st.markdown("---")
    st.markdown("### 🏢 Company Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
        st.markdown(f"**Country:** {info.get('country', 'N/A')}")
    
    with col2:
        website = info.get('website', '')
        if website:
            st.markdown(f"**Website:** [{website}]({website})")
        
        st.markdown(f"**Exchange:** {info.get('exchange', 'N/A')}")
        st.markdown(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
    
    if info.get('longBusinessSummary'):
        st.markdown("### 📝 About")
        st.write(info.get('longBusinessSummary', ''))
