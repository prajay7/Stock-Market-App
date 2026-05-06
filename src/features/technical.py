from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return pd.DataFrame({"macd": macd, "macd_signal": signal_line, "macd_hist": hist})


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def build_technical_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["return_1d"] = df.groupby("symbol")["close"].pct_change()
    df["log_return_1d"] = np.log1p(df["return_1d"])

    for span in [9, 20, 50]:
        df[f"ema_{span}"] = df.groupby("symbol")["close"].transform(lambda s: s.ewm(span=span, adjust=False).mean())

    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df.groupby("symbol")["return_1d"].shift(lag)

    for win in [5, 10, 20, 30, 60, 120]:
        df[f"return_{win}d"] = df.groupby("symbol")["close"].pct_change(win)
        df[f"log_return_{win}d"] = np.log1p(df[f"return_{win}d"])

    for win in [5, 10, 20, 50, 200]:
        df[f"sma_{win}"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(win, min_periods=win).mean())
        df[f"close_to_sma_{win}"] = df["close"] / df[f"sma_{win}"]

    for win in [20, 50, 100, 200]:
        rolling_high = df.groupby("symbol")["high"].transform(lambda s: s.rolling(win, min_periods=max(5, win // 4)).max())
        rolling_low = df.groupby("symbol")["low"].transform(lambda s: s.rolling(win, min_periods=max(5, win // 4)).min())
        df[f"close_to_high_{win}"] = df["close"] / rolling_high.replace(0, np.nan) - 1.0
        df[f"close_to_low_{win}"] = df["close"] / rolling_low.replace(0, np.nan) - 1.0
        df[f"range_position_{win}"] = (df["close"] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)

    for win in [5, 10, 20, 50]:
        df[f"vol_mean_{win}"] = df.groupby("symbol")["volume"].transform(lambda s: s.rolling(win, min_periods=win).mean())
        df[f"return_std_{win}"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(win, min_periods=win).std())

    df["volume_change_1d"] = df.groupby("symbol")["volume"].pct_change()
    df["dollar_volume"] = df["close"] * df["volume"]
    df["volume_ratio_20"] = df["volume"] / df["vol_mean_20"].replace(0, np.nan)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = (
        (typical_price * df["volume"]).groupby(df["symbol"]).cumsum()
        / df.groupby("symbol")["volume"].cumsum().replace(0, np.nan)
    )

    session_key = df["date"].dt.floor("D")
    df["session_date"] = session_key
    df["bar_in_session"] = df.groupby(["symbol", "session_date"]).cumcount() + 1
    opening_range_window = 3
    df["opening_range_high"] = df.groupby(["symbol", "session_date"])["high"].transform(
        lambda s: s.iloc[:opening_range_window].max() if len(s) else np.nan
    )
    df["opening_range_low"] = df.groupby(["symbol", "session_date"])["low"].transform(
        lambda s: s.iloc[:opening_range_window].min() if len(s) else np.nan
    )
    df["opening_range_breakout"] = (df["high"] >= df["opening_range_high"]).astype(int)
    df["opening_range_breakdown"] = (df["low"] <= df["opening_range_low"]).astype(int)

    prev_day_high = df.groupby("symbol")["high"].shift(1)
    prev_day_low = df.groupby("symbol")["low"].shift(1)
    df["prev_day_high"] = prev_day_high
    df["prev_day_low"] = prev_day_low
    df["prev_day_high_breakout"] = (df["high"] > prev_day_high).astype(int)
    df["prev_day_low_breakdown"] = (df["low"] < prev_day_low).astype(int)

    for win in [5, 20, 50]:
        prior_mean = df.groupby("symbol")["volume"].transform(lambda s: s.shift(1).rolling(win, min_periods=max(2, win // 2)).mean())
        prior_std = df.groupby("symbol")["volume"].transform(lambda s: s.shift(1).rolling(win, min_periods=max(2, win // 2)).std())
        safe_prior_mean = prior_mean.replace(0, np.nan)
        safe_prior_std = prior_std.replace(0, np.nan)
        ratio_col = f"volume_spike_ratio_{win}"
        df[ratio_col] = df["volume"] / safe_prior_mean
        df[f"volume_spike_pct_{win}"] = df[ratio_col] - 1.0
        df[f"volume_zscore_{win}"] = (df["volume"] - prior_mean) / safe_prior_std
        df[f"volume_spike_flag_{win}"] = (df[ratio_col] >= 2.0).astype(int)

    df["volume_trend_5_20"] = df["vol_mean_5"] / df["vol_mean_20"].replace(0, np.nan)
    df["volume_price_pressure_20"] = df["return_1d"] * df["volume_spike_ratio_20"]
    df["volume_momentum_pressure_20"] = df["return_20d"] * df["volume_spike_ratio_20"]

    df["rsi_14"] = df.groupby("symbol")["close"].transform(lambda s: compute_rsi(s, 14))
    macd_df = df.groupby("symbol", group_keys=False)["close"].apply(lambda s: compute_macd(s))
    df[["macd", "macd_signal", "macd_hist"]] = macd_df[["macd", "macd_signal", "macd_hist"]].values

    atr_parts = []
    for _, group in df.groupby("symbol", sort=False):
        atr_series = compute_atr(group, 14)
        atr_series.index = group.index
        atr_parts.append(atr_series)
    if atr_parts:
        df["atr_14"] = pd.concat(atr_parts).sort_index()
    else:
        df["atr_14"] = np.nan
    df["atr_pct_14"] = df["atr_14"] / df["close"].replace(0, np.nan)
    df["intraday_range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["close_position_in_day"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    prev_close = df.groupby("symbol")["close"].shift(1)
    df["gap_pct"] = (df["open"] - prev_close) / prev_close.replace(0, np.nan)
    df["trend_strength_20_50"] = df["sma_20"] / df["sma_50"].replace(0, np.nan) - 1.0
    df["trend_strength_50_200"] = df["sma_50"] / df["sma_200"].replace(0, np.nan) - 1.0

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Defragment once after many feature insertions to avoid repeated PerformanceWarning noise.
    df = df.copy()

    benchmark_candidates = ["^NSEI", "NIFTY", "NIFTY50", "NIFTY 50", "SPY"]
    benchmark_symbol = next((symbol for symbol in benchmark_candidates if symbol in set(df["symbol"].astype(str).str.upper())), "SPY")
    benchmark = (
        df[df["symbol"].astype(str).str.upper() == benchmark_symbol][["date", "return_1d"]]
        .rename(columns={"return_1d": "benchmark_return_1d"})
        .drop_duplicates(subset=["date"])
    )
    if not benchmark.empty:
        df = df.merge(benchmark, on="date", how="left")
        df["relative_return_1d"] = df["return_1d"] - df["benchmark_return_1d"]
        df["relative_strength_nifty"] = df["relative_return_1d"]
    else:
        df["benchmark_return_1d"] = np.nan
        df["relative_return_1d"] = np.nan
        df["relative_strength_nifty"] = np.nan

    return df
