from __future__ import annotations

import pandas as pd


SENTIMENT_FEATURE_COLUMNS = [
    "sentiment_same_day",
    "sentiment_3d",
    "sentiment_7d",
    "sentiment_momentum_3d",
    "news_count",
    "news_count_3d",
    "news_count_7d",
    "positive_news_count",
    "negative_news_count",
    "news_impact_mean",
    "news_impact_max",
    "news_impact_3d",
    "news_signal_score",
    "news_signal_3d",
    "news_signal_7d",
    "news_buzz_3d",
    "relevance_mean",
]


def merge_sentiment_features(features_df: pd.DataFrame, sentiment_daily_df: pd.DataFrame) -> pd.DataFrame:
    if sentiment_daily_df.empty:
        out = features_df.copy()
        for column in SENTIMENT_FEATURE_COLUMNS:
            out[column] = 0.0
        out["news_count"] = out["news_count"].astype(int)
        return out

    left = features_df.copy()
    right = sentiment_daily_df.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.floor("D")
    right["date"] = pd.to_datetime(right["date"]).dt.floor("D")
    for column in SENTIMENT_FEATURE_COLUMNS:
        if column not in right.columns:
            right[column] = 0.0

    merged = left.merge(
        right[["symbol", "date", *SENTIMENT_FEATURE_COLUMNS]],
        on=["symbol", "date"],
        how="left",
    )
    for c in SENTIMENT_FEATURE_COLUMNS:
        merged[c] = merged[c].fillna(0.0)
    for c in ["news_count", "news_count_3d", "news_count_7d", "positive_news_count", "negative_news_count"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)
    return merged
