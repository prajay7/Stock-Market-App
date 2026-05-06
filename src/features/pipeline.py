from __future__ import annotations

import pandas as pd

from src.data.news_loader import aggregate_daily_sentiment
from src.features.sentiment import merge_sentiment_features
from src.features.technical import build_technical_features
from src.features.targets import add_classification_target, add_regression_target


def build_feature_pipeline(price_df: pd.DataFrame, news_df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    tech = build_technical_features(price_df)
    sentiment_daily = aggregate_daily_sentiment(news_df)
    merged = merge_sentiment_features(tech, sentiment_daily)
    # Build all supported targets once so train/predict modes can reuse the same dataset.
    merged = add_classification_target(merged, horizon_days=horizon_days)
    merged = add_regression_target(merged, horizon_days=horizon_days)
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    return merged
