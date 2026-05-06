import pandas as pd

from src.data.news_loader import aggregate_daily_sentiment
from src.features.news_scoring import score_news_text
from src.features.technical import build_technical_features


def test_build_technical_features_basic_columns():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=260, freq="D"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": range(260),
            "volume": 1000,
            "symbol": "AAPL",
        }
    )
    spy = df.copy()
    spy["symbol"] = "SPY"
    inp = pd.concat([df, spy], ignore_index=True)

    out = build_technical_features(inp)
    for col in [
        "rsi_14",
        "macd",
        "sma_20",
        "relative_return_1d",
        "day_of_week",
        "return_30d",
        "close_to_high_50",
        "range_position_50",
        "atr_pct_14",
        "gap_pct",
        "trend_strength_20_50",
        "volume_spike_ratio_20",
        "volume_spike_pct_20",
        "volume_zscore_20",
        "volume_spike_flag_20",
        "volume_trend_5_20",
        "volume_price_pressure_20",
        "volume_momentum_pressure_20",
    ]:
        assert col in out.columns


def test_news_scoring_and_daily_aggregation_adds_signal_features():
    score = score_news_text("Reliance wins order and profit rises after strong results")
    assert score["sentiment_score"] > 0
    assert score["news_impact_score"] > 0
    assert score["news_signal_score"] > 0

    news = pd.DataFrame(
        {
            "symbol": ["RELIANCE", "RELIANCE"],
            "title": [
                "Reliance wins order and profit rises after strong results",
                "Reliance shares fall after weak results",
            ],
            "summary": ["", ""],
            "url": ["https://example.com/a", "https://example.com/b"],
            "published_at": pd.to_datetime(["2026-04-20 09:00", "2026-04-21 09:00"]),
        }
    )
    daily = aggregate_daily_sentiment(news)
    for col in ["news_impact_mean", "news_signal_score", "news_signal_3d", "positive_news_count", "negative_news_count"]:
        assert col in daily.columns
    assert int(daily["news_count"].sum()) == 2
