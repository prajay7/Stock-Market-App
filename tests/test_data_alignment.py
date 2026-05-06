import pandas as pd

from src.features.sentiment import merge_sentiment_features


def test_sentiment_alignment_on_dates():
    features = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "close": [100, 101],
        }
    )
    sentiment = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": pd.to_datetime(["2024-01-02"]),
            "sentiment_same_day": [0.5],
            "sentiment_3d": [0.4],
            "sentiment_7d": [0.3],
            "sentiment_momentum_3d": [0.1],
            "news_count": [2],
            "news_signal_score": [0.25],
            "news_impact_mean": [0.6],
        }
    )
    out = merge_sentiment_features(features, sentiment)
    assert float(out.iloc[0]["sentiment_same_day"]) == 0.0
    assert float(out.iloc[1]["sentiment_same_day"]) == 0.5
    assert float(out.iloc[1]["news_signal_score"]) == 0.25
    assert float(out.iloc[1]["news_impact_mean"]) == 0.6
