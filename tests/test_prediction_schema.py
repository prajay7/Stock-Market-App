from datetime import datetime

from app.models.schemas import PredictResponse


def test_prediction_response_schema_parses():
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "predictions": [
            {
                "symbol": "NVDA",
                "prob_up": 0.71,
                "predicted_return": 0.014,
                "confidence": 0.71,
                "latest_sentiment": 0.28,
                "news_count": 3,
                "news_impact_score": 0.6,
                "news_signal_score": 0.22,
                "news_adjusted_prob_up": 0.75,
                "news_adjusted_predicted_return": 0.02,
                "news_adjusted_confidence": 0.78,
                "news_decision": "NEWS_BUY_CANDIDATE",
                "decision": "BUY_CANDIDATE",
            }
        ],
    }
    obj = PredictResponse(**payload)
    assert obj.predictions[0].symbol == "NVDA"
