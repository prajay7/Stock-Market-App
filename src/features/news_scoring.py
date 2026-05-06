from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


POSITIVE_TERMS: dict[str, float] = {
    "beats estimates": 2.0,
    "record profit": 2.0,
    "strong results": 1.8,
    "profit rises": 1.6,
    "revenue rises": 1.5,
    "upgrade": 1.4,
    "raises target": 1.4,
    "order win": 1.7,
    "wins order": 1.7,
    "contract": 1.2,
    "expansion": 1.2,
    "partnership": 1.1,
    "approval": 1.1,
    "dividend": 1.0,
    "buyback": 1.4,
    "surges": 1.3,
    "jumps": 1.2,
    "rallies": 1.2,
    "gains": 0.8,
    "growth": 0.8,
    "bullish": 1.0,
}

NEGATIVE_TERMS: dict[str, float] = {
    "misses estimates": 2.0,
    "weak results": 1.8,
    "profit falls": 1.6,
    "revenue falls": 1.5,
    "downgrade": 1.4,
    "cuts target": 1.4,
    "fraud": 2.0,
    "probe": 1.6,
    "investigation": 1.6,
    "penalty": 1.5,
    "fine": 1.3,
    "lawsuit": 1.4,
    "resigns": 1.2,
    "default": 2.0,
    "debt": 1.0,
    "loss": 1.2,
    "plunges": 1.5,
    "slumps": 1.4,
    "falls": 0.8,
    "drops": 0.8,
    "bearish": 1.0,
}

EVENT_TERMS: dict[str, float] = {
    "result": 0.65,
    "earnings": 0.65,
    "profit": 0.60,
    "revenue": 0.55,
    "order": 0.65,
    "contract": 0.60,
    "merger": 0.70,
    "acquisition": 0.70,
    "stake": 0.55,
    "fundraise": 0.55,
    "dividend": 0.45,
    "buyback": 0.60,
    "approval": 0.55,
    "launch": 0.45,
    "expansion": 0.50,
    "guidance": 0.60,
    "rating": 0.45,
    "target price": 0.45,
}


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower()).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text))


def _weighted_hits(text: str, terms: dict[str, float]) -> tuple[float, int]:
    score = 0.0
    count = 0
    for phrase, weight in terms.items():
        if _contains_phrase(text, phrase):
            score += float(weight)
            count += 1
    return score, count


def score_news_text(title: Any, summary: Any = "") -> dict[str, float | int | str]:
    text = f"{_clean_text(title)}. {_clean_text(summary)}".lower()
    if not text.strip(". "):
        return {
            "sentiment_score": 0.0,
            "news_impact_score": 0.0,
            "news_signal_score": 0.0,
            "news_positive_hits": 0,
            "news_negative_hits": 0,
            "news_event_score": 0.0,
            "news_sentiment_label": "neutral",
        }

    positive_score, positive_hits = _weighted_hits(text, POSITIVE_TERMS)
    negative_score, negative_hits = _weighted_hits(text, NEGATIVE_TERMS)
    event_score, _ = _weighted_hits(text, EVENT_TERMS)

    raw_sentiment = positive_score - negative_score
    sentiment_score = math.tanh(raw_sentiment / 3.0)
    event_component = min(1.0, event_score)
    emotion_component = min(1.0, abs(raw_sentiment) / 3.0)
    length_component = min(0.25, len(text) / 500.0)
    impact_score = max(0.05 if text else 0.0, min(1.0, 0.45 * event_component + 0.40 * emotion_component + length_component))
    signal_score = max(-1.0, min(1.0, sentiment_score * impact_score))

    if sentiment_score >= 0.12:
        label = "positive"
    elif sentiment_score <= -0.12:
        label = "negative"
    else:
        label = "neutral"

    return {
        "sentiment_score": float(sentiment_score),
        "news_impact_score": float(impact_score),
        "news_signal_score": float(signal_score),
        "news_positive_hits": int(positive_hits),
        "news_negative_hits": int(negative_hits),
        "news_event_score": float(event_component),
        "news_sentiment_label": label,
    }


def enrich_news_scores(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return news_df.copy()

    out = news_df.copy()
    if "summary" not in out.columns:
        out["summary"] = ""
    if "title" not in out.columns:
        out["title"] = ""

    scored = pd.DataFrame(
        [
            score_news_text(row.get("title"), row.get("summary"))
            for row in out[["title", "summary"]].to_dict(orient="records")
        ]
    )
    for column in scored.columns:
        if column == "sentiment_score" and column in out.columns:
            existing = pd.to_numeric(out[column], errors="coerce")
            out[column] = existing.fillna(pd.to_numeric(scored[column], errors="coerce")).fillna(0.0)
        elif column in out.columns and pd.api.types.is_numeric_dtype(scored[column]):
            existing = pd.to_numeric(out[column], errors="coerce")
            out[column] = existing.fillna(pd.to_numeric(scored[column], errors="coerce")).fillna(0.0)
        elif column not in out.columns:
            out[column] = scored[column].values

    if "overall_sentiment_score" in out.columns:
        overall = pd.to_numeric(out["overall_sentiment_score"], errors="coerce")
        out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce").fillna(overall).fillna(0.0)

    return out
