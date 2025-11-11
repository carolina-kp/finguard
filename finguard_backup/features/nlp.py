from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from transformers import pipeline

DECAY_WINDOWS = [
    (30, 1.0),
    (60, 0.7),
    (90, 0.5),
]

NEGATIVE_LABELS = {"negative"}
POSITIVE_LABELS = {"positive"}
NEUTRAL_LABELS = {"neutral"}

CONTROVERSY_TERMS = {
    "probe",
    "lawsuit",
    "recall",
    "ban",
    "leak",
    "outage",
    "strike",
    "fraud",
    "regulator",
    "investigation",
}

_nlp_pipeline = None


@dataclass
class HeadlineSentiment:
    weighted_score_sum: float = 0.0
    weight_sum: float = 0.0
    negative_weight_sum: float = 0.0
    negative_titles: List[str] = None
    controversy_count: int = 0

    def __post_init__(self) -> None:
        if self.negative_titles is None:
            self.negative_titles = []


def _load_pipeline():
    global _nlp_pipeline  # pylint: disable=global-statement
    if _nlp_pipeline is None:
        _nlp_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return _nlp_pipeline


def _parse_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _decay_weight(published_at: Optional[str]) -> float:
    if not published_at:
        return DECAY_WINDOWS[-1][1]
    parsed = _parse_date(published_at)
    if not parsed:
        return DECAY_WINDOWS[-1][1]
    delta_days = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).days
    for max_days, weight in DECAY_WINDOWS:
        if delta_days <= max_days:
            return weight
    return DECAY_WINDOWS[-1][1]


def _sentiment_weight(label: str, score: float) -> float:
    if label.lower() in POSITIVE_LABELS:
        return score
    if label.lower() in NEGATIVE_LABELS:
        return -score
    return 0.0


def _count_controversy(text: str) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    return sum(term in text_lower for term in CONTROVERSY_TERMS)


def _sort_negatives(titles: Iterable[Dict[str, float]]) -> List[str]:
    sorted_items = sorted(titles, key=lambda item: item["weighted"], reverse=False)
    return [item["title"] for item in sorted_items[:5]]


def score_headlines(headlines: List[Dict[str, str]]) -> Dict[str, object]:
    if not headlines:
        return {
            "avg_sentiment": 0.0,
            "neg_share": 0.0,
            "controversy_count": 0,
            "top_negative_titles": [],
        }

    classifier = _load_pipeline()
    aggregator = HeadlineSentiment()
    negative_scores: List[Dict[str, float]] = []

    for headline in headlines:
        title = headline.get("title")
        if not title:
            continue

        sentiment = classifier(title)[0]
        weight = _decay_weight(headline.get("publishedAt"))
        signed_weight = _sentiment_weight(sentiment["label"], sentiment["score"]) * weight

        if signed_weight < 0:
            aggregator.negative_weight_sum += abs(signed_weight)
            negative_scores.append({"title": title, "weighted": signed_weight})

        aggregator.weighted_score_sum += signed_weight
        aggregator.weight_sum += weight
        aggregator.controversy_count += _count_controversy(title)

    avg_sentiment = aggregator.weighted_score_sum / aggregator.weight_sum if aggregator.weight_sum else 0.0
    neg_share = aggregator.negative_weight_sum / aggregator.weight_sum if aggregator.weight_sum else 0.0

    return {
        "avg_sentiment": max(-1.0, min(1.0, avg_sentiment)),
        "neg_share": max(0.0, min(1.0, neg_share)),
        "controversy_count": aggregator.controversy_count,
        "top_negative_titles": _sort_negatives(negative_scores),
    }

