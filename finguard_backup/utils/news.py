from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
import yfinance as yf


CACHE_DIR = Path("data/cache/news")
CACHE_TTL_HOURS = 72
NEWS_API_URL = "https://newsapi.org/v2/everything"


def _load_cached(ticker: str) -> List[Dict[str, Any]] | None:
    path = CACHE_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        expires_at = datetime.fromisoformat(payload.get("expires_at", "1970-01-01T00:00:00+00:00"))
        if expires_at < datetime.now(timezone.utc):
            return None
        return payload.get("items", [])
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _store_cached(ticker: str, items: List[Dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=CACHE_TTL_HOURS)).isoformat(),
        "items": items,
    }
    path = CACHE_DIR / f"{ticker.upper()}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _resolve_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
    except Exception:  # pylint: disable=broad-except
        info = {}
    return info.get("longName") or ticker.upper()


def get_headlines(ticker: str, limit: int = 30, lookback_days: int = 90) -> List[Dict[str, Any]]:
    ticker_upper = ticker.upper()
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        cached = _load_cached(ticker_upper)
        return cached or []

    cached_items = _load_cached(ticker_upper)
    if cached_items is not None:
        return cached_items

    company_name = _resolve_company_name(ticker_upper)
    from_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    params = {
        "q": f'"{company_name}" OR {ticker_upper}',
        "language": "en",
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": limit,
        "apiKey": api_key,
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok":
            raise RuntimeError(data.get("message") or "News API error")
        articles = data.get("articles", [])
        items = [
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "publishedAt": article.get("publishedAt"),
                "source": (article.get("source") or {}).get("name"),
            }
            for article in articles
        ]
        _store_cached(ticker_upper, items)
        return items
    except Exception:  # pylint: disable=broad-except
        return cached_items or []

