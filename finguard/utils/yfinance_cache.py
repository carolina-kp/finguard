"""Robust yfinance fetching with retry, caching, and logging."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


CACHE_DIR = Path("data/cache/prices")
CACHE_STALE_DAYS = 2


def _ensure_cache_dir() -> None:
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(ticker: str) -> Path:
    """Get cache file path for a ticker."""
    return CACHE_DIR / f"{ticker.upper()}.parquet"


def _is_cache_stale(cache_path: Path) -> bool:
    """Check if cache is stale (>2 days old)."""
    if not cache_path.exists():
        return True
    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age > timedelta(days=CACHE_STALE_DAYS)
    except Exception:
        return True


def _load_cached_history(ticker: str) -> Optional[pd.DataFrame]:
    """Load cached price history if available and fresh."""
    cache_path = _get_cache_path(ticker)
    if not cache_path.exists():
        return None
    
    if _is_cache_stale(cache_path):
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        if df.empty:
            return None
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
                df.index = pd.to_datetime(df.index)
            else:
                return None
        # Ensure tz-naive
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def _save_cached_history(ticker: str, history: pd.DataFrame) -> None:
    """Save price history to cache."""
    if history.empty:
        return
    
    try:
        _ensure_cache_dir()
        cache_path = _get_cache_path(ticker)
        # Save with index as Date column for parquet
        cache_df = history.copy()
        if isinstance(cache_df.index, pd.DatetimeIndex):
            cache_df = cache_df.reset_index()
            # Rename the first column (index) to Date
            first_col = cache_df.columns[0]
            cache_df = cache_df.rename(columns={first_col: "Date"})
        cache_df.to_parquet(cache_path, index=False)
    except Exception:
        pass  # Silently fail cache write


def fetch_history_with_retry(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch price history with retry and caching.
    
    Args:
        ticker: Ticker symbol
        period: Period string (e.g., "5y", "1y")
        interval: Interval string (e.g., "1d", "1h")
        start: Start date (optional, alternative to period)
        end: End date (optional)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
    
    Returns:
        DataFrame with price history
    
    Raises:
        ValueError: If all retries fail
    """
    ticker_upper = ticker.upper()
    
    # Try cache first
    cached = _load_cached_history(ticker_upper)
    if cached is not None:
        # Filter by date range if needed
        if start is not None:
            cached = cached[cached.index >= start]
        if end is not None:
            cached = cached[cached.index <= end]
        if not cached.empty:
            return cached
    
    # Fetch from yfinance with retry
    ticker_obj = yf.Ticker(ticker_upper)
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            if start is not None or end is not None:
                history = ticker_obj.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                )
            else:
                history = ticker_obj.history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                )
            
            if not history.empty:
                # Normalize timezone
                if hasattr(history.index, "tz") and history.index.tz is not None:
                    history.index = history.index.tz_convert("UTC").tz_localize(None)
                
                # Save to cache
                _save_cached_history(ticker_upper, history)
                return history
            
            # Empty history - might be temporary, retry
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            
        except Exception as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
    
    # All retries failed
    if last_exception:
        raise ValueError(f"Failed to fetch history for {ticker_upper} after {max_retries} attempts: {last_exception}")
    raise ValueError(f"No price history available for {ticker_upper}")


def fetch_info_with_retry(
    ticker: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict:
    """
    Fetch ticker info with retry.
    
    Args:
        ticker: Ticker symbol
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
    
    Returns:
        Dictionary with ticker info
    """
    ticker_upper = ticker.upper()
    ticker_obj = yf.Ticker(ticker_upper)
    
    for attempt in range(max_retries):
        try:
            info = ticker_obj.info
            if info:
                return info
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
        except Exception:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    
    return {}

