from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_YEAR = 252
FIVE_YEARS_DAYS = TRADING_DAYS_YEAR * 5
SPY_CACHE_PATH = Path("data/cache/spy_history.json")


def _naive_ts(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    timestamp = pd.Timestamp(ts)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _tz_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def _ensure_cache_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clip(value: Optional[float], low: float, high: float) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return float(max(low, min(high, value)))


def _safe_float(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    try:
        if math.isnan(value):  # type: ignore[arg-type]
            return 0.0
    except TypeError:
        pass
    return float(value)


def _to_date(asof: Optional[str], fallback: datetime) -> datetime:
    if not asof:
        return fallback
    try:
        return datetime.strptime(asof, "%Y-%m-%d")
    except ValueError:
        return fallback


def _fetch_history(ticker: str, asof: Optional[str]) -> pd.DataFrame:
    from utils.yfinance_cache import fetch_history_with_retry
    
    try:
        history = fetch_history_with_retry(ticker, period="5y", interval="1d")
    except ValueError as exc:
        raise ValueError(f"No price history for {ticker}: {exc}") from exc
    
    history = history.sort_index()
    # Normalize timezone immediately after yfinance download
    history = _tz_naive_index(history)
    if asof:
        cutoff = _to_date(asof, history.index.max().to_pydatetime())
        cutoff_ts = _naive_ts(pd.Timestamp(cutoff))
        if cutoff_ts is not None:
            # Ensure both sides of comparison are tz-naive
            history = history[history.index <= cutoff_ts]
    return history


def _load_spy_history(asof: Optional[str]) -> pd.DataFrame:
    latest_needed = _to_date(asof, datetime.now()) if asof else datetime.now()
    cutoff_date = _naive_ts(pd.Timestamp(latest_needed))
    if cutoff_date is None:
        cutoff_date = pd.Timestamp.now().normalize()
    
    if SPY_CACHE_PATH.exists():
        try:
            with SPY_CACHE_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            df = pd.DataFrame(payload)
            if df.empty:
                raise ValueError("Empty cache")
            if "date" not in df.columns:
                raise ValueError("Missing date column in cache")
            # Parse ISO strings back to Timestamps
            df["date"] = pd.to_datetime(df["date"])
            # Ensure tz-naive
            if getattr(df["date"].dt, "tz", None) is not None:
                df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
            df = df.set_index("date").sort_index()
            df = _tz_naive_index(df)
            # Ensure cutoff_date is tz-naive for comparison
            if cutoff_date is not None and cutoff_date.tz is not None:
                cutoff_date = _naive_ts(cutoff_date)
            if not df.empty and cutoff_date is not None and df.index.max() >= cutoff_date - pd.Timedelta(days=5):
                return df[df.index <= cutoff_date]
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            pass

    spy_history = _fetch_history("SPY", None)
    spy_history = spy_history.sort_index()
    spy_history = _tz_naive_index(spy_history)

    _ensure_cache_dir(SPY_CACHE_PATH)
    cache_df = spy_history[["Close"]].copy().reset_index()
    index_col = cache_df.columns[0]
    cache_df = cache_df.rename(columns={index_col: "date"})
    # Ensure date is tz-naive before converting to ISO string
    cache_df["date"] = pd.to_datetime(cache_df["date"])
    if getattr(cache_df["date"].dt, "tz", None) is not None:
        cache_df["date"] = cache_df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    # Convert to ISO format strings for JSON serialization
    cache_df["date"] = cache_df["date"].dt.strftime("%Y-%m-%d")
    with SPY_CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(cache_df.to_dict(orient="records"), handle)
    
    if cutoff_date is not None:
        return spy_history[spy_history.index <= cutoff_date]
    return spy_history


def _cagr(series: pd.Series, periods: int) -> float:
    if series.empty:
        return 0.0
    data = series.tail(periods)
    if data.empty or data.iloc[0] <= 0:
        return 0.0
    years = len(data) / TRADING_DAYS_YEAR
    if years <= 0:
        return 0.0
    return float((data.iloc[-1] / data.iloc[0]) ** (1 / years) - 1)


def _momentum(series: pd.Series, days: int) -> float:
    if len(series) < days:
        return 0.0
    start = series.iloc[-days]
    end = series.iloc[-1]
    if start == 0:
        return 0.0
    return float((end / start) - 1)


def _annualized_vol(series: pd.Series, window: int) -> float:
    returns = series.pct_change().dropna()
    if returns.empty:
        return 0.0
    windowed = returns.tail(window)
    if windowed.empty:
        return 0.0
    return float(windowed.std() * math.sqrt(TRADING_DAYS_YEAR))


def _max_drawdown(series: pd.Series, days: int) -> float:
    if series.empty:
        return 0.0
    windowed = series.tail(days)
    if windowed.empty:
        return 0.0
    running_max = windowed.cummax()
    drawdowns = (windowed / running_max) - 1
    return float(drawdowns.min())


def _beta_vs_spy(series: pd.Series, spy_series: pd.Series, days: int) -> float:
    returns = series.pct_change().dropna().tail(days)
    spy_returns = spy_series.pct_change().dropna().tail(days)
    joined = returns.to_frame("stock").join(spy_returns.to_frame("spy"), how="inner")
    if joined.empty or joined["spy"].var() == 0:
        return 0.0
    cov = joined["stock"].cov(joined["spy"])
    var = joined["spy"].var()
    return float(cov / var)


def _safe_info(ticker: str) -> Dict[str, Optional[float]]:
    from utils.yfinance_cache import fetch_info_with_retry
    return fetch_info_with_retry(ticker)


def build_features(ticker: str, asof: str | None = None) -> Dict[str, object]:
    ticker_upper = ticker.upper()
    price_history = _fetch_history(ticker_upper, asof)
    close_prices = price_history["Close"]
    info = _safe_info(ticker_upper)

    asof_date = close_prices.index.max().date()

    spy_history = _load_spy_history(asof)
    spy_close = spy_history["Close"]

    features = {
        "px_cagr_2y": _cagr(close_prices, TRADING_DAYS_YEAR * 2),
        "px_cagr_1y": _cagr(close_prices, TRADING_DAYS_YEAR * 1),
        "px_momentum_3m": _momentum(close_prices, 63),
        "px_momentum_6m": _momentum(close_prices, 126),
        "px_momentum_12m": _momentum(close_prices, 252),
        "px_vol_60d": _annualized_vol(close_prices, 60),
        "px_vol_250d": _annualized_vol(close_prices, 250),
        "px_maxdd_2y": _max_drawdown(close_prices, TRADING_DAYS_YEAR * 2),
        "px_beta_spy_1y": _beta_vs_spy(close_prices, spy_close, TRADING_DAYS_YEAR),
    }

    market_cap = info.get("marketCap")
    if not market_cap:
        shares = info.get("sharesOutstanding")
        price = close_prices.iloc[-1] if not close_prices.empty else 0
        if shares and price:
            market_cap = shares * price
    if market_cap and market_cap > 0:
        features["size_log_mcap"] = float(math.log(market_cap))
    else:
        features["size_log_mcap"] = 0.0

    features["valuation_pe"] = _clip(info.get("trailingPE"), 0, 200)
    features["valuation_ps"] = _clip(info.get("priceToSalesTrailing12Months"), 0, 50)
    features["valuation_pb"] = _clip(info.get("priceToBook"), 0, 30)

    features["quality_roe"] = _safe_float(info.get("returnOnEquity"))
    features["quality_profit_margin"] = _safe_float(info.get("profitMargins"))
    features["quality_op_margin"] = _safe_float(info.get("operatingMargins"))

    leverage = info.get("debtToEquity")
    features["leverage_dte"] = _clip(leverage if leverage is not None else 0.0, 0, 400)

    return {
        "ticker": ticker_upper,
        "asof": asof_date.isoformat(),
        **features,
    }

