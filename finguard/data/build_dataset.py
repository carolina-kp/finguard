from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

from features.core import build_features


def _load_config() -> dict:
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


CONFIG = _load_config()
DATASET_DIR = Path(CONFIG["paths"]["datasets_dir"])
UNIVERSE_TICKERS: List[str] = CONFIG["universe"]["tickers"]
FORWARD_HORIZON_DAYS: int = CONFIG.get("forward_horizon_days", 90)


def _month_end_dates(years: int = 3) -> List[pd.Timestamp]:
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=years)
    return list(pd.date_range(start=start, end=today, freq="ME"))


def _fetch_price_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from utils.yfinance_cache import fetch_history_with_retry
    
    try:
        history = fetch_history_with_retry(
            ticker,
            start=start - pd.Timedelta(days=30),
            end=end + pd.Timedelta(days=FORWARD_HORIZON_DAYS + 30),
            interval="1d",
        )
    except ValueError as exc:
        raise ValueError(f"No price history for {ticker}: {exc}") from exc
    
    history = history.sort_index()
    # Ensure tz-naive
    if hasattr(history.index, "tz") and history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    return history


def _price_on_or_before(history: pd.DataFrame, date: pd.Timestamp) -> float | None:
    subset = history[history.index <= date]
    if subset.empty:
        return None
    return float(subset["Close"].iloc[-1])


def _forward_return(history: pd.DataFrame, date: pd.Timestamp, days: int | None = None) -> float | None:
    if days is None:
        days = FORWARD_HORIZON_DAYS
    current_price = _price_on_or_before(history, date)
    if not current_price:
        return None
    future_date = date + pd.Timedelta(days=days)
    subset = history[(history.index > date) & (history.index <= future_date)]
    if subset.empty:
        return None
    future_price = float(subset["Close"].iloc[-1])
    if current_price == 0:
        return None
    return (future_price - current_price) / current_price


def _trailing_sharpe(history: pd.DataFrame, date: pd.Timestamp) -> float | None:
    end = history[history.index <= date]
    window = end[end.index >= date - pd.Timedelta(days=365)]
    if len(window) < 30:
        return None
    returns = window["Close"].pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return None
    sharpe = returns.mean() / returns.std() * math.sqrt(252)
    return float(sharpe)


def _realized_vol(history: pd.DataFrame, date: pd.Timestamp, days: int | None = None) -> float | None:
    if days is None:
        days = FORWARD_HORIZON_DAYS
    window = history[(history.index <= date) & (history.index > date - pd.Timedelta(days=days))]
    if len(window) < 20:
        return None
    returns = window["Close"].pct_change().dropna()
    if returns.empty:
        return None
    return float(returns.std() * math.sqrt(252))


def _non_null_ratio(row: pd.Series, feature_cols: Iterable[str]) -> float:
    subset = row[list(feature_cols)]
    return float(subset.notna().sum() / len(subset))


def _get_build_log_path() -> Path:
    """Get path to build log file."""
    from models.train_lightgbm import ARTIFACT_DIR
    return ARTIFACT_DIR / "build_log.txt"


def _log_skip(ticker: str, reason: str) -> None:
    """Log a skipped ticker to build log."""
    try:
        log_path = _get_build_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} [SKIP] {ticker}: {reason}\n")
    except Exception:
        pass  # Silently fail logging


def build_dataset() -> pd.DataFrame:
    month_ends = _month_end_dates()
    # Cutoff needs to be at least forward_horizon_days before today to have forward returns
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=FORWARD_HORIZON_DAYS + 30)
    month_ends = [d for d in month_ends if d <= cutoff]
    if not month_ends:
        raise RuntimeError("No month-end dates available after applying cutoff.")
    start_date = month_ends[0] - pd.DateOffset(years=1)
    end_date = pd.Timestamp.today()

    records: List[Dict[str, object]] = []
    tickers_ok = 0
    tickers_fail = 0
    rows_built = 0

    # Clear previous log
    log_path = _get_build_log_path()
    if log_path.exists():
        log_path.unlink()

    for ticker in UNIVERSE_TICKERS:
        try:
            history = _fetch_price_history(ticker, start_date, end_date)
        except ValueError as exc:
            tickers_fail += 1
            reason = f"No price history: {exc}"
            print(f"[skip history] {ticker} -> {reason}")
            _log_skip(ticker, reason)
            continue

        rows_before = rows_built

        for month_end in month_ends:
            asof_str = month_end.date().isoformat()
            try:
                features = build_features(ticker, asof=asof_str)
            except Exception as exc:  # pylint: disable=broad-except
                reason = f"Feature build failed: {exc}"
                print(f"[skip features] {ticker} {month_end.date()} -> {reason}")
                _log_skip(ticker, reason)
                continue

            asof_actual = pd.Timestamp(features["asof"])
            target = _forward_return(history, asof_actual)
            trailing_sharpe = _trailing_sharpe(history, asof_actual)
            realized_vol = _realized_vol(history, asof_actual)

            if target is None:
                reason = f"No {FORWARD_HORIZON_DAYS}d forward return for {month_end.date()}"
                print(f"[skip target] {ticker} {month_end.date()} no {FORWARD_HORIZON_DAYS}d forward")
                _log_skip(ticker, reason)
                continue

            # Use dynamic column name based on horizon
            forward_return_col = f"forward_{FORWARD_HORIZON_DAYS}d_return"
            record = {
                **features,
                forward_return_col: target,
                "trailing_sharpe_1y": trailing_sharpe,
                f"realized_vol_{FORWARD_HORIZON_DAYS}d": realized_vol,
            }
            records.append(record)
            rows_built += 1

        if rows_built > rows_before:
            tickers_ok += 1
        else:
            tickers_fail += 1

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(
            "Dataset is empty. Unable to build snapshot. "
            f"tickers_ok={tickers_ok}, tickers_fail={tickers_fail}, rows_built={rows_built}"
        )

    forward_return_col = f"forward_{FORWARD_HORIZON_DAYS}d_return"
    df = df.dropna(subset=[forward_return_col])
    feature_columns = [col for col in df.columns if col not in {"ticker", "asof"}]
    non_null_mask = df.apply(lambda row: _non_null_ratio(row, feature_columns), axis=1) >= 0.8
    df = df[non_null_mask]

    return df.reset_index(drop=True)


def save_dataset(df: pd.DataFrame) -> Path:
    os.makedirs(DATASET_DIR, exist_ok=True)
    snapshot_date = datetime.today().strftime("%Y%m%d")
    path = DATASET_DIR / f"snapshot_{snapshot_date}.csv"
    df.to_csv(path, index=False)
    return path


def main() -> None:
    df = build_dataset()
    path = save_dataset(df)
    print(f"Dataset saved with shape {df.shape} to {path}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().transpose()
        pd.set_option("display.max_columns", None)
        print("Basic stats:")
        print(stats)


if __name__ == "__main__":
    main()

