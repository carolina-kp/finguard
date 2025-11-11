from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import yaml

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from features.core import build_features
from features.governance import governance_index
from features.nlp import score_headlines
from utils.news import get_headlines


TRADING_DAYS_PER_YEAR = 252
def _load_config() -> dict:
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


CONFIG = _load_config()
ARTIFACT_DIR = Path(CONFIG["paths"]["artifacts_dir"])
MODEL_NAME = CONFIG["model_name"]
LATEST_DIR = ARTIFACT_DIR / "latest"
MODEL_ARTIFACT_PATH = (LATEST_DIR / f"{MODEL_NAME}.pkl") if (LATEST_DIR / f"{MODEL_NAME}.pkl").exists() else (ARTIFACT_DIR / f"{MODEL_NAME}.pkl")


@dataclass
class TickerMetrics:
    ticker: str
    cagr: Optional[float]
    max_drawdown: Optional[float]
    ann_vol: Optional[float]
    return_on_equity: Optional[float]
    debt_to_equity: Optional[float]
    long_business_summary: Optional[str]


def _fetch_history(ticker: str) -> pd.DataFrame:
    history = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
    if history.empty:
        raise ValueError(f"No price history available for {ticker}")
    return history


def _cagr(prices: pd.Series) -> Optional[float]:
    if prices.empty or prices.iloc[0] <= 0:
        return None
    total_periods = len(prices)
    total_years = total_periods / TRADING_DAYS_PER_YEAR
    if total_years <= 0:
        return None
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / total_years) - 1


def _max_drawdown(prices: pd.Series) -> Optional[float]:
    if prices.empty:
        return None
    running_max = prices.cummax()
    drawdowns = (prices / running_max) - 1
    return drawdowns.min()


def _ann_vol(prices: pd.Series) -> Optional[float]:
    if len(prices) < 2:
        return None
    returns = prices.pct_change().dropna()
    if returns.empty:
        return None
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def _score_positive_metric(value: Optional[float], low: float, high: float) -> float:
    if value is None or np.isnan(value):
        return 50.0
    score = (value - low) / (high - low)
    return float(np.clip(score * 100, 0, 100))


def _score_negative_metric(value: Optional[float], low: float, high: float) -> float:
    if value is None or np.isnan(value):
        return 50.0
    score = (high - value) / (high - low)
    return float(np.clip(score * 100, 0, 100))


def _collect_metrics(ticker: str) -> TickerMetrics:
    history = _fetch_history(ticker)
    close_prices = history["Close"]
    cagr = _cagr(close_prices)
    max_dd = _max_drawdown(close_prices)
    ann_vol = _ann_vol(close_prices)

    info = yf.Ticker(ticker).info or {}
    return_on_equity = info.get("returnOnEquity")
    debt_to_equity = info.get("debtToEquity")
    long_business_summary = info.get("longBusinessSummary")

    return TickerMetrics(
        ticker=ticker.upper(),
        cagr=cagr,
        max_drawdown=max_dd,
        ann_vol=ann_vol,
        return_on_equity=return_on_equity,
        debt_to_equity=debt_to_equity,
        long_business_summary=long_business_summary,
    )


def _build_subscores(metrics: TickerMetrics, sentiment_score: float, governance_score: float) -> Dict[str, float]:
    returns_quality = _score_positive_metric(metrics.cagr, low=-0.2, high=0.25)
    risk_from_vol = _score_negative_metric(metrics.ann_vol, low=0.1, high=0.6)
    risk_from_drawdown = _score_negative_metric(abs(metrics.max_drawdown) if metrics.max_drawdown is not None else None, low=0.1, high=0.7)
    risk = float(np.clip((risk_from_vol + risk_from_drawdown) / 2, 0, 100))

    profitability = _score_positive_metric(metrics.return_on_equity, low=0, high=0.3)
    leverage = _score_negative_metric(metrics.debt_to_equity, low=0, high=2.0)

    subscores = {
        "ReturnsQuality": returns_quality,
        "Risk": risk,
        "Profitability": profitability,
        "Leverage": leverage,
        "Sentiment": float(np.clip(sentiment_score, 0, 100)),
        "Governance": float(np.clip(governance_score, 0, 100)),
    }
    return subscores


def _clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def score_ticker(ticker: str) -> Dict[str, object]:
    metrics = _collect_metrics(ticker)
    headlines = get_headlines(ticker)
    sentiment_data = score_headlines(headlines)

    controversy_penalty = min(sentiment_data.get("controversy_count", 0) * 2, 30)
    sentiment_score = _clamp(
        (sentiment_data.get("avg_sentiment", 0.0) + 1.0) * 50.0
        - sentiment_data.get("neg_share", 0.0) * 20.0
        - controversy_penalty,
        0.0,
        100.0,
    )

    texts: List[str] = [item["title"] for item in headlines if item.get("title")]
    if metrics.long_business_summary:
        texts.append(metrics.long_business_summary)
    governance_data = governance_index(texts)
    governance_score = min(float(governance_data.get("index", 0.0) or 0.0), 100.0)

    subscores = _build_subscores(metrics, sentiment_score, governance_score)
    score = float(np.mean(list(subscores.values())))

    features = {
        "cagr": metrics.cagr,
        "annualized_volatility": metrics.ann_vol,
        "max_drawdown": metrics.max_drawdown,
        "return_on_equity": metrics.return_on_equity,
        "debt_to_equity": metrics.debt_to_equity,
    }

    return {
        "ticker": metrics.ticker,
        "score": score,
        "subscores": subscores,
        "features": features,
        "shap_top": [],
        "shap_plot_url": None,
        "headlines": headlines[:8],
        "governance": governance_data,
    }


def portfolio_summary(tickers: List[str]) -> Dict[str, object]:
    if not tickers:
        raise ValueError("Tickers list cannot be empty")

    details = [score_ticker(t) for t in tickers]

    avg_score = float(mean(d["score"] for d in details if d.get("score") is not None))

    def _extract_metric(key: str) -> List[float]:
        values = []
        for item in details:
            value = item["features"].get(key)
            if value is not None and not np.isnan(value):
                values.append(float(value))
        return values

    cagr_values = _extract_metric("cagr")
    vol_values = _extract_metric("annualized_volatility")
    dd_values = _extract_metric("max_drawdown")

    portfolio_metrics = {
        "average_score": avg_score,
        "cagr": float(mean(cagr_values)) if cagr_values else None,
        "max_drawdown": float(mean(dd_values)) if dd_values else None,
        "volatility": float(mean(vol_values)) if vol_values else None,
        "details": details,
    }

    return portfolio_metrics


def _latest_month_end() -> str:
    today = pd.Timestamp.today().normalize()
    month_end = today - pd.offsets.MonthEnd(0)
    return month_end.date().isoformat()


def _load_model() -> Any:
    # Prefer latest run; fallback to legacy root artifact
    preferred = LATEST_DIR / f"{MODEL_NAME}.pkl"
    path = preferred if preferred.exists() else MODEL_ARTIFACT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    model = joblib.load(path)
    return model


def predict_forward_return(ticker: str, asof: Optional[str] = None) -> float:
    target_asof = asof or _latest_month_end()
    features = build_features(ticker, asof=target_asof)

    model = _load_model()
    feature_frame = pd.DataFrame([features])
    feature_frame = feature_frame.drop(columns=["ticker", "asof"], errors="ignore")

    if hasattr(model, "feature_name_"):
        feature_names = list(model.feature_name_)
        missing_cols = [col for col in feature_names if col not in feature_frame.columns]
        for column in missing_cols:
            feature_frame[column] = np.nan
        feature_frame = feature_frame[feature_names]

    prediction = model.predict(feature_frame)[0]
    return float(prediction)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict 90-day return for a ticker using the trained LightGBM model.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to score.")
    parser.add_argument("--asof", help="As-of date (YYYY-MM-DD). Defaults to latest month-end if omitted.")
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()
    ticker = args.ticker.upper()
    asof = args.asof

    if asof:
        asof_ts = pd.Timestamp(asof)
        if asof_ts.tz is not None:
            asof_ts = asof_ts.tz_convert("UTC").tz_localize(None)
        asof = asof_ts.date().isoformat()

    predicted_return = predict_forward_return(ticker, asof=asof)
    active_asof = asof or _latest_month_end()
    print(f"{ticker} ({active_asof}) predicted forward 90d return: {predicted_return:.4%}")


if __name__ == "__main__":
    main()

