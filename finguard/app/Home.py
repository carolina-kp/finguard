import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from finguard.models.predict import predict_forward_return, CONFIG


st.set_page_config(page_title="FinGuard Dashboard", layout="wide")

ARTIFACT_DIR = Path(CONFIG["paths"]["artifacts_dir"])
LATEST_DIR = ARTIFACT_DIR / "latest"


@st.cache_data(show_spinner=False)
def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def _fetch_api_predictions(api_url: str, tickers: List[str], asof: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    try:
        payload = {"tickers": tickers}
        if asof:
            payload["asof"] = asof
        resp = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _ui_header() -> None:
    st.title("FinGuard - Model Dashboard")
    st.caption("Latest artifacts, diagnostics, and quick predictions")


def _ui_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Settings")
    api_url = st.sidebar.text_input("API URL (optional)", "http://127.0.0.1:8000")
    tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL, MSFT, NVDA")
    asof = st.sidebar.text_input("As-of (YYYY-MM-DD, optional)", "")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    return {"api_url": api_url, "tickers": tickers, "asof": asof or None}


def _section_artifacts() -> None:
    st.subheader("Latest Artifacts")
    if not LATEST_DIR.exists():
        st.warning("No latest artifacts found. Train a model first.")
        return

    fi = _load_csv(LATEST_DIR / "feature_importance.csv")
    deciles = _load_csv(LATEST_DIR / "decile_report.csv")
    backtest_summary = _load_json(LATEST_DIR / "backtest_summary.json") or _load_json(ARTIFACT_DIR / "backtest_summary.json")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### Top Feature Importances")
        if fi is None or fi.empty:
            st.info("feature_importance.csv not found.")
        else:
            st.dataframe(fi.head(20), use_container_width=True)
    with cols[1]:
        st.markdown("#### Decile Error Report")
        if deciles is None or deciles.empty:
            st.info("decile_report.csv not found.")
        else:
            st.dataframe(deciles, use_container_width=True)

    st.markdown("#### Backtest Summary")
    if backtest_summary is None:
        st.info("backtest_summary.json not found.")
    else:
        st.json(backtest_summary, expanded=False)


def _section_predict(api_url: str, tickers: List[str], asof: Optional[str]) -> None:
    st.subheader("Quick Predictions")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(f"Tickers: {', '.join(tickers) if tickers else '-'}")
    with col2:
        st.write(f"As-of: {asof or 'latest month-end'}")
    run = st.button("Predict", type="primary", use_container_width=True)

    if not run:
        return
    if not tickers:
        st.warning("Provide at least one ticker.")
        return

    # Try API first if provided; fallback to local predictor
    results = None
    if api_url:
        with st.spinner("Calling API..."):
            results = _fetch_api_predictions(api_url, tickers, asof)
    if results is None:
        with st.spinner("Predicting locally..."):
            preds: List[Dict[str, Any]] = []
            for t in tickers:
                try:
                    yhat = predict_forward_return(t, asof=asof)
                    preds.append({"ticker": t, "predicted_return": yhat, "asof": asof or ""})
                except Exception as exc:
                    preds.append({"ticker": t, "error": str(exc), "asof": asof or ""})
            results = preds

    # Display
    df = pd.DataFrame(results)
    if "predicted_return" in df.columns:
        df["predicted_return"] = df["predicted_return"].astype(float)
    st.dataframe(df, use_container_width=True)


def main() -> None:
    _ui_header()
    params = _ui_sidebar()
    _section_artifacts()
    _section_predict(params["api_url"], params["tickers"], params["asof"])


if __name__ == "__main__":
    main()


