import json
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="FinGuard", layout="wide")


def _parse_tickers(raw: str) -> List[str]:
    return [ticker.strip().upper() for ticker in raw.split(",") if ticker.strip()]


def _fetch_portfolio(api_base: str, tickers: List[str]) -> Dict[str, Any]:
    response = requests.post(f"{api_base}/portfolio", json={"tickers": tickers}, timeout=30)
    response.raise_for_status()
    return response.json()


def _fetch_score(api_base: str, ticker: str) -> Dict[str, Any]:
    response = requests.get(f"{api_base}/score", params={"ticker": ticker}, timeout=15)
    response.raise_for_status()
    return response.json()


st.sidebar.header("FinGuard Settings")
tickers_input = st.sidebar.text_area("Tickers", "AAPL, MSFT, NVDA")
api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = None

if st.sidebar.button("Fetch & Score"):
    tickers = _parse_tickers(tickers_input)
    if not tickers:
        st.sidebar.error("Please provide at least one ticker.")
    else:
        try:
            st.session_state["portfolio"] = _fetch_portfolio(api_url, tickers)
            st.sidebar.success("Portfolio updated.")
        except requests.RequestException as exc:
            st.sidebar.error(f"Failed to fetch portfolio: {exc}")


portfolio_tab, explain_tab = st.tabs(["Portfolio", "Explain"])

with portfolio_tab:
    data = st.session_state.get("portfolio")
    if not data:
        st.info("Fetch portfolio data to see metrics.")
    else:
        cols = st.columns(3)
        cols[0].metric("Average Score", f"{data.get('average_score', 0):.1f}")
        cagr = data.get("cagr")
        max_dd = data.get("max_drawdown")
        cols[1].metric("CAGR", f"{cagr:.2%}" if cagr is not None else "N/A")
        cols[2].metric("Max Drawdown", f"{max_dd:.2%}" if max_dd is not None else "N/A")

        details = data.get("details", [])
        if details:
            df = pd.DataFrame(details)
            df["score"] = df["score"].round(1)
            st.dataframe(df)


with explain_tab:
    data = st.session_state.get("portfolio")
    if not data or not data.get("details"):
        st.info("Fetch portfolio data first.")
    else:
        tickers = [item["ticker"] for item in data["details"]]
        selected = st.selectbox("Select ticker", tickers)
        if selected:
            try:
                score_data = _fetch_score(api_url, selected)
                st.json(score_data, expanded=True)
            except requests.RequestException as exc:
                st.error(f"Failed to fetch ticker score: {exc}")

