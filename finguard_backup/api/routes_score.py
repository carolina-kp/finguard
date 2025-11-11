from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from models.predict import portfolio_summary, score_ticker


class PortfolioRequest(BaseModel):
    tickers: list[str]


router = APIRouter()


@router.get("/score")
async def get_score(ticker: str = Query(..., min_length=1, description="Ticker symbol to score")) -> Any:
    try:
        return score_ticker(ticker)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/portfolio")
async def post_portfolio(request: PortfolioRequest) -> Any:
    try:
        return portfolio_summary(request.tickers)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc

