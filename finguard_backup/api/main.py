import json
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from api.routes_score import router as score_router
from features.core import build_features
from models.predict import _load_model, MODEL_ARTIFACT_PATH
import pandas as pd
import numpy as np


app = FastAPI(title="FinGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_model_version() -> str:
    """Get model version from metadata if available."""
    meta_path = MODEL_ARTIFACT_PATH.parent / "meta.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            return meta.get("trained_at", "unknown")
        except Exception:
            return "unknown"
    return "unknown"


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_version": _get_model_version(),
    }


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    tickers: List[str] = Field(..., min_length=1, description="List of ticker symbols")
    asof: Optional[str] = Field(None, description="As-of date (YYYY-MM-DD). Defaults to latest month-end if omitted.")

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: List[str]) -> List[str]:
        """Validate ticker list."""
        if not v:
            raise ValueError("Tickers list cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 tickers allowed per request")
        # Normalize tickers
        return [ticker.upper().strip() for ticker in v if ticker.strip()]

    @field_validator("asof")
    @classmethod
    def validate_asof(cls, v: Optional[str]) -> Optional[str]:
        """Validate asof date format."""
        if v is None:
            return None
        try:
            pd.Timestamp(v)
            return v
        except Exception as exc:
            raise ValueError(f"Invalid date format: {v}. Expected YYYY-MM-DD") from exc


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    ticker: str
    predicted_return: float
    asof: str


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictRequest) -> List[PredictionResponse]:
    """
    Predict 90-day forward returns for multiple tickers.
    
    Accepts a list of tickers and optional asof date.
    Returns predictions for each ticker.
    """
    if not MODEL_ARTIFACT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_ARTIFACT_PATH}. Please train the model first."
        )
    
    try:
        model = _load_model()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(exc)}"
        ) from exc
    
    results: List[PredictionResponse] = []
    errors: List[str] = []
    
    for ticker in request.tickers:
        try:
            # Build features
            features = build_features(ticker, asof=request.asof)
            active_asof = features.get("asof", request.asof or "latest")
            
            # Prepare feature frame
            feature_frame = pd.DataFrame([features])
            feature_frame = feature_frame.drop(columns=["ticker", "asof"], errors="ignore")
            
            # Align with model's expected features
            if hasattr(model, "feature_name_"):
                feature_names = list(model.feature_name_)
                missing_cols = [col for col in feature_names if col not in feature_frame.columns]
                for column in missing_cols:
                    feature_frame[column] = np.nan
                feature_frame = feature_frame[feature_names]
            
            # Predict
            prediction = model.predict(feature_frame)[0]
            
            results.append(PredictionResponse(
                ticker=ticker,
                predicted_return=float(prediction),
                asof=active_asof,
            ))
        except Exception as exc:
            errors.append(f"{ticker}: {str(exc)}")
    
    if not results:
        raise HTTPException(
            status_code=400,
            detail=f"All predictions failed. Errors: {', '.join(errors)}"
        )
    
    if errors:
        # Return partial results with warning in response
        # (FastAPI doesn't support warnings easily, so we'll include in detail)
        pass
    
    return results


app.include_router(score_router)

