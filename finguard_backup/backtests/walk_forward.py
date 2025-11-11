from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
import lightgbm as lgb

from models.train_lightgbm import _load_config, _create_model


def _load_config_local() -> dict:
    """Load config for backtest."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


CONFIG = _load_config_local()
DATASET_DIR = Path(CONFIG["paths"]["datasets_dir"])
ARTIFACT_DIR = Path(CONFIG["paths"]["artifacts_dir"])
RANDOM_SEED = CONFIG["random_seed"]
LGBM_PARAMS = CONFIG["lightgbm"]


def _latest_dataset() -> Path:
    """Get the latest dataset CSV."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    datasets = sorted(DATASET_DIR.glob("snapshot_*.csv"))
    if not datasets:
        raise FileNotFoundError("No dataset CSV files found in data/datasets/")
    return datasets[-1]


def walk_forward_backtest(min_train_months: int = 12) -> tuple[dict, pd.DataFrame]:
    """
    Perform walk-forward backtest with monthly rebalancing.
    
    Strategy:
    - Top decile predictions: Long (equal-weight)
    - Bottom decile predictions: Short (equal-weight)
    - Returns computed from forward_90d_return
    """
    dataset_path = _latest_dataset()
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    if "asof" not in df.columns:
        raise ValueError("Dataset must contain 'asof' column.")
    if "forward_90d_return" not in df.columns:
        raise ValueError("Dataset must contain 'forward_90d_return' column.")
    
    df["asof"] = pd.to_datetime(df["asof"])
    df = df.sort_values("asof").reset_index(drop=True)
    
    # Get feature columns
    feature_columns = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"forward_90d_return", "target"}
    ]
    
    # Get unique month-end dates
    df["year_month"] = df["asof"].dt.to_period("M")
    unique_months = sorted(df["year_month"].unique())
    
    if len(unique_months) < min_train_months + 1:
        raise ValueError(f"Insufficient months ({len(unique_months)}) for backtest. Need at least {min_train_months + 1}.")
    
    print(f"\nWalk-forward backtest:")
    print(f"  Total months: {len(unique_months)}")
    print(f"  Training months: {min_train_months}")
    print(f"  Test months: {len(unique_months) - min_train_months}")
    
    equity_curve: List[Dict[str, object]] = []
    
    # Walk forward: for each test month
    for test_month_idx in range(min_train_months, len(unique_months)):
        test_month = unique_months[test_month_idx]
        train_months = unique_months[:test_month_idx]
        
        # Split data
        train_mask = df["year_month"].isin(train_months)
        test_mask = df["year_month"] == test_month
        
        X_train = df.loc[train_mask, feature_columns]
        y_train = df.loc[train_mask, "forward_90d_return"]
        X_test = df.loc[test_mask, feature_columns]
        y_test = df.loc[test_mask, "forward_90d_return"]
        test_asof = df.loc[test_mask, "asof"].min()
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  Skipping {test_month}: insufficient data")
            continue
        
        # Train model
        model = _create_model()
        callbacks = [
            lgb.early_stopping(stopping_rounds=LGBM_PARAMS["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=LGBM_PARAMS["log_evaluation_period"]),
        ]
        
        # Use a validation set from training data for early stopping
        if len(X_train) > 100:
            val_size = int(len(X_train) * 0.2)
            X_train_inner = X_train.iloc[:-val_size]
            y_train_inner = y_train.iloc[:-val_size]
            X_val_inner = X_train.iloc[-val_size:]
            y_val_inner = y_train.iloc[-val_size:]
            model.fit(
                X_train_inner,
                y_train_inner,
                eval_set=[(X_val_inner, y_val_inner)],
                callbacks=callbacks,
            )
        else:
            model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Rank predictions into deciles
        pred_series = pd.Series(y_pred, index=X_test.index)
        decile_edges = np.percentile(y_pred, np.linspace(0, 100, 11))
        decile_labels = np.digitize(y_pred, decile_edges, right=False)
        decile_labels = np.clip(decile_labels, 1, 10)
        
        # Top decile (D10) = Long, Bottom decile (D1) = Short
        long_mask = decile_labels == 10
        short_mask = decile_labels == 1
        
        # Calculate returns
        if long_mask.sum() > 0:
            long_returns = y_test[long_mask].values
            long_return = float(np.mean(long_returns))
        else:
            long_return = 0.0
        
        if short_mask.sum() > 0:
            short_returns = y_test[short_mask].values
            short_return = float(np.mean(short_returns))
        else:
            short_return = 0.0
        
        # Long-short return
        long_short_return = long_return - short_return
        
        equity_curve.append({
            "asof": test_asof.date().isoformat(),
            "year_month": str(test_month),
            "n_long": int(long_mask.sum()),
            "n_short": int(short_mask.sum()),
            "n_total": int(len(X_test)),
            "long_return": long_return,
            "short_return": short_return,
            "long_short_return": long_short_return,
        })
        
        print(f"  {test_month}: Long={long_return:.4f}, Short={short_return:.4f}, L-S={long_short_return:.4f}")
    
    equity_df = pd.DataFrame(equity_curve)
    
    if equity_df.empty:
        raise ValueError("No backtest periods completed.")
    
    # Calculate summary statistics
    long_returns = equity_df["long_return"].values
    short_returns = equity_df["short_return"].values
    ls_returns = equity_df["long_short_return"].values
    
    # Annualize (assuming monthly returns, 12 months per year)
    months_per_year = 12.0
    
    # Annualized return
    long_ann_return = float(np.mean(long_returns) * months_per_year)
    short_ann_return = float(np.mean(short_returns) * months_per_year)
    ls_ann_return = float(np.mean(ls_returns) * months_per_year)
    
    # Annualized volatility
    long_ann_vol = float(np.std(long_returns) * math.sqrt(months_per_year))
    short_ann_vol = float(np.std(short_returns) * math.sqrt(months_per_year))
    ls_ann_vol = float(np.std(ls_returns) * math.sqrt(months_per_year))
    
    # Sharpe ratio (risk-free rate = 0)
    long_sharpe = long_ann_return / long_ann_vol if long_ann_vol > 0 else 0.0
    short_sharpe = short_ann_return / short_ann_vol if short_ann_vol > 0 else 0.0
    ls_sharpe = ls_ann_return / ls_ann_vol if ls_ann_vol > 0 else 0.0
    
    # Cumulative returns
    equity_df["cumulative_long"] = (1 + equity_df["long_return"]).cumprod()
    equity_df["cumulative_short"] = (1 + equity_df["short_return"]).cumprod()
    equity_df["cumulative_long_short"] = (1 + equity_df["long_short_return"]).cumprod()
    
    summary = {
        "backtest_period": {
            "start": equity_df["asof"].min(),
            "end": equity_df["asof"].max(),
            "n_months": int(len(equity_df)),
        },
        "long": {
            "annualized_return": long_ann_return,
            "annualized_volatility": long_ann_vol,
            "sharpe_ratio": float(long_sharpe),
            "total_return": float(equity_df["cumulative_long"].iloc[-1] - 1.0),
        },
        "short": {
            "annualized_return": short_ann_return,
            "annualized_volatility": short_ann_vol,
            "sharpe_ratio": float(short_sharpe),
            "total_return": float(equity_df["cumulative_short"].iloc[-1] - 1.0),
        },
        "long_short": {
            "annualized_return": ls_ann_return,
            "annualized_volatility": ls_ann_vol,
            "sharpe_ratio": float(ls_sharpe),
            "total_return": float(equity_df["cumulative_long_short"].iloc[-1] - 1.0),
        },
        "backtest_date": datetime.utcnow().isoformat(),
        "dataset": dataset_path.name,
    }
    
    return summary, equity_df


def main() -> None:
    """Run walk-forward backtest and save results."""
    print("=" * 60)
    print("Walk-Forward Backtest")
    print("=" * 60)
    
    summary, equity_df = walk_forward_backtest(min_train_months=12)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Backtest Summary")
    print("=" * 60)
    print(f"Period: {summary['backtest_period']['start']} to {summary['backtest_period']['end']}")
    print(f"Months: {summary['backtest_period']['n_months']}")
    print("\nLong-Short Strategy:")
    print(f"  Annualized Return: {summary['long_short']['annualized_return']:.4f}")
    print(f"  Annualized Volatility: {summary['long_short']['annualized_volatility']:.4f}")
    print(f"  Sharpe Ratio: {summary['long_short']['sharpe_ratio']:.4f}")
    print(f"  Total Return: {summary['long_short']['total_return']:.4f}")
    print("\nLong Only:")
    print(f"  Annualized Return: {summary['long']['annualized_return']:.4f}")
    print(f"  Sharpe Ratio: {summary['long']['sharpe_ratio']:.4f}")
    print("\nShort Only:")
    print(f"  Annualized Return: {summary['short']['annualized_return']:.4f}")
    print(f"  Sharpe Ratio: {summary['short']['sharpe_ratio']:.4f}")
    print("=" * 60)
    
    # Save results
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    summary_path = ARTIFACT_DIR / "backtest_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nBacktest summary saved to: {summary_path}")
    
    equity_path = ARTIFACT_DIR / "backtest_equity_curve.csv"
    equity_df.to_csv(equity_path, index=False)
    print(f"Equity curve saved to: {equity_path}")


if __name__ == "__main__":
    main()

