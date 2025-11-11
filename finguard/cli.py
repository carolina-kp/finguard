"""FinGuard CLI entrypoint using Typer."""

from __future__ import annotations

import sys
from typing import Optional

import typer

app = typer.Typer(help="FinGuard: Financial Guardrails for Portfolio Management")


@app.command()
def dataset(
    build: bool = typer.Option(False, "--build", help="Build the dataset"),
) -> None:
    """Dataset management commands."""
    if build:
        typer.echo("Building dataset...")
        from data.build_dataset import main as build_dataset_main
        build_dataset_main()
    else:
        typer.echo("Use --build to build the dataset")


@app.command()
def train(
    run: bool = typer.Option(False, "--run", help="Train the model"),
) -> None:
    """Training commands."""
    if run:
        typer.echo("Training model...")
        from models.train_lightgbm import main as train_main
        # Override sys.argv to remove typer arguments
        original_argv = sys.argv
        sys.argv = ["train_lightgbm.py"]
        try:
            train_main()
        finally:
            sys.argv = original_argv
    else:
        typer.echo("Use --run to train the model")


@app.command()
def eval(
    cv: bool = typer.Option(False, "--cv", help="Run cross-validation"),
) -> None:
    """Evaluation commands."""
    if cv:
        typer.echo("Running cross-validation...")
        from models.train_lightgbm import cross_validate, ARTIFACT_DIR
        import json
        
        cv_metrics = cross_validate(n_folds=5)
        
        # Print summary
        typer.echo("\n" + "=" * 60)
        typer.echo("Cross-Validation Results (5-fold expanding window):")
        typer.echo("=" * 60)
        typer.echo(f"R²: {cv_metrics['mean_R2']:.4f} ± {cv_metrics['std_R2']:.4f}")
        typer.echo(f"MAE: {cv_metrics['mean_MAE']:.6f} ± {cv_metrics['std_MAE']:.6f}")
        typer.echo("=" * 60)

        # Save CV metrics
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        cv_metrics_path = ARTIFACT_DIR / "cv_metrics.json"
        with cv_metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(cv_metrics, handle, indent=2)
        typer.echo(f"\nCV metrics saved to: {cv_metrics_path}")
    else:
        typer.echo("Use --cv to run cross-validation")


@app.command()
def backtest(
    walk: bool = typer.Option(False, "--walk", help="Run walk-forward backtest"),
) -> None:
    """Backtesting commands."""
    if walk:
        typer.echo("Running walk-forward backtest...")
        from backtests.walk_forward import main as backtest_main
        backtest_main()
    else:
        typer.echo("Use --walk to run walk-forward backtest")


@app.command()
def predict(
    one: bool = typer.Option(False, "--one", help="Make a single prediction"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Ticker symbol"),
    asof: Optional[str] = typer.Option(None, "--asof", "-a", help="As-of date (YYYY-MM-DD)"),
) -> None:
    """Prediction commands."""
    if one:
        if not ticker:
            typer.echo("Error: --ticker is required for prediction", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"Predicting 90-day return for {ticker}...")
        from models.predict import predict_forward_return
        
        try:
            predicted_return = predict_forward_return(ticker, asof=asof)
            active_asof = asof or "latest month-end"
            typer.echo(f"\n{ticker} ({active_asof})")
            typer.echo(f"Predicted forward 90d return: {predicted_return:.4%}")
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo("Use --one with --ticker to make a prediction")


if __name__ == "__main__":
    app()
