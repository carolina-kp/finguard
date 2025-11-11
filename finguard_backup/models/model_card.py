from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .train_lightgbm import ARTIFACT_DIR, DATASET_DIR


MODEL_DIR = Path("finguard/models")
MODEL_CARD_PATH = MODEL_DIR / "MODEL_CARD.md"


def _read_latest_meta() -> Tuple[Optional[dict], Optional[Path]]:
    latest_dir = ARTIFACT_DIR / "latest"
    meta_path = latest_dir / "meta.json"
    if not meta_path.exists():
        return None, None
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        return meta, latest_dir
    except Exception:
        return None, None


def _dataset_stats(dataset_filename: str) -> Optional[dict]:
    ds_path = DATASET_DIR / dataset_filename
    if not ds_path.exists():
        return None
    try:
        df = pd.read_csv(ds_path, usecols=["asof"], dtype=str)
        df["asof"] = pd.to_datetime(df["asof"], errors="coerce")
        df = df.dropna(subset=["asof"])
        if df.empty:
            return None
        return {
            "rows": int(len(df)),
            "asof_start": df["asof"].min().date().isoformat(),
            "asof_end": df["asof"].max().date().isoformat(),
        }
    except Exception:
        return None


def _read_compare_table() -> Optional[pd.DataFrame]:
    compare_path = ARTIFACT_DIR / "model_compare.csv"
    if not compare_path.exists():
        return None
    try:
        return pd.read_csv(compare_path)
    except Exception:
        return None


def _top_features(latest_dir: Path, n: int = 10) -> Optional[pd.DataFrame]:
    fi_path = latest_dir / "feature_importance.csv"
    if not fi_path.exists():
        return None
    try:
        df = pd.read_csv(fi_path)
        if "feature" in df.columns and "importance" in df.columns:
            return df.sort_values("importance", ascending=False).head(n)
        return None
    except Exception:
        return None


def _format_metrics(meta: dict, compare_df: Optional[pd.DataFrame]) -> str:
    r2 = meta.get("val_R2")
    mae = meta.get("val_MAE")
    lines = []
    if r2 is not None or mae is not None:
        lines.append(f"- Validation R²: {r2:.4f}" if r2 is not None else "- Validation R²: N/A")
        lines.append(f"- Validation MAE: {mae:.6f}" if mae is not None else "- Validation MAE: N/A")
    if compare_df is not None and not compare_df.empty:
        lines.append("")
        lines.append("Model comparison (top by R²):")
        df_sorted = compare_df.sort_values("R2", ascending=False)
        for _, row in df_sorted.iterrows():
            lines.append(f"- {row.get('model', 'model')}: R²={row.get('R2', float('nan')):.4f}, MAE={row.get('MAE', float('nan')):.6f}")
    return "\n".join(lines)


def generate_model_card() -> Optional[Path]:
    meta, latest_dir = _read_latest_meta()
    if meta is None or latest_dir is None:
        print("MODEL_CARD: No latest meta.json found; skipping generation.")
        return None

    dataset_name = meta.get("dataset")
    ds_stats = _dataset_stats(dataset_name) if dataset_name else None
    compare_df = _read_compare_table()
    top_feats = _top_features(latest_dir, n=10)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    goal = "Predict 90-day forward equity returns to support portfolio construction and ranking."
    data_sources = "- Price history and fundamentals via yfinance (with local caching)\n- Derived monthly snapshots from `data/datasets/`"
    features_used = (
        "\n".join(
            [f"- {row['feature']} (importance: {row['importance']})" for _, row in top_feats.iterrows()]
        )
        if top_feats is not None and not top_feats.empty
        else "- See latest run artifacts for feature_importance.csv"
    )
    target = "Forward 90-day return computed as price change over 90 calendar days from each as-of date."
    training_window = (
        f"{ds_stats['asof_start']} to {ds_stats['asof_end']} ({ds_stats['rows']} rows)"
        if ds_stats is not None
        else "Unknown (dataset stats unavailable)"
    )
    metrics = _format_metrics(meta, compare_df)
    limitations = (
        "- Dependent on data availability/quality from yfinance; corporate actions and anomalies may affect inputs\n"
        "- Monthly as-of snapshots; higher-frequency effects are not modeled\n"
        "- Model trained on past data and may underperform during regime shifts"
    )
    failure_modes = (
        "- Illiquid or recently listed tickers with sparse history produce weak or defaulted features\n"
        "- Extreme market events outside training distribution can degrade performance\n"
        "- Features with missing fundamentals may default to zeros and bias predictions"
    )
    usage_notes = (
        "- Ensure `data/datasets/` contains a recent snapshot and train before serving\n"
        "- Use `finguard.models.train_lightgbm --compare` to refresh comparison table\n"
        "- Latest artifacts at `finguard/models/artifacts/latest/`"
    )

    card = []
    card.append("## FinGuard Model Card")
    card.append("")
    card.append(f"- Run ID: {meta.get('run_id', 'unknown')}")
    card.append(f"- Trained at: {meta.get('trained_at', 'unknown')}")
    card.append(f"- Git commit: {meta.get('git_commit', 'unknown')}")
    card.append(f"- Dataset: {dataset_name or 'unknown'}")
    card.append("")
    card.append("### Goal")
    card.append(goal)
    card.append("")
    card.append("### Data sources")
    card.append(data_sources)
    card.append("")
    card.append("### Features used (top 10 by importance)")
    card.append(features_used)
    card.append("")
    card.append("### Target")
    card.append(target)
    card.append("")
    card.append("### Training window")
    card.append(training_window)
    card.append("")
    card.append("### Evaluation metrics")
    card.append(metrics or "- N/A")
    card.append("")
    card.append("### Limitations")
    card.append(limitations)
    card.append("")
    card.append("### Known failure modes")
    card.append(failure_modes)
    card.append("")
    card.append("### Usage notes")
    card.append(usage_notes)
    card.append("")

    MODEL_CARD_PATH.write_text("\n".join(card), encoding="utf-8")
    print(f"MODEL_CARD: Wrote {MODEL_CARD_PATH}")
    return MODEL_CARD_PATH


def main() -> None:
    generate_model_card()


if __name__ == "__main__":
    main()


