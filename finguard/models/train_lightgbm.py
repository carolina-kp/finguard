# Top 10 features by importance (latest artifact):
# forward_90d_return, px_vol_250d, px_momentum_3m, px_momentum_6m,
# px_maxdd_2y, px_vol_60d, px_cagr_2y, realized_vol_90d,
# valuation_ps, px_cagr_1y

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


def _load_config() -> dict:
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


CONFIG = _load_config()
DATASET_DIR = Path(CONFIG["paths"]["datasets_dir"])
ARTIFACT_DIR = Path(CONFIG["paths"]["artifacts_dir"])
RANDOM_SEED = CONFIG["random_seed"]
TEST_SIZE_BY_DATE = CONFIG["test_size_by_date"]
MODEL_NAME = CONFIG["model_name"]
LGBM_PARAMS = CONFIG["lightgbm"]


def _latest_dataset() -> Path:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    datasets = sorted(DATASET_DIR.glob("snapshot_*.csv"))
    if not datasets:
        raise FileNotFoundError("No dataset CSV files found in data/datasets/")
    return datasets[-1]


def _new_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _git_commit_hash() -> str | None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return commit or None
    except Exception:
        return None


def _prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if "asof" not in df.columns:
        raise ValueError("Dataset must contain 'asof' column for time-based split.")
    df["asof"] = pd.to_datetime(df["asof"])
    df = df.sort_values("asof").reset_index(drop=True)

    if "target" not in df.columns:
        # Try to find forward return column dynamically
        forward_cols = [col for col in df.columns if col.startswith("forward_") and col.endswith("_return")]
        if forward_cols:
            df["target"] = df[forward_cols[0]]
        elif "forward_90d_return" in df.columns:
            df["target"] = df["forward_90d_return"]
        else:
            raise ValueError("Dataset must contain 'target' or a 'forward_*_return' column.")

    # Get unique asof dates and sort chronologically
    unique_asofs = sorted(df["asof"].unique())
    n_asofs = len(unique_asofs)
    
    # Split by date: last TEST_SIZE_BY_DATE% of unique asof dates go to validation
    split_idx = int(n_asofs * (1 - TEST_SIZE_BY_DATE))
    if split_idx == 0:
        split_idx = 1
    if split_idx >= n_asofs:
        split_idx = n_asofs - 1
    
    train_asof_cutoff = unique_asofs[split_idx]
    
    # Ensure strict separation: train < cutoff, valid >= cutoff
    train_mask = df["asof"] < train_asof_cutoff
    valid_mask = df["asof"] >= train_asof_cutoff

    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError("Insufficient data for time-based split.")

    # Guardrail 1: Verify no row in valid set has asof earlier than or equal to max asof in train
    train_max_asof = df.loc[train_mask, "asof"].max()
    valid_min_asof = df.loc[valid_mask, "asof"].min()
    
    print(f"Time-based split:")
    print(f"  Train: {train_mask.sum()} rows, asof range: {df.loc[train_mask, 'asof'].min().date()} to {train_max_asof.date()}")
    print(f"  Valid: {valid_mask.sum()} rows, asof range: {valid_min_asof.date()} to {df.loc[valid_mask, 'asof'].max().date()}")
    
    if valid_min_asof <= train_max_asof:
        raise ValueError(
            f"DATA LEAKAGE DETECTED (Guardrail 1): "
            f"Valid set contains rows with asof ({valid_min_asof.date()}) <= train max asof ({train_max_asof.date()}). "
            f"All valid rows must have asof strictly after train max asof."
        )
    
    print(f"  ✓ Guardrail 1 passed: valid min asof ({valid_min_asof.date()}) > train max asof ({train_max_asof.date()})")
    
    # Guardrail 2: Verify no forward labels in train use dates overlapping the valid window
    # Forward returns are calculated from asof to asof + forward_horizon_days
    # We need to ensure train's forward return windows don't overlap with valid asof dates
    forward_horizon_days = CONFIG.get("forward_horizon_days", 90)
    train_forward_end_dates = df.loc[train_mask, "asof"] + pd.Timedelta(days=forward_horizon_days)
    train_max_forward_end = train_forward_end_dates.max()
    
    if train_max_forward_end >= valid_min_asof:
        overlapping_train_rows = df.loc[train_mask][train_forward_end_dates >= valid_min_asof]
        overlapping_count = len(overlapping_train_rows)
        overlapping_example = overlapping_train_rows.iloc[0] if overlapping_count > 0 else None
        
        raise ValueError(
            f"DATA LEAKAGE DETECTED (Guardrail 2): "
            f"Training set contains {overlapping_count} row(s) whose forward return calculation window "
            f"overlaps with validation period. "
            f"Train max forward end date ({train_max_forward_end.date()}) >= valid min asof ({valid_min_asof.date()}). "
            f"Forward returns are calculated from asof to asof + {forward_horizon_days} days. "
            f"Example overlapping row: ticker={overlapping_example.get('ticker', 'N/A')}, "
            f"asof={overlapping_example.get('asof', 'N/A') if overlapping_example is not None else 'N/A'}. "
            f"Training aborted to prevent data leakage."
        )
    
    print(f"  ✓ Guardrail 2 passed: train max forward end date ({train_max_forward_end.date()}) < valid min asof ({valid_min_asof.date()})")
    print(f"  ✓ All guardrails passed. Safe to proceed with training.")

    feature_columns = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"target"}
    ]
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    X_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, "target"]
    X_valid = df.loc[valid_mask, feature_columns]
    y_valid = df.loc[valid_mask, "target"]
    return X_train, X_valid, y_train, y_valid


def _create_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective=LGBM_PARAMS["objective"],
        n_estimators=LGBM_PARAMS["n_estimators"],
        learning_rate=LGBM_PARAMS["learning_rate"],
        num_leaves=LGBM_PARAMS["num_leaves"],
        max_depth=LGBM_PARAMS["max_depth"],
        subsample=LGBM_PARAMS["subsample"],
        colsample_bytree=LGBM_PARAMS["colsample_bytree"],
        reg_alpha=LGBM_PARAMS["reg_alpha"],
        reg_lambda=LGBM_PARAMS["reg_lambda"],
        random_state=RANDOM_SEED,
        n_jobs=LGBM_PARAMS["n_jobs"],
    )


def train_model() -> Tuple[LGBMRegressor, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.Series]:
    dataset_path = _latest_dataset()
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    X_train, X_valid, y_train, y_valid = _prepare_data(df)

    model = _create_model()

    callbacks = [
        lgb.early_stopping(stopping_rounds=LGBM_PARAMS["early_stopping_rounds"], verbose=True),
        lgb.log_evaluation(period=LGBM_PARAMS["log_evaluation_period"]),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=callbacks,
    )

    # Log best iteration if available (from early stopping)
    if hasattr(model, "best_iteration_"):
        print(f"Early stopping: best iteration = {model.best_iteration_}")

    y_pred = model.predict(X_valid)
    val_r2 = r2_score(y_valid, y_pred)
    val_mae = mean_absolute_error(y_valid, y_pred)

    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "rows": int(len(df)),
        "features": len(X_train.columns),
        "val_R2": float(val_r2),
        "val_MAE": float(val_mae),
        "dataset": dataset_path.name,
        "params": {
            "model_name": MODEL_NAME,
            "random_seed": RANDOM_SEED,
            "lightgbm": LGBM_PARAMS,
            "test_size_by_date": TEST_SIZE_BY_DATE,
        },
    }

    return model, feature_importance, metadata, X_train, X_valid, y_valid


def cross_validate(n_folds: int = 5) -> dict:
    """Perform expanding window time series cross-validation (walk-forward)."""
    dataset_path = _latest_dataset()
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    if "asof" not in df.columns:
        raise ValueError("Dataset must contain 'asof' column for time-based CV.")
    df["asof"] = pd.to_datetime(df["asof"])
    df = df.sort_values("asof").reset_index(drop=True)

    if "target" not in df.columns:
        # Try to find forward return column dynamically
        forward_cols = [col for col in df.columns if col.startswith("forward_") and col.endswith("_return")]
        if forward_cols:
            df["target"] = df[forward_cols[0]]
        elif "forward_90d_return" in df.columns:
            df["target"] = df["forward_90d_return"]
        else:
            raise ValueError("Dataset must contain 'target' or a 'forward_*_return' column.")

    # Get unique asof dates and sort chronologically
    unique_asofs = sorted(df["asof"].unique())
    n_asofs = len(unique_asofs)

    if n_asofs < n_folds + 1:
        raise ValueError(f"Insufficient unique asof dates ({n_asofs}) for {n_folds} folds. Need at least {n_folds + 1}.")

    # Create expanding window folds
    # Each fold uses all data up to a certain point as train, and a chunk as test
    fold_size = n_asofs // (n_folds + 1)  # Reserve some initial data for first fold
    if fold_size < 1:
        fold_size = 1

    feature_columns = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"target"}
    ]

    fold_results: List[dict] = []

    for fold_idx in range(n_folds):
        # Expanding window: train uses all data up to fold_idx * fold_size + initial_size
        # Test uses the next fold_size dates
        initial_size = fold_size  # Minimum training size
        train_end_idx = initial_size + fold_idx * fold_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, n_asofs)

        if test_start_idx >= n_asofs:
            break

        train_asof_cutoff = unique_asofs[train_end_idx]
        test_start_asof = unique_asofs[test_start_idx]
        test_end_asof = unique_asofs[test_end_idx - 1] if test_end_idx > test_start_idx else unique_asofs[-1]

        train_mask = df["asof"] < train_asof_cutoff
        test_mask = (df["asof"] >= test_start_asof) & (df["asof"] <= test_end_asof)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"Fold {fold_idx + 1}: Skipping (insufficient data)")
            continue

        # Guardrail checks for this fold
        train_max_asof = df.loc[train_mask, "asof"].max()
        test_min_asof = df.loc[test_mask, "asof"].min()
        
        if test_min_asof <= train_max_asof:
            raise ValueError(
                f"DATA LEAKAGE DETECTED in Fold {fold_idx + 1} (Guardrail 1): "
                f"Test set contains rows with asof ({test_min_asof.date()}) <= train max asof ({train_max_asof.date()})."
            )
        
        forward_horizon_days = CONFIG.get("forward_horizon_days", 90)
        train_forward_end_dates = df.loc[train_mask, "asof"] + pd.Timedelta(days=forward_horizon_days)
        train_max_forward_end = train_forward_end_dates.max()
        
        if train_max_forward_end >= test_min_asof:
            raise ValueError(
                f"DATA LEAKAGE DETECTED in Fold {fold_idx + 1} (Guardrail 2): "
                f"Training set forward return windows overlap with test period. "
                f"Train max forward end ({train_max_forward_end.date()}) >= test min asof ({test_min_asof.date()})."
            )

        X_train = df.loc[train_mask, feature_columns]
        y_train = df.loc[train_mask, "target"]
        X_test = df.loc[test_mask, feature_columns]
        y_test = df.loc[test_mask, "target"]

        print(f"\nFold {fold_idx + 1}/{n_folds}:")
        print(f"  Train: {train_mask.sum()} rows, asof < {train_asof_cutoff.date()}")
        print(f"  Test: {test_mask.sum()} rows, asof range: {test_start_asof.date()} to {test_end_asof.date()}")
        print(f"  ✓ Guardrails passed for fold {fold_idx + 1}")

        model = _create_model()
        callbacks = [
            lgb.early_stopping(stopping_rounds=LGBM_PARAMS["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=LGBM_PARAMS["log_evaluation_period"]),
        ]

        # Use a small validation set from training data for early stopping
        train_unique_asofs = sorted(df.loc[train_mask, "asof"].unique())
        if len(train_unique_asofs) > 2:
            val_split_idx = int(len(train_unique_asofs) * 0.8)
            val_asof_cutoff = train_unique_asofs[val_split_idx]
            train_inner_mask = df.loc[train_mask].index[df.loc[train_mask, "asof"] < val_asof_cutoff]
            val_inner_mask = df.loc[train_mask].index[
                (df.loc[train_mask, "asof"] >= val_asof_cutoff) & (df.loc[train_mask, "asof"] < train_asof_cutoff)
            ]
            if len(val_inner_mask) > 0:
                X_train_inner = df.loc[train_inner_mask, feature_columns]
                y_train_inner = df.loc[train_inner_mask, "target"]
                X_val_inner = df.loc[val_inner_mask, feature_columns]
                y_val_inner = df.loc[val_inner_mask, "target"]
                eval_set = [(X_val_inner, y_val_inner)]
            else:
                eval_set = None
        else:
            eval_set = None

        if eval_set:
            model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        fold_r2 = r2_score(y_test, y_pred)
        fold_mae = mean_absolute_error(y_test, y_pred)

        fold_results.append({
            "fold": fold_idx + 1,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_asof_max": train_asof_cutoff.date().isoformat(),
            "test_asof_min": test_start_asof.date().isoformat(),
            "test_asof_max": test_end_asof.date().isoformat(),
            "R2": float(fold_r2),
            "MAE": float(fold_mae),
        })

        print(f"  R²: {fold_r2:.4f}, MAE: {fold_mae:.6f}")

    if not fold_results:
        raise ValueError("No valid folds were created.")

    # Calculate statistics
    r2_values = [f["R2"] for f in fold_results]
    mae_values = [f["MAE"] for f in fold_results]

    cv_metrics = {
        "n_folds": len(fold_results),
        "folds": fold_results,
        "mean_R2": float(np.mean(r2_values)),
        "std_R2": float(np.std(r2_values)),
        "mean_MAE": float(np.mean(mae_values)),
        "std_MAE": float(np.std(mae_values)),
        "trained_at": datetime.utcnow().isoformat(),
        "dataset": dataset_path.name,
    }

    return cv_metrics


def compare_models() -> pd.DataFrame:
    """Train and compare ElasticNet, RandomForest, and LightGBM on the same validation set."""
    dataset_path = _latest_dataset()
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    X_train, X_valid, y_train, y_valid = _prepare_data(df)

    print("\n" + "=" * 60)
    print("Training models for comparison...")
    print("=" * 60)

    results = []

    # 1. ElasticNet (baseline)
    print("\n1. Training ElasticNet...")
    elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=2000)
    elastic_net.fit(X_train, y_train)
    y_pred_en = elastic_net.predict(X_valid)
    r2_en = r2_score(y_valid, y_pred_en)
    mae_en = mean_absolute_error(y_valid, y_pred_en)
    rmse_en = np.sqrt(mean_squared_error(y_valid, y_pred_en))
    results.append({
        "model": "ElasticNet",
        "R2": float(r2_en),
        "MAE": float(mae_en),
        "RMSE": float(rmse_en),
    })
    print(f"  R²: {r2_en:.4f}, MAE: {mae_en:.6f}, RMSE: {rmse_en:.6f}")

    # 2. RandomForestRegressor (baseline)
    print("\n2. Training RandomForestRegressor...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_valid)
    r2_rf = r2_score(y_valid, y_pred_rf)
    mae_rf = mean_absolute_error(y_valid, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_valid, y_pred_rf))
    results.append({
        "model": "RandomForest",
        "R2": float(r2_rf),
        "MAE": float(mae_rf),
        "RMSE": float(rmse_rf),
    })
    print(f"  R²: {r2_rf:.4f}, MAE: {mae_rf:.6f}, RMSE: {rmse_rf:.6f}")

    # 3. LightGBM (current best)
    print("\n3. Training LightGBM...")
    lgbm_model = _create_model()
    callbacks = [
        lgb.early_stopping(stopping_rounds=LGBM_PARAMS["early_stopping_rounds"], verbose=False),
        lgb.log_evaluation(period=LGBM_PARAMS["log_evaluation_period"]),
    ]
    lgbm_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=callbacks,
    )
    if hasattr(lgbm_model, "best_iteration_"):
        print(f"  Early stopping: best iteration = {lgbm_model.best_iteration_}")
    y_pred_lgbm = lgbm_model.predict(X_valid)
    r2_lgbm = r2_score(y_valid, y_pred_lgbm)
    mae_lgbm = mean_absolute_error(y_valid, y_pred_lgbm)
    rmse_lgbm = np.sqrt(mean_squared_error(y_valid, y_pred_lgbm))
    results.append({
        "model": "LightGBM",
        "R2": float(r2_lgbm),
        "MAE": float(mae_lgbm),
        "RMSE": float(rmse_lgbm),
    })
    print(f"  R²: {r2_lgbm:.4f}, MAE: {mae_lgbm:.6f}, RMSE: {rmse_lgbm:.6f}")

    # Create comparison DataFrame
    compare_df = pd.DataFrame(results)
    compare_df = compare_df.sort_values("R2", ascending=False)

    print("\n" + "=" * 60)
    print("Model Comparison Results:")
    print("=" * 60)
    print(compare_df.to_string(index=False))
    print("=" * 60)

    return compare_df


def _compute_diagnostics(
    model: LGBMRegressor,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    feature_columns: pd.Index,
    artifact_dir: Path,
) -> None:
    """Compute permutation importance and SHAP values for model diagnostics."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Computing model diagnostics...")
    print("=" * 60)
    
    # 1. Permutation Importance
    print("\n1. Computing permutation importance...")
    try:
        perm_result = permutation_importance(
            model,
            X_valid,
            y_valid,
            n_repeats=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        
        perm_df = pd.DataFrame({
            "feature": feature_columns,
            "importance_mean": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
        }).sort_values("importance_mean", ascending=False)
        
        perm_path = artifact_dir / "perm_importance.csv"
        perm_df.to_csv(perm_path, index=False)
        print(f"  ✓ Saved to {perm_path}")
        print(f"  Top 5 features by permutation importance:")
        for _, row in perm_df.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")
    except Exception as exc:
        print(f"  ✗ Permutation importance failed: {exc}")
    
    # 2. SHAP Values
    print("\n2. Computing SHAP values...")
    if not SHAP_AVAILABLE:
        print("  ✗ SHAP not available (install with: pip install shap)")
        return
    
    try:
        # Use a sample of validation data for SHAP (faster computation)
        n_shap_samples = min(100, len(X_valid))
        X_shap = X_valid.iloc[:n_shap_samples].copy()
        
        # Check if model is tree-based for appropriate explainer
        is_tree_based = isinstance(model, (LGBMRegressor, RandomForestRegressor))
        
        if is_tree_based:
            print(f"  Using TreeExplainer for tree-based model...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
        else:
            print(f"  Using KernelExplainer (slower, but works for any model)...")
            # Use a smaller background set for KernelExplainer
            n_background = min(50, len(X_train))
            background = X_train.iloc[:n_background]
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_shap)
        
        # Save SHAP values as parquet
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        shap_df = pd.DataFrame(
            shap_values,
            columns=feature_columns,
            index=X_shap.index,
        )
        
        shap_path = artifact_dir / "shap_values.parquet"
        shap_df.to_parquet(shap_path, index=True)
        print(f"  ✓ Saved SHAP values to {shap_path}")
        
        # Create summary plot
        print("  Creating SHAP summary plot...")
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap, feature_names=feature_columns, show=False)
            summary_path = artifact_dir / "shap_summary.png"
            plt.savefig(summary_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  ✓ Saved SHAP summary plot to {summary_path}")
        except Exception as plot_exc:
            print(f"  ✗ SHAP plot creation failed: {plot_exc}")
            
    except Exception as shap_exc:
        print(f"  ✗ SHAP computation failed: {shap_exc}")
        print(f"    (This is expected for non-tree-based models or if SHAP encounters issues)")


def _error_analysis_by_deciles(
    y_actual: pd.Series,
    y_pred: np.ndarray | pd.Series,
    artifact_dir: Path,
) -> None:
    """Analyze prediction errors by deciles of predicted return."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n3. Computing error analysis by deciles...")
    
    try:
        # Convert to numpy arrays for easier manipulation
        if isinstance(y_pred, pd.Series):
            y_pred_array = y_pred.values
        else:
            y_pred_array = y_pred
        
        if isinstance(y_actual, pd.Series):
            y_actual_array = y_actual.values
        else:
            y_actual_array = y_actual
        
        # Create decile buckets based on predicted values
        decile_edges = np.percentile(y_pred_array, np.linspace(0, 100, 11))
        
        # Assign each prediction to a decile (1-10)
        decile_labels = np.digitize(y_pred_array, decile_edges, right=False)
        # Handle edge case where value equals max (should be in D10)
        decile_labels = np.clip(decile_labels, 1, 10)
        
        # Compute metrics for each decile
        decile_results = []
        for decile in range(1, 11):
            mask = decile_labels == decile
            if mask.sum() == 0:
                continue
            
            actual_decile = y_actual_array[mask]
            pred_decile = y_pred_array[mask]
            
            decile_results.append({
                "decile": f"D{decile}",
                "count": int(mask.sum()),
                "mean_actual": float(np.mean(actual_decile)),
                "mean_predicted": float(np.mean(pred_decile)),
                "mae": float(mean_absolute_error(actual_decile, pred_decile)),
                "pred_min": float(np.min(pred_decile)),
                "pred_max": float(np.max(pred_decile)),
            })
        
        decile_df = pd.DataFrame(decile_results)
        
        # Save to CSV
        decile_path = artifact_dir / "decile_report.csv"
        decile_df.to_csv(decile_path, index=False)
        print(f"  ✓ Saved to {decile_path}")
        
        # Print summary
        print(f"  Decile analysis summary:")
        print(f"    Total predictions: {len(y_pred_array)}")
        print(f"    Deciles analyzed: {len(decile_df)}")
        print(f"    Mean MAE across deciles: {decile_df['mae'].mean():.6f}")
        print(f"    Decile with highest MAE: {decile_df.loc[decile_df['mae'].idxmax(), 'decile']} ({decile_df['mae'].max():.6f})")
        print(f"    Decile with lowest MAE: {decile_df.loc[decile_df['mae'].idxmin(), 'decile']} ({decile_df['mae'].min():.6f})")
        
    except Exception as exc:
        print(f"  ✗ Error analysis failed: {exc}")


def _update_latest_pointer(run_dir: Path) -> None:
    latest_path = ARTIFACT_DIR / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        try:
            if latest_path.is_symlink():
                latest_path.unlink()
            elif latest_path.is_dir():
                shutil.rmtree(latest_path)
            else:
                latest_path.unlink()
        except Exception:
            pass
    if os.name != "nt":
        try:
            os.symlink(run_dir.resolve(), latest_path, target_is_directory=True)
            print(f"Updated latest symlink -> {run_dir.name}")
            return
        except Exception as exc:
            print(f"Symlink failed ({exc}); falling back to copy.")
    shutil.copytree(run_dir, latest_path)
    print(f"Updated latest copy at {latest_path} from {run_dir.name}")


def save_artifacts(
    model: LGBMRegressor,
    feature_importance: pd.DataFrame,
    metadata: dict,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = _new_run_id()
    run_dir = ARTIFACT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Enrich metadata
    meta_ext = {
        **metadata,
        "run_id": run_id,
        "git_commit": _git_commit_hash(),
    }

    feature_importance_path = run_dir / "feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)

    model_path = run_dir / f"{MODEL_NAME}.pkl"
    joblib.dump(model, model_path)

    meta_path = run_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_ext, handle, indent=2)

    # Diagnostics
    _compute_diagnostics(
        model=model,
        X_train=X_train,
        X_valid=X_valid,
        y_valid=y_valid,
        feature_columns=X_train.columns,
        artifact_dir=run_dir,
    )
    # Error analysis
    y_pred = model.predict(X_valid)
    _error_analysis_by_deciles(y_actual=y_valid, y_pred=y_pred, artifact_dir=run_dir)

    _update_latest_pointer(run_dir)

    print(f"Artifacts saved to {run_dir}:")
    print(f"  - {model_path}")
    print(f"  - {feature_importance_path}")
    print(f"  - {meta_path}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM model or perform cross-validation")
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Perform 5-fold expanding window time series cross-validation",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare ElasticNet, RandomForest, and LightGBM models",
    )
    parser.add_argument(
        "--acceptance-test",
        action="store_true",
        help="Run guardrail acceptance test: expect pass on current data; expect abort when dates are shuffled",
    )
    args = parser.parse_args()

    if args.compare:
        compare_df = compare_models()
        
        # Save comparison results
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        compare_path = ARTIFACT_DIR / "model_compare.csv"
        compare_df.to_csv(compare_path, index=False)
        print(f"\nModel comparison saved to: {compare_path}")
    elif args.acceptance_test:
        # 1) Expect guardrails to pass on current dataset
        dataset_path = _latest_dataset()
        print(f"Acceptance test: loading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)

        print("\n[Step 1/2] Verifying guardrails pass on current dataset...")
        try:
            _ = _prepare_data(df)
            print("  ✓ Guardrails passed on current dataset.")
        except Exception as exc:  # pylint: disable=broad-except
            print("  ✗ Guardrails failed on current dataset (unexpected).")
            print(f"    Error: {exc}")
            raise SystemExit(1) from exc

        # 2) Intentionally shuffle 'asof' to force a failure
        print("\n[Step 2/2] Shuffling 'asof' to verify training aborts with a clear message...")
        if "asof" not in df.columns:
            print("  ✗ Dataset missing 'asof' column; cannot perform shuffle test.")
            raise SystemExit(1)
        df_shuffled = df.copy()
        rng = np.random.default_rng(seed=RANDOM_SEED)
        shuffled_asof = pd.Series(df_shuffled["asof"]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        df_shuffled["asof"] = shuffled_asof
        try:
            _ = _prepare_data(df_shuffled)
            print("  ✗ Guardrails did not abort after shuffling dates (unexpected).")
            raise SystemExit(1)
        except ValueError as exc:
            msg = str(exc)
            expected = ("DATA LEAKAGE DETECTED (Guardrail 1)" in msg) or ("DATA LEAKAGE DETECTED (Guardrail 2)" in msg)
            if expected:
                print("  ✓ Training aborted as expected with a clear message:")
                print(f"    {exc}")
                print("\nAcceptance test PASSED.")
                raise SystemExit(0)
            print("  ✗ Training aborted, but message did not reference guardrails clearly.")
            print(f"    Error: {exc}")
            raise SystemExit(1) from exc
    elif args.cv:
        cv_metrics = cross_validate(n_folds=5)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Cross-Validation Results (5-fold expanding window):")
        print("=" * 60)
        print(f"R²: {cv_metrics['mean_R2']:.4f} ± {cv_metrics['std_R2']:.4f}")
        print(f"MAE: {cv_metrics['mean_MAE']:.6f} ± {cv_metrics['std_MAE']:.6f}")
        print("=" * 60)

        # Save CV metrics
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        cv_metrics_path = ARTIFACT_DIR / "cv_metrics.json"
        with cv_metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(cv_metrics, handle, indent=2)
        print(f"\nCV metrics saved to: {cv_metrics_path}")
    else:
        model, feature_importance, metadata, X_train, X_valid, y_valid = train_model()
        run_dir = save_artifacts(model, feature_importance, metadata, X_train, X_valid, y_valid)
        print(f"Validation R2: {metadata['val_R2']:.4f}")
        print(f"Validation MAE: {metadata['val_MAE']:.6f}")
        print(f"Run artifacts directory: {run_dir}")
        # Generate/update model card from latest artifacts
        try:
            from .model_card import generate_model_card
            generate_model_card()
        except Exception as exc:
            print(f"MODEL_CARD: generation failed: {exc}")


if __name__ == "__main__":
    main()

