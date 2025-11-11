import pandas as pd
import numpy as np

from finguard.models.train_lightgbm import _prepare_data, CONFIG, TEST_SIZE_BY_DATE


def _make_synth_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=24, freq="M")
    tickers = ["AAA", "BBB", "CCC"]
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        for t in tickers:
            rows.append(
                {
                    "ticker": t,
                    "asof": d.date().isoformat(),
                    "feat1": rng.normal(),
                    "feat2": rng.normal(),
                    # Make a deterministic forward return to avoid randomness in tests
                    "forward_90d_return": float(((hash((t, d)) % 1000) - 500) / 10000.0),
                }
            )
    df = pd.DataFrame(rows)
    return df


def test_time_split_has_no_date_leakage():
    df = _make_synth_df()
    # Should not raise due to guardrails inside _prepare_data
    X_train, X_valid, y_train, y_valid = _prepare_data(df.copy())

    # Recompute masks the same way as in _prepare_data to validate properties
    df2 = df.copy()
    df2["asof"] = pd.to_datetime(df2["asof"])
    df2 = df2.sort_values("asof").reset_index(drop=True)

    unique_asofs = sorted(df2["asof"].unique())
    n_asofs = len(unique_asofs)
    split_idx = int(n_asofs * (1 - TEST_SIZE_BY_DATE))
    if split_idx == 0:
        split_idx = 1
    if split_idx >= n_asofs:
        split_idx = n_asofs - 1
    train_asof_cutoff = unique_asofs[split_idx]
    train_mask = df2["asof"] < train_asof_cutoff
    valid_mask = df2["asof"] >= train_asof_cutoff

    train_max_asof = df2.loc[train_mask, "asof"].max()
    valid_min_asof = df2.loc[valid_mask, "asof"].min()

    assert valid_min_asof > train_max_asof, "Validation min asof must be strictly after train max asof"

    forward_horizon_days = CONFIG.get("forward_horizon_days", 90)
    train_forward_end_dates = df2.loc[train_mask, "asof"] + pd.Timedelta(days=forward_horizon_days)
    train_max_forward_end = train_forward_end_dates.max()

    assert train_max_forward_end < valid_min_asof, "Train forward label window must not overlap validation period"


