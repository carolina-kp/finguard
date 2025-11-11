import pytest

from finguard.features.core import build_features


@pytest.mark.slow
def test_build_features_expected_keys_and_non_nulls():
    # Known, liquid ticker and a recent month-end date string format
    features = build_features("AAPL", asof=None)
    required_keys = [
        "ticker",
        "asof",
        "px_cagr_1y",
        "px_cagr_2y",
        "px_momentum_3m",
        "px_momentum_6m",
        "px_vol_60d",
        "px_vol_250d",
        "px_maxdd_2y",
        "px_beta_spy_1y",
        "size_log_mcap",
        "valuation_pe",
        "valuation_ps",
        "valuation_pb",
        "quality_roe",
        "quality_profit_margin",
        "quality_op_margin",
        "leverage_dte",
    ]
    for key in required_keys:
        assert key in features, f"Missing feature key: {key}"
        assert features[key] is not None, f"Feature {key} is None"


