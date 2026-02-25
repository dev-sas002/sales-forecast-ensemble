"""Tests for raking(): convergence with simple marginals, validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.weighting.raking import raking


def test_raking_converges_simple() -> None:
    df = pd.DataFrame({"region": ["NE", "NE", "SE"], "age_group": ["18-34", "35-54", "35-54"]})
    marginals = {
        "region": {"NE": 2.0, "SE": 1.0},
        "age_group": {"18-34": 1.0, "35-54": 2.0},
    }
    out = raking(df, marginals, max_iter=50, tol=1e-6)
    assert "weight" in out.columns
    assert len(out) == 3
    # Weights should sum by region to match marginals (approximately)
    by_region = out.groupby("region")["weight"].sum()
    assert abs(by_region["NE"] - 2.0) < 0.01
    assert abs(by_region["SE"] - 1.0) < 0.01


def test_raking_empty_df_raises() -> None:
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="non-empty"):
        raking(df, {"region": {"NE": 1}})


def test_raking_marginal_key_not_column_raises() -> None:
    df = pd.DataFrame({"region": ["NE"]})
    with pytest.raises(ValueError, match="not a column"):
        raking(df, {"region": {"NE": 1}, "nonexistent": {"x": 1}})


def test_a_category_absent_from_the_marginals_keeps_a_usable_weight() -> None:
    """
    A category present in the panel but not in the marginals used to get a
    target of 0.0, which scaled its weights to zero and left them there for
    good — so those panel members contributed nothing to any weighted total,
    silently, while the loop emitted a 0.0/0.0 RuntimeWarning every pass.
    """
    df = pd.DataFrame({"region": ["NE", "NE", "XX"]})
    out = raking(df, {"region": {"NE": 2.0}}, max_iter=10)

    assert out["weight"].notna().all(), "raking produced NaN weights"
    assert (out["weight"] > 0).all()
    assert out[out["region"] == "NE"]["weight"].sum() == pytest.approx(2.0)


def test_raking_does_not_mutate_the_input_frame() -> None:
    df = pd.DataFrame({"region": ["NE", "SE"]})
    raking(df, {"region": {"NE": 1.0, "SE": 1.0}})
    assert "weight" not in df.columns


def test_weights_are_all_one_when_the_marginals_already_match() -> None:
    df = pd.DataFrame({"region": ["NE", "NE", "SE"]})
    out = raking(df, {"region": {"NE": 2.0, "SE": 1.0}}, max_iter=10)
    assert out["weight"].tolist() == pytest.approx([1.0, 1.0, 1.0])
