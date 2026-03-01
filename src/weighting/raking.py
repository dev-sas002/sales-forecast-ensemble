"""
Simple raking / iterative proportional fitting to compute weights.
Inputs:
 - panel_parquet with demographic columns (age_group, region, income_bin)
 - population_marginals: a dict of {col: {value: total_count}}
Outputs:
 - panel_with_weights.parquet
"""

from __future__ import annotations

import pandas as pd

from src.config import LANDING_DIR


def raking(
    df: pd.DataFrame, marginals: dict, max_iter: int = 50, tol: float = 1e-6
) -> pd.DataFrame:
    """Iterative proportional fitting: scale weights so group sums match marginals. Returns df with 'weight' column."""
    if df.empty:
        raise ValueError("raking requires a non-empty DataFrame")
    for col in marginals:
        if col not in df.columns:
            raise ValueError(f"marginals key '{col}' is not a column in df")
    # Start with unit weights
    df = df.copy()
    df["weight"] = 1.0
    for it in range(max_iter):
        max_change = 0.0
        for col, target in marginals.items():
            current = df.groupby(col)["weight"].sum()
            # Compute scaling per observed category.
            #
            # A category present in df but absent from the marginals used to
            # take target.get(k, 0.0) == 0.0, which scaled its weights to zero
            # and — since zero times anything stays zero — kept them there for
            # every later iteration. Those rows then contributed nothing to
            # any weighted total, silently. A category with no target is left
            # alone (scale 1.0) instead. The denominator is guarded too: once
            # a group summed to zero the next pass computed 0.0/0.0 and raised
            # a RuntimeWarning on every iteration.
            scale = {}
            for k in current.index:
                if k not in target:
                    scale[k] = 1.0
                    continue
                denom = float(current.get(k, 0.0))
                scale[k] = float(target[k]) / (denom if abs(denom) > 1e-9 else 1e-9)
            m = df[col].map(scale).fillna(1.0)
            df["weight"] *= m
            max_change = max(max_change, float((abs(1 - m)).max()))
        if max_change < tol:
            print("Converged at iteration:", it)
            break
    return df


def main():
    panel = pd.read_parquet(LANDING_DIR / "panel.parquet")
    # Example marginals (toy). Replace with census/retailer derived targets.
    marginals = {
        "region": {"NE": 1000000, "SE": 800000, "MW": 1200000, "W": 900000},
        "age_group": {"18-34": 1000000, "35-54": 1500000, "55+": 800000},
    }
    panel_w = raking(panel, marginals)
    panel_w.to_parquet(LANDING_DIR / "panel_weighted.parquet", index=False)
    print("Saved weighted panel.")


if __name__ == "__main__":
    main()
