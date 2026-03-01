"""Validation helpers for DataFrames and cross-source checks."""

from __future__ import annotations

import pandas as pd


def check_required_columns(df: pd.DataFrame, cols: list[str], name: str = "df") -> None:
    """Raise ValueError if df is missing any of the required columns."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def check_totals_close(panel_df: pd.DataFrame, credit_df: pd.DataFrame, tol: float = 0.25) -> None:
    """Check that panel and credit quantity totals are within tolerance (rough cross-source check)."""
    p = float(panel_df.get("quantity", pd.Series([0])).sum())
    c = float(credit_df.get("quantity", pd.Series([0])).sum())
    if c == 0:
        return
    ratio = abs(p - c) / c
    if ratio > tol:
        raise ValueError(f"Panel vs credit totals deviate {ratio:.2%} > tol {tol:.2%}")
