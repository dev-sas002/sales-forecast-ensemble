"""Tests for src.utils.validation: check_required_columns, check_totals_close."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.validation import check_required_columns, check_totals_close


def test_check_required_columns_pass() -> None:
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    check_required_columns(df, ["a", "b"])
    check_required_columns(df, ["a", "b", "c"])


def test_check_required_columns_missing() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="missing required columns: \\['c'\\]"):
        check_required_columns(df, ["a", "b", "c"])
    with pytest.raises(ValueError, match="missing required columns: \\['x', 'y'\\]"):
        check_required_columns(df, ["x", "y"], name="mydf")


def test_check_totals_close_within_tol() -> None:
    panel = pd.DataFrame({"quantity": [10, 20]})
    credit = pd.DataFrame({"quantity": [28, 2]})  # 30 vs 30
    check_totals_close(panel, credit, tol=0.25)
    check_totals_close(panel, credit, tol=0.0)


def test_check_totals_close_credit_zero() -> None:
    panel = pd.DataFrame({"quantity": [0]})
    credit = pd.DataFrame({"quantity": [0]})
    check_totals_close(panel, credit)


def test_check_totals_close_exceeds_tol() -> None:
    panel = pd.DataFrame({"quantity": [10, 20]})  # 30
    credit = pd.DataFrame({"quantity": [100]})  # 100 -> ratio 0.7
    with pytest.raises(ValueError, match="totals deviate"):
        check_totals_close(panel, credit, tol=0.25)
