"""Tests for src.utils.metrics: wape and quantile_loss with known values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import quantile_loss, wape


def test_wape_known() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    # sum|y_true - y_pred| = 2+2+0 = 4, sum|y_true| = 60
    assert abs(wape(y_true, y_pred) - 4 / 60) < 1e-9


def test_wape_accepts_series() -> None:
    y_true = pd.Series([10.0, 20.0])
    y_pred = pd.Series([10.0, 20.0])
    assert wape(y_true, y_pred) == 0.0


def test_wape_zero_actual_guarded() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    # denom uses max(sum|y_true|, 1e-9) so we get a finite value
    w = wape(y_true, y_pred)
    assert np.isfinite(w)


def test_quantile_loss_known() -> None:
    # e = y_true - y_pred. For q=0.5, loss = mean(|e|)/2 * something; pinball: max(0.5*e, -0.5*e) = 0.5|e|
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert quantile_loss(y_true, y_pred, 0.5) == 0.0
    e = np.array([1.0, -1.0, 0.0])
    q = 0.5
    expected = np.mean(np.maximum(q * e, (q - 1) * e))
    assert abs(quantile_loss(y_true, y_pred + (-e), 0.5) - expected) < 1e-9


def test_quantile_loss_accepts_series() -> None:
    y_true = pd.Series([1.0, 2.0])
    y_pred = pd.Series([1.0, 2.0])
    assert quantile_loss(y_true, y_pred, 0.1) == 0.0


def test_quantile_loss_is_asymmetric_for_a_low_quantile() -> None:
    # The whole point of pinball loss: at q=0.1 an over-prediction must cost
    # more than an under-prediction of the same size, otherwise the fitted
    # "10th percentile" is just a mean. A symmetric implementation (a stray
    # abs(), or q and 1-q swapped) passes every equality test and fails this.
    y_true = np.array([100.0])
    over = quantile_loss(y_true, np.array([110.0]), 0.1)
    under = quantile_loss(y_true, np.array([90.0]), 0.1)
    assert over > under
    assert over == pytest.approx(0.9 * 10)
    assert under == pytest.approx(0.1 * 10)


def test_quantile_loss_is_asymmetric_the_other_way_for_a_high_quantile() -> None:
    y_true = np.array([100.0])
    over = quantile_loss(y_true, np.array([110.0]), 0.9)
    under = quantile_loss(y_true, np.array([90.0]), 0.9)
    assert under > over
    assert under == pytest.approx(0.9 * 10)
    assert over == pytest.approx(0.1 * 10)


def test_quantile_loss_at_the_median_is_half_the_absolute_error() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([14.0, 18.0, 30.0])
    assert quantile_loss(y_true, y_pred, 0.5) == pytest.approx(np.mean(np.abs(y_true - y_pred)) / 2)


def test_wape_is_weighted_not_averaged_per_row() -> None:
    # WAPE exists because MAPE explodes on small actuals. A one-unit miss on a
    # week that sold 1 unit must not dominate a one-unit miss on a week that
    # sold 1000; per-row MAPE would average 100% and 0.1%.
    y_true = np.array([1.0, 1000.0])
    y_pred = np.array([2.0, 1001.0])
    assert wape(y_true, y_pred) == pytest.approx(2 / 1001)


def test_wape_handles_a_zero_week_without_dividing_by_it() -> None:
    # A single zero-demand week is normal in retail; MAPE would be inf here.
    y_true = np.array([0.0, 50.0])
    y_pred = np.array([3.0, 45.0])
    assert wape(y_true, y_pred) == pytest.approx(8 / 50)


def test_wape_of_an_all_zero_actual_is_finite_not_nan() -> None:
    assert np.isfinite(wape(np.zeros(3), np.array([1.0, 2.0, 3.0])))
    assert wape(np.zeros(3), np.zeros(3)) == 0.0
