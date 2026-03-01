"""Forecast evaluation metrics (WAPE, quantile loss)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Weighted absolute percentage error: sum|y_true - y_pred| / sum|y_true|."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.sum(np.abs(y_true)), 1e-9)
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def quantile_loss(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, q: float
) -> float:
    """Pinball loss for quantile q: mean of max(q*e, (q-1)*e) where e = y_true - y_pred."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))
