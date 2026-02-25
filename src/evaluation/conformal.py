"""
Conformalised quantile regression — making the interval mean what it says.

The backtest reported a WAPE about 27% better than seasonal-naive and an 80%
prediction interval that covered roughly 72% of held-out actuals. The first
number is the headline; the second is the one that would cause damage. A
planner reading "80% interval" and sizing safety stock against it is being
told the tail risk is smaller than it is. Quantile regression minimises
pinball loss; nothing in that objective constrains the *frequency* with which
the realised value lands inside the band.

The fix is split-conformal prediction applied to quantile regression — CQR,
Romano, Patterson & Candès (2019). Fit the quantile models on one slice of
data, then on a disjoint calibration slice score each row by how far outside
the band it fell:

    E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))

E is negative when the actual was comfortably inside the interval and positive
by the size of the miss when it was outside. Take the empirical
(1-alpha)(1 + 1/n) quantile of those scores and widen both ends by it. Under
exchangeability that gives finite-sample coverage of at least 1-alpha,
whatever the underlying model does — the guarantee is a property of the
procedure, not an assumption about the data.

**The honest caveat.** Exchangeability does not hold exactly for a time
series: the calibration slice is the recent past and the rows being predicted
are the future, and a regime change between them breaks the guarantee. So this
is not a proof of coverage, it is a well-founded correction whose effect has
to be *measured out of sample* — which is what `src.evaluation.backtest`
does, refitting the offset inside each fold's training window and scoring the
calibrated interval on the weeks after the cutoff. The measured numbers are in
the README.

The split is by time, never at random. A random split would put a week's
neighbours on both sides of the boundary and quietly restore the leak the rest
of this repository is built to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import QUANTILES

#: Fraction of the training window held back to calibrate on. A fifth is the
#: usual compromise: enough rows for the empirical quantile to be stable,
#: little enough that the model is not meaningfully weaker for losing them.
CALIBRATION_FRACTION = 0.2


@dataclass(frozen=True, slots=True)
class ConformalResult:
    """What calibration learned, and what it did on the calibration slice."""

    offset: float
    nominal: float
    raw_coverage: float
    calibrated_coverage: float
    n_calibration_rows: int

    @property
    def improved(self) -> bool:
        return abs(self.calibrated_coverage - self.nominal) <= abs(self.raw_coverage - self.nominal)


def nominal_coverage(quantiles: tuple[float, ...] = QUANTILES) -> float:
    """The coverage the outer quantiles claim, e.g. 0.9 - 0.1 = 0.8."""
    return float(quantiles[-1] - quantiles[0])


def conformity_scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """CQR scores: how far outside the interval each actual fell (negative if inside)."""
    return np.maximum(lo - y, y - hi)


def conformal_offset(scores: np.ndarray, nominal: float) -> float:
    """
    The finite-sample-corrected empirical quantile of the conformity scores.

    The `(1 + 1/n)` inflation is what makes the guarantee hold at finite n
    rather than only asymptotically; without it the interval is very slightly
    too narrow, in exactly the direction the whole exercise is correcting.
    Clipped at zero: conformal calibration here only ever widens an interval,
    never narrows one, because a narrower-than-nominal band is the failure
    mode being fixed.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, nominal * (1.0 + 1.0 / n))
    return float(max(0.0, np.quantile(scores, level, method="higher")))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of actuals inside [lo, hi]."""
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((y >= lo) & (y <= hi)))


def time_split(
    panel: pd.DataFrame, fraction: float = CALIBRATION_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a supervised panel into (fit, calibrate) at a week boundary.

    Disjoint on both ends: the fit half keeps rows whose *target* falls at or
    before the boundary, the calibration half keeps rows whose *origin* falls
    after it. A row straddling the boundary belongs to neither, which costs a
    few rows and removes any possibility of the calibration slice having been
    trained on.
    """
    weeks = np.sort(pd.to_datetime(panel["week"]).unique())
    if len(weeks) < 4:
        return panel.iloc[0:0], panel.iloc[0:0]

    boundary = weeks[max(1, int(len(weeks) * (1.0 - fraction))) - 1]
    fit = panel[pd.to_datetime(panel["target_week"]) <= boundary]
    calibrate = panel[pd.to_datetime(panel["week"]) > boundary]
    return fit, calibrate


def offset_from_predictions(
    actual: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    nominal: float = nominal_coverage(),
) -> ConformalResult:
    """Compute the offset and both coverages from an already-predicted slice."""
    actual, lo, hi = (np.asarray(a, dtype=float) for a in (actual, lo, hi))
    offset = conformal_offset(conformity_scores(actual, lo, hi), nominal)
    return ConformalResult(
        offset=offset,
        nominal=nominal,
        raw_coverage=coverage(actual, lo, hi),
        calibrated_coverage=coverage(actual, np.clip(lo - offset, 0.0, None), hi + offset),
        n_calibration_rows=len(actual),
    )


def fit_conformal_offset(
    panel: pd.DataFrame,
    quantiles: tuple[float, ...] = QUANTILES,
    num_boost_round: int = 300,
    seed: int = 7,
) -> ConformalResult:
    """
    Learn the interval offset from a held-out slice of `panel`.

    Fits a throwaway model on the earlier part so the calibration slice is
    genuinely unseen. The offset it returns is then applied to the model
    trained on the *whole* panel, which is the standard split-conformal
    arrangement: a marginally stronger model wearing a marginally conservative
    correction.
    """
    # Imported here: conformal is a property of the protocol, and importing the
    # trainer at module scope would make the backtest's import graph circular.
    from src.models.train_lgb_quantile import (
        TARGET,
        predict_quantiles,
        train_quantile_models,
    )

    fit, calibrate = time_split(panel)
    nominal = nominal_coverage(quantiles)
    if fit.empty or calibrate.empty:
        return ConformalResult(0.0, nominal, float("nan"), float("nan"), 0)

    models, _ = train_quantile_models(
        fit, quantiles=quantiles, num_boost_round=num_boost_round, seed=seed
    )
    preds = predict_quantiles(models, calibrate)

    return offset_from_predictions(
        calibrate[TARGET].to_numpy(dtype=float),
        preds[f"q{quantiles[0]}_lgb"].to_numpy(dtype=float),
        preds[f"q{quantiles[-1]}_lgb"].to_numpy(dtype=float),
        nominal=nominal,
    )
