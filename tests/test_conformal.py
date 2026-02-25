"""
Tests for conformal interval calibration.

The property that matters is not "the code runs" but "the interval ends up
covering what it claims". These check the arithmetic of the procedure on
constructed data where the right answer is known by hand, and check the
time-split that keeps the calibration set genuinely unseen — a random split
would restore exactly the leak this repository is built to avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.conformal import (
    ConformalResult,
    conformal_offset,
    conformity_scores,
    coverage,
    nominal_coverage,
    offset_from_predictions,
    time_split,
)


class TestConformityScores:
    def test_a_value_inside_the_interval_scores_negative(self):
        score = conformity_scores(np.array([50.0]), np.array([10.0]), np.array([90.0]))
        # 40 below the top and 40 above the bottom: the score is the smaller
        # slack, negated.
        assert score[0] == pytest.approx(-40.0)

    def test_a_value_above_the_interval_scores_the_overshoot(self):
        score = conformity_scores(np.array([100.0]), np.array([10.0]), np.array([90.0]))
        assert score[0] == pytest.approx(10.0)

    def test_a_value_below_the_interval_scores_the_undershoot(self):
        score = conformity_scores(np.array([4.0]), np.array([10.0]), np.array([90.0]))
        assert score[0] == pytest.approx(6.0)


class TestTheOffset:
    def test_a_perfectly_calibrated_interval_needs_no_widening(self):
        # Every actual comfortably inside: all scores negative, offset clipped
        # to zero. Conformal calibration here only ever widens.
        scores = np.linspace(-50, -5, 100)
        assert conformal_offset(scores, 0.8) == 0.0

    def test_a_too_narrow_interval_is_widened(self):
        # 40 of 100 actuals fall outside by up to 20 units. An 80% interval
        # needs the 80th percentile of the scores, which is positive here.
        scores = np.concatenate([np.full(60, -5.0), np.linspace(0.5, 20.0, 40)])
        offset = conformal_offset(scores, 0.8)
        assert offset > 0

    def test_the_offset_achieves_nominal_coverage_on_its_own_scores(self):
        # The defining property: widen by the offset and at least `nominal` of
        # the calibration rows are covered.
        rng = np.random.default_rng(0)
        y = rng.normal(100, 30, 500)
        lo, hi = np.full(500, 90.0), np.full(500, 110.0)

        result = offset_from_predictions(y, lo, hi, nominal=0.8)
        assert result.raw_coverage < 0.8
        assert result.calibrated_coverage >= 0.8
        assert result.improved

    def test_no_scores_is_a_zero_offset_not_a_crash(self):
        assert conformal_offset(np.array([]), 0.8) == 0.0

    def test_non_finite_scores_are_ignored(self):
        scores = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
        assert np.isfinite(conformal_offset(scores, 0.8))

    def test_the_finite_sample_correction_never_exceeds_the_maximum_score(self):
        # (1 + 1/n) can push the requested level above 1; it must clamp.
        scores = np.array([1.0, 2.0, 3.0])
        assert conformal_offset(scores, 0.99) <= scores.max()


class TestCoverage:
    def test_coverage_counts_the_boundaries_as_covered(self):
        y = np.array([10.0, 20.0])
        assert coverage(y, np.array([10.0, 5.0]), np.array([15.0, 20.0])) == 1.0

    def test_coverage_of_nothing_is_nan_not_zero(self):
        # Zero rows means "not measured", not "covered nothing".
        assert np.isnan(coverage(np.array([]), np.array([]), np.array([])))


class TestTheTimeSplit:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        weeks = pd.date_range("2024-01-01", periods=40, freq="W-MON")
        rows = []
        for i, week in enumerate(weeks):
            for h in (1, 2, 3):
                if i + h < len(weeks):
                    rows.append(
                        {
                            "week": week,
                            "target_week": weeks[i + h],
                            "horizon": h,
                            "units_sold": float(i + h),
                        }
                    )
        return pd.DataFrame(rows)

    def test_the_split_is_chronological_and_disjoint(self, panel):
        fit, calibrate = time_split(panel, fraction=0.25)
        assert not fit.empty and not calibrate.empty
        # Nothing the model fitted on may reappear in calibration.
        assert pd.to_datetime(fit["target_week"]).max() < pd.to_datetime(
            calibrate["week"]
        ).min() + pd.Timedelta(weeks=1)

    def test_no_calibration_origin_precedes_the_boundary(self, panel):
        fit, calibrate = time_split(panel, fraction=0.25)
        boundary = pd.to_datetime(fit["target_week"]).max()
        assert (pd.to_datetime(calibrate["week"]) > boundary).all()

    def test_a_panel_too_short_to_split_yields_nothing_rather_than_leaking(self):
        tiny = pd.DataFrame(
            {
                "week": pd.to_datetime(["2024-01-01", "2024-01-08"]),
                "target_week": pd.to_datetime(["2024-01-08", "2024-01-15"]),
                "units_sold": [1.0, 2.0],
            }
        )
        fit, calibrate = time_split(tiny)
        assert fit.empty and calibrate.empty


class TestTheResultObject:
    def test_nominal_coverage_comes_from_the_configured_quantiles(self):
        assert nominal_coverage((0.1, 0.5, 0.9)) == pytest.approx(0.8)
        assert nominal_coverage((0.05, 0.5, 0.95)) == pytest.approx(0.9)

    def test_improved_is_false_when_calibration_overshoots_badly(self):
        result = ConformalResult(
            offset=100.0,
            nominal=0.8,
            raw_coverage=0.78,
            calibrated_coverage=1.0,
            n_calibration_rows=10,
        )
        # 0.78 is closer to nominal than 1.00 — over-covering is also wrong.
        assert not result.improved
