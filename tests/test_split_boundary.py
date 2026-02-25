"""
Leakage regression tests for the train/test split itself.

`tests/test_no_leakage.py` checks that a *feature* never reaches forward in
time. That is only half the property. The other half lives in the backtest: a
fold may train on nothing whose target falls after its cutoff, and must score
nothing that falls on or before it. Both halves fail silently — the score just
gets better — so both are pinned here by construction rather than by reading
the code.

Everything runs on a tiny synthetic panel: no real dataset, a handful of boost
rounds, sub-second fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import backtest as bt
from src.features.build_features import add_history_features
from src.models.train_lgb_quantile import KEYS, TARGET, make_supervised


def synthetic_features(n_weeks: int = 90, start: str = "2023-01-02") -> pd.DataFrame:
    """Two deterministic seasonal series, long enough to clear MIN_TRAIN_WEEKS."""
    rng = np.random.default_rng(11)
    weeks = pd.date_range(start, periods=n_weeks, freq="W-MON")
    rows = []
    for sku, base in (("SKU-A", 80.0), ("SKU-B", 30.0)):
        for i, week in enumerate(weeks):
            seasonal = 1 + 0.25 * np.cos(2 * np.pi * (i % 52) / 52)
            units = max(0.0, base * seasonal + rng.normal(0, base * 0.05))
            rows.append(
                {
                    "sku": sku,
                    "region": "NE",
                    "week": week,
                    "units_sold": float(round(units)),
                    "sales_dollars": units * 2.5,
                    "n_transactions": units / 3,
                    "avg_price": 2.5,
                }
            )
    return add_history_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return synthetic_features()


@pytest.fixture(scope="module")
def cutoff(features) -> pd.Timestamp:
    # Far enough in that MIN_TRAIN_WEEKS is satisfied, early enough to leave
    # real weeks on the far side to score.
    return pd.to_datetime(features["week"]).sort_values().unique()[75]


class TestFoldNeverTrainsOnWhatItScores:
    """The regression that would make every reported WAPE meaningless."""

    @pytest.fixture
    def captured(self, monkeypatch, features, cutoff):
        """Run one fold, intercepting the panels handed to fit and to predict."""
        seen: dict[str, pd.DataFrame] = {}

        real_train = bt.train_quantile_models
        real_predict = bt.predict_quantiles

        def spy_train(panel, *args, **kwargs):
            seen["train"] = panel.copy()
            kwargs.setdefault("num_boost_round", 20)
            return real_train(panel, *args, **kwargs)

        def spy_predict(models, panel, *args, **kwargs):
            seen["test"] = panel.copy()
            return real_predict(models, panel, *args, **kwargs)

        monkeypatch.setattr(bt, "train_quantile_models", spy_train)
        monkeypatch.setattr(bt, "predict_quantiles", spy_predict)

        fold = bt.evaluate_fold(features, pd.Timestamp(cutoff), horizon=4, rounds=20, seed=1)
        assert fold is not None, "fixture must produce a scored fold"
        seen["fold"] = fold
        return seen

    def test_no_training_target_falls_after_the_cutoff(self, captured, cutoff):
        targets = pd.to_datetime(captured["train"]["target_week"])
        assert (targets <= cutoff).all(), (
            "the fold trained on weeks it goes on to score — every WAPE it reports is contaminated"
        )

    def test_no_training_origin_falls_after_the_cutoff(self, captured, cutoff):
        origins = pd.to_datetime(captured["train"]["week"])
        assert (origins <= cutoff).all()

    def test_every_scored_target_falls_strictly_after_the_cutoff(self, captured, cutoff):
        targets = pd.to_datetime(captured["test"]["target_week"])
        assert len(targets) > 0
        assert (targets > cutoff).all()

    def test_train_and_test_rows_are_disjoint(self, captured):
        def keyset(df):
            return {
                (r.sku, r.region, pd.Timestamp(r.target_week))
                for r in df[KEYS + ["target_week"]].itertuples(index=False)
            }

        assert not keyset(captured["train"]) & keyset(captured["test"])

    def test_the_fold_is_scored_on_the_rows_it_predicted(self, captured):
        assert captured["fold"].n_rows == len(captured["test"])


class TestFeaturesAtTheBoundaryIgnoreTheFuture:
    """
    The subtler half: the fold reads a feature table built over the *whole*
    series, including weeks after the cutoff. That is only safe because every
    feature is strictly backward-looking. Rebuild the table from a series
    truncated at the cutoff and the rows up to it must be byte-identical.
    """

    def test_truncating_the_series_changes_no_feature_before_the_cutoff(self, features, cutoff):
        raw = features[
            ["sku", "region", "week", TARGET, "sales_dollars", "n_transactions", "avg_price"]
        ]
        truncated = add_history_features(
            raw[pd.to_datetime(raw["week"]) <= cutoff].reset_index(drop=True)
        )
        full = features[pd.to_datetime(features["week"]) <= cutoff]

        cols = [c for c in full.columns if c.startswith(("lag_", "roll_", "trend_", "price_"))]
        assert cols
        pd.testing.assert_frame_equal(
            full[cols].reset_index(drop=True),
            truncated[cols].reset_index(drop=True),
            check_dtype=False,
        )

    def test_a_supervised_row_never_uses_a_feature_from_after_its_origin(self, features):
        # make_supervised must carry the origin week's features forward, not
        # the target week's. If it ever joined on target_week instead, the
        # model would be reading the answer.
        panel = make_supervised(features, range(1, 4))
        row = panel.iloc[len(panel) // 2]
        origin = features[
            (features["sku"] == row["sku"])
            & (features["region"] == row["region"])
            & (features["week"] == row["week"])
        ].iloc[0]
        for col in ("lag_1_units", "roll_4w_mean", "lag_1_price"):
            assert row[col] == pytest.approx(origin[col], nan_ok=True)


class TestFoldsThatCannotBeScoredAreSkipped:
    def test_a_cutoff_before_the_data_returns_none(self, features):
        # An empty training frame used to slip past the history check, because
        # groupby().size().min() is NaN and `NaN < MIN_TRAIN_WEEKS` is False.
        assert (
            bt.evaluate_fold(features, pd.Timestamp("2010-01-04"), horizon=4, rounds=10, seed=1)
            is None
        )

    def test_a_cutoff_with_less_than_min_train_weeks_returns_none(self, features):
        weeks = pd.to_datetime(features["week"]).sort_values().unique()
        too_early = pd.Timestamp(weeks[bt.MIN_TRAIN_WEEKS - 2])
        assert bt.evaluate_fold(features, too_early, horizon=4, rounds=10, seed=1) is None

    def test_a_cutoff_at_the_very_last_week_has_nothing_to_score(self, features):
        last = pd.Timestamp(pd.to_datetime(features["week"]).max())
        assert bt.evaluate_fold(features, last, horizon=4, rounds=10, seed=1) is None


class TestRunBacktestProtocol:
    def test_folds_come_back_in_cutoff_order_and_leave_room_to_score(self, features):
        results = bt.run_backtest(features, n_folds=2, horizon=4, step_weeks=4, rounds=15, seed=1)
        assert results, "expected at least one scorable fold"
        assert [r.cutoff for r in results] == sorted(r.cutoff for r in results)
        last_week = pd.to_datetime(features["week"]).max()
        for r in results:
            # Every cutoff must leave `horizon` weeks of actuals behind it.
            assert r.cutoff <= last_week - pd.Timedelta(weeks=4)
            assert r.n_rows > 0
