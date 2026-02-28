"""
The feature registry: the seam for adding a history feature, and the mechanism
that makes "no look-ahead" structural instead of a promise.

The 2026-09-22 audit established that no feature reaches forward in time and
pinned it with tests. Those tests check the *current* set of columns. The
weakness in that arrangement is obvious once you name it: the next person to
add a feature writes it by hand against the raw frame, and nothing stops them
reaching for `df["units_sold"].rolling(4)` — the exact line that caused the
original leak. The test suite would still pass, because it enumerates the
columns it already knows about.

So the registry does two jobs at once.

**It is the extension point.** Adding a trailing statistic is one decorated
function; nothing else in the codebase changes. `build_features` iterates the
registry, the trainer selects columns by asking the registry what it produced,
and the leakage tests enumerate it too — so a feature added tomorrow is
covered by the invariance check the moment it is registered.

**It makes the leak unreachable.** A registered feature is not handed the
DataFrame. It is handed a :class:`History`, whose only data are the target and
price series *already shifted one period* and grouped by series. There is no
accessor on it that returns the current week. `History.lag(1)` is `prior`;
`History.lag(n)` is `prior.shift(n - 1)`, which is the raw series shifted `n`.
Every trailing window is a window over `prior`. You cannot write a leaking
feature through this interface without going around it, and going around it is
visible in review in a way that `.rolling(4)` on the wrong column is not.

Calendar features are registered separately and deliberately: the week of the
year is known before the week starts, so describing the row's own week is
legitimate — and confining that legitimacy to one short, named list is how it
stays legitimate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Prefixes that mark a column as a model input. The trainer selects on these,
#: so a registered feature must use one — `validate_registry` enforces it.
HISTORY_PREFIXES = ("lag_", "roll_", "trend_", "price_")


class History:
    """
    Everything a history feature is allowed to see.

    Constructed from series that have *already* been shifted one period and
    grouped by (sku, region), so "this week" is not reachable through any
    method here. That is the whole point of the class: the guarantee lives in
    the type, not in the discipline of whoever writes the next feature.
    """

    __slots__ = ("_price", "_units")

    def __init__(
        self, prior_units: pd.core.groupby.SeriesGroupBy, prior_price: pd.core.groupby.SeriesGroupBy
    ) -> None:
        self._units = prior_units
        self._price = prior_price

    # -- target history ---------------------------------------------------

    def lag(self, n: int) -> pd.Series:
        """Units `n` weeks before the row's own week. `lag(1)` is last week."""
        if n < 1:
            raise ValueError(f"lag must be at least 1 week back, got {n}")
        # The series is already shifted by one, so shifting it n-1 more gives
        # the raw series shifted n. n=0 — the current week — is unreachable.
        return self._units.shift(n - 1)

    def mean(self, window: int, min_periods: int = 1) -> pd.Series:
        """Mean units over the `window` weeks strictly before the row's week."""
        return self._units.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())

    def std(self, window: int, min_periods: int = 2) -> pd.Series:
        """Standard deviation of units over the weeks strictly before this one."""
        return self._units.transform(lambda s: s.rolling(window, min_periods=min_periods).std())

    # -- price history ----------------------------------------------------

    def price_lag(self, n: int) -> pd.Series:
        """
        Average unit price `n` weeks back.

        The price a planner *intends* to charge is known before the week
        starts, but the realised average price is not — it reflects the mix
        that actually sold. Only the lagged one is available here.
        """
        if n < 1:
            raise ValueError(f"price lag must be at least 1 week back, got {n}")
        return self._price.shift(n - 1)

    def price_mean(self, window: int, min_periods: int = 1) -> pd.Series:
        """Mean price over the `window` weeks strictly before the row's week."""
        return self._price.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """
        Divide, guarding the zero denominator with np.nan.

        np.nan and not pd.NA. pd.NA cannot live in a float64 column, so
        `.replace(0, pd.NA)` silently promotes the column to *object* dtype —
        and only when the denominator actually contains a zero, which is what
        a SKU with a quiet month produces. LightGBM then refuses the whole
        feature matrix at fit time ("DataFrame.dtypes for data must be int,
        float or bool"), so the pipeline worked on dense series and failed on
        sparse ones. `test_derived_feature_columns_stay_numeric` pins it.
        """
        return numerator / denominator.replace(0, np.nan)


@dataclass(frozen=True, slots=True)
class HistoryFeature:
    """One registered backward-looking feature."""

    name: str
    description: str
    build: Callable[[History], pd.Series]


@dataclass(frozen=True, slots=True)
class CalendarFeature:
    """
    One registered feature describing the row's own week.

    Legitimate because a calendar is known in advance. Kept in a separate,
    short list so that "reads the current row" is an explicit category with
    named members rather than something a history feature might drift into.
    """

    name: str
    description: str
    build: Callable[[pd.Series], pd.Series]


_HISTORY: list[HistoryFeature] = []
_CALENDAR: list[CalendarFeature] = []


def history_feature(name: str, description: str):
    """Register a backward-looking feature. The decorated function takes a History."""

    def decorate(fn: Callable[[History], pd.Series]) -> Callable[[History], pd.Series]:
        _HISTORY.append(HistoryFeature(name=name, description=description, build=fn))
        return fn

    return decorate


def calendar_feature(name: str, description: str):
    """Register a feature computed from the row's own week timestamp."""

    def decorate(fn: Callable[[pd.Series], pd.Series]) -> Callable[[pd.Series], pd.Series]:
        _CALENDAR.append(CalendarFeature(name=name, description=description, build=fn))
        return fn

    return decorate


def history_features() -> tuple[HistoryFeature, ...]:
    return tuple(_HISTORY)


def calendar_features() -> tuple[CalendarFeature, ...]:
    return tuple(_CALENDAR)


def history_feature_names() -> list[str]:
    """Sorted names of every registered history feature."""
    return sorted(f.name for f in _HISTORY)


def describe() -> dict[str, str]:
    """name → description, for the dashboard and the docs. One source of truth."""
    return {f.name: f.description for f in (*_HISTORY, *_CALENDAR)}


def validate_registry(features: Iterable[HistoryFeature] | None = None) -> None:
    """
    Check the registry's own invariants. Called at import; also called by tests.

    Two things can silently go wrong when someone adds a feature: a duplicate
    name (the second one overwrites the first in the output frame, and the
    model quietly loses an input) and a name the trainer's prefix filter does
    not select (the feature is computed, written to parquet, and never used).
    Neither raises on its own.
    """
    features = list(features if features is not None else _HISTORY)
    names = [f.name for f in features]

    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate history feature names in the registry: {duplicates}")

    unselectable = sorted(n for n in names if not n.startswith(HISTORY_PREFIXES))
    if unselectable:
        raise ValueError(
            f"history features {unselectable} do not start with one of "
            f"{HISTORY_PREFIXES}; the trainer selects model inputs by prefix, so "
            "they would be computed and then ignored"
        )


# ---------------------------------------------------------------------------
# The registered feature set.
#
# Adding one is a decorated function. It cannot see the current week, because
# the only argument it gets is the History.
# ---------------------------------------------------------------------------

#: Trailing windows in weeks. 52 gives the model a same-week-last-year level.
ROLLING_WINDOWS = (4, 13, 52)
LAGS = (1, 2, 3, 4, 52)


def _register_lags() -> None:
    for lag in LAGS:
        history_feature(
            f"lag_{lag}_units",
            f"Units sold {lag} week{'s' if lag != 1 else ''} before this one",
        )(lambda h, n=lag: h.lag(n))


def _register_windows() -> None:
    for window in ROLLING_WINDOWS:
        history_feature(
            f"roll_{window}w_mean",
            f"Mean weekly units over the {window} weeks before this one",
        )(lambda h, w=window: h.mean(w))
        history_feature(
            f"roll_{window}w_std",
            f"Volatility of weekly units over the {window} weeks before this one",
        )(lambda h, w=window: h.std(w))


_register_lags()
_register_windows()


@history_feature("trend_1_over_4", "Last week against the trailing 4-week mean")
def _trend_1_over_4(h: History) -> pd.Series:
    return History.ratio(h.lag(1), h.mean(4))


@history_feature("lag_1_price", "Average unit price last week")
def _lag_1_price(h: History) -> pd.Series:
    return h.price_lag(1)


@history_feature("price_vs_13w", "Last week's price against its trailing quarter")
def _price_vs_13w(h: History) -> pd.Series:
    return History.ratio(h.price_lag(1), h.price_mean(13))


@calendar_feature("weekofyear", "ISO week of the year (annual seasonality)")
def _weekofyear(week: pd.Series) -> pd.Series:
    return week.dt.isocalendar().week.astype(int)


@calendar_feature("month", "Calendar month (seasonality)")
def _month(week: pd.Series) -> pd.Series:
    return week.dt.month.astype(int)


@calendar_feature("year", "Calendar year (level drift)")
def _year(week: pd.Series) -> pd.Series:
    return week.dt.year.astype(int)


validate_registry()
