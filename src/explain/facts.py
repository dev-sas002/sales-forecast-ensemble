"""
The facts behind a forecast, computed from the model and the data.

This module is the honest half of forecast explanation, and it exists before
the language-model half for a reason. The failure mode of "explain this
prediction with an LLM" is a model writing a fluent, plausible story that has
no causal relationship to what the forecaster actually did. The reader cannot
tell the difference, which makes a confident wrong explanation worse than no
explanation.

So nothing here is generated. Every field is arithmetic on the feature table,
the forecast, the backtest report, or LightGBM's own `pred_contrib` — exact
tree SHAP values, which sum with the base value to the prediction itself. The
narrative layer is handed this object and is allowed to phrase it. It is not
allowed to compute anything, and it is not given the model.

The consequence worth stating: with no API key set, the facts are still
complete, and `src.explain.narrative` renders them as prose from a template.
The AI makes the output nicer to read; it is not what makes it true.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.quality.anomalies import detect

KEYS = ["sku", "region"]


def _json_safe(value):
    """Recursively replace non-finite floats with None."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class Driver:
    """One feature's contribution to the median forecast, in units."""

    feature: str
    contribution: float
    description: str


@dataclass(frozen=True, slots=True)
class ForecastFacts:
    """
    Everything known about one series' forecast, computed and not inferred.

    Deliberately flat and JSON-serialisable: it is what the API returns, what
    the dashboard renders, and what the narrator is given. One shape, three
    consumers, no second source of truth.
    """

    sku: str
    region: str
    horizon_weeks: int
    total_forecast: float
    mean_weekly: float
    recent_mean_weekly: float
    change_vs_recent: float
    interval_width_pct: float
    trend_direction: str
    seasonal_position: str
    drivers: list[Driver] = field(default_factory=list)
    anomaly_count: int = 0
    anomaly_notes: list[str] = field(default_factory=list)
    backtest_wape: float | None = None
    backtest_lift: float | None = None
    backtest_coverage: float | None = None

    def to_dict(self) -> dict:
        """
        JSON-safe: NaN becomes None.

        "Not measured" is a real state here — without a history table there is
        no recent average to compare against — and NaN is how pandas spells
        it. JSON has no NaN, so `json.dumps` raises rather than encoding one,
        which turned a missing feature table into a 500 on `/explain`. null is
        what the field means.
        """
        return _json_safe(asdict(self))


#: Plain-language names for the registry's feature columns. The narrator is
#: given these rather than `roll_13w_mean`, because "the trailing quarter" is
#: what the planner calls it.
FEATURE_PROSE = {
    "lag_1_units": "last week's units",
    "lag_2_units": "units two weeks ago",
    "lag_3_units": "units three weeks ago",
    "lag_4_units": "units four weeks ago",
    "lag_52_units": "the same week last year",
    "roll_4w_mean": "the trailing four-week average",
    "roll_13w_mean": "the trailing quarter's average",
    "roll_52w_mean": "the trailing year's average",
    "roll_4w_std": "recent week-to-week volatility",
    "roll_13w_std": "volatility over the quarter",
    "roll_52w_std": "volatility over the year",
    "trend_1_over_4": "last week against its four-week average",
    "lag_1_price": "last week's price",
    "price_vs_13w": "price against its quarterly average",
    "horizon": "how far ahead the week is",
    "target_weekofyear": "where the week falls in the year",
    "target_month": "the month being forecast",
    "sku": "which SKU it is",
    "region": "which region it is",
}

#: Weeks of history treated as "recent" when describing a change in level.
RECENT_WEEKS = 8


def _seasonal_position(history: pd.Series, forecast_weeks: pd.Series) -> str:
    """Where the forecast window sits against this series' own annual shape."""
    if history.empty or forecast_weeks.empty:
        return "unknown"
    weeks = pd.to_datetime(forecast_weeks).dt.isocalendar().week
    mid = int(weeks.median())
    # Compare the same weeks of previous years against the series' own median.
    same_period = history[
        pd.to_datetime(history.index).isocalendar().week.between(mid - 2, mid + 2)
    ]
    if same_period.empty:
        return "unknown"
    ratio = float(same_period.median()) / max(float(history.median()), 1e-9)
    if ratio > 1.1:
        return "a seasonally strong part of the year for this series"
    if ratio < 0.9:
        return "a seasonally weak part of the year for this series"
    return "a seasonally average part of the year for this series"


def build_facts(
    sku: str,
    region: str,
    forecast: pd.DataFrame,
    history: pd.DataFrame | None = None,
    drivers: list[tuple[str, float]] | None = None,
    backtest: dict | None = None,
) -> ForecastFacts:
    """
    Assemble the facts for one series.

    `forecast` is the served forecast rows (date + q0.1/q0.5/q0.9). `history`
    is the weekly feature table for the same series, when available — without
    it the comparison against recent trading is simply omitted rather than
    guessed. `drivers` are (feature, contribution) pairs from
    `feature_contributions`; `backtest` is the saved report's summary block.
    """
    fc = forecast.sort_values("date")
    median = fc["q0.5"].astype(float)
    lo = fc["q0.1"].astype(float)
    hi = fc["q0.9"].astype(float)

    total = float(median.sum())
    mean_weekly = float(median.mean()) if len(median) else 0.0
    width_pct = float(((hi - lo) / median.clip(lower=1e-9)).median()) if len(median) else 0.0

    recent_mean = float("nan")
    change = float("nan")
    trend = "unknown"
    seasonal = "unknown"
    anomaly_count = 0
    notes: list[str] = []

    if history is not None and not history.empty:
        hist = history.sort_values("week")
        recent = hist["units_sold"].tail(RECENT_WEEKS).astype(float)
        if len(recent):
            recent_mean = float(recent.mean())
            change = (mean_weekly - recent_mean) / max(abs(recent_mean), 1e-9)
            if change > 0.05:
                trend = "above"
            elif change < -0.05:
                trend = "below"
            else:
                trend = "in line with"

        indexed = hist.set_index("week")["units_sold"].astype(float)
        seasonal = _seasonal_position(indexed, fc["date"])

        found = detect(hist.assign(sku=sku, region=region))
        anomaly_count = len(found)
        notes = [str(d) for d in found["detail"].head(3)] if anomaly_count else []

    driver_objects = [
        Driver(
            feature=name,
            contribution=round(float(value), 2),
            description=FEATURE_PROSE.get(name, name),
        )
        for name, value in (drivers or [])
    ]

    summary = backtest or {}
    lift = None
    if summary.get("wape_naive"):
        lift = (summary["wape_naive"] - summary["wape_model"]) / summary["wape_naive"]

    return ForecastFacts(
        sku=sku,
        region=region,
        horizon_weeks=len(fc),
        total_forecast=round(total, 1),
        mean_weekly=round(mean_weekly, 1),
        recent_mean_weekly=(round(recent_mean, 1) if np.isfinite(recent_mean) else float("nan")),
        change_vs_recent=round(change, 3) if np.isfinite(change) else float("nan"),
        interval_width_pct=round(width_pct, 3),
        trend_direction=trend,
        seasonal_position=seasonal,
        drivers=driver_objects,
        anomaly_count=anomaly_count,
        anomaly_notes=notes,
        backtest_wape=summary.get("wape_model"),
        backtest_lift=lift,
        backtest_coverage=(summary.get("calibrated_coverage") or summary.get("coverage")),
    )
