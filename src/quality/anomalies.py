"""
Anomaly flagging on the *inputs*, before anything is fitted to them.

A demand forecaster is a machine for extrapolating the past. That makes it
maximally vulnerable to the one thing nobody checks: the past being wrong. A
week of missing feed looks exactly like a week of zero demand, and the model
learns the wrong thing from it without any part of the pipeline raising. A
stockout is worse than useless as a training signal — it is demand the
business could not serve, recorded as demand that did not exist, which teaches
the model to under-forecast the SKUs that sell out.

Three checks, all robust statistics on the weekly table, all arithmetic:

**Level spikes and collapses.** A modified z-score built on the median and the
median absolute deviation rather than the mean and standard deviation. The
mean and SD are themselves dragged by the outlier you are looking for, so a
single 10x week inflates the SD enough to hide itself; the median and MAD are
not. The 0.6745 factor makes MAD a consistent estimator of sigma for normal
data, so the threshold reads on the familiar scale.

**Zero runs.** Consecutive zero-unit weeks in a series that otherwise sells.
One quiet week is seasonality; four in a row for a SKU that averages 200 units
is a data feed that stopped, a delisting, or a stockout, and all three mean the
model should not be told that demand was zero.

**Price jumps.** A week-on-week change in average unit price beyond a
threshold. Genuine promotions do this on purpose — which is why the output
carries the size and direction rather than a bare flag, so a planner can tell
a promotion from a currency error or a units-vs-pence mistake.

Deliberately *not* a model. The point of this module is to be checkable by
hand: every number in its output can be recomputed from the weekly table with
a calculator, which is what makes it trustworthy as a gate on the thing that
is a model. It also means it needs no training, no artifact and no API key.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KEYS = ["sku", "region"]

#: |modified z| above this is reported. 3.5 is the conventional cut for the
#: MAD-based score — roughly a 1-in-2000 week under normality.
Z_THRESHOLD = 3.5

#: Consecutive zero weeks before a run is worth a planner's attention.
ZERO_RUN_WEEKS = 3

#: Fractional week-on-week price move worth reporting.
PRICE_JUMP = 0.25

#: MAD is a consistent estimator of sigma for normal data after this scaling.
_MAD_TO_SIGMA = 0.6745

#: The mean-absolute-deviation fallback's scaling, so both forms of the
#: modified z-score read on the same scale (Iglewicz & Hoaglin, 1993).
_MEANAD_TO_SIGMA = 1.253314


@dataclass(frozen=True, slots=True)
class AnomalyRules:
    """The thresholds, in one place, so a caller can tune them without forking."""

    z_threshold: float = Z_THRESHOLD
    zero_run_weeks: int = ZERO_RUN_WEEKS
    price_jump: float = PRICE_JUMP


ANOMALY_COLUMNS = ["sku", "region", "week", "kind", "severity", "detail", "value"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ANOMALY_COLUMNS})


def modified_zscore(values: pd.Series) -> pd.Series:
    """
    Iglewicz–Hoaglin modified z-score, with the zero-MAD fallback.

    The MAD form is the one worth having: the mean and standard deviation are
    themselves dragged by the outlier you are looking for, so a single 100x
    week inflates the SD enough to hide itself, while the median and MAD are
    not moved by it.

    But MAD has a failure of its own that matters here specifically. A series
    that sits on a flat baseline — 150 units a week, every week, then one week
    of 1,500 — has a MAD of exactly zero, because more than half its
    deviations are zero. Dividing by it yields inf or nan, and the spike that
    is blindingly obvious to a human goes unreported. Flat baselines are
    common in weekly retail data, so this is not a corner case.

    Iglewicz & Hoaglin's own remedy is to fall back to the *mean* absolute
    deviation, scaled by 1.253314 so the two forms read on the same scale. A
    genuinely constant series still scores zero everywhere, which is correct:
    a series that never moves has no outliers.
    """
    v = values.astype(float)
    median = v.median()
    deviations = (v - median).abs()

    mad = deviations.median()
    if np.isfinite(mad) and mad > 0:
        return _MAD_TO_SIGMA * (v - median) / mad

    mean_ad = deviations.mean()
    if np.isfinite(mean_ad) and mean_ad > 0:
        return (v - median) / (_MEANAD_TO_SIGMA * mean_ad)

    return pd.Series(np.zeros(len(v)), index=v.index)


def _level_anomalies(group: pd.DataFrame, rules: AnomalyRules) -> list[dict]:
    z = modified_zscore(group["units_sold"])
    out = []
    for idx, score in z.items():
        if abs(score) < rules.z_threshold:
            continue
        row = group.loc[idx]
        direction = "spike" if score > 0 else "collapse"
        out.append(
            {
                "kind": f"level_{direction}",
                "week": row["week"],
                "severity": round(float(abs(score)), 2),
                "detail": (
                    f"{row['units_sold']:.0f} units is {abs(score):.1f} robust "
                    f"deviations {'above' if score > 0 else 'below'} this series' median"
                ),
                "value": float(row["units_sold"]),
            }
        )
    return out


def _zero_run_anomalies(group: pd.DataFrame, rules: AnomalyRules) -> list[dict]:
    units = group["units_sold"].to_numpy(dtype=float)
    weeks = group["week"].tolist()
    if units.size == 0 or np.median(units) <= 0:
        # A series that is mostly zero is a slow mover, not a broken feed.
        return []

    out = []
    start = None
    for i, value in enumerate([*units, 1.0]):  # sentinel closes a trailing run
        if value == 0 and start is None:
            start = i
        elif value != 0 and start is not None:
            length = i - start
            if length >= rules.zero_run_weeks:
                out.append(
                    {
                        "kind": "zero_run",
                        "week": weeks[start],
                        "severity": float(length),
                        "detail": (
                            f"{length} consecutive weeks at zero units in a series "
                            f"whose median week is {np.median(units):.0f} — a stopped "
                            "feed, a delisting or a stockout, not demand"
                        ),
                        "value": 0.0,
                    }
                )
            start = None
    return out


def _price_anomalies(group: pd.DataFrame, rules: AnomalyRules) -> list[dict]:
    if "avg_price" not in group.columns:
        return []
    price = group["avg_price"].astype(float)
    change = price.pct_change()

    out = []
    for idx, delta in change.items():
        if not np.isfinite(delta) or abs(delta) < rules.price_jump:
            continue
        row = group.loc[idx]
        out.append(
            {
                "kind": "price_jump",
                "week": row["week"],
                "severity": round(float(abs(delta)), 3),
                "detail": (
                    f"average price moved {delta:+.0%} week-on-week to "
                    f"{row['avg_price']:.2f} — expected on a promotion, a data "
                    "error otherwise"
                ),
                "value": float(row["avg_price"]),
            }
        )
    return out


def detect(weekly: pd.DataFrame, rules: AnomalyRules | None = None) -> pd.DataFrame:
    """
    Flag suspect weeks in a weekly SKU x region table.

    Returns one tidy row per finding — sku, region, week, kind, severity,
    a sentence a planner can act on, and the offending value — sorted with the
    most severe first so a dashboard can show the top of the list and a caller
    can threshold it.
    """
    rules = rules or AnomalyRules()
    required = {"sku", "region", "week", "units_sold"}
    missing = sorted(required - set(weekly.columns))
    if missing:
        raise ValueError(f"anomaly detection needs columns {missing}")
    if weekly.empty:
        return _empty()

    weekly = weekly.sort_values(KEYS + ["week"])
    findings = []
    for (sku, region), group in weekly.groupby(KEYS, sort=False):
        for record in (
            *_level_anomalies(group, rules),
            *_zero_run_anomalies(group, rules),
            *_price_anomalies(group, rules),
        ):
            findings.append({"sku": sku, "region": region, **record})

    if not findings:
        return _empty()

    out = pd.DataFrame(findings)[ANOMALY_COLUMNS]
    return out.sort_values("severity", ascending=False).reset_index(drop=True)


def summarise(anomalies: pd.DataFrame) -> dict[str, int]:
    """Counts by kind, for a headline row. Always includes every known kind."""
    kinds = ["level_spike", "level_collapse", "zero_run", "price_jump"]
    counts = dict.fromkeys(kinds, 0)
    if not anomalies.empty:
        counts.update(anomalies["kind"].value_counts().to_dict())
    counts["total"] = len(anomalies)
    return {k: int(v) for k, v in counts.items()}
