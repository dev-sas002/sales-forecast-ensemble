"""
Aggregate linked transactions to SKU x region x week and attach features.

The rule this module exists to enforce: **a feature attached to week t may
only use information available strictly before week t.** The original version
broke it. `rolling(4).mean()` was computed on the raw target column, so the
window at week t included week t's own units — the model was handed a quarter
of the answer as an input. A leak like that does not raise; it quietly makes
backtest scores look excellent and live forecasts look broken, which is the
single most expensive failure mode in forecasting work.

The guarantee is now structural rather than careful. Features are not written
here at all: they live in :mod:`src.features.registry`, and each one receives
a :class:`~src.features.registry.History` — the target and price series
already shifted one period — instead of the DataFrame. There is no accessor on
that object that returns the current week, so the leaking line cannot be
written through the interface. `tests/test_no_leakage.py` enumerates the
registry, so a feature added tomorrow is covered by the invariance check the
moment it is registered.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_DIR, LANDING_DIR
from src.features.registry import (
    History,
    calendar_features,
    history_feature_names,
    history_features,
)
from src.utils.validation import check_required_columns

REQUIRED_LINKED_COLS = [
    "txn_date",
    "sku",
    "region",
    "quantity",
    "amount",
    "txn_id",
    "price",
]

KEYS = ["sku", "region"]

#: Bumped when the set of feature columns changes in a way that makes an older
#: parquet unsafe to train on. `load_feature_table` refuses a mismatch rather
#: than letting a stale table reach the model — which is exactly how a feature
#: table carrying the pre-2026-09 leaking `rolling_4w_mean` column stayed
#: consumable long after the code that wrote it was gone.
FEATURE_SCHEMA_VERSION = 2

_SIDECAR_SUFFIX = ".schema.json"


def aggregate_weekly(linked: pd.DataFrame) -> pd.DataFrame:
    """Collapse transactions to one row per SKU x region x week."""
    linked = linked.copy()
    linked["txn_date"] = pd.to_datetime(linked["txn_date"])
    # Monday-anchored weeks so a week label is the week's first day.
    linked["week"] = linked["txn_date"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly = (
        linked.groupby(KEYS + ["week"])
        .agg(
            units_sold=("quantity", "sum"),
            sales_dollars=("amount", "sum"),
            n_transactions=("txn_id", "nunique"),
            avg_price=("price", "mean"),
        )
        .reset_index()
    )
    return weekly.sort_values(KEYS + ["week"]).reset_index(drop=True)


def fill_missing_weeks(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Give every series a row for every week in its own span.

    A week with no transactions is absent from a groupby, not zero. Left as
    gaps, `shift(1)` would silently reach across them and call a value from
    three weeks ago "last week".

    Built as integer week offsets rather than dates. The original looped in
    Python and called `reindex` once per (sku, region); the obvious
    replacement — one `pd.date_range(freq="W-MON")` per span, then explode —
    is barely faster, because `W-MON` is a custom offset that pandas walks one
    step at a time in Python. Profiling a 2,048-series build put 2.7 of 3.7
    seconds inside `date_range` alone.

    Weeks are a regular 7-day grid, so none of that is necessary: number the
    weeks from a common Monday epoch, build the complete (series x week) index
    with numpy arithmetic, and convert back to dates once at the end. The
    measured effect is in the README; `scripts/benchmark.py` runs the old
    implementation beside the new one so the comparison is a measurement.
    """
    if weekly.empty:
        return weekly.copy()

    weekly = weekly.copy()
    weekly["week"] = pd.to_datetime(weekly["week"])

    epoch = weekly["week"].min()
    offsets = (weekly["week"] - epoch).dt.days
    # Every week label is a Monday (`aggregate_weekly` anchors them), so every
    # offset from the earliest Monday is a whole number of weeks. If that ever
    # stops being true the integer grid would silently collapse two weeks into
    # one, so it is checked rather than assumed.
    if (offsets % 7 != 0).any():
        raise ValueError(
            "week labels are not all on the same weekday; aggregate_weekly() "
            "anchors weeks to Monday and fill_missing_weeks() relies on it"
        )
    weekly["_w"] = (offsets // 7).astype("int64")

    spans = weekly.groupby(KEYS, sort=False)["_w"].agg(["min", "max"]).reset_index()
    counts = (spans["max"] - spans["min"] + 1).to_numpy(dtype=np.int64)

    # Position of each row within its own span: a global arange minus the
    # offset at which each span starts. No Python loop over series.
    starts = np.repeat(counts.cumsum() - counts, counts)
    within = np.arange(counts.sum(), dtype=np.int64) - starts
    grid = pd.DataFrame(
        {
            "sku": np.repeat(spans["sku"].to_numpy(), counts),
            "region": np.repeat(spans["region"].to_numpy(), counts),
            "_w": np.repeat(spans["min"].to_numpy(dtype=np.int64), counts) + within,
        }
    )

    filled = grid.merge(weekly.drop(columns="week"), on=KEYS + ["_w"], how="left")
    filled["week"] = epoch + pd.to_timedelta(filled["_w"] * 7, unit="D")
    filled = filled.drop(columns="_w")

    # No sales genuinely means zero units sold.
    for col in ("units_sold", "sales_dollars", "n_transactions"):
        filled[col] = filled[col].fillna(0.0)
    # Price is not zero in a week with no sales; carry the last known one.
    by_series = filled.groupby(KEYS, sort=False)["avg_price"]
    filled["avg_price"] = by_series.ffill()
    filled["avg_price"] = filled.groupby(KEYS, sort=False)["avg_price"].bfill()

    return filled.sort_values(KEYS + ["week"]).reset_index(drop=True)


def add_history_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Attach every registered feature.

    The two shifts below are the only place in the codebase where the target
    and price series are moved back a week, and the `History` handed to the
    registry is built from nothing else. Everything a feature can reach is
    therefore strictly in the past by construction.
    """
    df = weekly.sort_values(KEYS + ["week"]).copy()

    prior_units = df.groupby(KEYS, sort=False)["units_sold"].shift(1)
    prior_price = df.groupby(KEYS, sort=False)["avg_price"].shift(1)

    series_index = [df["sku"], df["region"]]
    history = History(
        prior_units=prior_units.groupby(series_index, sort=False),
        prior_price=prior_price.groupby(series_index, sort=False),
    )

    for feature in history_features():
        df[feature.name] = feature.build(history)

    week = pd.to_datetime(df["week"])
    for feature in calendar_features():
        df[feature.name] = feature.build(week)

    return df


def _sidecar_path(parquet: Path) -> Path:
    return parquet.with_suffix(parquet.suffix + _SIDECAR_SUFFIX)


def load_feature_table(path: Path | None = None) -> pd.DataFrame:
    """
    Read the feature table, refusing one this build of the code did not write.

    A feature parquet is derived data whose meaning is defined by the registry
    that produced it. Read one written by an older registry and the failure is
    invisible: the trainer selects columns by prefix, so a column that no
    longer exists is simply absent and a column that should no longer exist is
    silently fed to the model. Both cases are a wrong model, not an error.

    So the writer leaves a sidecar naming the schema version and the exact
    feature set, and this refuses anything that does not match. Regenerating
    is `python -m src.features.build_features`.
    """
    path = Path(path) if path is not None else FEATURE_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No feature table at {path}. Run `python -m scripts.run_pipeline` "
            "(or `make pipeline`) to build one."
        )

    sidecar = _sidecar_path(path)
    if not sidecar.exists():
        raise ValueError(
            f"{path} has no {sidecar.name} beside it, so it was written by a build "
            "of this pipeline that predates feature-schema checking. Its columns "
            "cannot be trusted to mean what the trainer thinks they mean — "
            "regenerate it with `python -m src.features.build_features`."
        )

    meta = json.loads(sidecar.read_text())
    expected = history_feature_names()
    if meta.get("schema_version") != FEATURE_SCHEMA_VERSION or meta.get("features") != expected:
        stale = sorted(set(meta.get("features") or []) - set(expected))
        missing = sorted(set(expected) - set(meta.get("features") or []))
        raise ValueError(
            f"{path} was built by a different feature registry "
            f"(schema v{meta.get('schema_version')}, this code is v{FEATURE_SCHEMA_VERSION})."
            + (f" Columns it has that the registry no longer defines: {stale}." if stale else "")
            + (f" Columns the registry defines that it lacks: {missing}." if missing else "")
            + " Regenerate it with `python -m src.features.build_features`."
        )

    return pd.read_parquet(path)


def build() -> Path:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    linked = pd.read_parquet(LANDING_DIR / "linked_panel_credit.parquet")
    check_required_columns(linked, REQUIRED_LINKED_COLS, "linked_panel_credit")

    weekly = aggregate_weekly(linked)
    weekly = fill_missing_weeks(weekly)
    features = add_history_features(weekly)

    out = FEATURE_DIR / "features.parquet"
    features.to_parquet(out, index=False)
    _sidecar_path(out).write_text(
        json.dumps(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "features": history_feature_names(),
                "rows": len(features),
                "series": int(features.groupby(KEYS).ngroups),
            },
            indent=2,
        )
    )
    print(
        f"Saved features: {out}  ({len(features):,} rows, "
        f"{features.groupby(KEYS).ngroups} series, "
        f"{len(history_feature_names())} history features)"
    )
    return out


if __name__ == "__main__":
    build()
