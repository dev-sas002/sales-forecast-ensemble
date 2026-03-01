"""
The forecast repository: how the API gets rows out of parquet.

This is where the serving bottleneck was. `/forecast` did, per request:

    df = pd.read_parquet(FEATURE_DIR / "forecast_ensemble.parquet")
    df = df[(df["sku"] == sku) & (df["region"] == region)]

— a full read of the whole forecast table, every column, every series, then a
boolean mask, to return at most twelve rows. The file the demo ships is small,
so nothing looked wrong; the cost is linear in the size of the catalogue, and
on a table of 25k series x 12 weeks each request reads several megabytes off
disk and materialises it just to throw ~0.003% of it away. `scripts/benchmark.py`
measures what that actually costs and what the change below saves.

Three things fix it, in order of how much they matter:

1. **Read once, not per request.** A forecast file changes when the pipeline
   runs, which is weekly, not per request. The table is cached in the process
   and invalidated on the file's `(mtime_ns, size)` — so a pipeline run is
   picked up on the next request without a restart, and nothing else triggers
   a re-read. Using the stat tuple rather than a TTL means the cache is never
   stale and never needlessly cold.

2. **Index it once.** On load the table is grouped by (sku, region) into a
   dict of small frames, each sorted by date. A lookup is then a dict hit and
   a `head(n)` rather than a scan of the whole table.

3. **Read only the columns needed.** Column projection is pushed into pyarrow,
   so the quantile columns and keys come off disk and the rest does not.

The cache is per process. That is the right scope here: it is a read-through
cache of an immutable-until-republished artifact, so N uvicorn workers holding
N copies is N times the memory and zero coordination, and the alternative
(Redis) would add a service to a system that does not otherwise need one. If
the catalogue grows past what a worker can hold, the answer is a columnar
store with predicate pushdown, not a cache — that trade is written up in the
README's design notes rather than pre-built here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import quantile_cols
from src.models import registry

#: Preference order when several forecast files exist. The ensemble is the
#: product of the pipeline; a single model's output is the fallback for a run
#: that trained but did not ensemble.
FORECAST_FILENAMES = (registry.ENSEMBLE_FILENAME, registry.output_filename("lgb"))

KEY_COLS = ["sku", "region", "date"]


class ForecastNotAvailable(FileNotFoundError):
    """No forecast file exists at all."""


class ForecastSchemaError(ValueError):
    """A forecast file exists but does not carry the configured quantiles."""


@dataclass(frozen=True, slots=True)
class _Stamp:
    """Identity of a file version: path plus what `stat` says about it."""

    path: str
    mtime_ns: int
    size: int

    @classmethod
    def of(cls, path: Path) -> _Stamp:
        st = path.stat()
        return cls(str(path), st.st_mtime_ns, st.st_size)


@dataclass(frozen=True, slots=True)
class ForecastTable:
    """One loaded, indexed forecast file."""

    source: str
    stamp: _Stamp
    by_series: dict[tuple[str, str], pd.DataFrame]
    quantiles: list[str]

    @property
    def n_rows(self) -> int:
        return sum(len(df) for df in self.by_series.values())

    @property
    def series(self) -> list[tuple[str, str]]:
        return sorted(self.by_series)

    def slice(self, sku: str, region: str, weeks: int) -> pd.DataFrame | None:
        """The first `weeks` forecast rows for one series, or None if unknown."""
        df = self.by_series.get((sku, region))
        if df is None:
            return None
        return df.head(weeks)


def _suffixed_match(columns: set[str], canonical: str) -> str | None:
    """
    Find `q0.5` or any `q0.5_<model>` among `columns`.

    Matched by pattern rather than against the model registry on purpose. The
    registry is populated by importing a trainer, so a registry lookup here
    would make the API's behaviour depend on whether LightGBM happened to have
    been imported yet — a serving layer that reads a file differently
    depending on an unrelated import is a bug waiting for a deployment.
    """
    if canonical in columns:
        return canonical
    prefix = f"{canonical}_"
    candidates = sorted(c for c in columns if c.startswith(prefix))
    if not candidates:
        return None
    # Prefer a model this build implements, if the registry happens to be
    # populated; otherwise take the first deterministically.
    for name in sorted(registry.registered()):
        if f"{prefix}{name}" in candidates:
            return f"{prefix}{name}"
    return candidates[0]


def normalize_forecast_columns(df: pd.DataFrame, quantiles: list[str]) -> pd.DataFrame:
    """
    Give the frame the canonical `q0.1 / q0.5 / q0.9` names.

    A single model writes `q0.5_lgb`; the ensemble writes `q0.5`. The API
    serves one shape regardless of which file it found, so a pipeline that
    trained but did not ensemble does not change the contract.
    """
    if all(c in df.columns for c in quantiles):
        return df
    columns = set(df.columns)
    renames = {}
    for canonical in quantiles:
        found = _suffixed_match(columns, canonical)
        if found is not None and found != canonical:
            renames[found] = canonical
    return df.rename(columns=renames) if renames else df


def _find_forecast_file(feature_dir: Path) -> Path:
    for name in FORECAST_FILENAMES:
        path = feature_dir / name
        if path.exists():
            return path
    raise ForecastNotAvailable(
        f"No forecast file in {feature_dir}. Run the pipeline "
        "(`make pipeline`, or `python -m scripts.run_pipeline`) first."
    )


def _load(path: Path) -> ForecastTable:
    quantiles = quantile_cols()
    # Column projection: read the keys and whatever quantile columns the file
    # has, under either naming. Everything else stays on disk — the schema is
    # read from the footer, which does not touch the row groups.
    schema_cols = set(pq.ParquetFile(path).schema.names)
    wanted = [c for c in KEY_COLS if c in schema_cols]
    for canonical in quantiles:
        found = _suffixed_match(schema_cols, canonical)
        if found is not None:
            wanted.append(found)

    df = pd.read_parquet(path, columns=wanted)
    df = normalize_forecast_columns(df, quantiles)

    missing = [c for c in quantiles if c not in df.columns]
    if missing:
        raise ForecastSchemaError(
            f"{path.name} is missing quantile columns {missing}. "
            "Re-run the pipeline to regenerate it."
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(KEY_COLS)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    by_series = {
        (str(sku), str(region)): group.reset_index(drop=True)
        for (sku, region), group in df.groupby(["sku", "region"], sort=False)
    }
    return ForecastTable(
        source=path.name,
        stamp=_Stamp.of(path),
        by_series=by_series,
        quantiles=quantiles,
    )


_cache: ForecastTable | None = None
_lock = threading.Lock()


def load_forecasts(feature_dir: Path) -> ForecastTable:
    """
    The cached, indexed forecast table for `feature_dir`.

    Re-reads only when the file's stat signature changes, so a pipeline run is
    visible to a running API on the next request with no restart and no TTL.
    """
    global _cache
    path = _find_forecast_file(feature_dir)
    stamp = _Stamp.of(path)

    cached = _cache
    if cached is not None and cached.stamp == stamp:
        return cached

    with _lock:
        # Re-check under the lock: two requests can race a pipeline run, and
        # loading twice is wasted work rather than a correctness problem.
        if _cache is not None and _cache.stamp == stamp:
            return _cache
        _cache = _load(path)
        return _cache


def clear_cache() -> None:
    """Drop the cached table. Used by tests, which move FEATURE_DIR under it."""
    global _cache
    with _lock:
        _cache = None
