"""
Combine per-model quantile forecasts into one set.

Two defects shaped this module, and both were the kind that produce
plausible-looking numbers rather than an error.

**It halved forecasts.** The original outer-joined the two models and averaged
with `.fillna(0)`:

    df["q0.5"] = 0.5 * df["q0.5_lgb"].fillna(0) + 0.5 * df["q0.5_deepar"].fillna(0)

Wherever one model had no row for a series-week — which is exactly what an
outer join is for — the missing side contributed a *zero*, and the ensemble
reported half of the forecast the other model had actually made. Silently
halving a demand forecast is the kind of error that reaches a purchase order.
Here a quantile is averaged over the models that produced it, and a row only
one model covers keeps that model's value.

**It blended an artifact with no producer.** The source list was a literal
dict naming `forecast_deepar.parquet`, and that file was committed to the
repository from a long-dead toy dataset. The DeepAR trainer has only ever been
a stub that raises `NotImplementedError`, so nothing in the codebase could
have written it — yet every local run loaded those 36 stale rows and averaged
them into the served forecast. Which models exist is not a fact about the
filesystem; it is a fact about the code, and the code now says so:
:mod:`src.models.registry` is the source of truth, and a `forecast_*.parquet`
no registered model produces is reported as an orphan and left out.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_DIR, QUANTILES
from src.models import registry

logger = logging.getLogger(__name__)

KEYS = ["sku", "region", "date"]


class OrphanForecastError(RuntimeError):
    """Raised when the only forecasts available have no registered producer."""


def load_optional(path: Path) -> pd.DataFrame | None:
    """Load a parquet, or None if it is simply absent."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except PermissionError:
        logger.warning("Permission denied reading %s", path)
        return None


def _tidy(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Reduce one model's output to keys plus its quantile columns."""
    renames = {}
    for q in QUANTILES:
        suffixed, bare = f"q{q}_{model}", f"q{q}"
        if suffixed in df.columns:
            renames[suffixed] = f"q{q}__{model}"
        elif bare in df.columns:
            renames[bare] = f"q{q}__{model}"
        else:
            raise KeyError(
                f"{model} forecast is missing a column for quantile {q} "
                f"(looked for {suffixed!r} and {bare!r}); got {list(df.columns)}"
            )
    out = df.rename(columns=renames)
    out["date"] = pd.to_datetime(out["date"])
    return out[KEYS + list(renames.values())]


def combine(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average each quantile across the models that actually produced it."""
    if not frames:
        raise FileNotFoundError(
            f"No forecast files found in {FEATURE_DIR}. "
            "Run the pipeline (python -m scripts.run_pipeline) first."
        )

    tidied = [_tidy(df, model) for model, df in frames.items()]
    merged = tidied[0]
    for other in tidied[1:]:
        merged = merged.merge(other, on=KEYS, how="outer")

    for q in QUANTILES:
        cols = [c for c in merged.columns if c.startswith(f"q{q}__")]
        # mean(skipna=True): a model that has no row for this week is absent
        # from the average rather than voting zero.
        merged[f"q{q}"] = merged[cols].mean(axis=1, skipna=True)

    quantile_cols = [f"q{q}" for q in QUANTILES]
    merged[quantile_cols] = merged[quantile_cols].clip(lower=0.0)
    # Averaging preserves ordering only if every input was ordered; sorting is
    # cheap insurance against an upstream model emitting crossed quantiles.
    #
    # np.sort rather than a row-wise `sorted()` apply: the apply was O(rows) in
    # Python, and on an empty frame it returns an empty DataFrame with no
    # columns, which made the assignment back fail on a pipeline that produced
    # no overlapping rows.
    merged[quantile_cols] = np.sort(merged[quantile_cols].to_numpy(dtype=float), axis=1)

    return merged[KEYS + quantile_cols].sort_values(KEYS).reset_index(drop=True)


def collect_sources(
    feature_dir: Path | None = None, include_unregistered: bool = False
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    """
    Gather the forecasts that may legitimately be blended.

    Returns the frames keyed by model name, plus the orphans that were found.
    An orphan is loaded only when `include_unregistered` is set, which exists
    for the case where someone genuinely does want to blend a forecast produced
    outside this repository — deliberately, having been told it is there.
    """
    feature_dir = feature_dir or FEATURE_DIR
    produced, orphans = registry.scan_forecasts(feature_dir)

    frames: dict[str, pd.DataFrame] = {}
    for model, path in produced.items():
        df = load_optional(path)
        if df is not None:
            frames[model] = df

    for path in orphans:
        stem = path.stem.removeprefix(registry.FORECAST_PREFIX)
        if include_unregistered:
            logger.warning("blending %s on request: no registered model produces it", path.name)
            df = load_optional(path)
            if df is not None:
                frames[stem] = df
        else:
            logger.warning(
                "ignoring %s: no registered forecaster produces it, so nothing in "
                "this codebase can have written it. Pass --include-unregistered if "
                "you know where it came from and want it blended.",
                path.name,
            )

    if not frames and orphans and not include_unregistered:
        raise OrphanForecastError(
            f"The only forecast files in {feature_dir} are "
            f"{[p.name for p in orphans]}, and no registered forecaster produces "
            "them. Refusing to build an ensemble out of artifacts with no "
            "producer — run the pipeline, or pass --include-unregistered."
        )

    return frames, orphans


def ensemble(include_unregistered: bool = False) -> Path:
    frames, orphans = collect_sources(include_unregistered=include_unregistered)
    out_df = combine(frames)

    out = FEATURE_DIR / registry.ENSEMBLE_FILENAME
    out_df.to_parquet(out, index=False)
    print(
        f"Saved ensemble forecasts: {out}  "
        f"({len(out_df):,} rows from {', '.join(frames) or 'nothing'})"
    )
    if orphans and not include_unregistered:
        print(
            f"  skipped {len(orphans)} forecast file(s) with no registered "
            f"producer: {', '.join(p.name for p in orphans)}"
        )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-unregistered",
        action="store_true",
        help="also blend forecast_*.parquet files no registered model produces",
    )
    args = parser.parse_args()
    ensemble(include_unregistered=args.include_unregistered)
