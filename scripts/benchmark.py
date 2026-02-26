"""
Measure the things the README claims about scalability.

Every performance number in the README comes from this script. It exists so
that "the feature build is the bottleneck" is a measurement rather than an
intuition, and so that a future change that makes something slower shows up as
a number rather than as a feeling.

It runs entirely on generated data — no real dataset, no network — and prints
a table per section. Nothing here runs in the test suite: it fits real models
and takes minutes, which is exactly what a test must not do.

    python -m scripts.benchmark                 # everything
    python -m scripts.benchmark --only features # one section
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import QUANTILES

SECTIONS = ("features", "training", "inference", "backtest", "serving")


@dataclass
class Row:
    label: str
    seconds: float
    extra: dict[str, float] = field(default_factory=dict)


def timed(fn: Callable[[], object], repeat: int = 1) -> tuple[float, object]:
    """Best-of-`repeat` wall time. Best-of, not mean: it is the least noisy."""
    best = float("inf")
    result = None
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def synthetic_weekly(n_series: int, n_weeks: int, seed: int = 7) -> pd.DataFrame:
    """
    A weekly SKU x region table of the shape `build_features` produces.

    Built directly rather than by running ingest, because what is being
    measured is the feature build, not CSV parsing.
    """
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    regions = ["NE", "SE", "MW", "W"]

    frames = []
    for s in range(n_series):
        base = float(rng.uniform(20, 400))
        seasonal = 1 + 0.3 * np.cos(2 * np.pi * (np.arange(n_weeks) % 52) / 52)
        units = np.maximum(0.0, base * seasonal + rng.normal(0, base * 0.1, n_weeks))
        frames.append(
            pd.DataFrame(
                {
                    "sku": f"SKU-{s:05d}",
                    "region": regions[s % len(regions)],
                    "week": weeks,
                    "units_sold": np.round(units),
                    "sales_dollars": units * 3.0,
                    "n_transactions": units / 4.0,
                    "avg_price": 3.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def print_table(title: str, rows: list[Row], unit: str = "s") -> None:
    print(f"\n{title}")
    print("-" * max(len(title), 62))
    keys: list[str] = []
    for r in rows:
        for k in r.extra:
            if k not in keys:
                keys.append(k)
    header = f"{'case':<28}{'time (' + unit + ')':>12}"
    for k in keys:
        header += f"{k:>16}"
    print(header)
    for r in rows:
        line = f"{r.label:<28}{r.seconds:>12.3f}"
        for k in keys:
            value = r.extra.get(k)
            # Three decimals: at one, every sub-second timing in the feature
            # table rounds to 0.0 and the comparison the table exists to make
            # disappears.
            line += f"{value:>16,.3f}" if value is not None else f"{'':>16}"
        print(line)


# --------------------------------------------------------------- features ---


def legacy_fill_missing_weeks(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    The pre-2026-09-23 gap-filler, kept here purely so the speedup is measured.

    Claiming "we vectorised it" without running the version being replaced is
    how benchmarks become marketing. This is a verbatim copy of the loop that
    `src.features.build_features.fill_missing_weeks` replaced, and it is not
    imported by anything but this script.
    """
    keys = ["sku", "region"]
    filled = []
    for (sku, region), group in weekly.groupby(keys, sort=False):
        span = pd.date_range(group["week"].min(), group["week"].max(), freq="W-MON")
        group = group.set_index("week").reindex(span).rename_axis("week").reset_index()
        group[keys] = [sku, region]
        group["units_sold"] = group["units_sold"].fillna(0.0)
        group["sales_dollars"] = group["sales_dollars"].fillna(0.0)
        group["n_transactions"] = group["n_transactions"].fillna(0.0)
        group["avg_price"] = group["avg_price"].ffill().bfill()
        filled.append(group)
    return pd.concat(filled, ignore_index=True).sort_values(keys + ["week"]).reset_index(drop=True)


def bench_features(sizes: list[int], n_weeks: int) -> list[Row]:
    """
    Feature-build cost against catalogue size, old gap-filler against new.

    `fill_missing_weeks` used to loop in Python and `reindex` once per series,
    which is linear in *series* on top of linear in rows. Both are run here so
    the difference is a measurement.
    """
    from src.features.build_features import add_history_features, fill_missing_weeks

    rows = []
    for n_series in sizes:
        weekly = synthetic_weekly(n_series, n_weeks)
        old_s, _ = timed(lambda w=weekly: legacy_fill_missing_weeks(w))
        fill_s, filled = timed(lambda w=weekly: fill_missing_weeks(w))
        hist_s, featured = timed(lambda f=filled: add_history_features(f))

        tracemalloc.start()
        add_history_features(filled)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows.append(
            Row(
                label=f"{n_series} series x {n_weeks}w",
                seconds=fill_s + hist_s,
                extra={
                    "rows": float(len(weekly)),
                    "gap-fill s": fill_s,
                    "was (loop) s": old_s,
                    "features s": hist_s,
                    "peak MB": peak / 1e6,
                    "cols": float(featured.shape[1]),
                },
            )
        )
    return rows


# --------------------------------------------------------------- training ---


def bench_training(sizes: list[int], n_weeks: int, horizon: int, rounds: int) -> list[Row]:
    """Training time against SKU count, at a fixed horizon and round count."""
    from src.features.build_features import add_history_features
    from src.models.train_lgb_quantile import make_supervised, train_quantile_models

    rows = []
    for n_series in sizes:
        features = add_history_features(synthetic_weekly(n_series, n_weeks))
        panel_s, panel = timed(lambda f=features: make_supervised(f, range(1, horizon + 1)))
        fit_s, _ = timed(lambda p=panel: train_quantile_models(p, num_boost_round=rounds))
        rows.append(
            Row(
                label=f"{n_series} series, h={horizon}",
                seconds=panel_s + fit_s,
                extra={
                    "panel rows": float(len(panel)),
                    "reshape s": panel_s,
                    "fit s": fit_s,
                    "s/series": (panel_s + fit_s) / n_series,
                },
            )
        )
    return rows


# -------------------------------------------------------------- inference ---


def bench_inference(n_series: int, n_weeks: int, horizons: list[int], rounds: int) -> list[Row]:
    """Forecast latency per horizon, once the models are already fitted."""
    from src.features.build_features import add_history_features
    from src.models.train_lgb_quantile import (
        forecast_from_last_origin,
        make_supervised,
        train_quantile_models,
    )

    features = add_history_features(synthetic_weekly(n_series, n_weeks))
    panel = make_supervised(features, range(1, max(horizons) + 1))
    models, _ = train_quantile_models(panel, num_boost_round=rounds)

    rows = []
    for h in horizons:
        seconds, out = timed(lambda hh=h: forecast_from_last_origin(models, features, hh), repeat=3)
        rows.append(
            Row(
                label=f"{n_series} series x {h}w ahead",
                seconds=seconds,
                extra={
                    "forecast rows": float(len(out)),
                    "ms/row": seconds * 1000 / max(len(out), 1),
                },
            )
        )
    return rows


# --------------------------------------------------------------- backtest ---


def bench_backtest(n_series: int, n_weeks: int, folds: int, rounds: int) -> list[Row]:
    """Serial vs parallel folds — is the harness actually parallelisable?"""
    import os

    from src.evaluation.backtest import run_backtest
    from src.features.build_features import add_history_features

    features = add_history_features(synthetic_weekly(n_series, n_weeks))
    rows = []
    for jobs in (1, min(folds, os.cpu_count() or 1)):
        seconds, results = timed(
            lambda j=jobs: run_backtest(
                features,
                n_folds=folds,
                horizon=4,
                step_weeks=4,
                rounds=rounds,
                jobs=j,
                calibrate=False,
            )
        )
        rows.append(
            Row(
                label=f"{folds} folds, jobs={jobs}",
                seconds=seconds,
                extra={"scored folds": float(len(results))},
            )
        )
    return rows


# ---------------------------------------------------------------- serving ---


def bench_serving(n_series: int, horizon: int, requests: int, tmp: str) -> list[Row]:
    """
    Per-request forecast latency: full parquet read vs the cached index.

    The "uncached" case is what the endpoint did before — read the whole file,
    mask it, take twelve rows — reproduced here so the comparison is against
    real behaviour rather than a strawman.
    """
    from pathlib import Path

    from src.serving import store

    out = Path(tmp)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    skus = [f"SKU-{i:05d}" for i in range(n_series)]
    frame = pd.DataFrame(
        {
            "sku": np.repeat(skus, horizon),
            "region": "NE",
            "date": np.tile(pd.date_range("2026-01-05", periods=horizon, freq="W-MON"), n_series),
            "q0.1": rng.uniform(10, 50, n_series * horizon),
            "q0.5": rng.uniform(50, 100, n_series * horizon),
            "q0.9": rng.uniform(100, 200, n_series * horizon),
        }
    )
    path = out / "forecast_ensemble.parquet"
    frame.to_parquet(path, index=False)
    picks = rng.choice(skus, size=requests)

    def uncached() -> int:
        n = 0
        for sku in picks:
            df = pd.read_parquet(path)
            df = df[(df["sku"] == sku) & (df["region"] == "NE")]
            n += len(df.sort_values("date").head(horizon))
        return n

    def cached() -> int:
        n = 0
        for sku in picks:
            table = store.load_forecasts(out)
            n += len(table.slice(sku, "NE", horizon))
        return n

    rows = []

    seconds, _ = timed(uncached)
    rows.append(
        Row(
            label="read parquet per request",
            seconds=seconds,
            extra={"requests": float(requests), "us/request": seconds * 1e6 / requests},
        )
    )

    # Cold start is its own number: the first request after a pipeline run
    # pays the load, and quoting only the warm figure would hide it.
    store.clear_cache()
    cold, _ = timed(lambda: store.load_forecasts(out))
    rows.append(
        Row(
            label="cached: first request (cold)",
            seconds=cold,
            extra={"requests": 1.0, "us/request": cold * 1e6},
        )
    )

    seconds, _ = timed(cached)
    rows.append(
        Row(
            label="cached + indexed (warm)",
            seconds=seconds,
            extra={"requests": float(requests), "us/request": seconds * 1e6 / requests},
        )
    )
    store.clear_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=SECTIONS, action="append")
    parser.add_argument("--weeks", type=int, default=130)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument(
        "--json", type=str, default=None, help="also write the measurements to this path"
    )
    parser.add_argument("--tmp", type=str, default="/tmp/sku-bench")
    args = parser.parse_args()

    wanted = args.only or list(SECTIONS)
    collected: dict[str, list[dict]] = {}

    print(
        f"Benchmark — {args.weeks} weeks of history, {args.rounds} boosting rounds, "
        f"quantiles {list(QUANTILES)}"
    )

    if "features" in wanted:
        rows = bench_features([8, 64, 256, 1024], args.weeks)
        print_table("Feature build vs catalogue size", rows)
        collected["features"] = [r.__dict__ for r in rows]

    if "training" in wanted:
        rows = bench_training([8, 32, 128], args.weeks, horizon=8, rounds=args.rounds)
        print_table("Training vs SKU count", rows)
        collected["training"] = [r.__dict__ for r in rows]

    if "inference" in wanted:
        rows = bench_inference(128, args.weeks, [1, 4, 12, 26], rounds=args.rounds)
        print_table("Forecast latency per horizon", rows)
        collected["inference"] = [r.__dict__ for r in rows]

    if "backtest" in wanted:
        rows = bench_backtest(256, args.weeks, folds=4, rounds=args.rounds)
        print_table("Backtest: serial vs parallel folds", rows)
        collected["backtest"] = [r.__dict__ for r in rows]

    if "serving" in wanted:
        rows = bench_serving(25_000, 12, requests=50, tmp=args.tmp)
        print_table("API: forecast lookup, 25,000 series x 12 weeks", rows)
        collected["serving"] = [r.__dict__ for r in rows]

    if args.json:
        from pathlib import Path

        Path(args.json).write_text(json.dumps(collected, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
