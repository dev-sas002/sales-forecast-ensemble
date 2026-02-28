"""
Rolling-origin backtesting.

A forecast number means nothing without three things, and most portfolio
forecasting projects supply none of them:

1. **An out-of-sample protocol.** Scores here come from retraining at several
   historical cutoffs and predicting forward from each. A single train/test
   split measures one lucky week.

2. **A baseline.** The seasonal-naive forecast — "this week last year" — is
   free, and a learned model that cannot beat it is a liability, not an asset.
   Every table below reports both, so the model has to earn its place.

3. **Calibration, not just accuracy.** A quantile forecast makes a claim about
   uncertainty: an 80% interval should contain the truth about 80% of the time.
   A model with excellent WAPE and 40% coverage is lying about its confidence,
   and only a coverage check catches it.

Since the 2026-09-23 pass the harness measures a fourth thing: whether
conformal calibration actually repairs the coverage shortfall. The offset is
refitted **inside each fold's own training window** and applied to the weeks
after the cutoff, so "calibrated coverage" is an out-of-sample number and not
the tautology you get by scoring a calibration set against the quantile that
defined it.

Folds are independent — different cutoffs, different models, no shared state —
so `--jobs` runs them in parallel processes. Results are sorted by cutoff
afterwards, so the report does not depend on completion order.

Every metric is arithmetic on held-out actuals. Nothing is judged by a model.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import FEATURE_DIR, QUANTILES
from src.evaluation.conformal import (
    coverage as interval_coverage,
)
from src.evaluation.conformal import (
    fit_conformal_offset,
    nominal_coverage,
)
from src.features.build_features import load_feature_table
from src.models.train_lgb_quantile import (
    KEYS,
    TARGET,
    make_supervised,
    predict_quantiles,
    train_quantile_models,
)
from src.utils.metrics import quantile_loss, wape

#: A fold is only scored if every series has at least this many weeks of
#: history at the cutoff. Below it the 52-week lag is still all-null and the
#: model has no seasonal signal, so the score would measure the fold, not the
#: model. Named rather than inlined so the threshold is visible and tunable.
MIN_TRAIN_WEEKS = 60


@dataclass
class FoldResult:
    cutoff: pd.Timestamp
    n_rows: int
    wape_model: float
    wape_naive: float
    pinball: dict[float, float]
    coverage: float
    calibrated_coverage: float
    conformal_offset: float
    per_horizon: pd.DataFrame = field(repr=False)


def seasonal_naive(features: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """
    "Same week last year", falling back to the last observed week.

    Deliberately the cheapest defensible baseline. Beating a hard baseline is
    interesting; beating a straw man is not, and beating nothing at all — which
    is what a project with no baseline does — tells the reader nothing.

    Implemented as two merges rather than a row loop. The previous version
    called `.get()` on a MultiIndex once per scored row inside `itertuples`,
    which is ~40x slower on a realistic evaluation set (measured; see
    `scripts/benchmark.py`) and grew linearly in Python rather than in pandas.
    """
    ordered = features.sort_values(KEYS + ["week"])
    # De-duplicated: a repeated (sku, region, week) made the old `.get()`
    # return a Series rather than a number.
    history = ordered.drop_duplicates(subset=KEYS + ["week"], keep="last")[
        KEYS + ["week", TARGET]
    ].rename(columns={"week": "_lookup_week", TARGET: "_last_year"})
    # Sorted before `.last()`: without it the fallback's "last observed week"
    # was whatever row happened to be last in the frame.
    fallback = ordered.groupby(KEYS, sort=False)[TARGET].last().rename("_fallback").reset_index()

    probe = targets[KEYS].copy()
    probe["_lookup_week"] = pd.to_datetime(targets["date"]) - pd.Timedelta(weeks=52)
    probe["_row"] = np.arange(len(probe))

    merged = probe.merge(history, on=KEYS + ["_lookup_week"], how="left").merge(
        fallback, on=KEYS, how="left"
    )
    values = merged["_last_year"].fillna(merged["_fallback"]).fillna(0.0)
    return pd.Series(
        values.to_numpy(dtype=float)[np.argsort(merged["_row"].to_numpy())],
        index=targets.index,
    )


def evaluate_fold(
    features: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    rounds: int,
    seed: int,
    calibrate: bool = True,
) -> FoldResult | None:
    """Train on everything up to `cutoff`, score the `horizon` weeks after it."""
    features = features.copy()
    features["week"] = pd.to_datetime(features["week"])

    train_features = features[features["week"] <= cutoff]
    if train_features.empty:
        # groupby().size().min() is NaN on an empty frame, and `NaN < 60` is
        # False — so without this an out-of-range cutoff fell through the
        # history check instead of being skipped.
        return None
    if train_features.groupby(KEYS).size().min() < MIN_TRAIN_WEEKS:
        return None

    # Train only on pairs whose target also falls at or before the cutoff —
    # otherwise the "training" set contains the weeks being scored.
    panel = make_supervised(train_features, range(1, horizon + 1))
    panel = panel[pd.to_datetime(panel["target_week"]) <= cutoff]
    if panel.empty:
        return None

    models, _ = train_quantile_models(panel, num_boost_round=rounds, seed=seed)

    # The conformal offset is learned from a split *inside* the training
    # panel, so it has seen nothing after the cutoff either.
    offset = 0.0
    if calibrate:
        offset = fit_conformal_offset(panel, num_boost_round=rounds, seed=seed).offset

    # The evaluation rows: origins at the cutoff, targets after it.
    full_panel = make_supervised(features, range(1, horizon + 1))
    full_panel["target_week"] = pd.to_datetime(full_panel["target_week"])
    test = full_panel[
        (pd.to_datetime(full_panel["week"]) == cutoff) & (full_panel["target_week"] > cutoff)
    ].copy()
    if test.empty:
        return None

    preds = predict_quantiles(models, test)
    scored = test.reset_index(drop=True).join(preds[[f"q{q}_lgb" for q in QUANTILES]])
    scored["date"] = scored["target_week"]
    scored["naive"] = seasonal_naive(train_features, scored)

    y = scored[TARGET].to_numpy(dtype=float)
    median = scored[f"q{QUANTILES[len(QUANTILES) // 2]}_lgb"].to_numpy(dtype=float)

    lo = scored[f"q{QUANTILES[0]}_lgb"].to_numpy(dtype=float)
    hi = scored[f"q{QUANTILES[-1]}_lgb"].to_numpy(dtype=float)
    coverage = interval_coverage(y, lo, hi)
    calibrated = interval_coverage(y, np.clip(lo - offset, 0.0, None), hi + offset)

    marked = scored.assign(
        abs_err=np.abs(y - median),
        actual=y,
        covered=((y >= lo) & (y <= hi)).astype(float),
    )
    per_horizon = (
        marked.groupby("horizon")
        .agg(abs_err=("abs_err", "sum"), actual=("actual", "sum"), coverage=("covered", "mean"))
        .reset_index()
    )
    per_horizon["wape"] = per_horizon["abs_err"] / per_horizon["actual"].clip(lower=1e-9)
    per_horizon = per_horizon[["horizon", "wape", "coverage"]]

    return FoldResult(
        cutoff=cutoff,
        n_rows=len(scored),
        wape_model=wape(y, median),
        wape_naive=wape(y, scored["naive"].to_numpy(dtype=float)),
        pinball={
            q: quantile_loss(y, scored[f"q{q}_lgb"].to_numpy(dtype=float), q) for q in QUANTILES
        },
        coverage=coverage,
        calibrated_coverage=calibrated,
        conformal_offset=offset,
        per_horizon=per_horizon,
    )


def _fold_cutoffs(
    features: pd.DataFrame, n_folds: int, horizon: int, step_weeks: int
) -> list[pd.Timestamp]:
    last_week = pd.to_datetime(features["week"]).max()
    # Leave `horizon` weeks of actuals after the newest cutoff to score on.
    return [last_week - pd.Timedelta(weeks=horizon + i * step_weeks) for i in range(n_folds)]


def run_backtest(
    features: pd.DataFrame,
    n_folds: int = 4,
    horizon: int = 8,
    step_weeks: int = 8,
    rounds: int = 200,
    seed: int = 7,
    calibrate: bool = True,
    jobs: int = 1,
) -> list[FoldResult]:
    """
    Score `n_folds` rolling origins, optionally in parallel.

    Each fold trains its own models from its own slice — nothing is shared, so
    the only reason this was ever serial is that nobody asked it not to be.
    `jobs=1` keeps the in-process path, which is what the tests use and what
    keeps a stack trace readable.
    """
    features = features.copy()
    features["week"] = pd.to_datetime(features["week"])
    cutoffs = _fold_cutoffs(features, n_folds, horizon, step_weeks)

    if jobs > 1 and len(cutoffs) > 1:
        # LightGBM is itself threaded; letting every worker take four threads
        # oversubscribes the machine and runs slower than serial.
        with ProcessPoolExecutor(
            max_workers=min(jobs, len(cutoffs)),
            initializer=_limit_thread_pools,
        ) as pool:
            futures = [
                pool.submit(evaluate_fold, features, cutoff, horizon, rounds, seed, calibrate)
                for cutoff in cutoffs
            ]
            results = [f.result() for f in futures]
    else:
        results = [
            evaluate_fold(features, cutoff, horizon, rounds, seed, calibrate) for cutoff in cutoffs
        ]

    return sorted((r for r in results if r is not None), key=lambda r: r.cutoff)


def _limit_thread_pools() -> None:
    """Keep each backtest worker to one BLAS/OpenMP thread."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def report(results: list[FoldResult], nominal: float) -> int:
    """Print the table and return a process exit code."""
    if not results:
        print("No folds could be evaluated — not enough history.")
        return 1

    print()
    print(
        f"{'cutoff':<12} {'rows':>6} {'WAPE':>8} {'naive':>8} {'lift':>7} {'cover':>7} {'cal':>7}"
    )
    print("-" * 62)
    for r in results:
        lift = (r.wape_naive - r.wape_model) / max(r.wape_naive, 1e-9)
        print(
            f"{r.cutoff.date()!s:<12} {r.n_rows:>6} {r.wape_model:>8.3f} "
            f"{r.wape_naive:>8.3f} {lift:>6.1%} {r.coverage:>7.1%} "
            f"{r.calibrated_coverage:>7.1%}"
        )

    wape_model = float(np.mean([r.wape_model for r in results]))
    wape_naive = float(np.mean([r.wape_naive for r in results]))
    coverage = float(np.mean([r.coverage for r in results]))
    calibrated = float(np.mean([r.calibrated_coverage for r in results]))
    lift = (wape_naive - wape_model) / max(wape_naive, 1e-9)

    print("-" * 62)
    print(
        f"{'mean':<12} {'':>6} {wape_model:>8.3f} {wape_naive:>8.3f} "
        f"{lift:>6.1%} {coverage:>7.1%} {calibrated:>7.1%}"
    )
    print()
    print("  pinball loss by quantile (lower is better):")
    for q in QUANTILES:
        print(f"    q{q}: {np.mean([r.pinball[q] for r in results]):.3f}")

    horizons = pd.concat([r.per_horizon for r in results]).groupby("horizon").mean()
    print()
    print("  by horizon:")
    for h, row in horizons.iterrows():
        bar = "█" * int(row["wape"] * 60)
        print(f"    +{int(h):>2}w  WAPE {row['wape']:.3f}  cover {row['coverage']:.0%}  {bar}")

    # Gates. These make the harness a check rather than a report nobody reads.
    print()
    ok = True
    if wape_model >= wape_naive:
        print(f"  FAIL: model WAPE {wape_model:.3f} does not beat seasonal-naive {wape_naive:.3f}")
        ok = False
    else:
        print(f"  PASS: beats seasonal-naive by {lift:.1%}")

    # Coverage is checked as a band, and checked on the *calibrated* interval,
    # because that is the one the API serves. Over-covering is a real fault
    # too: it means the intervals are wider than the data warrants and the
    # forecast is claiming less precision than it has.
    low, high = nominal - 0.15, nominal + 0.15
    if low <= calibrated <= high:
        print(
            f"  PASS: {nominal:.0%} interval covers {calibrated:.1%} after "
            f"conformal calibration ({coverage:.1%} raw, within ±15pp of nominal)"
        )
    else:
        print(
            f"  FAIL: {nominal:.0%} interval covers {calibrated:.1%} after "
            f"calibration — outside {low:.0%}–{high:.0%}"
        )
        ok = False

    return 0 if ok else 1


def save_report(results: list[FoldResult], path) -> None:
    """Write the scores to JSON so the dashboard can display measured accuracy."""
    import json

    payload = {
        "folds": [
            {
                "cutoff": str(r.cutoff.date()),
                "n_rows": r.n_rows,
                "wape_model": r.wape_model,
                "wape_naive": r.wape_naive,
                "coverage": r.coverage,
                "calibrated_coverage": r.calibrated_coverage,
                "conformal_offset": r.conformal_offset,
                "pinball": {str(q): v for q, v in r.pinball.items()},
            }
            for r in results
        ],
        "summary": {
            "wape_model": float(np.mean([r.wape_model for r in results])),
            "wape_naive": float(np.mean([r.wape_naive for r in results])),
            "coverage": float(np.mean([r.coverage for r in results])),
            "calibrated_coverage": float(np.mean([r.calibrated_coverage for r in results])),
            "conformal_offset": float(np.mean([r.conformal_offset for r in results])),
            "nominal_coverage": nominal_coverage(),
            "n_folds": len(results),
        },
        "per_horizon": (
            pd.concat([r.per_horizon for r in results])
            .groupby("horizon")
            .mean()
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rolling-origin backtest.")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--jobs", type=int, default=1, help="evaluate folds in this many parallel processes"
    )
    parser.add_argument(
        "--no-calibrate", action="store_true", help="report the raw quantile interval, uncalibrated"
    )
    args = parser.parse_args()

    features = load_feature_table()
    print(
        f"Backtesting {features.groupby(KEYS).ngroups} series, "
        f"{args.folds} folds, horizon {args.horizon}w, jobs {args.jobs}"
    )

    results = run_backtest(
        features,
        n_folds=args.folds,
        horizon=args.horizon,
        step_weeks=args.step,
        rounds=args.rounds,
        seed=args.seed,
        calibrate=not args.no_calibrate,
        jobs=args.jobs,
    )
    code = report(results, nominal_coverage())
    if results:
        save_report(results, FEATURE_DIR / "backtest_report.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
