"""
Generate a synthetic but realistic transaction dataset.

The repository shipped nine credit rows and four panel rows. You cannot fit a
forecaster on that, so nothing downstream could be demonstrated or measured —
the sample data was too small to reveal that the models were unimplemented.

What this generates has the structure a demand forecaster is supposed to find,
and no more: a per-SKU base rate, regional multipliers, a slow trend, annual
seasonality, promotional weeks that both cut price and lift volume, and
multiplicative noise. Nothing here is drawn from a distribution the model is
told about; the model has to recover the pattern from the series alone.

Deterministic given --seed, so a backtest score is reproducible and a change
in the score means a change in the code.

Usage:
    python -m scripts.generate_sample_data --weeks 130 --seed 7
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Eight products across four regions. Small enough to read in a table, large
# enough that per-series behaviour differs.
SKUS = {
    "SKU-1001": {"base": 140.0, "price": 4.50, "season": 0.35, "trend": +0.0022},
    "SKU-1002": {"base": 90.0, "price": 7.25, "season": 0.15, "trend": +0.0008},
    "SKU-1003": {"base": 220.0, "price": 2.10, "season": 0.05, "trend": -0.0011},
    "SKU-1004": {"base": 60.0, "price": 12.00, "season": 0.55, "trend": +0.0035},
    "SKU-2001": {"base": 175.0, "price": 3.75, "season": 0.25, "trend": +0.0004},
    "SKU-2002": {"base": 45.0, "price": 18.50, "season": 0.45, "trend": +0.0018},
    "SKU-2003": {"base": 310.0, "price": 1.60, "season": 0.10, "trend": -0.0006},
    "SKU-2004": {"base": 120.0, "price": 6.00, "season": 0.30, "trend": +0.0012},
}

REGIONS = {"NE": 1.25, "SE": 0.95, "MW": 1.05, "W": 1.15}

# Peak weeks differ per SKU so seasonality is not one shared national curve.
SEASON_PEAK_WEEK = {
    "SKU-1001": 48,
    "SKU-1002": 26,
    "SKU-1003": 30,
    "SKU-1004": 50,
    "SKU-2001": 22,
    "SKU-2002": 47,
    "SKU-2003": 33,
    "SKU-2004": 12,
}

AGE_GROUPS = ["18-29", "30-44", "45-59", "60+"]
INCOME_BINS = ["<40k", "40-75k", "75-120k", "120k+"]


def weekly_demand(rng, weeks: int, start: pd.Timestamp) -> pd.DataFrame:
    """Build the true weekly demand surface before it is broken into txns."""
    rows = []
    for sku, cfg in SKUS.items():
        peak = SEASON_PEAK_WEEK[sku]
        for region, region_mult in REGIONS.items():
            # A per-series level offset: two regions selling the same SKU are
            # not the same series with a constant applied.
            level = rng.uniform(0.85, 1.15)
            for w in range(weeks):
                date = start + pd.Timedelta(weeks=w)
                iso_week = date.isocalendar().week

                seasonal = 1.0 + cfg["season"] * math.cos(2 * math.pi * (iso_week - peak) / 52.0)
                trend = (1.0 + cfg["trend"]) ** w

                # Promotions: ~8% of weeks, price cut and a volume lift that is
                # larger than the discount — the elasticity the model can learn.
                on_promo = rng.random() < 0.08
                discount = rng.uniform(0.15, 0.35) if on_promo else 0.0
                promo_lift = 1.0 + (2.4 * discount if on_promo else 0.0)

                mean = cfg["base"] * region_mult * level * seasonal * trend * promo_lift
                # Multiplicative noise, then Poisson counts: variance grows with
                # the level, as unit sales actually do.
                noisy = mean * rng.lognormal(mean=0.0, sigma=0.12)
                units = int(rng.poisson(max(noisy, 1.0)))

                rows.append(
                    {
                        "week": date,
                        "sku": sku,
                        "region": region,
                        "units": units,
                        "price": round(cfg["price"] * (1.0 - discount), 2),
                        "on_promo": int(on_promo),
                    }
                )
    return pd.DataFrame(rows)


def explode_to_transactions(demand: pd.DataFrame, rng, customers) -> pd.DataFrame:
    """Break weekly units into individual basket-sized transactions."""
    records = []
    txn_id = 0
    by_region = {}
    for cid, region in customers:
        by_region.setdefault(region, []).append(cid)

    for row in demand.itertuples(index=False):
        remaining = row.units
        pool = by_region.get(row.region) or [c for c, _ in customers]
        while remaining > 0:
            qty = min(remaining, int(rng.integers(1, 6)))
            remaining -= qty
            txn_id += 1
            # Spread transactions across the days of the week.
            day_offset = int(rng.integers(0, 7))
            records.append(
                {
                    "txn_id": txn_id,
                    "customer_id": pool[int(rng.integers(0, len(pool)))],
                    "txn_date": (row.week + pd.Timedelta(days=day_offset)).date(),
                    "sku": row.sku,
                    "region": row.region,
                    "amount": round(qty * row.price, 2),
                    "price": row.price,
                    "quantity": qty,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=130, help="weeks of history (~2.5y)")
    parser.add_argument("--customers", type=int, default=1200)
    parser.add_argument("--panel-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from src.config import DATA_DIR

    out_dir = args.out or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    start = pd.Timestamp("2023-01-02")  # a Monday

    region_names = list(REGIONS)
    customers = [
        (f"C{i:05d}", region_names[int(rng.integers(0, len(region_names)))])
        for i in range(args.customers)
    ]

    demand = weekly_demand(rng, args.weeks, start)
    credit = explode_to_transactions(demand, rng, customers)

    # The panel is a *sample* of customers, which is the whole reason the
    # weighting step exists downstream.
    n_panel = int(len(customers) * args.panel_fraction)
    panel_ids = rng.choice(len(customers), size=n_panel, replace=False)
    panel = pd.DataFrame(
        [
            {
                "panel_id": f"P{i:05d}",
                "customer_id": customers[idx][0],
                "age_group": AGE_GROUPS[int(rng.integers(0, len(AGE_GROUPS)))],
                "region": customers[idx][1],
                "income_bin": INCOME_BINS[int(rng.integers(0, len(INCOME_BINS)))],
                "household_size": int(rng.integers(1, 6)),
            }
            for i, idx in enumerate(sorted(panel_ids))
        ]
    )

    credit.to_csv(out_dir / "credit.csv", index=False)
    panel.to_csv(out_dir / "panel.csv", index=False)
    # The ground truth, kept aside so a human can check what the model saw.
    demand.to_csv(out_dir / "true_weekly_demand.csv", index=False)

    print(f"  credit.csv  {len(credit):>7,} transactions")
    print(
        f"  panel.csv   {len(panel):>7,} panel members "
        f"({args.panel_fraction:.0%} of {args.customers:,} customers)"
    )
    print(
        f"  weeks       {args.weeks}  ({start.date()} → "
        f"{(start + pd.Timedelta(weeks=args.weeks - 1)).date()})"
    )
    print(f"  series      {len(SKUS)} SKUs x {len(REGIONS)} regions = {len(SKUS) * len(REGIONS)}")
    print(f"  → {out_dir}")


if __name__ == "__main__":
    main()
