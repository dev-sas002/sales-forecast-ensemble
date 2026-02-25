"""
Ingest raw CSVs (credit card, panel, demographics) and write Parquet landing files.
Usage:
  python -m src.ingest.ingest --credit_csv path/to/credit.csv --panel_csv path/to/panel.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import LANDING_DIR
from src.utils.validation import check_required_columns

# Required columns per README / downstream steps
CREDIT_COLS = [
    "txn_id",
    "customer_id",
    "txn_date",
    "sku",
    "region",
    "amount",
    "price",
    "quantity",
]
PANEL_COLS = [
    "panel_id",
    "customer_id",
    "age_group",
    "region",
    "income_bin",
    "household_size",
]


class IngestError(Exception):
    """Raised when ingest read/write or validation fails."""


def read_csv(path: str | Path) -> pd.DataFrame:
    """Thin wrapper: read CSV from path. Raises IngestError on file or parse errors."""
    path = Path(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        raise IngestError(f"File not found: {path}") from None
    except PermissionError:
        raise IngestError(f"Permission denied: {path}") from None
    except pd.errors.EmptyDataError:
        raise IngestError(f"Empty or invalid CSV: {path}") from None


def write_parquet(df: pd.DataFrame, name: str) -> None:
    """Thin wrapper: write DataFrame to LANDING_DIR / f'{name}.parquet'. Raises IngestError on failure."""
    LANDING_DIR.mkdir(parents=True, exist_ok=True)
    out = LANDING_DIR / f"{name}.parquet"
    try:
        df.to_parquet(out, index=False)
    except PermissionError:
        raise IngestError(f"Permission denied writing: {out}") from None
    except OSError as e:
        raise IngestError(f"Failed to write {out}: {e}") from e
    print(f"Wrote {out}")


def main(credit_csv: str, panel_csv: str, demo_csv: str | None = None) -> None:
    """Load credit and panel CSVs, validate required columns, write Parquet. Optionally load demographics."""
    credit = read_csv(credit_csv)
    check_required_columns(credit, CREDIT_COLS, "credit_csv")
    write_parquet(credit, "credit_txn")

    panel = read_csv(panel_csv)
    check_required_columns(panel, PANEL_COLS, "panel_csv")
    write_parquet(panel, "panel")

    if demo_csv:
        demo = read_csv(demo_csv)
        write_parquet(demo, "demographics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--credit_csv", required=True)
    parser.add_argument("--panel_csv", required=True)
    parser.add_argument("--demo_csv", required=False)
    args = parser.parse_args()
    main(args.credit_csv, args.panel_csv, args.demo_csv)
