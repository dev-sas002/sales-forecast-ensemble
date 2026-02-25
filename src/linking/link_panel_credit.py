"""
Link panel members to credit transactions using hashed customer IDs.
Input: landing/parquet files
Output: landing/linked_panel_credit.parquet
"""

from __future__ import annotations

import pandas as pd

from src.config import LANDING_DIR
from src.utils.hashing import hash_id
from src.utils.validation import check_required_columns

REQUIRED_LINK_COLS = ["customer_id"]


def _canonical_region(linked: pd.DataFrame) -> pd.DataFrame:
    """Set canonical 'region' from credit or panel if missing."""
    if "region" in linked.columns:
        return linked
    if "region_credit" in linked.columns or "region_panel" in linked.columns:
        linked = linked.copy()
        linked["region"] = linked.get("region_credit", pd.Series(index=linked.index)).fillna(
            linked.get("region_panel", pd.Series(index=linked.index))
        )
    return linked


def run() -> None:
    credit = pd.read_parquet(LANDING_DIR / "credit_txn.parquet")
    panel = pd.read_parquet(LANDING_DIR / "panel.parquet")
    check_required_columns(credit, REQUIRED_LINK_COLS, "credit_txn")
    check_required_columns(panel, REQUIRED_LINK_COLS, "panel")

    credit["cust_hash"] = credit["customer_id"].astype(str).apply(hash_id)
    panel["cust_hash"] = panel["customer_id"].astype(str).apply(hash_id)

    # Inner join: keep matched records only (simplified linkage)
    linked = panel.merge(credit, on="cust_hash", suffixes=("_panel", "_credit"))

    linked = _canonical_region(linked)
    linked.to_parquet(LANDING_DIR / "linked_panel_credit.parquet", index=False)
    print("Linked records:", len(linked))


if __name__ == "__main__":
    run()
