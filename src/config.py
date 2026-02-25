"""
Paths and settings, resolved from the environment.

Importing this module has no side effects. Every module and every test
imports it, so a mkdir here could fail with PermissionError on a read-only
path and take down test collection. Creating directories is an action; call
ensure_dirs() where you intend it.

**Inputs and outputs live in different places.** `DATA_DIR` is the pipeline's
working directory and is git-ignored; `EXAMPLES_DIR` holds the two committed
CSVs that document the expected input schema and is never written to, so no
derived artifact can be committed beside them, go stale and be read back.
"""

from __future__ import annotations

import os
from pathlib import Path

# ROOT is the repo root (the directory containing src/). Override via PROJECT_ROOT.
ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", str(ROOT / "data")))


#: Everything the pipeline writes. Git-ignored: it is derived, not source.
DATA_DIR = _data_dir()
RAW_DIR = DATA_DIR / "raw"
LANDING_DIR = DATA_DIR / "landing"
FEATURE_DIR = DATA_DIR / "features"
#: Trained boosters. Overridable so containers that share DATA_DIR on a volume
#: can share the model the seed job trained, too.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(ROOT / "artifacts" / "models")))

#: Committed input examples. Read-only by convention — the pipeline never
#: writes here, so nothing in it can go stale against the code.
EXAMPLES_DIR = ROOT / "examples" / "sample_data"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
SALT = os.environ.get("HASH_SALT", "CHANGE_ME_SECRET_SALT")

#: Quantiles the pipeline forecasts, in ascending order. Everything downstream
#: derives its column names from this rather than hardcoding "q0.1" in a dozen
#: places, so adding a quantile is one edit.
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def quantile_cols(suffix: str = "") -> list[str]:
    """Column names for the configured quantiles, e.g. ['q0.1', 'q0.5', 'q0.9']."""
    tail = f"_{suffix}" if suffix else ""
    return [f"q{q}{tail}" for q in QUANTILES]


def ensure_dirs() -> None:
    """Create the data directories. Call this from entry points, not at import."""
    for path in (RAW_DIR, LANDING_DIR, FEATURE_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
