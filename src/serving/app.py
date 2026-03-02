"""
FastAPI surface over the forecast artifacts.

Deliberately thin. Every handler here does three things — validate, call one
service, shape the response — and nothing else. Reading parquet lives in
:mod:`src.serving.store`, anomaly rules live in :mod:`src.quality.anomalies`,
explanation lives in :mod:`src.explain`. That is not ceremony: the previous
version did its own `pd.read_parquet` and its own filtering inside the route
function, which is why the caching fix had to be a rewrite of the endpoint
rather than a change to one module, and why none of that logic could be
exercised without an HTTP client.

Dependencies point inward. The store knows nothing about FastAPI; this module
knows nothing about parquet.
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import FEATURE_DIR, quantile_cols
from src.explain import build_facts, explain, narrator_status
from src.models import registry
from src.quality.anomalies import detect, summarise
from src.serving import store

app = FastAPI(
    title="SKU Forecast API",
    description=(
        "Quantile demand forecasts for SKU x region weekly sales, with input "
        "anomaly flags and plain-language explanation."
    ),
    version="0.2.0",
)

#: Derived from config.QUANTILES rather than hardcoded, so adding a quantile
#: to the pipeline does not silently leave the API serving the old three.
FORECAST_COLS = quantile_cols()

#: Upper bound on a request. The pipeline writes 12 weeks by default; this is
#: a sanity cap so a caller cannot ask for an unbounded slice.
MAX_HORIZON_WEEKS = 52


class Query(BaseModel):
    sku: str
    region: str
    # Unvalidated, a negative horizon reached `df.head(-n)`, which drops the
    # *last* n rows and returns a forecast rather than an error.
    horizon_weeks: int = Field(default=12, ge=1, le=MAX_HORIZON_WEEKS)


def _normalize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical quantile names, whichever model wrote the file."""
    return store.normalize_forecast_columns(df, FORECAST_COLS)


def _table() -> store.ForecastTable:
    """The cached forecast table, or the right HTTP error."""
    try:
        return store.load_forecasts(FEATURE_DIR)
    except store.ForecastNotAvailable as e:
        raise HTTPException(
            status_code=404,
            detail="No forecast data available. Please run the pipeline to generate forecasts.",
        ) from e
    except store.ForecastSchemaError as e:
        # A KeyError here would surface as a bare 500 with a pandas traceback;
        # the actual problem is a forecast file written by an older pipeline.
        raise HTTPException(
            status_code=500,
            detail=f"Forecast file is missing quantile columns {FORECAST_COLS}. "
            "Re-run the pipeline to regenerate it.",
        ) from e


def _series_rows(q: Query) -> pd.DataFrame:
    rows = _table().slice(q.sku, q.region, q.horizon_weeks)
    if rows is None or rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for sku={q.sku!r} and region={q.region!r}.",
        )
    return rows


@app.get("/health")
async def health() -> dict:
    """Liveness check: returns status ok."""
    return {"status": "ok"}


@app.get("/models")
async def models() -> dict:
    """
    Which forecasters this build implements, and what is on disk.

    Exposed because the ensemble's composition used to be invisible: a stale
    `forecast_deepar.parquet` with no producer was being blended into every
    served forecast and nothing said so. Orphans are now named here.
    """
    _, orphans = registry.scan_forecasts(FEATURE_DIR)
    return {
        "registered": sorted(registry.registered()),
        "orphan_forecast_files": [p.name for p in orphans],
        "narrator": narrator_status(),
    }


@app.post("/forecast")
async def forecast(q: Query):
    """Return quantile forecasts (q0.1, q0.5, q0.9) for the given sku/region and horizon."""
    rows = _series_rows(q)
    return rows[["sku", "region", "date"] + FORECAST_COLS].to_dict(orient="records")


@app.post("/explain")
async def explain_forecast(q: Query):
    """
    The forecast plus a plain-language note about it.

    The facts are computed here; the wording is either written by Claude or
    rendered from a template, and `source` says which. Never 500s because a
    language model was unavailable — see :mod:`src.explain.narrative`.
    """
    rows = _series_rows(q)

    history = None
    features = FEATURE_DIR / "features.parquet"
    if features.exists():
        try:
            table = pd.read_parquet(
                features, columns=["sku", "region", "week", "units_sold", "avg_price"]
            )
            history = table[(table["sku"] == q.sku) & (table["region"] == q.region)]
        except (OSError, ValueError, KeyError):
            # An unreadable or older feature table costs the comparison
            # against recent trading, not the endpoint.
            history = None

    backtest = _backtest_summary()
    facts = build_facts(q.sku, q.region, rows, history=history, backtest=backtest)
    note = explain(facts)

    return {
        "sku": q.sku,
        "region": q.region,
        "explanation": note.text,
        "source": note.source,
        "model": note.model,
        "facts": facts.to_dict(),
    }


@app.get("/anomalies")
async def anomalies(sku: str | None = None, region: str | None = None, limit: int = 50):
    """
    Suspect weeks in the input data the model was fitted on.

    Stockouts, stopped feeds and price errors all look like demand to a
    forecaster. This is the check that they are looked at before the forecast
    is believed. See :mod:`src.quality.anomalies` for the rules.
    """
    path = FEATURE_DIR / "features.parquet"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No feature table available. Run the pipeline first.",
        )

    limit = max(1, min(limit, 500))
    table = pd.read_parquet(path, columns=["sku", "region", "week", "units_sold", "avg_price"])
    if sku:
        table = table[table["sku"] == sku]
    if region:
        table = table[table["region"] == region]
    if table.empty:
        return {"summary": summarise(detect(table.head(0))), "anomalies": []}

    found = detect(table)
    payload = found.head(limit).copy()
    if not payload.empty:
        payload["week"] = pd.to_datetime(payload["week"]).dt.strftime("%Y-%m-%d")
    return {
        "summary": summarise(found),
        "returned": len(payload),
        "anomalies": payload.to_dict(orient="records"),
    }


def _backtest_summary() -> dict | None:
    import json

    path = FEATURE_DIR / "backtest_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("summary")
    except (OSError, ValueError):
        return None
