"""
Everything the dashboard needs to know, computed without importing Streamlit.

The dashboard was 615 lines with the loading, the derivation and the drawing
interleaved — `pd.read_parquet` inside a `with tab3:` block, a linkage rate
computed inline between two `st.metric` calls. That shape has two costs. None
of the logic can be tested without a browser, and every UI change risks a data
change, because they are the same lines.

So the split is: this module answers questions, `streamlit_demo.py` draws the
answers. Nothing here imports `streamlit`, which is the property that keeps
the split honest — it is checked by a test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.features.registry import describe as describe_features
from src.models import registry
from src.quality.anomalies import detect, summarise


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """One row of the sidebar's pipeline status."""

    name: str
    done: bool
    detail: str = ""


@dataclass
class DashboardData:
    """
    Whatever the pipeline has produced so far.

    Every field is optional on purpose: the dashboard must render something
    useful against a half-run pipeline, and "features exist but no forecast
    yet" is a normal state during a first run, not an error.
    """

    credit: pd.DataFrame | None = None
    panel: pd.DataFrame | None = None
    linked: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    forecast: pd.DataFrame | None = None
    backtest: dict | None = None
    orphan_forecasts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return any(df is not None for df in (self.credit, self.panel, self.linked, self.features))


def _read(path: Path, errors: list[str], columns: list[str] | None = None):
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {exc}")
        return None


def load_dashboard_data(landing_dir: Path, feature_dir: Path) -> DashboardData:
    """Read what exists; record what failed; never raise."""
    errors: list[str] = []

    data = DashboardData(
        credit=_read(landing_dir / "credit_txn.parquet", errors),
        panel=_read(landing_dir / "panel.parquet", errors),
        linked=_read(landing_dir / "linked_panel_credit.parquet", errors),
        features=_read(feature_dir / "features.parquet", errors),
        errors=errors,
    )

    produced, orphans = registry.scan_forecasts(feature_dir)
    data.orphan_forecasts = [p.name for p in orphans]

    ensemble = feature_dir / registry.ENSEMBLE_FILENAME
    if ensemble.exists():
        data.forecast = _read(ensemble, errors)
    elif produced:
        # Fall back to a single registered model's output. Never to an orphan:
        # a file no code in this build writes is not a forecast this dashboard
        # will present as one.
        first = sorted(produced)[0]
        raw = _read(produced[first], errors)
        if raw is not None:
            data.forecast = raw.rename(
                columns={c: c.removesuffix(f"_{first}") for c in raw.columns}
            )

    report = feature_dir / "backtest_report.json"
    if report.exists():
        try:
            data.backtest = json.loads(report.read_text())
        except (OSError, ValueError) as exc:
            errors.append(f"backtest_report.json: {exc}")

    return data


def pipeline_stages(data: DashboardData) -> list[PipelineStage]:
    """The sidebar checklist, with a number beside each done stage."""
    return [
        PipelineStage(
            "Ingest",
            data.credit is not None and data.panel is not None,
            f"{len(data.credit):,} transactions" if data.credit is not None else "",
        ),
        PipelineStage(
            "Link",
            data.linked is not None,
            f"{len(data.linked):,} linked rows" if data.linked is not None else "",
        ),
        PipelineStage(
            "Features",
            data.features is not None,
            f"{len(data.features):,} series-weeks" if data.features is not None else "",
        ),
        PipelineStage(
            "Forecast",
            data.forecast is not None,
            f"{len(data.forecast):,} forecast rows" if data.forecast is not None else "",
        ),
        PipelineStage(
            "Backtest",
            data.backtest is not None,
            (f"{data.backtest['summary']['n_folds']} folds" if data.backtest else "not scored"),
        ),
    ]


def linkage_rate(data: DashboardData) -> float | None:
    """Share of credit transactions matched to a panel member."""
    if data.linked is None or data.credit is None or len(data.credit) == 0:
        return None
    return len(data.linked) / len(data.credit)


def feature_glossary(features: pd.DataFrame) -> pd.DataFrame:
    """
    The feature table's columns with their registry descriptions.

    Read from `src.features.registry` rather than a dict maintained here, so
    a feature added tomorrow is documented in the UI automatically instead of
    appearing as a bare column name.
    """
    described = describe_features()
    rows = [
        {"feature": col, "description": described[col]}
        for col in features.columns
        if col in described
    ]
    return pd.DataFrame(rows, columns=["feature", "description"])


@dataclass(frozen=True, slots=True)
class ForecastView:
    """One series' history and forecast, ready to plot."""

    sku: str
    region: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    anomalies: pd.DataFrame

    @property
    def total(self) -> float:
        return float(self.forecast["q0.5"].sum())

    @property
    def mean_weekly(self) -> float:
        return float(self.forecast["q0.5"].mean()) if len(self.forecast) else 0.0

    @property
    def recent_mean(self) -> float | None:
        if self.history.empty:
            return None
        return float(self.history["units_sold"].tail(8).mean())

    @property
    def change_vs_recent(self) -> float | None:
        recent = self.recent_mean
        if recent is None or abs(recent) < 1e-9:
            return None
        return (self.mean_weekly - recent) / recent


def build_forecast_view(
    data: DashboardData, sku: str, region: str, history_weeks: int = 52
) -> ForecastView | None:
    """Slice one series out of the loaded tables. None if it is not present."""
    if data.forecast is None:
        return None

    forecast = data.forecast[
        (data.forecast["sku"] == sku) & (data.forecast["region"] == region)
    ].copy()
    if forecast.empty:
        return None
    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast = forecast.sort_values("date")

    history = pd.DataFrame(columns=["week", "units_sold"])
    anomalies = detect(history.assign(sku=sku, region=region).head(0))
    if data.features is not None:
        series = data.features[
            (data.features["sku"] == sku) & (data.features["region"] == region)
        ].copy()
        if not series.empty:
            series["week"] = pd.to_datetime(series["week"])
            series = series.sort_values("week")
            anomalies = detect(series)
            history = series.tail(history_weeks)

    return ForecastView(
        sku=sku, region=region, history=history, forecast=forecast, anomalies=anomalies
    )


def series_options(data: DashboardData) -> dict[str, list[str]]:
    """sku → the regions that have a forecast, so the pickers cannot pick a 404."""
    if data.forecast is None:
        return {}
    pairs = data.forecast[["sku", "region"]].drop_duplicates()
    return {
        str(sku): sorted(group["region"].astype(str).unique())
        for sku, group in pairs.groupby("sku", sort=True)
    }


def quality_summary(features: pd.DataFrame | None) -> tuple[dict[str, int], pd.DataFrame]:
    """Anomaly counts and the findings themselves, over the whole catalogue."""
    if features is None or features.empty:
        empty = detect(pd.DataFrame(columns=["sku", "region", "week", "units_sold"]))
        return summarise(empty), empty
    found = detect(features)
    return summarise(found), found


def backtest_frames(report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-horizon, per-fold) tables from a saved backtest report."""
    horizons = pd.DataFrame(report.get("per_horizon", []))
    folds = pd.DataFrame(report.get("folds", []))
    if not folds.empty:
        folds["improvement"] = (folds["wape_naive"] - folds["wape_model"]) / folds["wape_naive"]
    return horizons, folds
