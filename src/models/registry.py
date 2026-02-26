"""
The model registry: how a second learner gets added, and why the ensemble can
no longer blend a file nothing wrote.

The ensemble used to hold a literal dict of `{model name: filename}` and load
whatever was on disk:

    SOURCES = {"lgb": "forecast_lgb.parquet", "deepar": "forecast_deepar.parquet"}

Both halves of that are a problem. It is the wrong extension point — adding a
learner means editing a constant in a *downstream* module, which is exactly
backwards — and it trusts the filesystem. `forecast_deepar.parquet` was a
committed artifact from a toy dataset; the DeepAR trainer has never been more
than a stub that raises. So every local ensemble run blended 36 rows of stale
numbers for a model that does not exist, silently, and the resulting
`forecast_ensemble.parquet` was what the API served.

Here a forecast file is only trusted when a registered, *implemented*
forecaster claims it. A parquet matching the forecast naming convention that
no registered model produces is an orphan: reported by name, and excluded.
A blend can no longer depend on an artifact with no producer without someone
having asked for it in writing.

Adding a learner:

    @register
    class ProphetForecaster:
        name = "prophet"
        output_filename = "forecast_prophet.parquet"
        def fit(self, panel, quantiles): ...
        def predict(self, panel): ...          # keys + q{q}_prophet columns
        def forecast(self, features, horizon): ...

Nothing downstream changes: the ensemble, the orphan check and the pipeline
all read the registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

#: Every model writes `forecast_<name>.parquet` into FEATURE_DIR. The ensemble's
#: own output is excluded from orphan detection by name.
FORECAST_PREFIX = "forecast_"
FORECAST_SUFFIX = ".parquet"
ENSEMBLE_FILENAME = "forecast_ensemble.parquet"


@runtime_checkable
class Forecaster(Protocol):
    """
    What a learner has to provide to take part in the pipeline.

    Deliberately four members. Anything larger would be describing LightGBM
    rather than describing forecasting, and the next model would not fit.
    """

    #: Short identifier, e.g. "lgb". Also names the output file.
    name: str

    def fit(self, panel: pd.DataFrame, quantiles: tuple[float, ...]) -> Forecaster:
        """Fit on a supervised panel (one row per origin week x horizon)."""
        ...

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Predict for supervised rows. Returns keys plus `q{q}_{name}` columns."""
        ...

    def forecast(self, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Forecast `horizon` weeks forward from each series' last observed week."""
        ...


_REGISTRY: dict[str, Forecaster] = {}

#: Modules whose import registers a built-in forecaster. Registration is an
#: import side effect, which is convenient at the definition site and fragile
#: everywhere else: whether "lgb" was registered depended on whether some
#: unrelated module had already imported the trainer. `/models` reported an
#: empty list for exactly that reason. Every read of the registry now loads
#: these first, so the answer does not depend on import order.
_BUILTIN_MODULES = ("src.models.train_lgb_quantile",)
_loaded = False


def _ensure_loaded() -> None:
    """Import the built-in trainers once, lazily."""
    global _loaded
    if _loaded:
        return
    # Set before importing: the trainer imports this module to call register(),
    # and re-entering here would recurse.
    _loaded = True
    from importlib import import_module

    for module in _BUILTIN_MODULES:
        import_module(module)


def register(forecaster):
    """
    Register a forecaster class or instance. Usable as a decorator.

    Registration is a claim that the model is *implemented*. A stub that
    raises NotImplementedError must not be registered — the whole point of the
    registry is that its members are the models whose output can be believed.
    """
    instance = forecaster() if isinstance(forecaster, type) else forecaster
    name = instance.name
    if name in _REGISTRY:
        raise ValueError(f"a forecaster named {name!r} is already registered")
    _REGISTRY[name] = instance
    return forecaster


def registered() -> dict[str, Forecaster]:
    """Every implemented forecaster, by name."""
    _ensure_loaded()
    return dict(_REGISTRY)


def get(name: str) -> Forecaster:
    _ensure_loaded()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no forecaster named {name!r}; registered: {sorted(_REGISTRY)}") from None


def output_filename(name: str) -> str:
    """The parquet a model writes. Convention, not configuration."""
    return f"{FORECAST_PREFIX}{name}{FORECAST_SUFFIX}"


def expected_filenames() -> dict[str, str]:
    """name → filename for every registered model."""
    return {name: output_filename(name) for name in registered()}


def scan_forecasts(feature_dir: Path) -> tuple[dict[str, Path], list[Path]]:
    """
    Split the forecast files on disk into (produced by a registered model, orphans).

    An orphan is a `forecast_*.parquet` with no registered producer. It is not
    an error — a colleague may have dropped one there deliberately — but it is
    never blended without being asked for, and it is always named in the log.
    """
    known = {v: k for k, v in expected_filenames().items()}
    produced: dict[str, Path] = {}
    orphans: list[Path] = []

    if not feature_dir.exists():
        return produced, orphans

    for path in sorted(feature_dir.glob(f"{FORECAST_PREFIX}*{FORECAST_SUFFIX}")):
        if path.name == ENSEMBLE_FILENAME:
            continue
        model = known.get(path.name)
        if model is None:
            orphans.append(path)
        else:
            produced[model] = path
    return produced, orphans
