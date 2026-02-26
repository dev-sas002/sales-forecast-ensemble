"""
DeepAR (GluonTS) training stub. Not yet implemented.

**Deliberately not registered.** `src.models.registry.register` is a claim
that a model is implemented and its output can be believed, and this raises.
The stale `forecast_deepar.parquet` that used to be committed to this
repository was blended into every local ensemble run precisely because the
ensemble trusted a filename instead of the code — so leaving this unregistered
is what makes that impossible rather than merely unlikely.

To implement it: fit here, then add a class satisfying the `Forecaster`
protocol and decorate it with `@registry.register`, exactly as
`train_lgb_quantile.LightGBMQuantileForecaster` does. Nothing downstream needs
to change — the ensemble, the orphan check and `GET /models` all read the
registry. Add the `train_deepar` task back to `dags/forecast_dag.py` at the
same time.

Run from repo root: python -m src.models.train_deepar_gluonts --horizon 12
"""

from __future__ import annotations

import argparse


def main() -> None:
    raise NotImplementedError("train_deepar_gluonts is not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=12)
    parser.parse_args()
    main()
