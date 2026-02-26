"""
Run the whole pipeline in order: generate → ingest → link → features → train → ensemble.

One command, so "how do I run this?" has a one-line answer. Each step prints
what it produced; any step raising stops the run, because a pipeline that
carries on after a failed stage produces output that looks finished and isn't.

Everything is written under `DATA_DIR`, which defaults to `data/` at the repo
root and is git-ignored. The committed CSVs in `examples/sample_data/` are
input *schema* examples and are never written to — see that directory's README
for why that separation exists.

Usage:
    python -m scripts.run_pipeline --weeks 130 --horizon 12
"""

from __future__ import annotations

import argparse
import sys
import time

from src.config import DATA_DIR, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=130)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-generate", action="store_true", help="use the CSVs already in DATA_DIR"
    )
    parser.add_argument(
        "--no-calibrate", action="store_true", help="skip conformal interval calibration"
    )
    args = parser.parse_args()

    ensure_dirs()
    started = time.perf_counter()

    def step(n: int, name: str) -> None:
        print(f"\n── {n}. {name} " + "─" * (46 - len(name)))

    if not args.skip_generate:
        step(1, "generate sample data")
        from scripts import generate_sample_data

        argv = sys.argv
        sys.argv = ["gen", "--weeks", str(args.weeks), "--seed", str(args.seed)]
        try:
            generate_sample_data.main()
        finally:
            sys.argv = argv

    step(2, "ingest raw CSVs")
    from src.ingest.ingest import main as ingest

    ingest(str(DATA_DIR / "credit.csv"), str(DATA_DIR / "panel.csv"))

    step(3, "link panel to transactions")
    from src.linking.link_panel_credit import run as link

    link()

    step(4, "build features")
    from src.features.build_features import build

    build()

    step(5, "train quantile models")
    from src.models.train_lgb_quantile import main as train

    train(
        horizon=args.horizon,
        num_boost_round=args.rounds,
        seed=args.seed,
        calibrate=not args.no_calibrate,
    )

    step(6, "ensemble")
    from src.ensemble.ensemble_and_reconcile import ensemble

    ensemble()

    print(f"\nPipeline finished in {time.perf_counter() - started:.1f}s")
    print(f"Artifacts in {DATA_DIR}")


if __name__ == "__main__":
    main()
