# Input schema examples

These two CSVs document **what the pipeline expects to be handed**. They are
eight transactions and four panel members — far too small to fit a forecaster
on, and deliberately so. Their job is to show the column names and types, not
to be a dataset.

Nothing in the pipeline writes here. `DATA_DIR` defaults to `data/` at the repo
root, which is git-ignored, so a run cannot modify anything committed.

That separation is new, and it is the fix for a specific class of bug. This
directory used to be `DATA_DIR` as well, so every run wrote derived parquet
alongside these files — and two of those derived files were committed:

- `features/forecast_deepar.parquet`, which **no code in this repository has
  ever written** (the DeepAR trainer is a stub that raises), yet
  `ensemble_and_reconcile` loaded it and blended 36 rows of numbers from a
  dead toy dataset into every local ensemble run;
- `features/features.parquet`, which still carried the pre-2026-09
  `rolling_4w_mean` column — the *leaking* trailing window that included the
  week it described.

Both are deleted. Both also now have a structural defence, because deleting a
file does not stop the next one appearing:

- `src/models/registry.py` is the source of truth for which models exist. A
  `forecast_*.parquet` that no registered forecaster produces is reported as
  an orphan and excluded from the blend.
- `src/features/build_features.py` writes a `features.parquet.schema.json`
  sidecar naming the schema version and the exact feature set, and
  `load_feature_table()` refuses a table that does not match the registry that
  is compiled in.

## Getting real data

```bash
python -m scripts.generate_sample_data --weeks 130 --seed 7
```

writes 236k seeded synthetic transactions into `DATA_DIR`. `make pipeline`
does that and everything after it.

## credit.csv

| column | type | notes |
|---|---|---|
| `txn_id` | int | unique per transaction; counted with `nunique` |
| `customer_id` | str | hashed with HMAC-SHA256 before linkage |
| `txn_date` | date | any pandas-parseable date; aggregated to Monday-anchored weeks |
| `sku` | str | series key |
| `region` | str | series key |
| `amount` | float | line value, summed to weekly `sales_dollars` |
| `price` | float | unit price, averaged to weekly `avg_price` |
| `quantity` | int | units, summed to the target `units_sold` |

## panel.csv

| column | type | notes |
|---|---|---|
| `panel_id` | str | panel membership id |
| `customer_id` | str | joined to credit on the salted hash |
| `age_group` | str | raking dimension |
| `region` | str | raking dimension |
| `income_bin` | str | raking dimension |
| `household_size` | int | unused today; carried through linkage |

`demographics.csv` is accepted by `src.ingest.ingest --demo_csv` and is not
used downstream.
