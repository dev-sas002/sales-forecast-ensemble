# Input schema examples

These two CSVs document **what the pipeline expects to be handed**. They are
eight transactions and four panel members — far too small to fit a forecaster
on, and deliberately so. Their job is to show the column names and types, not
to be a dataset.

Nothing in the pipeline writes here. `DATA_DIR` defaults to `data/` at the repo
root, which is git-ignored, so a run cannot modify anything committed and no
derived file can sit beside these, go stale and be read back later. Two
structural checks back that up:

- `src/models/registry.py` is the source of truth for which models exist. A
  `forecast_*.parquet` that no registered forecaster produces is reported as
  an orphan and excluded from the blend.
- `src/features/build_features.py` writes a `features.parquet.schema.json`
  sidecar naming the schema version and the exact feature set, and
  `load_feature_table()` refuses a table that does not match the registry that
  is compiled in.

## Generating a full dataset

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
