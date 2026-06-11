# Importing Real UFC CSV Data

Codespaces may not be able to connect to UFCStats, so real historical data should be imported from local CSV files.

Place files here:

```text
data/raw/imports/
  fights.csv
  fighters.csv
  fight_stats.csv
  scorecards.csv        # optional
```

Then run:

```bash
ufc-predict validate-imports
ufc-predict import-csv
ufc-predict data-summary
ufc-predict build-dataset
```

`fights.csv`, `fighters.csv`, and `fight_stats.csv` are required for the real-data import workflow. `scorecards.csv` is optional.

Schema reference files:

- `docs/fights_schema.csv`
- `docs/fighters_schema.csv`
- `docs/fight_stats_schema.csv`
- `docs/scorecards_schema.csv`

Validation checks:

- Required files exist.
- Required columns exist.
- Fight dates parse.
- Winner matches `fighter_a` or `fighter_b`.
- Fighter names are consistent across fights, fighters, and fight stats.
- Fight stats usually have two rows per fight.
- The dataset has enough rows for modeling.

Dataset size guidance:

- Fewer than 500 fights: too small for meaningful model training.
- Fewer than 1000 fights: backtest reliability will be limited.

The import workflow treats these CSVs as real data. The bundled sample/dev data is only used when `--use-sample-data` is explicitly passed.
