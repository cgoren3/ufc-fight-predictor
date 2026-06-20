# Importing Real UFC CSV Data

Codespaces may not be able to connect to UFCStats, so real historical data should be imported from local CSV files.

Place files here:

```text
data/raw/imports/
  fights.csv
  fighters.csv
  fight_stats.csv
  fight_enrichment.csv # optional fight context overlay
  odds.csv             # optional market snapshots
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
- `docs/fight_enrichment_schema.csv`
- `docs/odds_schema.csv`

Validation checks:

- Required files exist.
- Required columns exist.
- Fight dates parse.
- Winner matches `fighter_a` or `fighter_b`.
- Fighter names are consistent across fights, fighters, and fight stats.
- Fight stats usually have two rows per fight.
- The dataset has enough rows for modeling.
- `main_event` and `title_fight` are 0/1 when supplied.
- `scheduled_rounds` is 3 or 5 when supplied.
- `event_location` is location/category text, not a numeric code.
- If `fight_enrichment.csv` provides a known `weight_class`, that value has been applied to `fights.csv`.

Dataset size guidance:

- Fewer than 500 fights: too small for meaningful model training.
- Fewer than 1000 fights: backtest reliability will be limited.

The import workflow treats these CSVs as real data. The bundled sample/dev data is only used when `--use-sample-data` is explicitly passed.

## Optional Fight Enrichment

Some third-party datasets have fight outcomes and stats but omit fight context such as weight class, location, and main-event status. Add those fields in:

```text
data/raw/imports/fight_enrichment.csv
```

Required columns:

```text
fight_date,event,fighter_a,fighter_b,weight_class,event_location,main_event,title_fight,scheduled_rounds
```

The enrichment join is keyed by `fight_date`, `event`, and the sorted fighter pair, so fighter order can be either orientation. Example:

```csv
fight_date,event,fighter_a,fighter_b,weight_class,event_location,main_event,title_fight,scheduled_rounds
2020-01-18,UFC 246: McGregor vs. Cowboy,Conor McGregor,Donald Cerrone,Welterweight,"Las Vegas, Nevada, USA",1,0,5
```

Apply it before importing normalized raw data:

```bash
ufc-predict build-enrichment-template
ufc-predict import-enrichment
ufc-predict validate-imports
ufc-predict import-csv
```

If validation says enrichment values have not been applied, rerun `ufc-predict import-enrichment`.

Use `ufc-predict enrichment-summary` to check coverage. See `docs/enrichment.md` for the full workflow and event-level enrichment rules.

## Optional Odds

Place pre-fight odds snapshots in:

```text
data/raw/imports/odds.csv
```

Schema:

```text
fight_date,fighter_a,fighter_b,sportsbook,fighter_a_odds,fighter_b_odds,timestamp
```

Example:

```csv
fight_date,fighter_a,fighter_b,sportsbook,fighter_a_odds,fighter_b_odds,timestamp
2020-01-18,Conor McGregor,Donald Cerrone,ExampleBook,-300,240,2020-01-17T12:00:00Z
```

Import odds separately:

```bash
ufc-predict import-odds
```

Odds are converted to implied probabilities for model-vs-market analysis only. The project does not recommend bets.

## Optional Scorecards

Place official or manually normalized scorecard rows in:

```text
data/raw/imports/scorecards.csv
```

Schema:

```text
event,fight_date,fighter_a,fighter_b,judge,round_1_a,round_1_b,round_2_a,round_2_b,round_3_a,round_3_b,round_4_a,round_4_b,round_5_a,round_5_b,total_a,total_b,decision_type
```

`ufc-predict import-csv` copies this file into `data/raw/scorecards.csv` when present.
