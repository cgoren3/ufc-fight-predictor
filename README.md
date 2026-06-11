# UFC Fight Predictor

Leakage-safe UFC fight outcome prediction project. It builds pre-fight matchup features, trains calibrated models, backtests them chronologically, and serves predictions through a CLI or Streamlit dashboard.

The project is designed around these sources:

- [UFCStats](https://ufcstats.com/) for event, fight, fighter, and fight-stat pages.
- [UFC Scorecards](https://www.ufc.com/scorecards) for official judges' scorecards that can be manually downloaded/imported.
- [MMA Decisions](https://mmadecisions.com/) for optional manually curated decision, judge, fan/media, and disputed-decision data.
- [SportsDataIO MMA/UFC API](https://sportsdata.io/mma-ufc-api) as an optional paid source for schedules, odds, live data, results, and historical feeds.

## Install

```bash
cd ufc-fight-predictor
pip install -e ".[dev]"
```

Python 3.11+ is required.

## New Codespace Setup

Open the repository in GitHub Codespaces. The devcontainer uses Python 3.11 and automatically runs:

```bash
python -m pip install --upgrade pip && python -m pip install -e ".[dev]"
```

After setup, the `ufc-predict` CLI should be available in the Codespace terminal:

```bash
ufc-predict --help
python -m pytest
```

## Data Layout

```text
data/
  raw/          # UFCStats CSV output and cache
  processed/    # feature datasets and backtest outputs
  external/     # manual scorecards, odds, injuries, camp notes
models/         # trained model bundle and metadata
```

## Ingest Data

Default ingestion is intentionally conservative and resumable.

```bash
ufc-predict ingest-ufcstats
```

For a small smoke test:

```bash
ufc-predict ingest-ufcstats --max-events 3
```

To also parse fight-detail pages and fighter profiles:

```bash
ufc-predict ingest-ufcstats --include-details
```

The scraper uses a user-agent header, local HTML cache, request retries, a configurable delay, and a resume file. It should not be used aggressively. If live UFCStats ingestion cannot produce valid fight rows, the command prints a clear warning and, by default, writes the bundled sample/dev data so the MVP pipeline can still run. Use strict mode when you do not want this fallback:

```bash
ufc-predict ingest-ufcstats --no-sample-on-failure
```

If a stale resume file or bad cached page is suspected:

```bash
ufc-predict ingest-ufcstats --ignore-resume
```

To diagnose live access from Codespaces or another environment:

```bash
ufc-predict check-ufcstats
```

The check prints the requested URL, HTTP status code, exception type/message, body preview, whether cached data was used, and attempt count.

Strict live scraping exits with code 1 on network/site failures and prints diagnostics instead of reporting a bad CLI argument:

```bash
ufc-predict ingest-ufcstats --no-sample-on-failure
```

If live discovery of the completed-events page is blocked, manually download the UFCStats completed events HTML page and place it at `data/raw/manual/ufcstats_completed_events.html`, then run:

```bash
ufc-predict ingest-ufcstats --from-html data/raw/manual/ufcstats_completed_events.html
```

This uses the local HTML file to discover event links. Event-detail pages still need to be reachable unless you use CSV imports.

To import real raw CSV data instead of scraping, place `fights.csv`, and optionally `fighters.csv`, `fight_stats.csv`, and `events.csv`, under `data/raw/imports/`, then run:

```bash
ufc-predict validate-imports
ufc-predict import-csv
```

Imported CSV data is treated as real raw input, not sample data. The bundled sample data is only for development and is clearly labeled in command output.

To summarize the active raw data:

```bash
ufc-predict data-summary
```

The summary prints row counts for fights, fighters, fight stats, scorecards, unique fighters, fight-date range, and whether the active data source is `live scrape`, `csv import`, `manual html`, `sample`, or `unknown`.

## Offline CSV Import Schema

Codespaces may not be able to connect to UFCStats reliably. The recommended real-data path is to export/download historical data as CSV files under `data/raw/imports/`, then run:

```bash
ufc-predict validate-imports
ufc-predict import-csv
ufc-predict data-summary
ufc-predict build-dataset
```

Required: `data/raw/imports/fights.csv`

```text
fight_id,event_id,event_name,fight_date,event_location,
fighter_a,fighter_b,winner,method,finish_round,finish_time,
weight_class,scheduled_rounds,main_event,title_fight,
catchweight,missed_weight,short_notice_replacement,source_url
```

Minimum required columns are `fighter_a`, `fighter_b`, `fight_date`, and `winner`. Missing optional columns are created as blanks during import.

Recommended: `data/raw/imports/fighters.csv`

```text
name,stance,height_in,weight_lb,reach_in,date_of_birth,country,state,source_url
```

Recommended: `data/raw/imports/fight_stats.csv`

```text
fight_id,fighter,opponent,knockdowns,
sig_str_landed,sig_str_attempted,total_str_landed,total_str_attempted,
takedowns_landed,takedowns_attempted,submission_attempts,reversals,
control_seconds,head_landed,body_landed,leg_landed
```

Optional: `data/raw/imports/scorecards.csv`

```text
event,fight_date,fighter_a,fighter_b,judge,
round_1_a,round_1_b,round_2_a,round_2_b,round_3_a,round_3_b,
round_4_a,round_4_b,round_5_a,round_5_b,total_a,total_b,decision_type
```

`build-dataset` automatically imports `data/raw/imports/fights.csv` when present. It refuses to build from bundled sample data unless `--use-sample-data` is passed, and it warns when fewer than 500 training rows are produced.

Full schema files live in:

- `docs/fights_schema.csv`
- `docs/fighters_schema.csv`
- `docs/fight_stats_schema.csv`
- `docs/scorecards_schema.csv`

Additional import guidance lives in `docs/import_data.md`.

## Adapt Kaggle or Third-Party CSV Data

If you download a UFC dataset from Kaggle or another source, unzip it under `data/raw/downloads/` and inspect the columns first:

```bash
ufc-predict dataset-columns --source data/raw/downloads/kaggle_ufc_dataset
```

Then adapt common UFC/Kaggle column names into the required import schema:

```bash
ufc-predict adapt-dataset --source data/raw/downloads/kaggle_ufc_dataset
ufc-predict validate-imports
ufc-predict import-csv
ufc-predict data-summary
```

The adapter writes real import files under `data/raw/imports/`; it does not treat adapted data as sample data. It is conservative: if required fight/date/winner columns cannot be mapped confidently, it prints a clear error instead of guessing. Full guidance is in `docs/kaggle_import.md`.

The adapter also supports long-format fighter-performance CSVs with columns such as `fight_fighter`, `opponent`, `fight_result`, `kd`, `str`, `td`, `sub`, `event`, `event_date`, `method`, `round`, and `time`.

## Manual Data

Official scorecards can be imported from CSV with these columns:

```text
event,fight_date,fighter_a,fighter_b,judge,
round_1_a,round_1_b,round_2_a,round_2_b,round_3_a,round_3_b,
round_4_a,round_4_b,round_5_a,round_5_b,total_a,total_b,decision_type
```

```bash
ufc-predict load-scorecards data/external/scorecards.csv
```

Optional manual files can be placed in `data/external/` for injuries, short-notice flags, missed weight, camp changes, altitude, betting odds, media/fan disputed decisions, or MMA Decisions exports. Join them into `fights.csv` or the processed dataset using stable keys such as `event`, `fight_date`, `fighter_a`, and `fighter_b`.

SportsDataIO is optional. Add a key to `.env` when available:

```text
SPORTS_DATA_IO_API_KEY=...
```

The project runs without the key.

## Build Features

```bash
ufc-predict build-dataset
```

`build-dataset` validates that `data/raw/fights.csv` exists, has headers, and has at least one data row before pandas reads it. If the file is missing, empty, headers-only, or missing required columns, the command explains the problem and points you to ingestion/cache checks or sample data.

For a fully local development run:

```bash
ufc-predict build-dataset --use-sample-data
```

Every training row is a fighter matchup as of the fight date. For historical rows, features are computed only from fights with `fight_date` strictly before the target fight. Same-day, future, final-career, closing-result, and post-fight information are not used.

Feature groups include biographical data, career experience, recent form, striking, grappling, style matchup interactions, durability, finishing, scorecard/judging features, contextual flags, Elo, and recency-weighted trendlines.

## Train

```bash
ufc-predict train
```

The trainer uses a chronological split, not a random split. It trains:

- logistic regression baseline
- random forest
- XGBoost when installed, otherwise scikit-learn `HistGradientBoostingClassifier`
- calibrated ensemble average

Models, feature lists, preprocessing, and metadata are saved in `models/`.

## Backtest

```bash
ufc-predict backtest
```

Rolling backtest trains on all fights before each period, predicts the next period, and rolls forward. Metrics include accuracy, log loss, Brier score, ROC AUC, calibration curve data, expected calibration error, performance by confidence tier, year, weight class, main event, men/women when supplied, underdog performance when odds are supplied, and baselines such as better record, higher Elo, and betting favorite when available.

## Predict

```bash
ufc-predict predict --fighter-a "Islam Makhachev" --fighter-b "Charles Oliveira" --date 2026-10-01 --weight-class "Lightweight" --scheduled-rounds 5
```

Example output:

```json
{
  "fighter_a": "Fighter A",
  "fighter_b": "Fighter B",
  "predicted_winner": "Fighter A",
  "fighter_a_win_probability": 0.63,
  "fighter_b_win_probability": 0.37,
  "confidence_score": 0.63,
  "confidence_tier": "Medium",
  "top_factors_for_prediction": [
    "Fighter A has +7.4 Elo advantage",
    "Fighter A has stronger striking differential"
  ],
  "warning": "Prediction is not guaranteed. Confidence is based on calibrated historical performance."
}
```

Confidence uses calibrated probability:

- Low: 50% to 57%
- Medium: 57% to 65%
- High: 65%+

Thresholds can be changed in `src/ufc_predictor/config.py` or `.env`.

## Streamlit

```bash
streamlit run src/ufc_predictor/app/streamlit_app.py
```

The dashboard lets you select fighters, fight date, weight class, and scheduled rounds, then shows predicted winner, probability bars, confidence tier, top features, fighter comparison, recent form, and backtest performance.

## Leakage Policy

The most important rule: a row for a fight may only use information available before that fight date.

This means:

- No future fights in rolling stats.
- No final career totals.
- No post-fight stats from the fight being predicted.
- No closing result fields in pre-fight features.
- Betting odds are optional and should be timestamped; use only odds available before prediction time.
- Manual notes must be dated or deliberately scoped to pre-fight knowledge.

## Tests

```bash
pytest
```

Tests verify no future fight leakage, Elo update timing, safe fighter-order handling, probability normalization, confidence tiers, and missing-data prediction behavior.

## Accuracy

Predictions are probabilistic, not guarantees. MMA outcomes are noisy, datasets can be incomplete, and late-breaking information such as injuries, weight misses, opponent changes, and judging variance can materially affect results. Treat the output as a calibrated research signal, not betting advice.
