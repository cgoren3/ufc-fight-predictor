# Adapting Kaggle or Third-Party UFC CSV Data

Many UFC datasets on Kaggle use their own column names, such as `R_fighter`, `B_fighter`, `Winner`, `R_SIG_STR.`, and `B_TD`. The adapter converts common fight-level CSV layouts into this project's import schema.

It also supports long-format fighter-performance files where each row is one fighter in one fight, such as:

```text
fight_fighter,opponent,fight_result,kd,str,td,sub,event,event_date,method,round,time
```

For that layout, mirrored rows are deduplicated into one `fights.csv` row per actual fight, while `fight_stats.csv` keeps one row per fighter performance.

Some long-format exports do not include `weight_class`, `event_location`, or `main_event`. The adapter leaves those values as `Unknown`, blank, or `0` instead of guessing. Add those fields later with `data/raw/imports/fight_enrichment.csv`.

Rows with missing or unparseable `event_date` are skipped with a warning because this project cannot build leakage-safe chronological features from undated fights.

## 1. Download Manually

Download the dataset from Kaggle or another provider using your browser or their CLI, then unzip it under:

```text
data/raw/downloads/
```

For example:

```text
data/raw/downloads/kaggle_ufc_dataset/
  kaggle_fights.csv
  fighters.csv
```

Kaggle licenses vary. Keep the original dataset files out of commits unless the license allows redistribution.

## 2. Inspect Columns

Before adapting, print every CSV and its columns:

```bash
ufc-predict dataset-columns --source data/raw/downloads/kaggle_ufc_dataset
```

Use this output to confirm the files contain fight-level columns such as date, fighters, winner, weight class, method, and stats.

## 3. Adapt Into Project Import Files

Run:

```bash
ufc-predict adapt-dataset --source data/raw/downloads/kaggle_ufc_dataset
```

The adapter writes:

```text
data/raw/imports/fights.csv
data/raw/imports/fighters.csv
data/raw/imports/fight_stats.csv
data/raw/imports/scorecards.csv    # only when found in schema
```

If the source folder already contains `fights.csv`, `fighters.csv`, and `fight_stats.csv` in this project's schema, the command copies them directly.

## 4. Validate and Import

```bash
ufc-predict validate-imports
ufc-predict import-csv
ufc-predict data-summary
```

Validation checks required files, required columns, parseable fight dates, winner names, fighter name consistency, two stats rows per fight when possible, and dataset size.

If you have fight context metadata, create:

```text
data/raw/imports/fight_enrichment.csv
```

with:

```text
fight_date,event,fighter_a,fighter_b,weight_class,event_location,main_event,title_fight,scheduled_rounds
```

Then run:

```bash
ufc-predict build-enrichment-template
ufc-predict import-enrichment
ufc-predict validate-imports
ufc-predict import-csv
```

## 5. Build and Model

```bash
ufc-predict build-dataset
ufc-predict train
ufc-predict backtest
```

The model is still only useful when the imported dataset is large enough. Fewer than 500 fights is too small for meaningful training, and fewer than 1000 fights limits backtest reliability.

## Conservative Mapping Rules

The adapter maps obvious aliases only. It supports common names including:

- `event`, `date`, `location`
- `R_fighter`, `B_fighter`, `red_fighter`, `blue_fighter`
- `Winner`, `winner`, `result`
- `weight_class`, `method`, `round`, `time`, `no_of_rounds`
- `stance`, `height`, `reach`, `dob`, `date_of_birth`
- `SIG_STR`, `TOTAL_STR`, `TD`, `SUB_ATT`, `REV`, `CTRL` with `R_`/`B_`, `red_`/`blue_`, or direct per-fighter columns
- Long-format columns: `fight_fighter`, `opponent`, `fight_result`, `kd`, `str`, `td`, `sub`, `event`, `event_date`, `method`, `round`, `time`

If the adapter cannot confidently identify the fight-level file or required fighter/date/winner columns, it exits with a clear error. Rename columns or provide normalized CSV files under `data/raw/imports/` when a dataset uses unusual naming.
