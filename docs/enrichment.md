# Fight Enrichment Workflow

Some imported datasets include fight outcomes and stats but omit context such as weight class, event location, main-event status, title-fight status, and scheduled rounds.

## 1. Create a Template

Start from the normalized imports:

```bash
ufc-predict build-enrichment-template
```

This writes:

```text
data/raw/imports/fight_enrichment_template.csv
```

with one row per fight and these columns:

```text
fight_date,event,fighter_a,fighter_b,weight_class,event_location,main_event,title_fight,scheduled_rounds
```

## 2. Fill Missing Fields

Edit the template and fill fields you can verify from reliable pre-fight or official sources.

Use:

- `weight_class`: UFC division text, such as `Lightweight` or `Women's Strawweight`.
- `event_location`: location text, such as `Las Vegas, Nevada, USA`.
- `main_event`: `1` for the main event, `0` for other fights.
- `title_fight`: `1` for title fights, `0` for non-title fights.
- `scheduled_rounds`: `3` or `5`.

Save the completed file as:

```text
data/raw/imports/fight_enrichment.csv
```

## 3. Auto-Enrich First

For a practical first pass, run:

```bash
ufc-predict auto-enrich
```

This reads `data/raw/imports/fight_enrichment_template.csv` and writes:

```text
data/raw/imports/fight_enrichment.csv
```

Auto-enrichment is intentionally conservative:

- `main_event` is set to `1` when the event title appears to name the same fighter matchup, such as `Fighter A vs. Fighter B`.
- Other fights on an event are set to `main_event = 0` only when exactly one headliner was detected for that event.
- `title_fight` is set to `1` when `scheduled_rounds` is `5` or when the event name indicates title/championship language.
- `title_fight` is set to `0` for known three-round fights.
- `scheduled_rounds` stays from the template.
- `weight_class` remains `Unknown` unless supplied by a source.
- `event_location` remains blank unless supplied by a source.

## 4. Optional External Sources

Drop CSV files into:

```text
data/raw/enrichment_sources/
```

Then rerun:

```bash
ufc-predict auto-enrich
```

The command reads every `.csv` file in that folder and tries to merge known fields. Supported fight-level columns include:

```text
event,event_date,fighter_a,fighter_b,weight_class,event_location,main_event,title_fight,scheduled_rounds
```

It also accepts common aliases such as:

```text
event,date,location,weight_class,bout,main_event
```

For `bout`, use text like:

```text
Fighter A vs Fighter B
```

Matching is conservative:

- exact date plus exact fighter pair is strongest;
- exact event plus exact fighter pair is also accepted;
- event-only rows can supply `event_location` to all fights on that event;
- event-only fight-specific fields are applied only to single-fight events;
- blank source values do not overwrite known values;
- unmatched rows are counted in command output.

## 5. Apply and Validate

```bash
ufc-predict import-enrichment
ufc-predict validate-imports
ufc-predict import-csv
ufc-predict build-dataset
ufc-predict train
ufc-predict backtest
ufc-predict report
```

## 6. Check Coverage

Before or after applying enrichment:

```bash
ufc-predict enrichment-summary --file data/raw/imports/fight_enrichment.csv
```

The command reports the percent of fights with known `weight_class`, `event_location`, `main_event`, `title_fight`, and `scheduled_rounds`.

## Event-Level Enrichment

If an external file only has event-level context, `import-enrichment` can also read:

```text
event,event_date,weight_class,event_location,main_event,title_fight,scheduled_rounds
```

Event-level `event_location` is applied to every fight on the matched event. Fight-specific fields such as `weight_class`, `main_event`, `title_fight`, and `scheduled_rounds` are only applied when the event has exactly one fight in `fights.csv`; otherwise they are skipped with a warning. Use the fight-level template for accurate per-fight values on multi-fight events.

## Matching Rules

Fight-level enrichment matches by:

- `fight_date`
- `event`
- sorted pair of `fighter_a` and `fighter_b`

That means the fighter order in `fight_enrichment.csv` can be either orientation.
