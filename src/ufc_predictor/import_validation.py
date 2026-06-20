from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.data_io import InputDataError, inspect_csv
from ufc_predictor.enrichment import (
    EnrichmentError,
    fight_identity_key,
    is_unknown_weight_class,
    normalize_enrichment_frame,
    validate_enrichment_frame,
)
from ufc_predictor.ingest.scorecards_loader import REQUIRED_COLUMNS as SCORECARD_COLUMNS
from ufc_predictor.ingest.ufcstats_scraper import FIGHT_STAT_COLUMNS, FIGHTER_COLUMNS, FIGHT_COLUMNS


MIN_MEANINGFUL_FIGHTS = 500
MIN_RELIABLE_BACKTEST_FIGHTS = 1000

REQUIRED_IMPORT_FILES = ["fights.csv", "fighters.csv", "fight_stats.csv"]
REQUIRED_FIGHT_COLUMNS = ["fighter_a", "fighter_b", "fight_date", "winner"]
REQUIRED_FIGHTER_COLUMNS = ["name"]
REQUIRED_FIGHT_STAT_COLUMNS = ["fight_id", "fighter", "opponent"]


@dataclass
class ImportValidationResult:
    import_dir: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    date_range: dict[str, str | None] = field(default_factory=lambda: {"start": None, "end": None})

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "import_dir": str(self.import_dir),
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "counts": self.counts,
            "date_range": self.date_range,
        }


def _read_csv_if_valid(path: Path, required_columns: list[str], label: str, result: ImportValidationResult) -> pd.DataFrame | None:
    try:
        inspect_csv(path, required_columns=required_columns, require_rows=True, label=label)
    except InputDataError as exc:
        result.errors.append(str(exc))
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        result.errors.append(f"Could not read {label} at {path}: {type(exc).__name__}: {exc}")
        return None


def _names_from_fights(fights: pd.DataFrame) -> set[str]:
    names: set[str] = set()
    for column in ["fighter_a", "fighter_b"]:
        if column in fights.columns:
            names.update(fights[column].dropna().astype(str).str.strip().loc[lambda s: s != ""].tolist())
    return names


def _numeric_location_values(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).str.strip()
    values = values.loc[values != ""]
    return values.loc[values.str.fullmatch(r"[-+]?\d+(\.\d+)?", na=False)].head(5).tolist()


def _validate_fight_context_fields(fights: pd.DataFrame, result: ImportValidationResult) -> None:
    if "main_event" in fights.columns:
        values = fights["main_event"].dropna().astype(str).str.strip()
        values = values.loc[values != ""]
        if not values.empty:
            numeric = pd.to_numeric(values, errors="coerce")
            invalid = values.loc[numeric.isna() | ~numeric.isin([0, 1])]
            if not invalid.empty:
                result.errors.append(f"main_event must contain only 0 or 1 values. Invalid sample: {invalid.head(5).tolist()}")
    if "title_fight" in fights.columns:
        values = fights["title_fight"].dropna().astype(str).str.strip()
        values = values.loc[values != ""]
        if not values.empty:
            numeric = pd.to_numeric(values, errors="coerce")
            invalid = values.loc[numeric.isna() | ~numeric.isin([0, 1])]
            if not invalid.empty:
                result.errors.append(f"title_fight must contain only 0 or 1 values. Invalid sample: {invalid.head(5).tolist()}")
    if "scheduled_rounds" in fights.columns:
        values = fights["scheduled_rounds"].dropna().astype(str).str.strip()
        values = values.loc[values != ""]
        if not values.empty:
            numeric = pd.to_numeric(values, errors="coerce")
            invalid = values.loc[numeric.isna() | ~numeric.isin([3, 5])]
            if not invalid.empty:
                result.errors.append(
                    f"scheduled_rounds must be 3 or 5 where known. Invalid sample: {invalid.head(5).tolist()}"
                )
    if "event_location" in fights.columns:
        numeric_locations = _numeric_location_values(fights["event_location"])
        if numeric_locations:
            result.errors.append(
                "event_location must be categorical location text, not numeric values. "
                f"Invalid sample: {numeric_locations}"
            )


def _validate_enrichment_applied(import_dir: Path, fights: pd.DataFrame, result: ImportValidationResult) -> None:
    enrichment_path = import_dir / "fight_enrichment.csv"
    if not enrichment_path.exists():
        return
    try:
        inspect_csv(enrichment_path, require_rows=True, label="fight_enrichment.csv")
        raw_enrichment = pd.read_csv(enrichment_path)
    except InputDataError as exc:
        result.errors.append(str(exc))
        return
    try:
        enrichment, _, warnings = normalize_enrichment_frame(fights, raw_enrichment)
        for warning in warnings:
            result.warnings.append(warning)
    except EnrichmentError as exc:
        result.errors.append(str(exc))
        return
    try:
        for warning in validate_enrichment_frame(enrichment):
            result.warnings.append(warning)
    except EnrichmentError as exc:
        result.errors.append(str(exc))
        return

    event_column = "event_name" if "event_name" in fights.columns else "event"
    fights_by_key = {fight_identity_key(row, event_column=event_column): row for _, row in fights.iterrows()}
    stale_rows = []
    for _, row in enrichment.iterrows():
        if is_unknown_weight_class(row.get("weight_class")):
            continue
        key = fight_identity_key(
            {
                "fight_date": row.get("fight_date"),
                "event": row.get("event"),
                "fighter_a": row.get("fighter_a"),
                "fighter_b": row.get("fighter_b"),
            },
            event_column="event",
        )
        fight_row = fights_by_key.get(key)
        if fight_row is not None and is_unknown_weight_class(fight_row.get("weight_class")):
            stale_rows.append(
                {
                    "fight_date": row.get("fight_date"),
                    "event": row.get("event"),
                    "fighter_a": row.get("fighter_a"),
                    "fighter_b": row.get("fighter_b"),
                }
            )
    if stale_rows:
        result.errors.append(
            "fight_enrichment.csv provides weight_class values that have not been applied to fights.csv. "
            f"Run `ufc-predict import-enrichment`. Sample rows: {stale_rows[:3]}"
        )


def validate_import_directory(import_dir: str | Path) -> ImportValidationResult:
    path = Path(import_dir)
    result = ImportValidationResult(import_dir=path)
    if not path.exists():
        result.errors.append(f"Import directory does not exist: {path}")
        return result

    for filename in REQUIRED_IMPORT_FILES:
        if not (path / filename).exists():
            result.errors.append(f"Missing required import file: {path / filename}")

    fights = _read_csv_if_valid(path / "fights.csv", REQUIRED_FIGHT_COLUMNS, "fights.csv", result)
    fighters = _read_csv_if_valid(path / "fighters.csv", REQUIRED_FIGHTER_COLUMNS, "fighters.csv", result)
    fight_stats = _read_csv_if_valid(path / "fight_stats.csv", REQUIRED_FIGHT_STAT_COLUMNS, "fight_stats.csv", result)
    scorecards = None
    if (path / "scorecards.csv").exists():
        scorecards = _read_csv_if_valid(path / "scorecards.csv", list(SCORECARD_COLUMNS), "scorecards.csv", result)

    if fights is not None:
        result.counts["fights"] = int(len(fights))
        missing_optional = [column for column in FIGHT_COLUMNS if column not in fights.columns]
        if missing_optional:
            result.warnings.append(f"fights.csv is missing optional columns that will be created as blanks: {', '.join(missing_optional)}")
        dates = pd.to_datetime(fights["fight_date"], errors="coerce")
        bad_dates = int(dates.isna().sum())
        if bad_dates:
            result.errors.append(f"fights.csv has {bad_dates} unparseable fight_date values.")
        else:
            result.date_range["start"] = dates.min().date().isoformat()
            result.date_range["end"] = dates.max().date().isoformat()

        winners = fights["winner"].fillna("").astype(str).str.strip()
        valid_winners = (winners == fights["fighter_a"].fillna("").astype(str).str.strip()) | (
            winners == fights["fighter_b"].fillna("").astype(str).str.strip()
        )
        invalid_winners = fights.loc[~valid_winners & (winners != ""), ["fighter_a", "fighter_b", "winner"]]
        if not invalid_winners.empty:
            sample = invalid_winners.head(3).to_dict(orient="records")
            result.errors.append(f"winner must match fighter_a or fighter_b. Invalid rows: {sample}")

        if len(fights) < MIN_MEANINGFUL_FIGHTS:
            result.warnings.append("Too small for meaningful model training")
        if len(fights) < MIN_RELIABLE_BACKTEST_FIGHTS:
            result.warnings.append("Backtest reliability will be limited")
        _validate_fight_context_fields(fights, result)
        _validate_enrichment_applied(path, fights, result)

    fight_names = _names_from_fights(fights) if fights is not None else set()
    if fighters is not None:
        result.counts["fighters"] = int(len(fighters))
        missing_optional = [column for column in FIGHTER_COLUMNS if column not in fighters.columns]
        if missing_optional:
            result.warnings.append(f"fighters.csv is missing optional columns: {', '.join(missing_optional)}")
        fighter_names = set(fighters["name"].dropna().astype(str).str.strip().loc[lambda s: s != ""].tolist())
        missing_fighters = sorted(fight_names - fighter_names)
        if missing_fighters:
            result.errors.append(f"fighters.csv is missing fighters referenced in fights.csv: {missing_fighters[:10]}")

    if fight_stats is not None:
        result.counts["fight_stats"] = int(len(fight_stats))
        missing_optional = [column for column in FIGHT_STAT_COLUMNS if column not in fight_stats.columns]
        if missing_optional:
            result.warnings.append(f"fight_stats.csv is missing optional columns: {', '.join(missing_optional)}")
        if fights is not None and "fight_id" in fights.columns:
            fight_ids = set(pd.to_numeric(fights["fight_id"], errors="coerce").dropna().astype(int).tolist())
            stat_ids = pd.to_numeric(fight_stats["fight_id"], errors="coerce")
            unknown_ids = sorted(set(stat_ids.dropna().astype(int).tolist()) - fight_ids)
            if unknown_ids:
                result.errors.append(f"fight_stats.csv references fight_id values not present in fights.csv: {unknown_ids[:10]}")
            stats_per_fight = stat_ids.dropna().astype(int).value_counts()
            missing_two_rows = sorted([int(fight_id) for fight_id in fight_ids if stats_per_fight.get(fight_id, 0) != 2])
            if missing_two_rows:
                result.warnings.append(
                    "fight_stats.csv should usually have two rows per fight. "
                    f"Nonconforming fight_id values: {missing_two_rows[:10]}"
                )
        stat_names = set()
        for column in ["fighter", "opponent"]:
            if column in fight_stats.columns:
                stat_names.update(fight_stats[column].dropna().astype(str).str.strip().loc[lambda s: s != ""].tolist())
        unknown_stat_names = sorted(stat_names - fight_names)
        if fights is not None and unknown_stat_names:
            result.errors.append(f"fight_stats.csv references names not present in fights.csv: {unknown_stat_names[:10]}")

    if scorecards is not None:
        result.counts["scorecards"] = int(len(scorecards))

    return result
