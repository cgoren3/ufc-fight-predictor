from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.data_io import InputDataError, inspect_csv


ENRICHMENT_COLUMNS = [
    "fight_date",
    "event",
    "fighter_a",
    "fighter_b",
    "weight_class",
    "event_location",
    "main_event",
    "title_fight",
    "scheduled_rounds",
]
EVENT_ENRICHMENT_COLUMNS = [
    "event",
    "event_date",
    "weight_class",
    "event_location",
    "main_event",
    "title_fight",
    "scheduled_rounds",
]
ENRICHABLE_FIELDS = ["weight_class", "event_location", "main_event", "title_fight", "scheduled_rounds"]
UNKNOWN_WEIGHT_VALUES = {"", "unknown", "unk", "n/a", "na", "none", "nan"}
MISSING_ENRICHMENT_GUIDANCE = (
    "Run ufc-predict build-enrichment-template, fill in the missing columns, "
    "save it as data/raw/imports/fight_enrichment.csv, then rerun import-enrichment."
)


class EnrichmentError(InputDataError):
    """Raised when fight enrichment cannot be safely applied."""


@dataclass
class EnrichmentReport:
    fights_path: Path
    enrichment_path: Path
    output_path: Path
    fights_rows: int = 0
    enrichment_rows: int = 0
    matched_rows: int = 0
    unmatched_enrichment_rows: int = 0
    updated_fields: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source_format: str = "fight"

    def as_dict(self) -> dict[str, Any]:
        return {
            "fights_path": str(self.fights_path),
            "enrichment_path": str(self.enrichment_path),
            "output_path": str(self.output_path),
            "fights_rows": self.fights_rows,
            "enrichment_rows": self.enrichment_rows,
            "matched_rows": self.matched_rows,
            "unmatched_enrichment_rows": self.unmatched_enrichment_rows,
            "updated_fields": self.updated_fields,
            "warnings": self.warnings,
            "source_format": self.source_format,
        }


@dataclass
class EnrichmentSummary:
    path: Path
    total_fights: int
    known_weight_class_pct: float
    known_event_location_pct: float
    known_main_event_pct: float
    known_title_fight_pct: float
    known_scheduled_rounds_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "total_fights": self.total_fights,
            "known_weight_class_pct": self.known_weight_class_pct,
            "known_event_location_pct": self.known_event_location_pct,
            "known_main_event_pct": self.known_main_event_pct,
            "known_title_fight_pct": self.known_title_fight_pct,
            "known_scheduled_rounds_pct": self.known_scheduled_rounds_pct,
        }


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _key_text(value: Any) -> str:
    return _clean(value).lower()


def _date_key(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return ""
    return date.date().isoformat()


def _pair_key(fighter_a: Any, fighter_b: Any) -> str:
    return "|".join(sorted([_key_text(fighter_a), _key_text(fighter_b)]))


def fight_identity_key(row: pd.Series | dict[str, Any], event_column: str = "event_name") -> str:
    """Stable fight key used for safe enrichment joins."""

    event = row.get(event_column, row.get("event", ""))
    return "||".join(
        [
            _date_key(row.get("fight_date")),
            _key_text(event),
            _pair_key(row.get("fighter_a"), row.get("fighter_b")),
        ]
    )


def _enrichment_key(row: pd.Series | dict[str, Any]) -> str:
    return "||".join(
        [
            _date_key(row.get("fight_date")),
            _key_text(row.get("event")),
            _pair_key(row.get("fighter_a"), row.get("fighter_b")),
        ]
    )


def _event_key(event: Any, fight_date: Any) -> str:
    return "||".join([_date_key(fight_date), _key_text(event)])


def is_unknown_weight_class(value: Any) -> bool:
    return _key_text(value) in UNKNOWN_WEIGHT_VALUES


def _non_empty(value: Any) -> bool:
    return _key_text(value) not in {"", "nan", "none"}


def _normalize_binary(value: Any, field_name: str) -> int | None:
    if not _non_empty(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or int(numeric) != float(numeric) or int(numeric) not in {0, 1}:
        raise EnrichmentError(f"{field_name} values must be 0 or 1. Invalid value: {value!r}")
    return int(numeric)


def _normalize_scheduled_rounds(value: Any) -> int | None:
    if not _non_empty(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or int(numeric) != float(numeric) or int(numeric) not in {3, 5}:
        raise EnrichmentError(f"scheduled_rounds values must be 3 or 5 where known. Invalid value: {value!r}")
    return int(numeric)


def _numeric_location_values(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).str.strip()
    values = values.loc[values != ""]
    numeric = values.loc[values.str.fullmatch(r"[-+]?\d+(\.\d+)?", na=False)]
    return numeric.head(5).tolist()


def _percent(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator) * 100.0, 2)


def _series_or_blank(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([""] * len(frame), index=frame.index)


def _has_positive_binary(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return bool(numeric.eq(1).any())


def _known_binary_mask(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return pd.Series([False] * len(series), index=series.index)
    numeric = pd.to_numeric(series, errors="coerce")
    if not bool(numeric.eq(1).any()):
        return pd.Series([False] * len(series), index=series.index)
    return numeric.isin([0, 1])


def enrichment_summary_from_frame(frame: pd.DataFrame, path: str | Path) -> EnrichmentSummary:
    total = int(len(frame))
    weight = _series_or_blank(frame, "weight_class")
    location = _series_or_blank(frame, "event_location")
    main_event = _series_or_blank(frame, "main_event")
    title_fight = _series_or_blank(frame, "title_fight")
    scheduled = pd.to_numeric(_series_or_blank(frame, "scheduled_rounds"), errors="coerce")
    known_weight = int(weight.map(lambda value: _non_empty(value) and not is_unknown_weight_class(value)).sum())
    known_location = int(location.map(_non_empty).sum())
    known_main = int(_known_binary_mask(main_event).sum())
    known_title = int(_known_binary_mask(title_fight).sum())
    known_scheduled = int(scheduled.isin([3, 5]).sum())
    return EnrichmentSummary(
        path=Path(path),
        total_fights=total,
        known_weight_class_pct=_percent(known_weight, total),
        known_event_location_pct=_percent(known_location, total),
        known_main_event_pct=_percent(known_main, total),
        known_title_fight_pct=_percent(known_title, total),
        known_scheduled_rounds_pct=_percent(known_scheduled, total),
    )


def summarize_enrichment_file(path: str | Path | None = None) -> EnrichmentSummary:
    summary_path = Path(path) if path else _default_enrichment_summary_path()
    inspect_csv(summary_path, require_rows=True, label="enrichment CSV")
    frame = pd.read_csv(summary_path)
    if "event_date" in frame.columns and "fight_date" not in frame.columns:
        frame = frame.rename(columns={"event_date": "fight_date"})
    return enrichment_summary_from_frame(frame, summary_path)


def _default_enrichment_summary_path() -> Path:
    import_dir = settings.raw_data_dir / "imports"
    enrichment = import_dir / "fight_enrichment.csv"
    template = import_dir / "fight_enrichment_template.csv"
    if enrichment.exists():
        return enrichment
    if template.exists():
        return template
    return import_dir / "fights.csv"


def _event_column(fights: pd.DataFrame) -> str:
    return "event_name" if "event_name" in fights.columns else "event"


def _template_binary_value(series: pd.Series, value: Any) -> Any:
    if not _has_positive_binary(series):
        return ""
    normalized = _normalize_binary(value, "binary")
    return "" if normalized is None else normalized


def build_enrichment_template(
    fights_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    fights_csv = Path(fights_path) if fights_path else settings.raw_data_dir / "imports" / "fights.csv"
    output = Path(output_path) if output_path else settings.raw_data_dir / "imports" / "fight_enrichment_template.csv"
    inspect_csv(fights_csv, required_columns=["fight_date", "fighter_a", "fighter_b"], require_rows=True, label="fights CSV")
    fights = pd.read_csv(fights_csv)
    event_column = _event_column(fights)
    main_event_series = _series_or_blank(fights, "main_event")
    title_fight_series = _series_or_blank(fights, "title_fight")

    rows: list[dict[str, Any]] = []
    for _, fight in fights.iterrows():
        weight_class = fight.get("weight_class", "Unknown")
        if not _non_empty(weight_class):
            weight_class = "Unknown"
        scheduled = fight.get("scheduled_rounds", "")
        try:
            scheduled = _normalize_scheduled_rounds(scheduled) or ""
        except EnrichmentError:
            scheduled = ""
        rows.append(
            {
                "fight_date": _date_key(fight.get("fight_date")) or fight.get("fight_date", ""),
                "event": fight.get(event_column, ""),
                "fighter_a": fight.get("fighter_a", ""),
                "fighter_b": fight.get("fighter_b", ""),
                "weight_class": weight_class,
                "event_location": _clean(fight.get("event_location", fight.get("location", ""))),
                "main_event": _template_binary_value(main_event_series, fight.get("main_event", "")),
                "title_fight": _template_binary_value(title_fight_series, fight.get("title_fight", "")),
                "scheduled_rounds": scheduled,
            }
        )
    frame = pd.DataFrame(rows, columns=ENRICHMENT_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame, output


def validate_enrichment_frame(frame: pd.DataFrame) -> list[str]:
    """Validate enrichment rows and return non-fatal warnings."""

    missing = [column for column in ENRICHMENT_COLUMNS if column not in frame.columns]
    if missing:
        raise EnrichmentError(f"fight_enrichment.csv is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise EnrichmentError("fight_enrichment.csv has headers but no data rows.")

    dates = pd.to_datetime(frame["fight_date"], errors="coerce")
    bad_dates = int(dates.isna().sum())
    if bad_dates:
        raise EnrichmentError(f"fight_enrichment.csv has {bad_dates} unparseable fight_date values.")

    for field in ["main_event", "title_fight"]:
        invalid = []
        for value in frame[field]:
            try:
                _normalize_binary(value, field)
            except EnrichmentError:
                invalid.append(value)
        if invalid:
            raise EnrichmentError(f"{field} values must be 0 or 1. Invalid sample: {invalid[:5]}")

    invalid_rounds = []
    for value in frame["scheduled_rounds"]:
        try:
            _normalize_scheduled_rounds(value)
        except EnrichmentError:
            invalid_rounds.append(value)
    if invalid_rounds:
        raise EnrichmentError(f"scheduled_rounds values must be 3 or 5 where known. Invalid sample: {invalid_rounds[:5]}")

    numeric_locations = _numeric_location_values(frame["event_location"])
    if numeric_locations:
        raise EnrichmentError(
            "event_location must be categorical location text, not numeric values. "
            f"Invalid sample: {numeric_locations}"
        )

    warnings: list[str] = []
    usable_weight = frame["weight_class"].map(lambda value: _non_empty(value) and not is_unknown_weight_class(value))
    if not bool(usable_weight.any()):
        warnings.append("fight_enrichment.csv does not contain any usable weight_class values.")
    return warnings


def validate_event_enrichment_frame(frame: pd.DataFrame) -> list[str]:
    missing = [column for column in EVENT_ENRICHMENT_COLUMNS if column not in frame.columns]
    if missing:
        raise EnrichmentError(f"event-level enrichment CSV is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise EnrichmentError("event-level enrichment CSV has headers but no data rows.")
    dates = pd.to_datetime(frame["event_date"], errors="coerce")
    bad_dates = int(dates.isna().sum())
    if bad_dates:
        raise EnrichmentError(f"event-level enrichment CSV has {bad_dates} unparseable event_date values.")
    for field in ["main_event", "title_fight"]:
        invalid = []
        for value in frame[field]:
            try:
                _normalize_binary(value, field)
            except EnrichmentError:
                invalid.append(value)
        if invalid:
            raise EnrichmentError(f"{field} values must be 0 or 1. Invalid sample: {invalid[:5]}")
    invalid_rounds = []
    for value in frame["scheduled_rounds"]:
        try:
            _normalize_scheduled_rounds(value)
        except EnrichmentError:
            invalid_rounds.append(value)
    if invalid_rounds:
        raise EnrichmentError(f"scheduled_rounds values must be 3 or 5 where known. Invalid sample: {invalid_rounds[:5]}")
    numeric_locations = _numeric_location_values(frame["event_location"])
    if numeric_locations:
        raise EnrichmentError(
            "event_location must be categorical location text, not numeric values. "
            f"Invalid sample: {numeric_locations}"
        )
    return []


def event_enrichment_to_fight_enrichment(
    fights: pd.DataFrame,
    event_enrichment: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    warnings = validate_event_enrichment_frame(event_enrichment)
    event_column = _event_column(fights)
    fight_groups: dict[str, pd.DataFrame] = {}
    event_keys = fights.apply(lambda row: _event_key(row.get(event_column), row.get("fight_date")), axis=1)
    for key, group in fights.groupby(event_keys):
        if key:
            fight_groups[str(key)] = group

    event_copy = event_enrichment.copy()
    event_copy["_event_key"] = event_copy.apply(lambda row: _event_key(row.get("event"), row.get("event_date")), axis=1)
    duplicate_keys = event_copy["_event_key"].loc[event_copy["_event_key"].duplicated()].unique().tolist()
    if duplicate_keys:
        raise EnrichmentError(f"event-level enrichment CSV has duplicate event keys. Duplicate sample: {duplicate_keys[:5]}")

    rows: list[dict[str, Any]] = []
    skipped_fight_specific = 0
    unmatched_events = 0
    for _, event_row in event_copy.iterrows():
        group = fight_groups.get(str(event_row["_event_key"]))
        if group is None or group.empty:
            unmatched_events += 1
            continue
        single_fight_event = len(group) == 1
        if not single_fight_event:
            fight_specific_values = [
                event_row.get("weight_class"),
                event_row.get("main_event"),
                event_row.get("title_fight"),
                event_row.get("scheduled_rounds"),
            ]
            if any(_non_empty(value) for value in fight_specific_values):
                skipped_fight_specific += len(group)
        for _, fight in group.iterrows():
            if single_fight_event:
                weight_class = event_row.get("weight_class", "")
                main_event = event_row.get("main_event", "")
                title_fight = event_row.get("title_fight", "")
                scheduled_rounds = event_row.get("scheduled_rounds", "")
            else:
                weight_class = ""
                main_event = ""
                title_fight = ""
                scheduled_rounds = ""
            rows.append(
                {
                    "fight_date": _date_key(fight.get("fight_date")),
                    "event": fight.get(event_column, ""),
                    "fighter_a": fight.get("fighter_a", ""),
                    "fighter_b": fight.get("fighter_b", ""),
                    "weight_class": weight_class,
                    "event_location": event_row.get("event_location", ""),
                    "main_event": main_event,
                    "title_fight": title_fight,
                    "scheduled_rounds": scheduled_rounds,
                }
            )
    if unmatched_events:
        warnings.append(f"Event-level enrichment had {unmatched_events} event rows that did not match fights.csv.")
    if skipped_fight_specific:
        warnings.append(
            "Event-level enrichment matched multi-fight events; applied event_location but skipped fight-specific "
            f"fields for {skipped_fight_specific} fight rows. Use fight-level enrichment for weight_class, "
            "main_event, title_fight, and scheduled_rounds on multi-fight events."
        )
    return pd.DataFrame(rows, columns=ENRICHMENT_COLUMNS), warnings


def normalize_enrichment_frame(fights: pd.DataFrame, enrichment: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    columns = set(enrichment.columns)
    if set(ENRICHMENT_COLUMNS) <= columns:
        return enrichment[ENRICHMENT_COLUMNS].copy(), "fight", []
    if set(EVENT_ENRICHMENT_COLUMNS) <= columns:
        frame, warnings = event_enrichment_to_fight_enrichment(fights, enrichment[EVENT_ENRICHMENT_COLUMNS].copy())
        return frame, "event", warnings
    raise EnrichmentError(
        "Enrichment CSV must use either fight-level columns "
        f"({', '.join(ENRICHMENT_COLUMNS)}) or event-level columns ({', '.join(EVENT_ENRICHMENT_COLUMNS)})."
    )


def merge_enrichment_into_fights(
    fights: pd.DataFrame,
    enrichment: pd.DataFrame,
    fights_path: str | Path,
    enrichment_path: str | Path,
    output_path: str | Path,
    source_format: str = "fight",
    extra_warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, EnrichmentReport]:
    warnings = [*validate_enrichment_frame(enrichment), *(extra_warnings or [])]
    merged = fights.copy()
    for field in ENRICHABLE_FIELDS:
        if field not in merged.columns:
            merged[field] = ""
    for field in ["weight_class", "event_location"]:
        merged[field] = merged[field].astype("object")

    fight_event_column = "event_name" if "event_name" in merged.columns else "event"
    merged["_enrichment_key"] = merged.apply(lambda row: fight_identity_key(row, event_column=fight_event_column), axis=1)
    enrichment_copy = enrichment.copy()
    enrichment_copy["_enrichment_key"] = enrichment_copy.apply(_enrichment_key, axis=1)

    duplicate_keys = enrichment_copy["_enrichment_key"].loc[enrichment_copy["_enrichment_key"].duplicated()].unique().tolist()
    if duplicate_keys:
        raise EnrichmentError(f"fight_enrichment.csv has duplicate fight keys. Duplicate sample: {duplicate_keys[:5]}")

    fight_key_to_index = dict(zip(merged["_enrichment_key"], merged.index))
    matched = 0
    unmatched = 0
    updated_fields = {field: 0 for field in ENRICHABLE_FIELDS}

    for _, row in enrichment_copy.iterrows():
        key = row["_enrichment_key"]
        if key not in fight_key_to_index:
            unmatched += 1
            continue
        matched += 1
        target_index = fight_key_to_index[key]
        for field in ENRICHABLE_FIELDS:
            raw_value = row.get(field)
            if field == "weight_class":
                if not _non_empty(raw_value) or is_unknown_weight_class(raw_value):
                    continue
                value: Any = _clean(raw_value)
            elif field == "event_location":
                if not _non_empty(raw_value):
                    continue
                value = _clean(raw_value)
            elif field in {"main_event", "title_fight"}:
                normalized = _normalize_binary(raw_value, field)
                if normalized is None:
                    continue
                value = normalized
            elif field == "scheduled_rounds":
                normalized = _normalize_scheduled_rounds(raw_value)
                if normalized is None:
                    continue
                value = normalized
            else:
                continue
            if merged.at[target_index, field] != value:
                merged.at[target_index, field] = value
                updated_fields[field] += 1

    if matched == 0:
        warnings.append("No enrichment rows matched fights.csv. Check fight_date, event, and fighter names.")
    provided_weight = enrichment_copy["weight_class"].map(lambda value: _non_empty(value) and not is_unknown_weight_class(value))
    if bool(provided_weight.any()) and updated_fields["weight_class"] == 0:
        warnings.append("Enrichment provided weight_class values, but no fights.csv weight_class values were updated.")

    merged = merged.drop(columns=["_enrichment_key"])
    report = EnrichmentReport(
        fights_path=Path(fights_path),
        enrichment_path=Path(enrichment_path),
        output_path=Path(output_path),
        fights_rows=int(len(fights)),
        enrichment_rows=int(len(enrichment)),
        matched_rows=matched,
        unmatched_enrichment_rows=unmatched,
        updated_fields=updated_fields,
        warnings=warnings,
        source_format=source_format,
    )
    return merged, report


def import_enrichment_csv(
    enrichment_path: str | Path | None = None,
    fights_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> EnrichmentReport:
    source = Path(enrichment_path) if enrichment_path else settings.raw_data_dir / "imports" / "fight_enrichment.csv"
    fights_csv = Path(fights_path) if fights_path else settings.raw_data_dir / "imports" / "fights.csv"
    output = Path(output_path) if output_path else fights_csv

    inspect_csv(fights_csv, required_columns=["fighter_a", "fighter_b", "fight_date"], require_rows=True, label="fights CSV")
    if not source.exists():
        raise EnrichmentError(f"Missing fight_enrichment.csv at {source}. {MISSING_ENRICHMENT_GUIDANCE}")
    inspect_csv(source, require_rows=True, label="fight_enrichment.csv")
    fights = pd.read_csv(fights_csv)
    enrichment, source_format, warnings = normalize_enrichment_frame(fights, pd.read_csv(source))
    merged, report = merge_enrichment_into_fights(
        fights=fights,
        enrichment=enrichment,
        fights_path=fights_csv,
        enrichment_path=source,
        output_path=output,
        source_format=source_format,
        extra_warnings=warnings,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return report
