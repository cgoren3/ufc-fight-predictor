from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv
import re
import unicodedata

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
STRING_ENRICHMENT_FIELDS = ["weight_class", "event_location"]
NUMERIC_ENRICHMENT_FIELDS = ["main_event", "title_fight", "scheduled_rounds"]
UNKNOWN_WEIGHT_VALUES = {"", "unknown", "unk", "n/a", "na", "none", "nan"}
EVENT_ID_ALIASES = ["event_id", "event_id_x", "event_id_y"]
FIGHT_DATE_ALIASES = ["fight_date", "event_date", "date", "card_date"]
EVENT_ALIASES = ["event", "event_name", "event_title", "name", "card", "card_name"]
FIGHTER_A_ALIASES = ["fighter_a", "red_fighter", "r_fighter", "fighter_1", "fight_fighter", "fighter"]
FIGHTER_B_ALIASES = ["fighter_b", "blue_fighter", "b_fighter", "fighter_2", "opponent"]
BOUT_ALIASES = ["bout", "matchup", "fight", "fighters"]
WEIGHT_CLASS_ALIASES = ["weight_class", "division", "class", "bout_weight", "weightclass", "weight"]
EVENT_LOCATION_ALIASES = [
    "event_location",
    "location",
    "venue",
    "place",
    "event_venue_name",
    "event_venue_city",
    "event_venue_state",
    "event_venue_country",
    "venue_name",
    "venue_city",
    "venue_state",
    "venue_country",
    "city",
    "state",
    "country",
]
MAIN_EVENT_ALIASES = ["main_event", "is_main_event", "bout_order", "fight_order"]
TITLE_FIGHT_ALIASES = ["title_fight", "is_title_fight", "championship", "title", "title_bout", "belt"]
DIRECT_TITLE_FIGHT_ALIASES = ["title_fight", "is_title_fight", "championship", "title", "title_bout"]
SCHEDULED_ROUNDS_ALIASES = ["scheduled_rounds", "no_of_rounds", "rounds", "time_format", "max_rounds"]
ODDS_ALIASES = [
    "fighter_a_odds",
    "fighter_b_odds",
    "red_fighter_moneyline_odds",
    "blue_fighter_moneyline_odds",
    "moneyline",
    "odds",
]
SCORECARD_ALIASES = ["score_cards", "scorecards", "judge", "round_1_a", "round_1_b", "total_a", "total_b"]
MISSING_ENRICHMENT_GUIDANCE = (
    "Run ufc-predict build-enrichment-template, fill in the missing columns, "
    "save it as data/raw/imports/fight_enrichment.csv, then rerun import-enrichment."
)
EXTERNAL_SOURCE_SUFFIXES = {".csv"}
TITLE_EVENT_TERMS = ("title", "championship", "belt", "interim")


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
    known_weight_class_count: int
    known_event_location_count: int
    known_main_event_count: int
    known_title_fight_count: int
    known_scheduled_rounds_count: int
    known_weight_class_pct: float
    known_event_location_pct: float
    known_main_event_pct: float
    known_title_fight_pct: float
    known_scheduled_rounds_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "total_fights": self.total_fights,
            "known_weight_class_count": self.known_weight_class_count,
            "known_event_location_count": self.known_event_location_count,
            "known_main_event_count": self.known_main_event_count,
            "known_title_fight_count": self.known_title_fight_count,
            "known_scheduled_rounds_count": self.known_scheduled_rounds_count,
            "known_weight_class_pct": self.known_weight_class_pct,
            "known_event_location_pct": self.known_event_location_pct,
            "known_main_event_pct": self.known_main_event_pct,
            "known_title_fight_pct": self.known_title_fight_pct,
            "known_scheduled_rounds_pct": self.known_scheduled_rounds_pct,
        }


@dataclass
class ExternalEnrichmentSourceReport:
    path: Path
    rows_read: int = 0
    matched_rows: int = 0
    unmatched_rows: int = 0
    columns: list[str] = field(default_factory=list)
    dataset_type: str = "unknown"
    usable_fields: list[str] = field(default_factory=list)
    updated_fields: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "rows_read": self.rows_read,
            "matched_rows": self.matched_rows,
            "unmatched_rows": self.unmatched_rows,
            "columns": self.columns,
            "dataset_type": self.dataset_type,
            "usable_fields": self.usable_fields,
            "updated_fields": self.updated_fields,
            "warnings": self.warnings,
        }


@dataclass
class EnrichmentSourceInspection:
    path: Path
    rows_read: int = 0
    columns: list[str] = field(default_factory=list)
    dataset_type: str = "unknown"
    usable_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "rows_read": self.rows_read,
            "columns": self.columns,
            "dataset_type": self.dataset_type,
            "usable_fields": self.usable_fields,
            "warnings": self.warnings,
        }


@dataclass
class AutoEnrichmentReport:
    template_path: Path
    output_path: Path
    total_rows: int
    inferred_main_event_rows: int = 0
    inferred_non_main_event_rows: int = 0
    inferred_title_fight_rows: int = 0
    inferred_non_title_fight_rows: int = 0
    external_sources: list[ExternalEnrichmentSourceReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_path": str(self.template_path),
            "output_path": str(self.output_path),
            "total_rows": self.total_rows,
            "inferred_main_event_rows": self.inferred_main_event_rows,
            "inferred_non_main_event_rows": self.inferred_non_main_event_rows,
            "inferred_title_fight_rows": self.inferred_title_fight_rows,
            "inferred_non_title_fight_rows": self.inferred_non_title_fight_rows,
            "external_sources": [source.as_dict() for source in self.external_sources],
            "warnings": self.warnings,
        }


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _key_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _clean(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


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


def _date_pair_key(fight_date: Any, fighter_a: Any, fighter_b: Any) -> str:
    return "||".join([_date_key(fight_date), _pair_key(fighter_a, fighter_b)])


def _event_pair_key(event: Any, fighter_a: Any, fighter_b: Any) -> str:
    return "||".join([_key_text(event), _pair_key(fighter_a, fighter_b)])


def is_unknown_weight_class(value: Any) -> bool:
    return _key_text(value) in UNKNOWN_WEIGHT_VALUES


def _non_empty(value: Any) -> bool:
    return _key_text(value) not in {"", "nan", "none"}


def _normalize_binary(value: Any, field_name: str) -> int | None:
    if not _non_empty(value):
        return None
    text = _key_text(value)
    if text in {"true", "yes", "y"}:
        return 1
    if text in {"false", "no", "n"}:
        return 0
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or int(numeric) != float(numeric) or int(numeric) not in {0, 1}:
        raise EnrichmentError(f"{field_name} values must be 0 or 1. Invalid value: {value!r}")
    return int(numeric)


def _normalize_scheduled_rounds(value: Any) -> int | None:
    if not _non_empty(value):
        return None
    text = _key_text(value)
    match = re.search(r"\b([35])\s*rnd\b", text)
    if match:
        return int(match.group(1))
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


def _first_present(row: pd.Series | dict[str, Any], columns: list[str]) -> Any:
    for column in columns:
        if column in row and _non_empty(row.get(column)):
            return row.get(column)
    return ""


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())).strip("_")
        for column in output.columns
    ]
    return output


def coerce_enrichment_dtypes(frame: pd.DataFrame, ensure_columns: bool = False) -> pd.DataFrame:
    """Coerce enrichment columns to assignment-safe dtypes.

    Pandas can read blank integer-looking columns as string-backed columns. This
    helper keeps fight-context text as object strings and flags/rounds as nullable
    integers so import-enrichment never writes Python ints into string columns.
    """

    output = frame.copy()
    if ensure_columns:
        for column in ENRICHABLE_FIELDS:
            if column not in output.columns:
                output[column] = pd.NA if column in NUMERIC_ENRICHMENT_FIELDS else ""
    for column in STRING_ENRICHMENT_FIELDS:
        if column in output.columns:
            output[column] = output[column].astype("object").where(pd.notna(output[column]), "")
    for column in NUMERIC_ENRICHMENT_FIELDS:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype("Int64")
    return output


def _has_positive_binary(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return bool(numeric.eq(1).any())


def _known_binary_mask(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return pd.Series([False] * len(series), index=series.index)
    numeric = pd.to_numeric(series, errors="coerce")
    return values.ne("") & numeric.isin([0, 1])


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
        known_weight_class_count=known_weight,
        known_event_location_count=known_location,
        known_main_event_count=known_main,
        known_title_fight_count=known_title,
        known_scheduled_rounds_count=known_scheduled,
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


def _known_field(value: Any, field: str) -> bool:
    if field == "weight_class":
        return _non_empty(value) and not is_unknown_weight_class(value)
    if field in {"main_event", "title_fight"}:
        return _normalize_binary(value, field) is not None if _non_empty(value) else False
    if field == "scheduled_rounds":
        return _normalize_scheduled_rounds(value) is not None if _non_empty(value) else False
    return _non_empty(value)


def _safe_field_value(value: Any, field: str) -> Any:
    if field == "weight_class":
        return _clean(value) if _known_field(value, field) else ""
    if field == "event_location":
        return _clean(value) if _non_empty(value) else ""
    if field in {"main_event", "title_fight"}:
        normalized = _normalize_binary(value, field)
        return "" if normalized is None else normalized
    if field == "scheduled_rounds":
        normalized = _normalize_scheduled_rounds(value)
        return "" if normalized is None else normalized
    return value


def _values_equal(left: Any, right: Any) -> bool:
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if bool(left_missing) and bool(right_missing):
        return True
    if bool(left_missing) or bool(right_missing):
        return False
    return left == right


def _set_if_known(frame: pd.DataFrame, index: Any, field: str, raw_value: Any) -> bool:
    value = _safe_field_value(raw_value, field)
    if not _non_empty(value) and value != 0:
        return False
    current = frame.at[index, field] if field in frame.columns else ""
    if _values_equal(current, value):
        return False
    frame.at[index, field] = value
    return True


def _fighter_aliases(name: Any) -> set[str]:
    text = _key_text(name)
    tokens = [token for token in text.split() if token not in {"jr", "sr", "ii", "iii", "iv"}]
    aliases = {text}
    if tokens:
        aliases.add(tokens[-1])
    if len(tokens) >= 2:
        aliases.add(" ".join(tokens[-2:]))
    return {alias for alias in aliases if alias}


def _parse_vs_pair(text: Any) -> tuple[str, str] | None:
    cleaned = _key_text(text)
    if not cleaned:
        return None
    match = re.search(r"\b(.+?)\s+(?:vs|v|versus)\s+(.+)$", cleaned)
    if not match:
        return None
    left = re.sub(r"\b\d+\b.*$", "", match.group(1)).strip()
    right = re.sub(r"\b\d+\b.*$", "", match.group(2)).strip()
    right = re.sub(r"\s+(?:fight night|ufc|dwcs|road to ufc).*$", "", right).strip()
    return (left, right) if left and right else None


def _alias_matches(candidate: str, aliases: set[str]) -> bool:
    candidate = _key_text(candidate)
    if not candidate:
        return False
    return any(candidate == alias or candidate in alias or alias in candidate for alias in aliases)


def _event_mentions_matchup(event: Any, fighter_a: Any, fighter_b: Any) -> bool:
    event_key = _key_text(event)
    aliases_a = _fighter_aliases(fighter_a)
    aliases_b = _fighter_aliases(fighter_b)
    if any(alias and alias in event_key for alias in aliases_a) and any(alias and alias in event_key for alias in aliases_b):
        return True
    parsed = _parse_vs_pair(event)
    if parsed is None:
        return False
    left, right = parsed
    return (_alias_matches(left, aliases_a) and _alias_matches(right, aliases_b)) or (
        _alias_matches(left, aliases_b) and _alias_matches(right, aliases_a)
    )


def _event_indicates_title(event: Any) -> bool:
    event_key = _key_text(event)
    return any(term in event_key.split() for term in TITLE_EVENT_TERMS)


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


def _ensure_working_enrichment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ENRICHMENT_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    output = output[ENRICHMENT_COLUMNS]
    output = output.astype("object").where(pd.notna(output), "")
    return coerce_enrichment_dtypes(output, ensure_columns=True)


def _infer_main_events(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    output = frame.copy()
    matches = output.apply(lambda row: _event_mentions_matchup(row.get("event"), row.get("fighter_a"), row.get("fighter_b")), axis=1)
    event_keys = output.apply(lambda row: _event_key(row.get("event"), row.get("fight_date")), axis=1)
    main_count = 0
    non_main_count = 0
    for _, group in output.groupby(event_keys):
        group_matches = matches.loc[group.index]
        if int(group_matches.sum()) != 1:
            continue
        main_index = group_matches.loc[group_matches].index[0]
        if not _known_field(output.at[main_index, "main_event"], "main_event"):
            output.at[main_index, "main_event"] = 1
            main_count += 1
        for index in group.index:
            if index == main_index:
                continue
            if not _known_field(output.at[index, "main_event"], "main_event"):
                output.at[index, "main_event"] = 0
                non_main_count += 1
    return output, main_count, non_main_count


def _infer_title_fights(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    output = frame.copy()
    title_count = 0
    non_title_count = 0
    for index, row in output.iterrows():
        if _known_field(row.get("title_fight"), "title_fight"):
            continue
        scheduled = _normalize_scheduled_rounds(row.get("scheduled_rounds")) if _non_empty(row.get("scheduled_rounds")) else None
        if scheduled == 5 or _event_indicates_title(row.get("event")):
            output.at[index, "title_fight"] = 1
            title_count += 1
        elif scheduled == 3:
            output.at[index, "title_fight"] = 0
            non_title_count += 1
    return output, title_count, non_title_count


def _split_bout(value: Any) -> tuple[str, str] | None:
    parsed = _parse_vs_pair(value)
    if parsed is not None:
        return parsed
    text = _clean(value)
    if not text:
        return None
    for delimiter in [" vs. ", " vs ", " v. ", " v ", " versus "]:
        if delimiter in text.lower():
            parts = re.split(re.escape(delimiter), text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2 and _non_empty(parts[0]) and _non_empty(parts[1]):
                return _clean(parts[0]), _clean(parts[1])
    return None


def _compose_location_from_row(row: pd.Series | dict[str, Any]) -> str:
    venue_parts = [
        _first_present(row, ["event_venue_name", "venue_name"]),
        _first_present(row, ["event_venue_city", "venue_city", "city"]),
        _first_present(row, ["event_venue_state", "venue_state", "state"]),
        _first_present(row, ["event_venue_country", "venue_country", "country"]),
    ]
    venue_location = ", ".join(_clean(part) for part in venue_parts if _non_empty(part))
    if venue_location:
        return venue_location
    return _first_present(row, EVENT_LOCATION_ALIASES)


def _event_id_value(row: pd.Series | dict[str, Any]) -> str:
    return _clean(_first_present(row, EVENT_ID_ALIASES))


def _event_metadata_payload(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    return {
        "event": _first_present(row, EVENT_ALIASES),
        "fight_date": _first_present(row, FIGHT_DATE_ALIASES),
        "event_location": _compose_location_from_row(row),
    }


def _source_csv_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in EXTERNAL_SOURCE_SUFFIXES)


def _build_event_metadata_maps(paths: list[Path]) -> dict[str, dict[str, Any]]:
    event_maps: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            raw = _canonical_columns(pd.read_csv(path))
        except Exception:
            continue
        if not any(alias in raw.columns for alias in EVENT_ID_ALIASES):
            continue
        for _, row in raw.iterrows():
            event_id = _event_id_value(row)
            if not event_id:
                continue
            payload = _event_metadata_payload(row)
            if not any(_non_empty(value) for value in payload.values()):
                continue
            target = event_maps.setdefault(event_id, {})
            for field, value in payload.items():
                if _non_empty(value) and not _non_empty(target.get(field, "")):
                    target[field] = value
    return event_maps


def _augment_with_event_metadata(frame: pd.DataFrame, event_maps: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if not event_maps or not any(alias in frame.columns for alias in EVENT_ID_ALIASES):
        return frame
    output = frame.copy()
    for column in ["event", "fight_date", "event_location"]:
        if column not in output.columns:
            output[column] = ""
    for index, row in output.iterrows():
        event_id = _event_id_value(row)
        metadata = event_maps.get(event_id)
        if not metadata:
            continue
        for field in ["event", "fight_date", "event_location"]:
            if _non_empty(output.at[index, field]):
                continue
            value = metadata.get(field, "")
            if _non_empty(value):
                output.at[index, field] = value
    return output


def _has_any_value(frame: pd.DataFrame, columns: list[str]) -> bool:
    for column in columns:
        if column in frame.columns and bool(frame[column].map(_non_empty).any()):
            return True
    return False


def _has_fighter_pair_columns(frame: pd.DataFrame) -> bool:
    return (
        (_has_any_value(frame, FIGHTER_A_ALIASES) and _has_any_value(frame, FIGHTER_B_ALIASES))
        or _has_any_value(frame, BOUT_ALIASES)
    )


def _column_present(columns: set[str], aliases: list[str]) -> bool:
    return any(alias in columns for alias in aliases)


def _source_fields_from_columns(columns: list[str]) -> list[str]:
    column_set = set(columns)
    fields: list[str] = []
    if _column_present(column_set, WEIGHT_CLASS_ALIASES):
        fields.append("weight_class")
    if _column_present(column_set, EVENT_LOCATION_ALIASES):
        fields.append("event_location")
    if _column_present(column_set, MAIN_EVENT_ALIASES):
        fields.append("main_event")
    has_context = _has_fighter_pair_columns_by_name(column_set) or (
        _column_present(column_set, EVENT_ALIASES) and _column_present(column_set, FIGHT_DATE_ALIASES)
    )
    if _column_present(column_set, DIRECT_TITLE_FIGHT_ALIASES) or ("belt" in column_set and has_context):
        fields.append("title_fight")
    if _column_present(column_set, SCHEDULED_ROUNDS_ALIASES):
        fields.append("scheduled_rounds")
    if _column_present(column_set, ODDS_ALIASES) or any(column.endswith("_odds") for column in column_set):
        fields.append("odds")
    if _column_present(column_set, SCORECARD_ALIASES):
        fields.append("scorecards")
    return fields


def _has_fighter_pair_columns_by_name(columns: set[str]) -> bool:
    return (_column_present(columns, FIGHTER_A_ALIASES) and _column_present(columns, FIGHTER_B_ALIASES)) or _column_present(columns, BOUT_ALIASES)


def _guess_source_dataset_type_from_columns(columns: list[str]) -> str:
    column_set = set(columns)
    fields = _source_fields_from_columns(columns)
    if "scorecards" in fields and not _has_fighter_pair_columns_by_name(column_set):
        return "scorecards"
    if "odds" in fields and not _has_fighter_pair_columns_by_name(column_set):
        return "odds"
    if _has_fighter_pair_columns_by_name(column_set) and any(field in fields for field in ENRICHABLE_FIELDS):
        return "fight-level enrichment"
    if _column_present(column_set, EVENT_ALIASES) and _column_present(column_set, FIGHT_DATE_ALIASES) and "event_location" in fields:
        return "event-level enrichment"
    if "scorecards" in fields:
        return "scorecards"
    if "odds" in fields:
        return "odds"
    return "unknown"


def _read_source_header_and_count(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader, [])
        rows = sum(1 for _ in reader)
    return [str(column) for column in columns], rows


def _usable_fields_for_source(frame: pd.DataFrame) -> list[str]:
    fields: list[str] = []
    if _has_any_value(frame, FIGHT_DATE_ALIASES):
        fields.append("fight_date")
    if _has_any_value(frame, EVENT_ALIASES):
        fields.append("event")
    if _has_fighter_pair_columns(frame):
        fields.append("fighter_pair")
    if _has_any_value(frame, WEIGHT_CLASS_ALIASES):
        fields.append("weight_class")
    if _has_any_value(frame, EVENT_LOCATION_ALIASES):
        fields.append("event_location")
    if _has_any_value(frame, MAIN_EVENT_ALIASES):
        fields.append("main_event")
    has_context = _has_fighter_pair_columns(frame) or (_has_any_value(frame, EVENT_ALIASES) and _has_any_value(frame, FIGHT_DATE_ALIASES))
    if _has_any_value(frame, DIRECT_TITLE_FIGHT_ALIASES) or ("belt" in frame.columns and has_context and _has_any_value(frame, ["belt"])):
        fields.append("title_fight")
    if _has_any_value(frame, SCHEDULED_ROUNDS_ALIASES):
        fields.append("scheduled_rounds")
    if _has_any_value(frame, ODDS_ALIASES) or any(column.endswith("_odds") for column in frame.columns):
        fields.append("odds")
    if _has_any_value(frame, SCORECARD_ALIASES):
        fields.append("scorecards")
    return fields


def _enrichment_fields_for_source(frame: pd.DataFrame) -> list[str]:
    return [field for field in _usable_fields_for_source(frame) if field in ENRICHABLE_FIELDS]


def _guess_source_dataset_type(frame: pd.DataFrame) -> str:
    fields = _usable_fields_for_source(frame)
    if "scorecards" in fields and not _has_fighter_pair_columns(frame):
        return "scorecards"
    if "odds" in fields and not _has_fighter_pair_columns(frame):
        return "odds"
    if _has_fighter_pair_columns(frame) and any(field in fields for field in ENRICHABLE_FIELDS):
        return "fight-level enrichment"
    if _has_any_value(frame, EVENT_ALIASES) and _has_any_value(frame, FIGHT_DATE_ALIASES) and "event_location" in fields:
        return "event-level enrichment"
    if "scorecards" in fields:
        return "scorecards"
    if "odds" in fields:
        return "odds"
    return "unknown"


def _source_inspection_for_path(path: Path, event_maps: dict[str, dict[str, Any]] | None = None) -> EnrichmentSourceInspection:
    report = EnrichmentSourceInspection(path=path.resolve())
    try:
        report.columns, report.rows_read = _read_source_header_and_count(path)
        canonical_columns = [
            re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())).strip("_")
            for column in report.columns
        ]
    except Exception as exc:
        report.warnings.append(f"Could not read enrichment source: {type(exc).__name__}: {exc}")
        return report
    report.usable_fields = _source_fields_from_columns(canonical_columns)
    report.dataset_type = _guess_source_dataset_type_from_columns(canonical_columns)
    if not report.usable_fields:
        report.warnings.append("No usable enrichment, odds, or scorecard columns found.")
    return report


def inspect_enrichment_sources(source_dir: str | Path | None = None) -> tuple[Path, bool, list[EnrichmentSourceInspection]]:
    directory = Path(source_dir) if source_dir else settings.raw_data_dir / "enrichment_sources"
    directory = directory.resolve()
    if not directory.exists():
        return directory, False, []
    source_paths = _source_csv_files(directory)
    return directory, True, [_source_inspection_for_path(path) for path in source_paths]


def _main_event_value_from_row(row: pd.Series | dict[str, Any]) -> Any:
    direct = _first_present(row, ["main_event", "is_main_event"])
    if _non_empty(direct):
        return direct
    order = _first_present(row, ["bout_order", "fight_order"])
    if not _non_empty(order):
        return ""
    order_text = _key_text(order)
    if "main" in order_text:
        return 1
    numeric = pd.to_numeric(pd.Series([order]), errors="coerce").iloc[0]
    if not pd.isna(numeric) and int(numeric) == float(numeric) and int(numeric) == 1:
        return 1
    return ""


def _external_row_to_payload(row: pd.Series) -> dict[str, Any]:
    fighter_a = _first_present(row, FIGHTER_A_ALIASES)
    fighter_b = _first_present(row, FIGHTER_B_ALIASES)
    bout = _first_present(row, BOUT_ALIASES)
    if (not _non_empty(fighter_a) or not _non_empty(fighter_b)) and _non_empty(bout):
        parsed = _split_bout(bout)
        if parsed is not None:
            fighter_a, fighter_b = parsed
    return {
        "fight_date": _first_present(row, FIGHT_DATE_ALIASES),
        "event": _first_present(row, EVENT_ALIASES),
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "weight_class": _first_present(row, WEIGHT_CLASS_ALIASES),
        "event_location": _compose_location_from_row(row),
        "main_event": _main_event_value_from_row(row),
        "title_fight": _first_present(row, TITLE_FIGHT_ALIASES),
        "scheduled_rounds": _first_present(row, SCHEDULED_ROUNDS_ALIASES),
    }


def _unique_mapping(keys: pd.Series) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    counts = keys.value_counts()
    for index, key in keys.items():
        if key and counts.get(key, 0) == 1:
            mapping[str(key)] = index
    return mapping


def _merge_external_fight_row(
    frame: pd.DataFrame,
    payload: dict[str, Any],
    date_pair_to_index: dict[str, Any],
    event_pair_to_index: dict[str, Any],
    updated_fields: dict[str, int],
) -> bool:
    index = None
    if _non_empty(payload.get("fighter_a")) and _non_empty(payload.get("fighter_b")):
        date_key = _date_pair_key(payload.get("fight_date"), payload.get("fighter_a"), payload.get("fighter_b"))
        event_key = _event_pair_key(payload.get("event"), payload.get("fighter_a"), payload.get("fighter_b"))
        index = date_pair_to_index.get(date_key)
        if index is None:
            index = event_pair_to_index.get(event_key)
    if index is None:
        return False
    for field in ENRICHABLE_FIELDS:
        try:
            if _set_if_known(frame, index, field, payload.get(field)):
                updated_fields[field] += 1
        except EnrichmentError:
            continue
    return True


def _merge_external_event_row(
    frame: pd.DataFrame,
    payload: dict[str, Any],
    event_to_indices: dict[str, list[Any]],
    updated_fields: dict[str, int],
) -> bool:
    event_key = _event_key(payload.get("event"), payload.get("fight_date"))
    indices = event_to_indices.get(event_key, [])
    if not indices:
        return False
    for index in indices:
        try:
            if _set_if_known(frame, index, "event_location", payload.get("event_location")):
                updated_fields["event_location"] += 1
        except EnrichmentError:
            continue
    if len(indices) == 1:
        index = indices[0]
        for field in ["weight_class", "main_event", "title_fight", "scheduled_rounds"]:
            try:
                if _set_if_known(frame, index, field, payload.get(field)):
                    updated_fields[field] += 1
            except EnrichmentError:
                continue
    return True


def merge_external_enrichment_sources(
    frame: pd.DataFrame,
    source_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[ExternalEnrichmentSourceReport]]:
    directory = Path(source_dir) if source_dir else settings.raw_data_dir / "enrichment_sources"
    output = frame.copy()
    if not directory.exists():
        return output, []

    date_pair_keys = output.apply(lambda row: _date_pair_key(row.get("fight_date"), row.get("fighter_a"), row.get("fighter_b")), axis=1)
    event_pair_keys = output.apply(lambda row: _event_pair_key(row.get("event"), row.get("fighter_a"), row.get("fighter_b")), axis=1)
    date_pair_to_index = _unique_mapping(date_pair_keys)
    event_pair_to_index = _unique_mapping(event_pair_keys)
    event_keys = output.apply(lambda row: _event_key(row.get("event"), row.get("fight_date")), axis=1)
    event_to_indices: dict[str, list[Any]] = {}
    for index, key in event_keys.items():
        event_to_indices.setdefault(str(key), []).append(index)

    reports: list[ExternalEnrichmentSourceReport] = []
    source_paths = _source_csv_files(directory)
    event_maps = _build_event_metadata_maps(source_paths)
    for path in source_paths:
        report = ExternalEnrichmentSourceReport(path=path.resolve(), updated_fields={field: 0 for field in ENRICHABLE_FIELDS})
        try:
            raw_input = pd.read_csv(path)
            report.columns = [str(column) for column in raw_input.columns]
            raw = _augment_with_event_metadata(_canonical_columns(raw_input), event_maps)
        except Exception as exc:
            report.warnings.append(f"Could not read enrichment source: {type(exc).__name__}: {exc}")
            reports.append(report)
            continue
        report.rows_read = int(len(raw))
        report.usable_fields = _usable_fields_for_source(raw)
        report.dataset_type = _guess_source_dataset_type(raw)
        if not report.usable_fields:
            report.warnings.append("No usable enrichment columns found.")
        for _, row in raw.iterrows():
            payload = _external_row_to_payload(row)
            has_pair = _non_empty(payload.get("fighter_a")) and _non_empty(payload.get("fighter_b"))
            matched = False
            if has_pair:
                matched = _merge_external_fight_row(output, payload, date_pair_to_index, event_pair_to_index, report.updated_fields)
            if not matched:
                matched = _merge_external_event_row(output, payload, event_to_indices, report.updated_fields)
            if matched:
                report.matched_rows += 1
            else:
                report.unmatched_rows += 1
        reports.append(report)
    return output, reports


def auto_enrich(
    template_path: str | Path | None = None,
    output_path: str | Path | None = None,
    source_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, AutoEnrichmentReport]:
    template = Path(template_path) if template_path else settings.raw_data_dir / "imports" / "fight_enrichment_template.csv"
    output = Path(output_path) if output_path else settings.raw_data_dir / "imports" / "fight_enrichment.csv"
    inspect_csv(template, required_columns=ENRICHMENT_COLUMNS, require_rows=True, label="fight enrichment template CSV")
    frame = _ensure_working_enrichment_frame(pd.read_csv(template))
    frame, inferred_main, inferred_non_main = _infer_main_events(frame)
    frame, inferred_title, inferred_non_title = _infer_title_fights(frame)
    frame, source_reports = merge_external_enrichment_sources(frame, source_dir=source_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    report = AutoEnrichmentReport(
        template_path=template,
        output_path=output,
        total_rows=int(len(frame)),
        inferred_main_event_rows=inferred_main,
        inferred_non_main_event_rows=inferred_non_main,
        inferred_title_fight_rows=inferred_title,
        inferred_non_title_fight_rows=inferred_non_title,
        external_sources=source_reports,
    )
    return frame, report


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
        return coerce_enrichment_dtypes(enrichment[ENRICHMENT_COLUMNS].copy(), ensure_columns=True), "fight", []
    if set(EVENT_ENRICHMENT_COLUMNS) <= columns:
        frame, warnings = event_enrichment_to_fight_enrichment(fights, enrichment[EVENT_ENRICHMENT_COLUMNS].copy())
        return coerce_enrichment_dtypes(frame, ensure_columns=True), "event", warnings
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
    merged = coerce_enrichment_dtypes(fights.copy(), ensure_columns=True)
    for field in ENRICHABLE_FIELDS:
        if field not in merged.columns:
            merged[field] = ""
    enrichment = coerce_enrichment_dtypes(enrichment, ensure_columns=True)

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
            if not _values_equal(merged.at[target_index, field], value):
                merged.at[target_index, field] = value
                updated_fields[field] += 1

    if matched == 0:
        warnings.append("No enrichment rows matched fights.csv. Check fight_date, event, and fighter names.")
    provided_weight = enrichment_copy["weight_class"].map(lambda value: _non_empty(value) and not is_unknown_weight_class(value))
    populated_weight_after = merged["weight_class"].map(lambda value: _non_empty(value) and not is_unknown_weight_class(value))
    if bool(provided_weight.any()) and updated_fields["weight_class"] == 0 and not bool(populated_weight_after.any()):
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
