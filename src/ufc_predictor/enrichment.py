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
ENRICHABLE_FIELDS = ["weight_class", "event_location", "main_event", "title_fight", "scheduled_rounds"]
UNKNOWN_WEIGHT_VALUES = {"", "unknown", "unk", "n/a", "na", "none", "nan"}


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


def merge_enrichment_into_fights(
    fights: pd.DataFrame,
    enrichment: pd.DataFrame,
    fights_path: str | Path,
    enrichment_path: str | Path,
    output_path: str | Path,
) -> tuple[pd.DataFrame, EnrichmentReport]:
    warnings = validate_enrichment_frame(enrichment)
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
    inspect_csv(source, required_columns=ENRICHMENT_COLUMNS, require_rows=True, label="fight_enrichment.csv")
    fights = pd.read_csv(fights_csv)
    enrichment = pd.read_csv(source)
    merged, report = merge_enrichment_into_fights(
        fights=fights,
        enrichment=enrichment,
        fights_path=fights_csv,
        enrichment_path=source,
        output_path=output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return report
