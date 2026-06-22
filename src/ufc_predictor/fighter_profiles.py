from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.data_io import InputDataError, read_optional_csv, read_required_csv
from ufc_predictor.ingest.ufcstats_scraper import UFCStatsScraper


PROFILE_COLUMNS = [
    "name",
    "nickname",
    "height_in",
    "weight_lb",
    "reach_in",
    "stance",
    "date_of_birth",
    "weight_class",
    "source_url",
    "source_file",
]
UPDATE_FIELDS = ["nickname", "height_in", "weight_lb", "reach_in", "stance", "date_of_birth", "weight_class", "source_url"]
CORE_COVERAGE_FIELDS = ["reach_in", "height_in", "stance", "date_of_birth"]
MANUAL_PROFILE_COLUMNS = ["fighter_name", "height", "weight", "reach", "stance", "dob", "weight_class", "source"]


@dataclass
class FighterProfileEnrichmentReport:
    fighters_read: int
    enrichment_rows: int
    fighters_updated: int = 0
    fields_updated: dict[str, int] = field(default_factory=dict)
    source_files_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    matched_fighters: int = 0
    unmatched_enrichment_rows: int = 0
    examples_filled: list[dict[str, Any]] = field(default_factory=list)
    unmatched_examples: list[str] = field(default_factory=list)
    output_path: Path | None = None
    report_path: Path | None = None
    fighters_path: Path | None = None
    backup_dir: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fighters_read": self.fighters_read,
            "enrichment_rows": self.enrichment_rows,
            "fighters_updated": self.fighters_updated,
            "fields_updated": self.fields_updated,
            "source_files_used": self.source_files_used,
            "warnings": self.warnings,
            "coverage": self.coverage,
            "matched_fighters": self.matched_fighters,
            "unmatched_enrichment_rows": self.unmatched_enrichment_rows,
            "examples_filled": self.examples_filled,
            "unmatched_examples": self.unmatched_examples,
            "output_path": str(self.output_path) if self.output_path else None,
            "report_path": str(self.report_path) if self.report_path else None,
            "fighters_path": str(self.fighters_path) if self.fighters_path else None,
            "backup_dir": str(self.backup_dir) if self.backup_dir else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _is_empty(value: Any) -> bool:
    return _clean(value).lower() in {"", "nan", "none", "null", "unknown", "<na>", "no nickname"}


def _name_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _clean(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"\([^)]*\)|\"[^\"]*\"|'[^']*'", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    tokens = []
    raw_tokens = text.split()
    for index, token in enumerate(raw_tokens):
        if token in {"jr", "sr", "ii", "iii", "iv", "v", "the"}:
            continue
        if len(token) == 1 and 0 < index < len(raw_tokens) - 1:
            continue
        tokens.append(token)
    return " ".join(tokens)


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())).strip("_")
        for column in output.columns
    ]
    return output


def _first_present(row: pd.Series | dict[str, Any], aliases: list[str]) -> Any:
    for alias in aliases:
        if alias in row and not _is_empty(row.get(alias)):
            return row.get(alias)
    return ""


def _parse_height(value: Any) -> str:
    text = _clean(value).replace('"', "")
    if not text:
        return ""
    if "'" in text:
        feet, inches = text.split("'", 1)
        inches = inches.replace("in", "").strip()
        try:
            return str(round(float(feet) * 12.0 + float(inches), 2))
        except ValueError:
            return text
    if re.fullmatch(r"\d+\.\d+", text):
        feet_text, inch_text = text.split(".", 1)
        try:
            feet = int(feet_text)
            inches = int(inch_text)
            if inches >= 12 and len(inch_text) == 2 and inch_text.endswith("0"):
                inches = int(inch_text[0])
            return str(float(feet * 12 + inches))
        except ValueError:
            return text
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric) > 12:
        return str(float(numeric))
    return text


def _parse_weight(value: Any) -> str:
    text = _clean(value).replace("lbs.", "").replace("lbs", "").replace("lb", "").strip()
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return str(float(numeric)) if pd.notna(numeric) else text


def _parse_reach(value: Any) -> str:
    text = _clean(value).replace('"', "").strip()
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return str(float(numeric)) if pd.notna(numeric) else text


def _profile_from_row(row: pd.Series, source_file: str) -> dict[str, Any] | None:
    name = _first_present(row, ["name", "fighter", "fighter_name", "full_name", "red_fighter", "blue_fighter", "fighter_full_name"])
    if _is_empty(name):
        return None
    return {
        "name": _clean(name),
        "nickname": _clean(_first_present(row, ["nickname", "nick_name"])),
        "height_in": _parse_height(_first_present(row, ["height_in", "height", "ht", "fighter_height"])),
        "weight_lb": _parse_weight(_first_present(row, ["weight_lb", "weight", "wt", "fighter_weight"])),
        "reach_in": _parse_reach(_first_present(row, ["reach_in", "reach", "fighter_reach"])),
        "stance": _clean(_first_present(row, ["stance", "fighter_stance"])),
        "date_of_birth": _clean(_first_present(row, ["date_of_birth", "dob", "birth_date", "fighter_dob"])),
        "weight_class": _clean(_first_present(row, ["weight_class", "division", "bout_weight", "weightclass"])),
        "source_url": _clean(_first_present(row, ["source_url", "url", "fighter_url"])),
        "source_file": source_file,
    }


def _source_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return sorted(path for path in source_dir.rglob("*.csv") if path.is_file())


def _profiles_from_sources(source_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for path in _source_files(source_dir):
        try:
            frame = _canonical_columns(pd.read_csv(path))
        except Exception:
            continue
        usable_columns = {
            "name",
            "fighter",
            "fighter_name",
            "full_name",
            "height",
            "height_in",
            "ht",
            "reach",
            "reach_in",
            "stance",
            "dob",
            "date_of_birth",
            "nickname",
            "weight_class",
            "wt",
        }
        if not (usable_columns & set(frame.columns)):
            continue
        for _, row in frame.iterrows():
            profile = _profile_from_row(row, str(path))
            if profile is None:
                continue
            if any(not _is_empty(profile.get(field)) for field in UPDATE_FIELDS):
                rows.append(profile)
                used.add(str(path))
    return rows, sorted(used)


def _profiles_from_manual(imports_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    used: list[str] = []
    for filename in ["fighter_profile_enrichment_manual.csv", "fighter_profiles.csv"]:
        path = imports_dir / filename
        frame = read_optional_csv(path, label="manual fighter profile enrichment CSV")
        if frame is None:
            continue
        frame = _canonical_columns(frame)
        for _, row in frame.iterrows():
            profile = _profile_from_row(row, str(path))
            if profile is not None:
                rows.append(profile)
        used.append(str(path))
    return rows, used


def _profiles_from_ufcstats(fighters: pd.DataFrame, source: str, cache_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if source not in {"auto", "ufcstats"} or "source_url" not in fighters.columns:
        return [], [], []
    scraper = UFCStatsScraper(cache_dir=cache_dir)
    rows: list[dict[str, Any]] = []
    used: list[str] = []
    warnings: list[str] = []
    for _, row in fighters.iterrows():
        url = _clean(row.get("source_url"))
        if not url or "/fighter-details/" not in url:
            continue
        if not any(_is_empty(row.get(field)) for field in ["reach_in", "height_in", "stance", "date_of_birth"]):
            continue
        try:
            profile = scraper.scrape_fighter(url)
        except Exception as exc:  # pragma: no cover - network/layout dependent
            warnings.append(f"Could not fetch fighter profile {url}: {type(exc).__name__}: {exc}")
            continue
        profile["source_file"] = url
        rows.append({column: profile.get(column, "") for column in PROFILE_COLUMNS})
        used.append(url)
    return rows, used, warnings


def _dedupe_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _name_key(row.get("name"))
        if not key:
            continue
        target = by_name.setdefault(key, {"name": _clean(row.get("name"))})
        for column in PROFILE_COLUMNS:
            value = row.get(column)
            if not _is_empty(value) and _is_empty(target.get(column)):
                target[column] = value
    return [{column: row.get(column, "") for column in PROFILE_COLUMNS} for row in by_name.values()]


def _coverage(frame: pd.DataFrame) -> dict[str, Any]:
    total = int(len(frame))
    output: dict[str, Any] = {"fighters": total}
    for field in CORE_COVERAGE_FIELDS:
        if field not in frame.columns:
            count = 0
        else:
            count = int((~frame[field].map(_is_empty)).sum())
        output[f"{field}_count"] = count
        output[f"{field}_pct"] = round(count / total * 100.0, 2) if total else 0.0
    return output


def _proposal(
    fighters: pd.DataFrame,
    enrichment: pd.DataFrame,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, int, dict[str, int], int, list[dict[str, Any]], list[str]]:
    output = fighters.copy()
    if "name" not in output.columns:
        raise InputDataError("fighters.csv must contain a name column.")
    for field in UPDATE_FIELDS:
        if field not in output.columns:
            output[field] = pd.NA
        output[field] = output[field].astype("object")
        if field not in enrichment.columns:
            enrichment[field] = pd.NA
    name_to_index = {_name_key(row.get("name")): index for index, row in output.iterrows() if _name_key(row.get("name"))}
    fields_updated: dict[str, int] = {}
    fighters_updated: set[int] = set()
    unmatched = 0
    examples: list[dict[str, Any]] = []
    unmatched_examples: list[str] = []
    for _, row in enrichment.iterrows():
        key = _name_key(row.get("name"))
        if not key or key not in name_to_index:
            unmatched += 1
            if len(unmatched_examples) < 10 and not _is_empty(row.get("name")):
                unmatched_examples.append(_clean(row.get("name")))
            continue
        target = name_to_index[key]
        for field in UPDATE_FIELDS:
            value = row.get(field)
            current = output.at[target, field]
            if _is_empty(value):
                continue
            if overwrite or _is_empty(current):
                if _clean(current) == _clean(value):
                    continue
                output.at[target, field] = value
                fields_updated[field] = fields_updated.get(field, 0) + 1
                fighters_updated.add(target)
                if len(examples) < 10:
                    examples.append(
                        {
                            "fighter": output.at[target, "name"],
                            "field": field,
                            "old_value": "" if _is_empty(current) else current,
                            "new_value": value,
                        }
                    )
    return output, len(fighters_updated), fields_updated, unmatched, examples, unmatched_examples


def merge_profile_enrichment(fighters: pd.DataFrame, enrichment: pd.DataFrame, overwrite: bool = False) -> tuple[pd.DataFrame, int, dict[str, int]]:
    merged, fighters_updated, fields_updated, _, _, _ = _proposal(fighters, enrichment, overwrite=overwrite)
    return merged, fighters_updated, fields_updated


def _report_for_frames(
    fighters: pd.DataFrame,
    enrichment: pd.DataFrame,
    *,
    source_files_used: list[str],
    warnings: list[str],
    output_path: Path,
    report_path: Path,
    fighters_path: Path,
    overwrite: bool = False,
) -> FighterProfileEnrichmentReport:
    merged, fighters_updated, fields_updated, unmatched, examples, unmatched_examples = _proposal(
        fighters,
        enrichment,
        overwrite=overwrite,
    )
    coverage_before = _coverage(fighters)
    coverage_after = _coverage(merged)
    matched = int(len(enrichment) - unmatched)
    coverage = {
        "existing_fighters_count": int(len(fighters)),
        "existing_reach_coverage_before": coverage_before.get("reach_in_pct", 0.0),
        "existing_height_coverage_before": coverage_before.get("height_in_pct", 0.0),
        "existing_stance_coverage_before": coverage_before.get("stance_pct", 0.0),
        "existing_dob_coverage_before": coverage_before.get("date_of_birth_pct", 0.0),
        "enrichment_rows_with_reach": int((~enrichment.get("reach_in", pd.Series(dtype=object)).map(_is_empty)).sum()) if not enrichment.empty else 0,
        "enrichment_rows_with_height": int((~enrichment.get("height_in", pd.Series(dtype=object)).map(_is_empty)).sum()) if not enrichment.empty else 0,
        "enrichment_rows_with_stance": int((~enrichment.get("stance", pd.Series(dtype=object)).map(_is_empty)).sum()) if not enrichment.empty else 0,
        "enrichment_rows_with_dob": int((~enrichment.get("date_of_birth", pd.Series(dtype=object)).map(_is_empty)).sum()) if not enrichment.empty else 0,
        "matched_fighters": matched,
        "unmatched_enrichment_rows": unmatched,
        "reach_coverage_after": coverage_after.get("reach_in_pct", 0.0),
        "height_coverage_after": coverage_after.get("height_in_pct", 0.0),
        "stance_coverage_after": coverage_after.get("stance_pct", 0.0),
        "dob_coverage_after": coverage_after.get("date_of_birth_pct", 0.0),
        "reach_count_before": coverage_before.get("reach_in_count", 0),
        "reach_count_after": coverage_after.get("reach_in_count", 0),
        "height_count_before": coverage_before.get("height_in_count", 0),
        "height_count_after": coverage_after.get("height_in_count", 0),
        "stance_count_before": coverage_before.get("stance_count", 0),
        "stance_count_after": coverage_after.get("stance_count", 0),
        "dob_count_before": coverage_before.get("date_of_birth_count", 0),
        "dob_count_after": coverage_after.get("date_of_birth_count", 0),
    }
    return FighterProfileEnrichmentReport(
        fighters_read=int(len(fighters)),
        enrichment_rows=int(len(enrichment)),
        fighters_updated=fighters_updated,
        fields_updated=fields_updated,
        source_files_used=sorted(set(source_files_used)),
        warnings=warnings,
        coverage=coverage,
        matched_fighters=matched,
        unmatched_enrichment_rows=unmatched,
        examples_filled=examples,
        unmatched_examples=unmatched_examples,
        output_path=output_path,
        report_path=report_path,
        fighters_path=fighters_path,
    )


def enrich_fighter_profiles(
    source: str = "auto",
    fighters_path: str | Path | None = None,
    imports_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> FighterProfileEnrichmentReport:
    source_name = source.strip().lower()
    if source_name not in {"auto", "local", "ufcstats"}:
        raise InputDataError("--source must be one of: auto, local, ufcstats.")
    imports = Path(imports_dir) if imports_dir else settings.raw_data_dir / "imports"
    fighters_csv = Path(fighters_path) if fighters_path else imports / "fighters.csv"
    sources = Path(source_dir) if source_dir else settings.raw_data_dir / "enrichment_sources"
    output = Path(output_path) if output_path else imports / "fighter_profile_enrichment.csv"
    report_output = Path(report_path) if report_path else settings.raw_data_dir / "staging" / "fighter_profile_enrichment_report.json"
    fighters = read_required_csv(fighters_csv, required_columns=["name"], label="fighters CSV")
    rows: list[dict[str, Any]] = []
    used: list[str] = []
    warnings: list[str] = []
    manual_rows, manual_used = _profiles_from_manual(imports)
    rows.extend(manual_rows)
    used.extend(manual_used)
    if source_name in {"auto", "local"}:
        local_rows, local_used = _profiles_from_sources(sources)
        rows.extend(local_rows)
        used.extend(local_used)
    cache_dir = settings.raw_data_dir / "staging" / "fighter_pages_cache"
    ufcstats_rows, ufcstats_used, ufcstats_warnings = _profiles_from_ufcstats(fighters, source_name, cache_dir)
    rows.extend(ufcstats_rows)
    used.extend(ufcstats_used)
    warnings.extend(ufcstats_warnings)
    enrichment = pd.DataFrame(_dedupe_profiles(rows), columns=PROFILE_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    enrichment.to_csv(output, index=False)
    report = _report_for_frames(
        fighters,
        enrichment,
        source_files_used=used,
        warnings=warnings,
        output_path=output,
        report_path=report_output,
        fighters_path=fighters_csv,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return report


def validate_fighter_profile_enrichment(
    fighters_path: str | Path | None = None,
    enrichment_path: str | Path | None = None,
    report_path: str | Path | None = None,
    overwrite: bool = False,
) -> FighterProfileEnrichmentReport:
    fighters_csv = Path(fighters_path) if fighters_path else settings.raw_data_dir / "imports" / "fighters.csv"
    enrichment_csv = Path(enrichment_path) if enrichment_path else settings.raw_data_dir / "imports" / "fighter_profile_enrichment.csv"
    report_output = Path(report_path) if report_path else settings.raw_data_dir / "staging" / "fighter_profile_enrichment_validation_report.json"
    fighters = read_required_csv(fighters_csv, required_columns=["name"], label="fighters CSV")
    enrichment = read_required_csv(enrichment_csv, required_columns=["name"], label="fighter profile enrichment CSV")
    report = _report_for_frames(
        fighters,
        enrichment,
        source_files_used=[],
        warnings=[],
        output_path=enrichment_csv,
        report_path=report_output,
        fighters_path=fighters_csv,
        overwrite=overwrite,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return report


def apply_fighter_profile_enrichment(
    fighters_path: str | Path | None = None,
    enrichment_path: str | Path | None = None,
    backup_root: str | Path | None = None,
    overwrite: bool = False,
) -> FighterProfileEnrichmentReport:
    fighters_csv = Path(fighters_path) if fighters_path else settings.raw_data_dir / "imports" / "fighters.csv"
    enrichment_csv = Path(enrichment_path) if enrichment_path else settings.raw_data_dir / "imports" / "fighter_profile_enrichment.csv"
    backup_base = Path(backup_root) if backup_root else settings.raw_data_dir / "backups"
    fighters = read_required_csv(fighters_csv, required_columns=["name"], label="fighters CSV")
    enrichment = read_required_csv(enrichment_csv, required_columns=["name"], label="fighter profile enrichment CSV")
    merged, fighters_updated, fields_updated, unmatched, examples, unmatched_examples = _proposal(fighters, enrichment, overwrite=overwrite)
    backup_dir = backup_base / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    if fighters_csv.exists():
        shutil.copy2(fighters_csv, backup_dir / fighters_csv.name)
    merged.to_csv(fighters_csv, index=False)
    report_output = settings.raw_data_dir / "staging" / "fighter_profile_enrichment_apply_report.json"
    report = _report_for_frames(
        fighters,
        enrichment,
        source_files_used=[],
        warnings=[],
        output_path=enrichment_csv,
        report_path=report_output,
        fighters_path=fighters_csv,
        overwrite=overwrite,
    )
    report.fighters_updated = fighters_updated
    report.fields_updated = fields_updated
    report.unmatched_enrichment_rows = unmatched
    report.examples_filled = examples
    report.unmatched_examples = unmatched_examples
    report.backup_dir = backup_dir
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return report


def import_fighter_profile_csv(
    file: str | Path,
    output_path: str | Path | None = None,
    append: bool = True,
) -> tuple[pd.DataFrame, Path]:
    source = Path(file)
    if not source.exists():
        raise InputDataError(f"Manual fighter profile CSV not found: {source}")
    frame = pd.read_csv(source)
    missing = [column for column in MANUAL_PROFILE_COLUMNS if column not in frame.columns]
    if missing:
        raise InputDataError("Manual fighter profile CSV is missing required columns: " + ", ".join(missing))
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "name": _clean(row.get("fighter_name")),
                "nickname": "",
                "height_in": _parse_height(row.get("height")),
                "weight_lb": _parse_weight(row.get("weight")),
                "reach_in": _parse_reach(row.get("reach")),
                "stance": _clean(row.get("stance")),
                "date_of_birth": _clean(row.get("dob")),
                "weight_class": _clean(row.get("weight_class")),
                "source_url": "",
                "source_file": _clean(row.get("source")) or str(source),
            }
        )
    output = Path(output_path) if output_path else settings.raw_data_dir / "imports" / "fighter_profile_enrichment.csv"
    manual = pd.DataFrame(rows, columns=PROFILE_COLUMNS)
    if append and output.exists() and output.stat().st_size > 0:
        existing = pd.read_csv(output)
        combined = pd.concat([existing, manual], ignore_index=True)
        manual = pd.DataFrame(_dedupe_profiles(combined.to_dict(orient="records")), columns=PROFILE_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    manual.to_csv(output, index=False)
    return manual, output
