from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.data_io import InputDataError, read_optional_csv, read_required_csv
from ufc_predictor.ingest.ufcstats_scraper import UFCStatsScraper


PROFILE_COLUMNS = ["name", "nickname", "height_in", "weight_lb", "reach_in", "stance", "date_of_birth", "weight_class", "source_url", "source_file"]
UPDATE_FIELDS = ["nickname", "height_in", "weight_lb", "reach_in", "stance", "date_of_birth", "weight_class", "source_url"]


@dataclass
class FighterProfileEnrichmentReport:
    fighters_read: int
    enrichment_rows: int
    fighters_updated: int
    fields_updated: dict[str, int] = field(default_factory=dict)
    source_files_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_path: Path | None = None
    report_path: Path | None = None
    fighters_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fighters_read": self.fighters_read,
            "enrichment_rows": self.enrichment_rows,
            "fighters_updated": self.fighters_updated,
            "fields_updated": self.fields_updated,
            "source_files_used": self.source_files_used,
            "warnings": self.warnings,
            "output_path": str(self.output_path) if self.output_path else None,
            "report_path": str(self.report_path) if self.report_path else None,
            "fighters_path": str(self.fighters_path) if self.fighters_path else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _normal(value: Any) -> str:
    return _clean(value).lower()


def _is_empty(value: Any) -> bool:
    return _clean(value).lower() in {"", "nan", "none", "null", "unknown", "<na>"}


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    import re

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


def _source_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return sorted(path for path in source_dir.rglob("*.csv") if path.is_file())


def _profile_from_row(row: pd.Series, source_file: str) -> dict[str, Any] | None:
    name = _first_present(row, ["name", "fighter", "fighter_name", "full_name", "red_fighter", "blue_fighter"])
    if _is_empty(name):
        return None
    return {
        "name": _clean(name),
        "nickname": _clean(_first_present(row, ["nickname", "nick_name"])),
        "height_in": _clean(_first_present(row, ["height_in", "height", "fighter_height"])),
        "weight_lb": _clean(_first_present(row, ["weight_lb", "weight", "fighter_weight"])),
        "reach_in": _clean(_first_present(row, ["reach_in", "reach", "fighter_reach"])),
        "stance": _clean(_first_present(row, ["stance", "fighter_stance"])),
        "date_of_birth": _clean(_first_present(row, ["date_of_birth", "dob", "birth_date", "fighter_dob"])),
        "weight_class": _clean(_first_present(row, ["weight_class", "division", "bout_weight", "weightclass"])),
        "source_url": _clean(_first_present(row, ["source_url", "url", "fighter_url"])),
        "source_file": source_file,
    }


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
            "reach",
            "reach_in",
            "stance",
            "dob",
            "date_of_birth",
            "nickname",
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


def _profiles_from_ufcstats(fighters: pd.DataFrame, source: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if source not in {"auto", "ufcstats"}:
        return [], [], []
    if "source_url" not in fighters.columns:
        return [], [], []
    scraper = UFCStatsScraper()
    rows: list[dict[str, Any]] = []
    used: list[str] = []
    warnings: list[str] = []
    for _, row in fighters.iterrows():
        url = _clean(row.get("source_url"))
        if not url or "/fighter-details/" not in url:
            continue
        missing_profile = any(_is_empty(row.get(field)) for field in ["reach_in", "height_in", "stance", "date_of_birth"])
        if not missing_profile:
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
        key = _normal(row.get("name"))
        if not key:
            continue
        target = by_name.setdefault(key, {"name": _clean(row.get("name"))})
        for column in PROFILE_COLUMNS:
            value = row.get(column)
            if not _is_empty(value) and _is_empty(target.get(column)):
                target[column] = value
    return [{column: row.get(column, "") for column in PROFILE_COLUMNS} for row in by_name.values()]


def merge_profile_enrichment(fighters: pd.DataFrame, enrichment: pd.DataFrame) -> tuple[pd.DataFrame, int, dict[str, int]]:
    output = fighters.copy()
    if "name" not in output.columns:
        raise InputDataError("fighters.csv must contain a name column.")
    for field in UPDATE_FIELDS:
        if field not in output.columns:
            output[field] = pd.NA
        output[field] = output[field].astype("object")
        if field not in enrichment.columns:
            enrichment[field] = pd.NA
    name_to_index = {_normal(row.get("name")): index for index, row in output.iterrows() if _normal(row.get("name"))}
    fields_updated: dict[str, int] = {}
    fighters_updated: set[int] = set()
    for _, row in enrichment.iterrows():
        key = _normal(row.get("name"))
        if not key or key not in name_to_index:
            continue
        target = name_to_index[key]
        for field in UPDATE_FIELDS:
            value = row.get(field)
            if not _is_empty(value) and _is_empty(output.at[target, field]):
                output.at[target, field] = value
                fields_updated[field] = fields_updated.get(field, 0) + 1
                fighters_updated.add(target)
    return output, len(fighters_updated), fields_updated


def enrich_fighter_profiles(
    source: str = "auto",
    fighters_path: str | Path | None = None,
    imports_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    apply: bool = True,
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
    ufcstats_rows, ufcstats_used, ufcstats_warnings = _profiles_from_ufcstats(fighters, source_name)
    rows.extend(ufcstats_rows)
    used.extend(ufcstats_used)
    warnings.extend(ufcstats_warnings)
    deduped = _dedupe_profiles(rows)
    enrichment = pd.DataFrame(deduped, columns=PROFILE_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    enrichment.to_csv(output, index=False)
    fighters_updated = 0
    fields_updated: dict[str, int] = {}
    if apply and not enrichment.empty:
        merged, fighters_updated, fields_updated = merge_profile_enrichment(fighters, enrichment)
        merged.to_csv(fighters_csv, index=False)
    report = FighterProfileEnrichmentReport(
        fighters_read=int(len(fighters)),
        enrichment_rows=int(len(enrichment)),
        fighters_updated=fighters_updated,
        fields_updated=fields_updated,
        source_files_used=sorted(set(used)),
        warnings=warnings,
        output_path=output,
        report_path=report_output,
        fighters_path=fighters_csv,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return report
