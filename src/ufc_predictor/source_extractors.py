from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re

import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.ingest.scorecards_loader import REQUIRED_COLUMNS as REQUIRED_SCORECARD_COLUMNS
from ufc_predictor.odds import is_valid_american_odds


ODDS_EXTRACT_COLUMNS = [
    "fight_date",
    "event",
    "fighter_a",
    "fighter_b",
    "sportsbook",
    "fighter_a_odds",
    "fighter_b_odds",
    "timestamp",
    "source_file",
]

SCORECARD_EXTRACT_COLUMNS = REQUIRED_SCORECARD_COLUMNS + [
    "winner",
    "card_type",
    "raw_scorecards",
    "source_file",
]


@dataclass
class ExtractionReport:
    output_path: Path
    rows_written: int
    source_files_scanned: int
    source_files_used: list[str] = field(default_factory=list)
    skipped_rows: int = 0
    warnings: list[str] = field(default_factory=list)


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())).strip("_")
        for column in output.columns
    ]
    return output


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _non_empty(value: Any) -> bool:
    return _clean(value).lower() not in {"", "nan", "none", "null"}


def _date_key(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return ""
    return date.date().isoformat()


def _source_csv_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return sorted(path for path in source_dir.rglob("*.csv") if path.is_file())


def _event_id(row: pd.Series | dict[str, Any]) -> str:
    for column in ["event_id", "event_id_x", "event_id_y"]:
        if column in row and _non_empty(row.get(column)):
            return _clean(row.get(column))
    return ""


def _first_present(row: pd.Series | dict[str, Any], columns: list[str]) -> Any:
    for column in columns:
        if column in row and _non_empty(row.get(column)):
            return row.get(column)
    return ""


def _build_event_map(paths: list[Path]) -> dict[str, dict[str, str]]:
    event_map: dict[str, dict[str, str]] = {}
    for path in paths:
        try:
            frame = _canonical_columns(pd.read_csv(path))
        except Exception:
            continue
        if "event_id" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            event_id = _event_id(row)
            if not event_id:
                continue
            event_name = _first_present(row, ["event", "event_name", "event_title", "name"])
            event_date = _first_present(row, ["fight_date", "event_date", "date"])
            if _non_empty(event_name) or _non_empty(event_date):
                event_map.setdefault(event_id, {})
                if _non_empty(event_name):
                    event_map[event_id].setdefault("event", _clean(event_name))
                if _non_empty(event_date):
                    event_map[event_id].setdefault("fight_date", _date_key(event_date))
    return event_map


def _read_source(path: Path) -> pd.DataFrame | None:
    try:
        return _canonical_columns(pd.read_csv(path))
    except Exception:
        return None


def _has_espn_fighter_columns(frame: pd.DataFrame) -> bool:
    return {"fight_date", "red_fighter", "blue_fighter"} <= set(frame.columns)


def _odds_value(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _source_event(row: pd.Series, event_map: dict[str, dict[str, str]]) -> str:
    event = _first_present(row, ["event", "event_name", "event_title"])
    if _non_empty(event):
        return _clean(event)
    event_id = _event_id(row)
    return event_map.get(event_id, {}).get("event", "")


def _source_fight_date(row: pd.Series, event_map: dict[str, dict[str, str]]) -> str:
    fight_date = _first_present(row, ["fight_date", "event_date", "date"])
    if _non_empty(fight_date):
        return _date_key(fight_date)
    event_id = _event_id(row)
    return event_map.get(event_id, {}).get("fight_date", "")


def extract_odds_from_sources(
    source_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, ExtractionReport]:
    directory = Path(source_dir) if source_dir else settings.raw_data_dir / "enrichment_sources"
    output = Path(output_path) if output_path else settings.raw_data_dir / "imports" / "odds.csv"
    paths = _source_csv_files(directory)
    event_map = _build_event_map(paths)
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    skipped = 0
    warnings: list[str] = []

    for path in paths:
        frame = _read_source(path)
        if frame is None or not _has_espn_fighter_columns(frame):
            continue
        if not {"red_fighter_moneyline_odds", "blue_fighter_moneyline_odds"} <= set(frame.columns):
            continue
        for _, row in frame.iterrows():
            red_odds = _odds_value(row.get("red_fighter_moneyline_odds"))
            blue_odds = _odds_value(row.get("blue_fighter_moneyline_odds"))
            if red_odds is None and blue_odds is None:
                skipped += 1
                continue
            if not is_valid_american_odds(red_odds) or not is_valid_american_odds(blue_odds):
                skipped += 1
                warnings.append(f"Skipped invalid American odds in {path.name}.")
                continue
            fight_date = _source_fight_date(row, event_map)
            fighter_a = _clean(row.get("red_fighter"))
            fighter_b = _clean(row.get("blue_fighter"))
            if not fight_date or not fighter_a or not fighter_b:
                skipped += 1
                continue
            rows.append(
                {
                    "fight_date": fight_date,
                    "event": _source_event(row, event_map),
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "sportsbook": _clean(row.get("sportsbook")) or "espn_source",
                    "fighter_a_odds": int(red_odds),
                    "fighter_b_odds": int(blue_odds),
                    "timestamp": "",
                    "source_file": str(path),
                }
            )
            used.add(str(path))

    frame = pd.DataFrame(rows, columns=ODDS_EXTRACT_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    report = ExtractionReport(
        output_path=output,
        rows_written=int(len(frame)),
        source_files_scanned=len(paths),
        source_files_used=sorted(used),
        skipped_rows=skipped,
        warnings=warnings[:10],
    )
    return frame, report


def extract_scorecards_from_sources(
    source_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, ExtractionReport]:
    directory = Path(source_dir) if source_dir else settings.raw_data_dir / "enrichment_sources"
    output = Path(output_path) if output_path else settings.raw_data_dir / "imports" / "scorecards.csv"
    paths = _source_csv_files(directory)
    event_map = _build_event_map(paths)
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    skipped = 0

    for path in paths:
        frame = _read_source(path)
        if frame is None or not _has_espn_fighter_columns(frame):
            continue
        if "score_cards" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            raw_scorecards = _clean(row.get("score_cards"))
            if not raw_scorecards:
                skipped += 1
                continue
            fight_date = _source_fight_date(row, event_map)
            fighter_a = _clean(row.get("red_fighter"))
            fighter_b = _clean(row.get("blue_fighter"))
            if not fight_date or not fighter_a or not fighter_b:
                skipped += 1
                continue
            rows.append(
                {
                    "fight_date": fight_date,
                    "event": _source_event(row, event_map),
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "judge": "raw_scorecards",
                    "round_1_a": pd.NA,
                    "round_1_b": pd.NA,
                    "round_2_a": pd.NA,
                    "round_2_b": pd.NA,
                    "round_3_a": pd.NA,
                    "round_3_b": pd.NA,
                    "round_4_a": pd.NA,
                    "round_4_b": pd.NA,
                    "round_5_a": pd.NA,
                    "round_5_b": pd.NA,
                    "total_a": pd.NA,
                    "total_b": pd.NA,
                    "decision_type": _clean(row.get("card_type")),
                    "winner": _clean(row.get("winner")),
                    "card_type": _clean(row.get("card_type")),
                    "raw_scorecards": raw_scorecards,
                    "source_file": str(path),
                }
            )
            used.add(str(path))

    frame = pd.DataFrame(rows, columns=SCORECARD_EXTRACT_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    report = ExtractionReport(
        output_path=output,
        rows_written=int(len(frame)),
        source_files_scanned=len(paths),
        source_files_used=sorted(used),
        skipped_rows=skipped,
    )
    return frame, report
