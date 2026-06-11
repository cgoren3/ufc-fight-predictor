from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.data_io import read_optional_csv


SOURCE_LIVE_SCRAPE = "live scrape"
SOURCE_CSV_IMPORT = "csv import"
SOURCE_MANUAL_HTML = "manual html"
SOURCE_SAMPLE = "sample"
SOURCE_UNKNOWN = "unknown"


def metadata_path(raw_dir: str | Path | None = None) -> Path:
    base = Path(raw_dir) if raw_dir else settings.raw_data_dir
    return base / "source_metadata.json"


def write_source_metadata(
    source: str,
    raw_dir: str | Path | None = None,
    details: dict[str, Any] | None = None,
) -> Path:
    path = metadata_path(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_source_metadata(raw_dir: str | Path | None = None) -> dict[str, Any]:
    path = metadata_path(raw_dir)
    if not path.exists():
        return {"source": SOURCE_UNKNOWN, "details": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"source": SOURCE_UNKNOWN, "details": {"error": f"Invalid metadata JSON at {path}"}}


def imports_dir_has_fights(import_dir: str | Path | None = None) -> bool:
    source = Path(import_dir) if import_dir else settings.raw_data_dir / "imports"
    return (source / "fights.csv").exists()


def summarize_raw_data(
    raw_dir: str | Path | None = None,
    scorecards_path: str | Path | None = None,
) -> dict[str, Any]:
    raw = Path(raw_dir) if raw_dir else settings.raw_data_dir
    scorecards = Path(scorecards_path) if scorecards_path else raw / "scorecards.csv"
    fights = read_optional_csv(raw / "fights.csv", label="fights CSV")
    fighters = read_optional_csv(raw / "fighters.csv", label="fighters CSV")
    fight_stats = read_optional_csv(raw / "fight_stats.csv", label="fight stats CSV")
    scorecards_frame = read_optional_csv(scorecards, label="scorecards CSV")

    unique_fighters: set[str] = set()
    date_min = None
    date_max = None
    if fights is not None and not fights.empty:
        for column in ["fighter_a", "fighter_b"]:
            if column in fights.columns:
                unique_fighters.update(fights[column].dropna().astype(str).str.strip().loc[lambda s: s != ""].tolist())
        if "fight_date" in fights.columns:
            dates = pd.to_datetime(fights["fight_date"], errors="coerce").dropna()
            if not dates.empty:
                date_min = dates.min().date().isoformat()
                date_max = dates.max().date().isoformat()
    if fighters is not None and "name" in fighters.columns:
        unique_fighters.update(fighters["name"].dropna().astype(str).str.strip().loc[lambda s: s != ""].tolist())

    metadata = read_source_metadata(raw)
    return {
        "data_source": metadata.get("source", SOURCE_UNKNOWN),
        "fights_row_count": 0 if fights is None else int(len(fights)),
        "fighters_row_count": 0 if fighters is None else int(len(fighters)),
        "fight_stats_row_count": 0 if fight_stats is None else int(len(fight_stats)),
        "scorecards_row_count": 0 if scorecards_frame is None else int(len(scorecards_frame)),
        "unique_fighters": len(unique_fighters),
        "date_range": {"start": date_min, "end": date_max},
        "metadata": metadata,
    }
