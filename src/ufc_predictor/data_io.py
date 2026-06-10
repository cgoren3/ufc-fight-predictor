from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class InputDataError(RuntimeError):
    """Raised when a required local data file is missing or unusable."""


@dataclass(frozen=True)
class CsvInspection:
    path: Path
    headers: list[str]
    has_rows: bool


def inspect_csv(
    path: str | Path,
    required_columns: list[str] | tuple[str, ...] | None = None,
    require_rows: bool = True,
    label: str = "CSV",
) -> CsvInspection:
    """Validate that a CSV exists, has headers, and optionally has rows.

    This intentionally uses the standard library instead of pandas so callers can
    produce useful messages before pandas raises EmptyDataError.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise InputDataError(
            f"Missing {label} at {csv_path}. Run `ufc-predict ingest-ufcstats` "
            "or use `ufc-predict build-dataset --use-sample-data` for a local dev dataset."
        )
    if csv_path.stat().st_size == 0:
        raise InputDataError(
            f"{label} at {csv_path} is empty. Rerun `ufc-predict ingest-ufcstats`, "
            "or check the UFCStats scraper cache/resume files under data/raw/cache. "
            "For development, run `ufc-predict build-dataset --use-sample-data`."
        )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = [column.strip() for column in next(reader)]
        except StopIteration as exc:
            raise InputDataError(f"{label} at {csv_path} has no header row.") from exc
        if not any(headers):
            raise InputDataError(f"{label} at {csv_path} has a blank header row.")
        missing = [column for column in required_columns or [] if column not in headers]
        if missing:
            raise InputDataError(f"{label} at {csv_path} is missing required columns: {', '.join(missing)}")
        has_rows = any(any(cell.strip() for cell in row) for row in reader)
        if require_rows and not has_rows:
            raise InputDataError(
                f"{label} at {csv_path} has headers but no data rows. Rerun "
                "`ufc-predict ingest-ufcstats`, check the scraper/cache, or use "
                "`ufc-predict build-dataset --use-sample-data` for development."
            )
    return CsvInspection(path=csv_path, headers=headers, has_rows=has_rows)


def read_required_csv(
    path: str | Path,
    required_columns: list[str] | tuple[str, ...] | None = None,
    label: str = "CSV",
) -> pd.DataFrame:
    inspect_csv(path, required_columns=required_columns, require_rows=True, label=label)
    return pd.read_csv(path)


def read_optional_csv(path: str | Path, label: str = "optional CSV") -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        inspection = inspect_csv(csv_path, require_rows=False, label=label)
    except InputDataError:
        return None
    if not inspection.has_rows:
        return None
    return pd.read_csv(csv_path)
