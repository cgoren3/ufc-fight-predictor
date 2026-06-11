from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.data_io import InputDataError, inspect_csv
from ufc_predictor.ingest.scorecards_loader import REQUIRED_COLUMNS as SCORECARD_COLUMNS
from ufc_predictor.ingest.ufcstats_scraper import FIGHT_COLUMNS, FIGHT_STAT_COLUMNS, FIGHTER_COLUMNS


class DatasetAdapterError(RuntimeError):
    """Raised when an external dataset cannot be mapped safely."""


@dataclass(frozen=True)
class CsvColumnInfo:
    path: Path
    columns: list[str]


@dataclass
class FightMapping:
    raw: pd.DataFrame
    output: pd.DataFrame
    source_path: Path
    red_col: str
    blue_col: str
    date_col: str
    source_id_col: str | None
    source_to_output_ids: dict[str, int]


@dataclass
class AdaptDatasetResult:
    output_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source_files: dict[str, Path] = field(default_factory=dict)
    copied_existing_schema: bool = False


EVENT_ALIASES = ["event_name", "event", "event_title", "event_name_title", "Event"]
DATE_ALIASES = ["fight_date", "date", "Date", "bout_date"]
LOCATION_ALIASES = ["event_location", "location", "Location", "venue"]
RED_FIGHTER_ALIASES = ["red_fighter", "r_fighter", "R_fighter", "RedFighter", "fighter_a"]
BLUE_FIGHTER_ALIASES = ["blue_fighter", "b_fighter", "B_fighter", "BlueFighter", "fighter_b"]
WINNER_ALIASES = ["winner", "Winner", "result", "Result"]
WEIGHT_CLASS_ALIASES = ["weight_class", "weightclass", "fight_type", "Fight_type", "division"]
METHOD_ALIASES = ["method", "finish_method", "win_method", "method_detailed"]
FINISH_ROUND_ALIASES = ["finish_round", "round", "Round", "last_round"]
FINISH_TIME_ALIASES = ["finish_time", "time", "Time", "last_round_time"]
SCHEDULED_ROUNDS_ALIASES = ["scheduled_rounds", "no_of_rounds", "rounds", "max_rounds"]
FIGHT_ID_ALIASES = ["fight_id", "bout_id", "id", "fightid"]
SOURCE_URL_ALIASES = ["source_url", "url", "fight_url"]

PROFILE_NAME_ALIASES = ["name", "fighter", "fighter_name", "Fighter"]
STANCE_ALIASES = ["stance", "Stance"]
HEIGHT_ALIASES = ["height_in", "height", "height_inches", "height_cms", "height_cm"]
WEIGHT_ALIASES = ["weight_lb", "weight", "weight_lbs", "weight_class_weight"]
REACH_ALIASES = ["reach_in", "reach", "reach_inches", "reach_cms", "reach_cm"]
DOB_ALIASES = ["date_of_birth", "dob", "DOB", "birth_date"]
RECORD_ALIASES = ["record", "pro_record"]

STAT_ALIASES = {
    "knockdowns": ["knockdowns", "knockdown", "kd"],
    "sig_str": ["sig_str", "sig_strikes", "significant_strikes", "sig_str_att"],
    "sig_str_landed": ["sig_str_landed", "significant_strikes_landed", "sig_landed", "sigstr_landed"],
    "sig_str_attempted": ["sig_str_attempted", "significant_strikes_attempted", "sig_attempted", "sigstr_attempted"],
    "total_str": ["total_str", "total_strikes", "tot_str"],
    "total_str_landed": ["total_str_landed", "total_strikes_landed", "total_landed", "tot_str_landed"],
    "total_str_attempted": ["total_str_attempted", "total_strikes_attempted", "total_attempted", "tot_str_attempted"],
    "takedowns": ["td", "takedowns", "take_downs"],
    "takedowns_landed": ["takedowns_landed", "td_landed", "tds_landed"],
    "takedowns_attempted": ["takedowns_attempted", "td_attempted", "tds_attempted"],
    "submission_attempts": ["submission_attempts", "sub_attempts", "sub_att", "sub"],
    "reversals": ["reversals", "rev"],
    "control_seconds": ["control_seconds", "control_time", "control", "ctrl"],
    "head_landed": ["head_landed", "head_strikes_landed", "head"],
    "body_landed": ["body_landed", "body_strikes_landed", "body"],
    "leg_landed": ["leg_landed", "leg_strikes_landed", "leg"],
}


def normalize_column(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _column_lookup(columns: list[str] | pd.Index) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for column in columns:
        lookup.setdefault(normalize_column(column), []).append(str(column))
    return lookup


def _find_column(
    columns: list[str] | pd.Index,
    aliases: list[str],
    label: str,
    *,
    required: bool = False,
    context: str = "CSV",
) -> str | None:
    lookup = _column_lookup(columns)
    for alias in aliases:
        matches = lookup.get(normalize_column(alias), [])
        if len(matches) > 1:
            raise DatasetAdapterError(f"{context} has duplicate candidate columns for {label}: {matches}")
        if matches:
            return matches[0]
    if required:
        raise DatasetAdapterError(
            f"{context} is missing required {label} column. Looked for aliases: {', '.join(aliases)}"
        )
    return None


def _find_side_column(columns: list[str] | pd.Index, side: str, aliases: list[str]) -> str | None:
    prefixes = ["r", "red"] if side == "red" else ["b", "blue"]
    side_aliases = []
    for prefix in prefixes:
        for alias in aliases:
            normalized = normalize_column(alias)
            side_aliases.extend([f"{prefix}_{normalized}", f"{prefix}{normalized}"])
    return _find_column(columns, side_aliases, f"{side} {aliases[0]}", required=False)


def _csv_files(source: str | Path) -> list[Path]:
    path = Path(source)
    if not path.exists():
        raise DatasetAdapterError(f"Source folder does not exist: {path}")
    if not path.is_dir():
        raise DatasetAdapterError(f"Source must be a folder containing CSV files: {path}")
    files = sorted(candidate for candidate in path.rglob("*.csv") if candidate.is_file())
    if not files:
        raise DatasetAdapterError(f"No CSV files found under source folder: {path}")
    return files


def list_csv_columns(source: str | Path) -> list[CsvColumnInfo]:
    infos: list[CsvColumnInfo] = []
    for path in _csv_files(source):
        try:
            headers = inspect_csv(path, require_rows=False, label=f"{path.name}").headers
        except InputDataError as exc:
            raise DatasetAdapterError(str(exc)) from exc
        infos.append(CsvColumnInfo(path=path, columns=headers))
    return infos


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copyfile(source, destination)


def _clear_known_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["fights.csv", "fighters.csv", "fight_stats.csv", "scorecards.csv"]:
        (output_dir / name).unlink(missing_ok=True)


def _copy_if_already_schema(source: Path, output_dir: Path) -> AdaptDatasetResult | None:
    required = {
        "fights": (source / "fights.csv", ["fighter_a", "fighter_b", "fight_date", "winner"]),
        "fighters": (source / "fighters.csv", ["name"]),
        "fight_stats": (source / "fight_stats.csv", ["fight_id", "fighter", "opponent"]),
    }
    if not all(path.exists() for path, _ in required.values()):
        return None

    try:
        for name, (path, columns) in required.items():
            inspect_csv(path, required_columns=columns, require_rows=True, label=f"{name}.csv")
        if (source / "scorecards.csv").exists():
            inspect_csv(
                source / "scorecards.csv",
                required_columns=list(SCORECARD_COLUMNS),
                require_rows=False,
                label="scorecards.csv",
            )
    except InputDataError:
        return None

    if source.resolve() != output_dir.resolve():
        _clear_known_outputs(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    result = AdaptDatasetResult(output_dir=output_dir, copied_existing_schema=True)
    for name in ["fights", "fighters", "fight_stats", "scorecards"]:
        source_file = source / f"{name}.csv"
        if source_file.exists():
            destination = output_dir / f"{name}.csv"
            _copy_file(source_file, destination)
            result.files[name] = destination
            result.source_files[name] = source_file
    return result


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DatasetAdapterError(f"Could not read CSV file {path}: {type(exc).__name__}: {exc}") from exc


def _has_pair_fight_columns(info: CsvColumnInfo) -> bool:
    columns = info.columns
    return all(
        _find_column(columns, aliases, label, required=False) is not None
        for aliases, label in [
            (RED_FIGHTER_ALIASES, "red fighter"),
            (BLUE_FIGHTER_ALIASES, "blue fighter"),
            (DATE_ALIASES, "fight date"),
            (WINNER_ALIASES, "winner"),
        ]
    )


def _select_fight_file(infos: list[CsvColumnInfo]) -> CsvColumnInfo:
    candidates = [info for info in infos if _has_pair_fight_columns(info)]
    if not candidates:
        raise DatasetAdapterError(
            "Could not find a fight-level CSV with red/blue fighter, date, and winner columns. "
            "Run `ufc-predict dataset-columns --source <folder>` and either rename columns or "
            "provide already-normalized fights.csv, fighters.csv, and fight_stats.csv files."
        )
    if len(candidates) > 1:
        names = ", ".join(str(info.path) for info in candidates)
        raise DatasetAdapterError(
            "Multiple fight-level CSV candidates were found. To avoid guessing, keep only one "
            f"candidate in the source folder or normalize the dataset first. Candidates: {names}"
        )
    return candidates[0]


def _date_series(frame: pd.DataFrame, column: str, path: Path) -> pd.Series:
    dates = pd.to_datetime(frame[column], errors="coerce")
    bad = frame.loc[dates.isna(), column].head(5).tolist()
    if bad:
        raise DatasetAdapterError(f"Could not parse fight dates in {path}. Bad examples: {bad}")
    return dates.dt.date.astype(str)


def _numeric_or_blank(value: Any) -> Any:
    text = _clean(value)
    if not text:
        return ""
    number = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return "" if pd.isna(number) else number


def _resolve_winner(row: pd.Series, winner_col: str, red_col: str, blue_col: str) -> str:
    winner = _clean(row.get(winner_col))
    red = _clean(row.get(red_col))
    blue = _clean(row.get(blue_col))
    winner_lower = winner.lower()
    if winner_lower in {"", "draw", "nc", "no contest", "no-contest"}:
        return ""
    if winner_lower in {"red", "r", "red fighter", "r_fighter"}:
        return red
    if winner_lower in {"blue", "b", "blue fighter", "b_fighter"}:
        return blue
    if winner_lower == red.lower():
        return red
    if winner_lower == blue.lower():
        return blue
    raise DatasetAdapterError(
        f"Winner value '{winner}' did not match red/blue fighter names or Red/Blue labels "
        f"for fight '{red}' vs '{blue}'."
    )


def _optional_value(row: pd.Series, column: str | None) -> Any:
    return _clean(row.get(column)) if column else ""


def _build_source_ids(frame: pd.DataFrame, fight_id_col: str | None) -> tuple[list[int], dict[str, int]]:
    generated_ids = list(range(1, len(frame) + 1))
    if not fight_id_col:
        return generated_ids, {}
    source_values = frame[fight_id_col].map(_clean).tolist()
    numeric = pd.to_numeric(pd.Series(source_values), errors="coerce")
    if numeric.isna().any() or numeric.duplicated().any():
        return generated_ids, {value: generated_ids[index] for index, value in enumerate(source_values) if value}
    output_ids = numeric.astype(int).tolist()
    return output_ids, {value: output_ids[index] for index, value in enumerate(source_values) if value}


def _adapt_fights(info: CsvColumnInfo) -> FightMapping:
    raw = _read_frame(info.path)
    red_col = _find_column(raw.columns, RED_FIGHTER_ALIASES, "red fighter", required=True, context=str(info.path))
    blue_col = _find_column(raw.columns, BLUE_FIGHTER_ALIASES, "blue fighter", required=True, context=str(info.path))
    date_col = _find_column(raw.columns, DATE_ALIASES, "fight date", required=True, context=str(info.path))
    winner_col = _find_column(raw.columns, WINNER_ALIASES, "winner", required=True, context=str(info.path))
    fight_id_col = _find_column(raw.columns, FIGHT_ID_ALIASES, "fight id", required=False)
    event_col = _find_column(raw.columns, EVENT_ALIASES, "event name", required=False)
    location_col = _find_column(raw.columns, LOCATION_ALIASES, "event location", required=False)
    weight_col = _find_column(raw.columns, WEIGHT_CLASS_ALIASES, "weight class", required=False)
    method_col = _find_column(raw.columns, METHOD_ALIASES, "method", required=False)
    round_col = _find_column(raw.columns, FINISH_ROUND_ALIASES, "finish round", required=False)
    time_col = _find_column(raw.columns, FINISH_TIME_ALIASES, "finish time", required=False)
    scheduled_col = _find_column(raw.columns, SCHEDULED_ROUNDS_ALIASES, "scheduled rounds", required=False)
    source_url_col = _find_column(raw.columns, SOURCE_URL_ALIASES, "source URL", required=False)

    fight_ids, source_to_output_ids = _build_source_ids(raw, fight_id_col)
    dates = _date_series(raw, date_col, info.path)
    rows: list[dict[str, Any]] = []
    for index, row in raw.iterrows():
        red = _clean(row.get(red_col))
        blue = _clean(row.get(blue_col))
        if not red or not blue:
            raise DatasetAdapterError(f"Missing fighter name in {info.path} row {index + 2}.")
        rows.append(
            {
                "fight_id": fight_ids[index],
                "event_name": _optional_value(row, event_col),
                "fight_date": dates.iloc[index],
                "event_location": _optional_value(row, location_col),
                "fighter_a": red,
                "fighter_b": blue,
                "winner": _resolve_winner(row, winner_col, red_col, blue_col),
                "weight_class": _optional_value(row, weight_col),
                "method": _optional_value(row, method_col),
                "finish_round": _optional_value(row, round_col),
                "finish_time": _optional_value(row, time_col),
                "scheduled_rounds": _optional_value(row, scheduled_col),
                "source_url": _optional_value(row, source_url_col),
            }
        )
    output = pd.DataFrame(rows)
    for column in FIGHT_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return FightMapping(
        raw=raw,
        output=output[FIGHT_COLUMNS],
        source_path=info.path,
        red_col=red_col,
        blue_col=blue_col,
        date_col=date_col,
        source_id_col=fight_id_col,
        source_to_output_ids=source_to_output_ids,
    )


def _measurement_to_inches(value: Any, *, kind: str) -> Any:
    text = _clean(value).lower().replace('"', "").replace("inches", "in").replace("inch", "in")
    if not text or text in {"--", "nan"}:
        return ""
    feet_match = re.search(r"(\d+)\s*'\s*(\d+)?", text)
    if feet_match:
        feet = int(feet_match.group(1))
        inches = int(feet_match.group(2) or 0)
        return feet * 12 + inches
    number_match = re.search(r"[-+]?\d*\.?\d+", text)
    if not number_match:
        return ""
    number = float(number_match.group(0))
    if "cm" in text:
        number *= 0.3937007874
    if kind == "weight" and "kg" in text:
        number *= 2.2046226218
    return round(number, 2)


def _profile_from_row(row: pd.Series, name_col: str, columns: dict[str, str | None]) -> dict[str, Any]:
    return {
        "name": _clean(row.get(name_col)),
        "stance": _optional_value(row, columns.get("stance")),
        "height_in": _measurement_to_inches(row.get(columns.get("height")), kind="height") if columns.get("height") else "",
        "weight_lb": _measurement_to_inches(row.get(columns.get("weight")), kind="weight") if columns.get("weight") else "",
        "reach_in": _measurement_to_inches(row.get(columns.get("reach")), kind="reach") if columns.get("reach") else "",
        "date_of_birth": _optional_value(row, columns.get("date_of_birth")),
        "record": _optional_value(row, columns.get("record")),
        "source_url": _optional_value(row, columns.get("source_url")),
    }


def _select_profile_file(infos: list[CsvColumnInfo], fight_file: Path) -> CsvColumnInfo | None:
    scored: list[tuple[int, CsvColumnInfo]] = []
    for info in infos:
        if info.path == fight_file:
            continue
        name_col = _find_column(info.columns, PROFILE_NAME_ALIASES, "fighter name", required=False)
        if name_col is None:
            continue
        score = 0
        for aliases in [STANCE_ALIASES, HEIGHT_ALIASES, WEIGHT_ALIASES, REACH_ALIASES, DOB_ALIASES, RECORD_ALIASES]:
            if _find_column(info.columns, aliases, aliases[0], required=False):
                score += 1
        if score:
            scored.append((score, info))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        names = ", ".join(str(item[1].path) for item in scored if item[0] == scored[0][0])
        raise DatasetAdapterError(f"Multiple fighter profile CSV candidates were found with equal confidence: {names}")
    return scored[0][1]


def _profiles_from_profile_file(info: CsvColumnInfo) -> pd.DataFrame:
    frame = _read_frame(info.path)
    name_col = _find_column(frame.columns, PROFILE_NAME_ALIASES, "fighter name", required=True, context=str(info.path))
    columns = {
        "stance": _find_column(frame.columns, STANCE_ALIASES, "stance", required=False),
        "height": _find_column(frame.columns, HEIGHT_ALIASES, "height", required=False),
        "weight": _find_column(frame.columns, WEIGHT_ALIASES, "weight", required=False),
        "reach": _find_column(frame.columns, REACH_ALIASES, "reach", required=False),
        "date_of_birth": _find_column(frame.columns, DOB_ALIASES, "date of birth", required=False),
        "record": _find_column(frame.columns, RECORD_ALIASES, "record", required=False),
        "source_url": _find_column(frame.columns, SOURCE_URL_ALIASES, "source URL", required=False),
    }
    rows = [_profile_from_row(row, name_col, columns) for _, row in frame.iterrows()]
    output = pd.DataFrame(rows).drop_duplicates(subset=["name"])
    return output.reindex(columns=FIGHTER_COLUMNS)


def _profiles_from_fight_rows(mapping: FightMapping) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    for side, name_col in [("red", mapping.red_col), ("blue", mapping.blue_col)]:
        columns = {
            "stance": _find_side_column(mapping.raw.columns, side, STANCE_ALIASES),
            "height": _find_side_column(mapping.raw.columns, side, HEIGHT_ALIASES),
            "weight": _find_side_column(mapping.raw.columns, side, WEIGHT_ALIASES),
            "reach": _find_side_column(mapping.raw.columns, side, REACH_ALIASES),
            "date_of_birth": _find_side_column(mapping.raw.columns, side, DOB_ALIASES),
            "record": _find_side_column(mapping.raw.columns, side, RECORD_ALIASES),
            "source_url": _find_side_column(mapping.raw.columns, side, SOURCE_URL_ALIASES),
        }
        for _, row in mapping.raw.iterrows():
            profile = _profile_from_row(row, name_col, columns)
            if profile["name"] and profile["name"] not in rows:
                rows[profile["name"]] = profile
    output = pd.DataFrame(rows.values()).reindex(columns=FIGHTER_COLUMNS)
    return output.sort_values("name").reset_index(drop=True)


def _parse_landed_attempted(value: Any) -> tuple[Any, Any]:
    text = _clean(value).lower()
    if not text:
        return "", ""
    match = re.search(r"([-+]?\d*\.?\d+)\s*(?:of|/|-)\s*([-+]?\d*\.?\d+)", text)
    if not match:
        return "", ""
    return float(match.group(1)), float(match.group(2))


def _time_to_seconds(value: Any) -> Any:
    text = _clean(value).lower()
    if not text or text in {"--", "nan"}:
        return ""
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        try:
            return float(minutes) * 60.0 + float(seconds)
        except ValueError:
            return ""
    return _numeric_or_blank(text)


def _side_stat_value(row: pd.Series, columns: list[str] | pd.Index, side: str, canonical: str) -> Any:
    column = _find_side_column(columns, side, STAT_ALIASES[canonical])
    if column is None:
        return ""
    value = row.get(column)
    if canonical == "control_seconds":
        return _time_to_seconds(value)
    return _numeric_or_blank(value)


def _side_landed_attempted(
    row: pd.Series,
    columns: list[str] | pd.Index,
    side: str,
    combined_key: str,
    landed_key: str,
    attempted_key: str,
) -> tuple[Any, Any]:
    landed_column = _find_side_column(columns, side, STAT_ALIASES[landed_key])
    attempted_column = _find_side_column(columns, side, STAT_ALIASES[attempted_key])
    if landed_column and attempted_column:
        return _numeric_or_blank(row.get(landed_column)), _numeric_or_blank(row.get(attempted_column))
    combined_column = _find_side_column(columns, side, STAT_ALIASES[combined_key])
    if combined_column:
        return _parse_landed_attempted(row.get(combined_column))
    return "", ""


def _has_side_stats(columns: list[str] | pd.Index, side: str) -> bool:
    return any(_find_side_column(columns, side, aliases) is not None for aliases in STAT_ALIASES.values())


def _stats_from_fight_rows(mapping: FightMapping) -> pd.DataFrame:
    if not (_has_side_stats(mapping.raw.columns, "red") or _has_side_stats(mapping.raw.columns, "blue")):
        return pd.DataFrame(columns=FIGHT_STAT_COLUMNS)
    rows: list[dict[str, Any]] = []
    for index, row in mapping.raw.iterrows():
        fight_id = mapping.output.iloc[index]["fight_id"]
        for side, fighter_col, opponent_col in [
            ("red", mapping.red_col, mapping.blue_col),
            ("blue", mapping.blue_col, mapping.red_col),
        ]:
            sig_landed, sig_attempted = _side_landed_attempted(
                row, mapping.raw.columns, side, "sig_str", "sig_str_landed", "sig_str_attempted"
            )
            total_landed, total_attempted = _side_landed_attempted(
                row, mapping.raw.columns, side, "total_str", "total_str_landed", "total_str_attempted"
            )
            td_landed, td_attempted = _side_landed_attempted(
                row, mapping.raw.columns, side, "takedowns", "takedowns_landed", "takedowns_attempted"
            )
            rows.append(
                {
                    "fight_id": fight_id,
                    "source_url": "",
                    "fighter": _clean(row.get(fighter_col)),
                    "opponent": _clean(row.get(opponent_col)),
                    "knockdowns": _side_stat_value(row, mapping.raw.columns, side, "knockdowns"),
                    "sig_str_landed": sig_landed,
                    "sig_str_attempted": sig_attempted,
                    "total_str_landed": total_landed,
                    "total_str_attempted": total_attempted,
                    "takedowns_landed": td_landed,
                    "takedowns_attempted": td_attempted,
                    "submission_attempts": _side_stat_value(row, mapping.raw.columns, side, "submission_attempts"),
                    "reversals": _side_stat_value(row, mapping.raw.columns, side, "reversals"),
                    "control_seconds": _side_stat_value(row, mapping.raw.columns, side, "control_seconds"),
                    "head_landed": _side_stat_value(row, mapping.raw.columns, side, "head_landed"),
                    "body_landed": _side_stat_value(row, mapping.raw.columns, side, "body_landed"),
                    "leg_landed": _side_stat_value(row, mapping.raw.columns, side, "leg_landed"),
                }
            )
    return pd.DataFrame(rows).reindex(columns=FIGHT_STAT_COLUMNS)


def _has_per_fighter_stats(info: CsvColumnInfo) -> bool:
    fighter = _find_column(info.columns, ["fighter", "Fighter"], "fighter", required=False)
    opponent = _find_column(info.columns, ["opponent", "Opponent"], "opponent", required=False)
    if fighter is None or opponent is None:
        return False
    return any(_find_column(info.columns, aliases, label, required=False) is not None for label, aliases in STAT_ALIASES.items())


def _select_stats_file(infos: list[CsvColumnInfo], fight_file: Path) -> CsvColumnInfo | None:
    candidates = [info for info in infos if info.path != fight_file and _has_per_fighter_stats(info)]
    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(str(info.path) for info in candidates)
        raise DatasetAdapterError(f"Multiple per-fighter stats CSV candidates were found: {names}")
    return candidates[0]


def _stats_from_per_fighter_file(info: CsvColumnInfo, mapping: FightMapping) -> pd.DataFrame:
    frame = _read_frame(info.path)
    fighter_col = _find_column(frame.columns, ["fighter", "Fighter"], "fighter", required=True, context=str(info.path))
    opponent_col = _find_column(frame.columns, ["opponent", "Opponent"], "opponent", required=True, context=str(info.path))
    fight_id_col = _find_column(frame.columns, FIGHT_ID_ALIASES, "fight id", required=False)
    date_col = _find_column(frame.columns, DATE_ALIASES, "fight date", required=False)

    if fight_id_col is None and date_col is None:
        raise DatasetAdapterError(
            f"{info.path} has per-fighter stats but no fight_id or fight_date column for joining to fights."
        )

    date_pair_to_id = {}
    for _, row in mapping.output.iterrows():
        key = (
            str(row["fight_date"]),
            tuple(sorted([_clean(row["fighter_a"]).lower(), _clean(row["fighter_b"]).lower()])),
        )
        date_pair_to_id[key] = int(row["fight_id"])

    rows: list[dict[str, Any]] = []
    missing_join = 0
    for _, row in frame.iterrows():
        fight_id = None
        if fight_id_col is not None:
            fight_id = mapping.source_to_output_ids.get(_clean(row.get(fight_id_col)))
        if fight_id is None and date_col is not None:
            parsed_date = pd.to_datetime(pd.Series([row.get(date_col)]), errors="coerce").iloc[0]
            if pd.notna(parsed_date):
                key = (
                    parsed_date.date().isoformat(),
                    tuple(sorted([_clean(row.get(fighter_col)).lower(), _clean(row.get(opponent_col)).lower()])),
                )
                fight_id = date_pair_to_id.get(key)
        if fight_id is None:
            missing_join += 1
            continue

        sig_landed, sig_attempted = _direct_landed_attempted(row, frame.columns, "sig_str", "sig_str_landed", "sig_str_attempted")
        total_landed, total_attempted = _direct_landed_attempted(
            row, frame.columns, "total_str", "total_str_landed", "total_str_attempted"
        )
        td_landed, td_attempted = _direct_landed_attempted(row, frame.columns, "takedowns", "takedowns_landed", "takedowns_attempted")
        rows.append(
            {
                "fight_id": fight_id,
                "source_url": "",
                "fighter": _clean(row.get(fighter_col)),
                "opponent": _clean(row.get(opponent_col)),
                "knockdowns": _direct_stat_value(row, frame.columns, "knockdowns"),
                "sig_str_landed": sig_landed,
                "sig_str_attempted": sig_attempted,
                "total_str_landed": total_landed,
                "total_str_attempted": total_attempted,
                "takedowns_landed": td_landed,
                "takedowns_attempted": td_attempted,
                "submission_attempts": _direct_stat_value(row, frame.columns, "submission_attempts"),
                "reversals": _direct_stat_value(row, frame.columns, "reversals"),
                "control_seconds": _time_to_seconds(_direct_stat_value(row, frame.columns, "control_seconds")),
                "head_landed": _direct_stat_value(row, frame.columns, "head_landed"),
                "body_landed": _direct_stat_value(row, frame.columns, "body_landed"),
                "leg_landed": _direct_stat_value(row, frame.columns, "leg_landed"),
            }
        )
    if missing_join:
        raise DatasetAdapterError(
            f"Could not join {missing_join} per-fighter stats rows in {info.path} to adapted fights. "
            "Provide matching fight_id values or fight_date/fighter/opponent pairs."
        )
    return pd.DataFrame(rows).reindex(columns=FIGHT_STAT_COLUMNS)


def _direct_stat_value(row: pd.Series, columns: list[str] | pd.Index, canonical: str) -> Any:
    column = _find_column(columns, STAT_ALIASES[canonical], canonical, required=False)
    if column is None:
        return ""
    if canonical == "control_seconds":
        return _time_to_seconds(row.get(column))
    return _numeric_or_blank(row.get(column))


def _direct_landed_attempted(
    row: pd.Series,
    columns: list[str] | pd.Index,
    combined_key: str,
    landed_key: str,
    attempted_key: str,
) -> tuple[Any, Any]:
    landed_column = _find_column(columns, STAT_ALIASES[landed_key], landed_key, required=False)
    attempted_column = _find_column(columns, STAT_ALIASES[attempted_key], attempted_key, required=False)
    if landed_column and attempted_column:
        return _numeric_or_blank(row.get(landed_column)), _numeric_or_blank(row.get(attempted_column))
    combined_column = _find_column(columns, STAT_ALIASES[combined_key], combined_key, required=False)
    if combined_column:
        return _parse_landed_attempted(row.get(combined_column))
    return "", ""


def _minimal_stats_from_fights(fights: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, fight in fights.iterrows():
        rows.append({"fight_id": fight["fight_id"], "fighter": fight["fighter_a"], "opponent": fight["fighter_b"]})
        rows.append({"fight_id": fight["fight_id"], "fighter": fight["fighter_b"], "opponent": fight["fighter_a"]})
    return pd.DataFrame(rows).reindex(columns=FIGHT_STAT_COLUMNS)


def _find_scorecards_file(infos: list[CsvColumnInfo]) -> CsvColumnInfo | None:
    for info in infos:
        if set(SCORECARD_COLUMNS).issubset(set(info.columns)):
            return info
    return None


def adapt_dataset(source: str | Path, output_dir: str | Path) -> AdaptDatasetResult:
    source_path = Path(source)
    output_path = Path(output_dir)
    copied = _copy_if_already_schema(source_path, output_path)
    if copied is not None:
        return copied

    infos = list_csv_columns(source_path)
    fight_info = _select_fight_file(infos)
    mapping = _adapt_fights(fight_info)
    result = AdaptDatasetResult(output_dir=output_path, source_files={"fights": fight_info.path})

    profile_info = _select_profile_file(infos, fight_info.path)
    if profile_info is not None:
        fighters = _profiles_from_profile_file(profile_info)
        result.source_files["fighters"] = profile_info.path
    else:
        fighters = _profiles_from_fight_rows(mapping)
        result.warnings.append("No separate fighter profile CSV detected; wrote fighters.csv from fight rows.")

    stats_info = _select_stats_file(infos, fight_info.path)
    if stats_info is not None:
        fight_stats = _stats_from_per_fighter_file(stats_info, mapping)
        result.source_files["fight_stats"] = stats_info.path
    else:
        fight_stats = _stats_from_fight_rows(mapping)
        result.source_files["fight_stats"] = fight_info.path
    if fight_stats.empty:
        fight_stats = _minimal_stats_from_fights(mapping.output)
        result.warnings.append("No mapped fight stat columns detected; wrote fight_stats.csv with fighter/opponent rows only.")

    if source_path.resolve() != output_path.resolve():
        _clear_known_outputs(output_path)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    files = {
        "fights": output_path / "fights.csv",
        "fighters": output_path / "fighters.csv",
        "fight_stats": output_path / "fight_stats.csv",
    }
    mapping.output.to_csv(files["fights"], index=False)
    fighters.to_csv(files["fighters"], index=False)
    fight_stats.to_csv(files["fight_stats"], index=False)
    result.files.update(files)

    scorecards = _find_scorecards_file(infos)
    if scorecards is not None:
        destination = output_path / "scorecards.csv"
        _copy_file(scorecards.path, destination)
        result.files["scorecards"] = destination
        result.source_files["scorecards"] = scorecards.path

    return result
