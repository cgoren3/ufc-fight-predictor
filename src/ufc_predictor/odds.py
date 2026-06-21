from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re
import unicodedata

import numpy as np
import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.data_io import InputDataError, inspect_csv


ODDS_COLUMNS = [
    "fight_date",
    "fighter_a",
    "fighter_b",
    "sportsbook",
    "fighter_a_odds",
    "fighter_b_odds",
    "timestamp",
]


@dataclass
class OddsCoverageReport:
    raw_odds_rows_found: int = 0
    odds_rows_parsed: int = 0
    odds_rows_matched: int = 0
    odds_rows_unmatched: int = 0
    fights_count: int = 0
    final_odds_coverage_count: int = 0
    final_odds_coverage_percent: float = 0.0
    match_reasons: dict[str, int] = field(default_factory=dict)
    unmatched_examples: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_odds_rows_found": self.raw_odds_rows_found,
            "odds_rows_parsed": self.odds_rows_parsed,
            "odds_rows_matched": self.odds_rows_matched,
            "odds_rows_unmatched": self.odds_rows_unmatched,
            "fights_count": self.fights_count,
            "final_odds_coverage_count": self.final_odds_coverage_count,
            "final_odds_coverage_percent": self.final_odds_coverage_percent,
            "match_reasons": self.match_reasons,
            "unmatched_examples": self.unmatched_examples,
        }


def _is_missing(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _normal(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _name_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _normal(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"\([^)]*\)|\"[^\"]*\"|'[^']*'", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    raw_tokens = text.split()
    tokens = []
    for index, token in enumerate(raw_tokens):
        if token in {"jr", "sr", "ii", "iii", "iv", "v", "the"}:
            continue
        is_middle_initial = len(token) == 1 and 0 < index < len(raw_tokens) - 1
        if is_middle_initial:
            continue
        tokens.append(token)
    return " ".join(tokens)


def _pair_key_from_names(fighter_a: Any, fighter_b: Any) -> str:
    return "|".join(sorted([_name_key(fighter_a), _name_key(fighter_b)]))


def _event_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _normal(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _date_key(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return ""
    return date.date().isoformat()


def _date_variants(value: Any) -> list[str]:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return []
    base = pd.Timestamp(date).normalize()
    return [(base + pd.Timedelta(days=offset)).date().isoformat() for offset in [0, -1, 1]]


def american_odds_to_implied_probability(odds: Any) -> float | None:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or value == 0:
        return None
    value = float(value)
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def is_valid_american_odds(odds: Any) -> bool:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value):
        return False
    value = float(value)
    return value != 0 and value == int(value) and abs(value) >= 100.0


def import_odds_csv(
    import_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    source = Path(import_path) if import_path else settings.raw_data_dir / "imports" / "odds.csv"
    output = Path(output_path) if output_path else settings.raw_data_dir / "odds.csv"
    try:
        inspect_csv(source, required_columns=ODDS_COLUMNS, require_rows=True, label="odds CSV")
    except InputDataError:
        raise
    frame = pd.read_csv(source)
    invalid_rows = []
    for column in ["fighter_a_odds", "fighter_b_odds"]:
        for index, value in frame[column].items():
            if not _is_missing(value) and not is_valid_american_odds(value):
                invalid_rows.append(f"row {index + 2} {column}={value!r}")
    if invalid_rows:
        raise InputDataError("Invalid American odds values: " + "; ".join(invalid_rows[:10]))
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce").dt.date.astype("string")
    frame["fighter_a_implied_probability"] = frame["fighter_a_odds"].map(american_odds_to_implied_probability)
    frame["fighter_b_implied_probability"] = frame["fighter_b_odds"].map(american_odds_to_implied_probability)
    frame["market_probability_sum"] = frame["fighter_a_implied_probability"].fillna(0.0) + frame[
        "fighter_b_implied_probability"
    ].fillna(0.0)
    valid_sum = frame["fighter_a_implied_probability"].notna() & frame["fighter_b_implied_probability"].notna()
    frame.loc[valid_sum, "fighter_a_no_vig_probability"] = (
        frame.loc[valid_sum, "fighter_a_implied_probability"] / frame.loc[valid_sum, "market_probability_sum"]
    )
    frame.loc[valid_sum, "fighter_b_no_vig_probability"] = (
        frame.loc[valid_sum, "fighter_b_implied_probability"] / frame.loc[valid_sum, "market_probability_sum"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def load_odds(path: str | Path | None = None) -> pd.DataFrame | None:
    odds_path = Path(path) if path else settings.raw_data_dir / "odds.csv"
    if not odds_path.exists():
        return None
    inspection = inspect_csv(odds_path, required_columns=ODDS_COLUMNS, require_rows=False, label="odds CSV")
    if not inspection.has_rows:
        return None
    return pd.read_csv(odds_path)


def _prepare_odds_for_matching(odds: pd.DataFrame) -> pd.DataFrame:
    odds_frame = odds.copy()
    if odds_frame.empty:
        return odds_frame
    for column in ["event", "source_file"]:
        if column not in odds_frame.columns:
            odds_frame[column] = ""
    if "fighter_a_no_vig_probability" not in odds_frame.columns or "fighter_b_no_vig_probability" not in odds_frame.columns:
        odds_frame["fighter_a_implied_probability"] = odds_frame["fighter_a_odds"].map(american_odds_to_implied_probability)
        odds_frame["fighter_b_implied_probability"] = odds_frame["fighter_b_odds"].map(american_odds_to_implied_probability)
        probability_sum = odds_frame["fighter_a_implied_probability"].fillna(0.0) + odds_frame["fighter_b_implied_probability"].fillna(0.0)
        valid = odds_frame["fighter_a_implied_probability"].notna() & odds_frame["fighter_b_implied_probability"].notna() & probability_sum.gt(0)
        odds_frame.loc[valid, "fighter_a_no_vig_probability"] = odds_frame.loc[valid, "fighter_a_implied_probability"] / probability_sum.loc[valid]
        odds_frame.loc[valid, "fighter_b_no_vig_probability"] = odds_frame.loc[valid, "fighter_b_implied_probability"] / probability_sum.loc[valid]
    odds_frame["_fight_date_key"] = odds_frame["fight_date"].map(_date_key)
    odds_frame["_pair_key"] = odds_frame.apply(lambda row: _pair_key_from_names(row.get("fighter_a"), row.get("fighter_b")), axis=1)
    odds_frame["_event_key"] = odds_frame["event"].map(_event_key) if "event" in odds_frame.columns else ""
    if "timestamp" in odds_frame.columns:
        odds_frame["_timestamp"] = pd.to_datetime(odds_frame["timestamp"], errors="coerce")
        odds_frame = odds_frame.sort_values("_timestamp")
    return odds_frame


def _prepare_fights_for_matching(fights: pd.DataFrame) -> pd.DataFrame:
    frame = fights.copy()
    if frame.empty:
        return frame
    event_column = "event_name" if "event_name" in frame.columns else "event" if "event" in frame.columns else ""
    frame["_fight_date_key"] = frame["fight_date"].map(_date_key)
    frame["_pair_key"] = frame.apply(lambda row: _pair_key_from_names(row.get("fighter_a"), row.get("fighter_b")), axis=1)
    frame["_event_key"] = frame[event_column].map(_event_key) if event_column else ""
    return frame


def _fight_lookup_maps(fights: pd.DataFrame) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[str]]:
    frame = _prepare_fights_for_matching(fights)
    if frame.empty:
        return set(), set(), set()
    date_pair = set(zip(frame["_fight_date_key"], frame["_pair_key"]))
    event_pair = set(zip(frame["_event_key"], frame["_pair_key"]))
    pairs = set(frame["_pair_key"].tolist())
    return date_pair, event_pair, pairs


def _valid_odds_frame(odds: pd.DataFrame) -> pd.DataFrame:
    odds_frame = _prepare_odds_for_matching(odds)
    if odds_frame.empty:
        return odds_frame
    valid = odds_frame[
        odds_frame["fighter_a_no_vig_probability"].notna()
        & odds_frame["fighter_b_no_vig_probability"].notna()
        & odds_frame["_fight_date_key"].ne("")
        & odds_frame["_pair_key"].ne("|")
    ].copy()
    if "_timestamp" in valid.columns:
        valid = valid.sort_values("_timestamp")
    return valid


def _build_odds_indexes(odds: pd.DataFrame) -> tuple[dict[tuple[str, str], pd.Series], dict[tuple[str, str], pd.Series]]:
    valid = _valid_odds_frame(odds)
    date_pair_index: dict[tuple[str, str], pd.Series] = {}
    event_pair_index: dict[tuple[str, str], pd.Series] = {}
    if valid.empty:
        return date_pair_index, event_pair_index
    for _, row in valid.iterrows():
        pair = row.get("_pair_key", "")
        for date_key in _date_variants(row.get("fight_date")):
            if date_key:
                date_pair_index[(date_key, pair)] = row
        event = row.get("_event_key", "")
        if event:
            event_pair_index[(event, pair)] = row
    return date_pair_index, event_pair_index


def _match_odds_row(
    fight_row: pd.Series,
    date_pair_index: dict[tuple[str, str], pd.Series],
    event_pair_index: dict[tuple[str, str], pd.Series],
) -> tuple[pd.Series | None, str]:
    pair = fight_row.get("_pair_key", "")
    exact_date = fight_row.get("_fight_date_key", "")
    if exact_date and (exact_date, pair) in date_pair_index:
        return date_pair_index[(exact_date, pair)], "normalized_date_pair"
    event = fight_row.get("_event_key", "")
    if event and (event, pair) in event_pair_index:
        return event_pair_index[(event, pair)], "event_pair"
    return None, "no_date_or_event_pair_match"


def _oriented_market_probabilities(fight_row: pd.Series, odds_row: pd.Series) -> tuple[float | None, float | None]:
    probability_a = pd.to_numeric(pd.Series([odds_row.get("fighter_a_no_vig_probability")]), errors="coerce").iloc[0]
    probability_b = pd.to_numeric(pd.Series([odds_row.get("fighter_b_no_vig_probability")]), errors="coerce").iloc[0]
    if pd.isna(probability_a) or pd.isna(probability_b):
        return None, None
    same_orientation = _name_key(fight_row.get("fighter_a")) == _name_key(odds_row.get("fighter_a"))
    if same_orientation:
        return float(probability_a), float(probability_b)
    return float(probability_b), float(probability_a)


def odds_coverage_report(fights: pd.DataFrame, odds: pd.DataFrame | None, max_examples: int = 10) -> OddsCoverageReport:
    report = OddsCoverageReport(fights_count=int(len(fights)))
    if odds is None or odds.empty:
        return report
    odds_frame = _prepare_odds_for_matching(odds)
    valid_odds = _valid_odds_frame(odds)
    report.raw_odds_rows_found = int(len(odds_frame))
    report.odds_rows_parsed = int(len(valid_odds))
    fights_frame = _prepare_fights_for_matching(fights)
    date_pair_to_fights: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    event_pair_to_fights: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    pairs: set[str] = set()
    if not fights_frame.empty:
        for _, fight in fights_frame.iterrows():
            fight_key = (
                str(fight.get("_fight_date_key", "")),
                str(fight.get("_pair_key", "")),
                str(fight.get("fight_id", fight.name)),
            )
            pair = fight.get("_pair_key", "")
            pairs.add(pair)
            date_pair_to_fights.setdefault((fight.get("_fight_date_key", ""), pair), set()).add(fight_key)
            event = fight.get("_event_key", "")
            if event:
                event_pair_to_fights.setdefault((event, pair), set()).add(fight_key)
    matched_fights: set[tuple[str, str, str]] = set()
    reasons: dict[str, int] = {}
    unmatched: list[dict[str, Any]] = []
    for _, row in valid_odds.iterrows():
        pair = row.get("_pair_key", "")
        dates = _date_variants(row.get("fight_date"))
        exact_date = dates[0] if dates else ""
        event = row.get("_event_key", "")
        reason = ""
        if (exact_date, pair) in date_pair_to_fights:
            reason = "exact_date_pair"
            matched_fights.update(date_pair_to_fights[(exact_date, pair)])
        elif event and (event, pair) in event_pair_to_fights:
            reason = "event_pair"
            matched_fights.update(event_pair_to_fights[(event, pair)])
        else:
            timezone_match = next((date for date in dates[1:] if (date, pair) in date_pair_to_fights), "")
            if timezone_match:
                reason = "timezone_adjusted_date_pair"
                matched_fights.update(date_pair_to_fights[(timezone_match, pair)])
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            report.odds_rows_matched += 1
        else:
            report.odds_rows_unmatched += 1
            if len(unmatched) < max_examples:
                unmatched.append(
                    {
                        "fight_date": row.get("fight_date"),
                        "fighter_a": row.get("fighter_a"),
                        "fighter_b": row.get("fighter_b"),
                        "event": row.get("event", ""),
                        "reason": "fighter pair not found for normalized date/event" if pair not in pairs else "date/event mismatch",
                    }
                )
    report.match_reasons = reasons
    report.unmatched_examples = unmatched
    report.final_odds_coverage_count = len(matched_fights)
    report.final_odds_coverage_percent = round((len(matched_fights) / len(fights) * 100.0), 2) if len(fights) else 0.0
    return report


def attach_odds_features(dataset: pd.DataFrame, odds: pd.DataFrame | None) -> pd.DataFrame:
    if dataset.empty or odds is None or odds.empty:
        return dataset
    frame = _prepare_fights_for_matching(dataset)
    result = dataset.copy()
    result["market_fighter_a_implied_probability"] = np.nan
    result["market_fighter_b_implied_probability"] = np.nan
    result["closing_odds_favorite_is_a"] = pd.Series([pd.NA] * len(result), dtype="boolean")
    date_pair_index, event_pair_index = _build_odds_indexes(odds)
    if not date_pair_index and not event_pair_index:
        return result
    for index, fight_row in frame.iterrows():
        odds_row, _ = _match_odds_row(fight_row, date_pair_index, event_pair_index)
        if odds_row is None:
            continue
        probability_a, probability_b = _oriented_market_probabilities(fight_row, odds_row)
        if probability_a is None or probability_b is None:
            continue
        result.at[index, "market_fighter_a_implied_probability"] = probability_a
        result.at[index, "market_fighter_b_implied_probability"] = probability_b
        result.at[index, "closing_odds_favorite_is_a"] = probability_a >= probability_b
    return result


def market_probability_for_fight(
    odds: pd.DataFrame | None,
    fighter_a: str,
    fighter_b: str,
    fight_date: object,
    event: str = "",
) -> dict[str, Any] | None:
    if odds is None or odds.empty:
        return None
    odds_frame = _prepare_odds_for_matching(odds)
    pair = _pair_key_from_names(fighter_a, fighter_b)
    event_key = _event_key(event)
    candidates = odds_frame[odds_frame["_pair_key"] == pair].copy()
    if candidates.empty:
        return None
    date_options = _date_variants(fight_date)
    matched = candidates[candidates["_fight_date_key"].isin(date_options)]
    if matched.empty and event_key:
        matched = candidates[candidates["_event_key"] == event_key]
    if matched.empty:
        return None
    row = matched.iloc[-1]
    same_orientation = _name_key(fighter_a) == _name_key(row.get("fighter_a"))
    probability_a = row.get("fighter_a_no_vig_probability") if same_orientation else row.get("fighter_b_no_vig_probability")
    probability_b = row.get("fighter_b_no_vig_probability") if same_orientation else row.get("fighter_a_no_vig_probability")
    if pd.isna(probability_a) or pd.isna(probability_b):
        return None
    return {
        "market_implied_probability": float(probability_a),
        "market_fighter_b_implied_probability": float(probability_b),
        "sportsbook": row.get("sportsbook", ""),
        "source_file": row.get("source_file", ""),
    }
