from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _normal(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _name_key(value: Any) -> str:
    return _normal(value).lower()


def american_odds_to_implied_probability(odds: Any) -> float | None:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or value == 0:
        return None
    value = float(value)
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


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
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce").dt.date.astype("string")
    frame["fighter_a_implied_probability"] = frame["fighter_a_odds"].map(american_odds_to_implied_probability)
    frame["fighter_b_implied_probability"] = frame["fighter_b_odds"].map(american_odds_to_implied_probability)
    frame["market_probability_sum"] = frame["fighter_a_implied_probability"].fillna(0.0) + frame[
        "fighter_b_implied_probability"
    ].fillna(0.0)
    valid_sum = frame["market_probability_sum"] > 0
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


def attach_odds_features(dataset: pd.DataFrame, odds: pd.DataFrame | None) -> pd.DataFrame:
    if dataset.empty or odds is None or odds.empty:
        return dataset
    frame = dataset.copy()
    odds_frame = odds.copy()
    frame["_fight_date_key"] = pd.to_datetime(frame["fight_date"], errors="coerce").dt.date.astype("string")
    odds_frame["_fight_date_key"] = pd.to_datetime(odds_frame["fight_date"], errors="coerce").dt.date.astype("string")
    odds_frame = odds_frame.rename(columns={"fighter_a": "_odds_fighter_a", "fighter_b": "_odds_fighter_b"})
    odds_frame["_pair_key"] = odds_frame.apply(
        lambda row: "|".join(sorted([_name_key(row.get("_odds_fighter_a")), _name_key(row.get("_odds_fighter_b"))])),
        axis=1,
    )
    frame["_pair_key"] = frame.apply(
        lambda row: "|".join(sorted([_name_key(row.get("fighter_a")), _name_key(row.get("fighter_b"))])), axis=1
    )
    if "timestamp" in odds_frame.columns:
        odds_frame["_timestamp"] = pd.to_datetime(odds_frame["timestamp"], errors="coerce")
        odds_frame = odds_frame.sort_values("_timestamp")
    odds_latest = odds_frame.drop_duplicates(subset=["_fight_date_key", "_pair_key"], keep="last")
    merged = frame.merge(
        odds_latest,
        on=["_fight_date_key", "_pair_key"],
        how="left",
    )
    same_orientation = merged["fighter_a"].map(_name_key) == merged["_odds_fighter_a"].map(_name_key)
    merged["market_fighter_a_implied_probability"] = merged["fighter_a_no_vig_probability"].where(
        same_orientation, merged["fighter_b_no_vig_probability"]
    )
    merged["market_fighter_b_implied_probability"] = merged["fighter_b_no_vig_probability"].where(
        same_orientation, merged["fighter_a_no_vig_probability"]
    )
    favorite = merged["market_fighter_a_implied_probability"] >= merged["market_fighter_b_implied_probability"]
    favorite = favorite.astype("boolean")
    favorite.loc[merged["market_fighter_a_implied_probability"].isna()] = pd.NA
    merged["closing_odds_favorite_is_a"] = favorite
    drop_columns = [
        column
        for column in [
            "_fight_date_key",
            "_pair_key",
            "_timestamp",
            "_odds_fighter_a",
            "_odds_fighter_b",
            "fight_date_y",
            "fight_date",
            "sportsbook",
            "fighter_a_odds",
            "fighter_b_odds",
            "fighter_a_implied_probability",
            "fighter_b_implied_probability",
            "fighter_a_no_vig_probability",
            "fighter_b_no_vig_probability",
            "market_probability_sum",
            "timestamp",
        ]
        if column in merged.columns
    ]
    if "fight_date_x" in merged.columns:
        merged = merged.rename(columns={"fight_date_x": "fight_date"})
    return merged.drop(columns=drop_columns)
