from __future__ import annotations

from pathlib import Path

import pandas as pd

from ufc_predictor.database import write_dataframe


REQUIRED_COLUMNS = [
    "event",
    "fight_date",
    "fighter_a",
    "fighter_b",
    "judge",
    "round_1_a",
    "round_1_b",
    "round_2_a",
    "round_2_b",
    "round_3_a",
    "round_3_b",
    "round_4_a",
    "round_4_b",
    "round_5_a",
    "round_5_b",
    "total_a",
    "total_b",
    "decision_type",
]

RAW_SCORECARD_COLUMNS = [
    "event",
    "fight_date",
    "fighter_a",
    "fighter_b",
    "winner",
    "card_type",
    "raw_scorecards",
    "source_file",
]

MMA_DECISIONS_OPTIONAL_COLUMNS = [
    "event",
    "fight_date",
    "fighter_a",
    "fighter_b",
    "winner",
    "decision_type",
    "judge_1",
    "judge_1_score_a",
    "judge_1_score_b",
    "judge_2",
    "judge_2_score_a",
    "judge_2_score_b",
    "judge_3",
    "judge_3_score_a",
    "judge_3_score_b",
    "media_score_a",
    "media_score_b",
    "fan_score_a",
    "fan_score_b",
    "disputed_decision_flag",
]


def load_scorecards_csv(path: str | Path) -> pd.DataFrame:
    """Load manually downloaded official scorecard data from CSV."""

    frame = pd.read_csv(path)
    if set(REQUIRED_COLUMNS) <= set(frame.columns):
        frame = frame.copy()
    elif {"fight_date", "fighter_a", "fighter_b", "raw_scorecards"} <= set(frame.columns):
        for column in REQUIRED_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        if "event" not in frame.columns:
            frame["event"] = ""
        frame["judge"] = frame["judge"].fillna("raw_scorecards")
        frame["decision_type"] = frame["decision_type"].fillna(frame.get("card_type", pd.Series([""] * len(frame))))
    else:
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        raise ValueError(f"Scorecard CSV is missing columns: {', '.join(missing)}")
    optional = [column for column in RAW_SCORECARD_COLUMNS if column in frame.columns and column not in REQUIRED_COLUMNS]
    frame = frame[REQUIRED_COLUMNS + optional].copy()
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce").dt.date.astype("string")
    numeric_columns = [column for column in REQUIRED_COLUMNS if column.startswith("round_") or column.startswith("total_")]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def import_scorecards(path: str | Path, db_path: str | Path | None = None) -> pd.DataFrame:
    frame = load_scorecards_csv(path)
    write_dataframe(frame[REQUIRED_COLUMNS], "scorecards", db_path=db_path, if_exists="append")
    return frame


def load_mmadecisions_csv(path: str | Path) -> pd.DataFrame:
    """Load manually exported MMA Decisions-style judging data.

    MMA Decisions pages are useful for split/close/disputed decision research,
    but the project treats them as optional external/manual data. The returned
    frame can be joined into downstream analysis or converted to scorecard rows.
    """

    frame = pd.read_csv(path)
    for column in MMA_DECISIONS_OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[MMA_DECISIONS_OPTIONAL_COLUMNS].copy()
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce").dt.date.astype("string")
    numeric_columns = [column for column in frame.columns if column.endswith("_score_a") or column.endswith("_score_b")]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["disputed_decision_flag"] = frame["disputed_decision_flag"].fillna(False).astype(bool)
    return frame
