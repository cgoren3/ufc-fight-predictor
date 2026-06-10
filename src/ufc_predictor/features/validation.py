from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


class DataLeakageError(ValueError):
    """Raised when future information enters a pre-fight feature calculation."""


@dataclass(frozen=True)
class LeakageCheck:
    as_of_date: pd.Timestamp
    offending_dates: list[pd.Timestamp]

    @property
    def passed(self) -> bool:
        return not self.offending_dates


def to_datetime_series(values: Iterable[object] | pd.Series) -> pd.Series:
    return pd.to_datetime(pd.Series(values), errors="coerce")


def check_history_is_before(
    history: pd.DataFrame,
    as_of_date: object,
    date_column: str = "fight_date",
) -> LeakageCheck:
    """Return a leakage check for rows that must all predate as_of_date."""

    if history.empty or date_column not in history.columns:
        return LeakageCheck(pd.Timestamp(as_of_date), [])
    cutoff = pd.Timestamp(as_of_date)
    dates = pd.to_datetime(history[date_column], errors="coerce")
    offending = dates[dates >= cutoff].dropna().sort_values().tolist()
    return LeakageCheck(cutoff, offending)


def assert_history_is_before(
    history: pd.DataFrame,
    as_of_date: object,
    date_column: str = "fight_date",
) -> None:
    check = check_history_is_before(history, as_of_date, date_column=date_column)
    if not check.passed:
        dates = ", ".join(date.strftime("%Y-%m-%d") for date in check.offending_dates[:5])
        raise DataLeakageError(
            f"Found {len(check.offending_dates)} future/current rows at or after "
            f"{check.as_of_date.date()}: {dates}"
        )


def validate_training_frame(frame: pd.DataFrame) -> None:
    required = {"fight_date", "fighter_a", "fighter_b", "fighter_a_win"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training frame is missing required columns: {sorted(missing)}")
    if frame["fighter_a_win"].isna().any():
        raise ValueError("Training frame contains missing targets.")
    if "max_history_date_used" in frame.columns:
        dates = pd.to_datetime(frame["max_history_date_used"], errors="coerce")
        fight_dates = pd.to_datetime(frame["fight_date"], errors="coerce")
        leaked = dates >= fight_dates
        if leaked.fillna(False).any():
            sample = frame.loc[leaked, ["fight_date", "fighter_a", "fighter_b", "max_history_date_used"]].head()
            raise DataLeakageError(f"Training frame contains future history rows:\n{sample}")
