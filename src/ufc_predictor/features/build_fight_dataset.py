from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.features.elo import build_elo_features
from ufc_predictor.features.fighter_history import FighterHistoryContext, SNAPSHOT_NUMERIC_DEFAULTS, compute_fighter_snapshot
from ufc_predictor.features.style_features import compute_style_matchup_features
from ufc_predictor.features.validation import validate_training_frame


METADATA_COLUMNS = {
    "fight_id",
    "event_id",
    "event_name",
    "fight_date",
    "fighter_a",
    "fighter_b",
    "winner",
    "target",
    "fighter_a_win",
    "max_history_date_used",
    "source_url",
}


@dataclass(frozen=True)
class BuildProgress:
    processed_fights: int
    total_fights: int
    elapsed_seconds: float
    estimated_remaining_seconds: float


@dataclass
class BuildDatasetReport:
    total_fights_read: int = 0
    total_fights_processed: int = 0
    training_rows_written: int = 0
    skipped_rows_count: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)


class BuildDatasetError(RuntimeError):
    """Raised with fight context when feature generation fails mid-build."""

    def __init__(self, message: str, current_fight: dict[str, Any] | None = None):
        super().__init__(message)
        self.current_fight = current_fight or {}


ProgressCallback = Callable[[BuildProgress], None]


def _normal(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _prepare_fights(fights: pd.DataFrame) -> pd.DataFrame:
    if fights.empty:
        return fights.copy()
    frame = fights.copy()
    required = {"fighter_a", "fighter_b", "fight_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Fights frame is missing columns: {sorted(missing)}")
    if "fight_id" not in frame.columns:
        frame["fight_id"] = np.arange(len(frame))
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce")
    frame["fighter_a"] = frame["fighter_a"].map(_normal)
    frame["fighter_b"] = frame["fighter_b"].map(_normal)
    return frame.sort_values(["fight_date", "fight_id"]).reset_index(drop=True)


def _target(row: pd.Series, fighter_a: str, fighter_b: str) -> int | None:
    target, _ = _target_with_reason(row, fighter_a, fighter_b)
    return target


def _target_with_reason(row: pd.Series, fighter_a: str, fighter_b: str) -> tuple[int | None, str | None]:
    winner = _normal(row.get("winner"))
    if not winner or winner.lower() in {"draw", "nc", "no contest"}:
        return None, "missing_or_non_decisive_winner"
    if winner == fighter_a:
        return 1, None
    if winner == fighter_b:
        return 0, None
    return None, "winner_not_in_fight"


def _prefix_snapshot(prefix: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in snapshot.items()
        if key not in {"fighter", "max_history_date_used"}
    }


def _difference_features(a_snapshot: dict[str, Any], b_snapshot: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in SNAPSHOT_NUMERIC_DEFAULTS:
        try:
            output[f"diff_{key}"] = float(a_snapshot.get(key, 0.0)) - float(b_snapshot.get(key, 0.0))
        except (TypeError, ValueError):
            output[f"diff_{key}"] = 0.0
    output["diff_reach_advantage"] = output.get("diff_reach_in", 0.0)
    return output


def _row_for_order(
    fight_row: pd.Series,
    fighter_a: str,
    fighter_b: str,
    all_fights: pd.DataFrame,
    fight_stats: pd.DataFrame | None,
    fighters: pd.DataFrame | None,
    scorecards: pd.DataFrame | None,
    context: FighterHistoryContext | None = None,
) -> dict[str, Any] | None:
    target = _target(fight_row, fighter_a, fighter_b)
    if target is None:
        return None
    fight_date = fight_row["fight_date"]
    weight_class = fight_row.get("weight_class", "")
    a_snapshot = compute_fighter_snapshot(
        all_fights,
        fighter_a,
        fight_date,
        fight_stats=fight_stats,
        fighters=fighters,
        scorecards=scorecards,
        weight_class=weight_class,
        context=context,
    )
    b_snapshot = compute_fighter_snapshot(
        all_fights,
        fighter_b,
        fight_date,
        fight_stats=fight_stats,
        fighters=fighters,
        scorecards=scorecards,
        weight_class=weight_class,
        context=context,
    )
    style = compute_style_matchup_features(a_snapshot, b_snapshot)
    max_dates = [
        pd.to_datetime(a_snapshot.get("max_history_date_used"), errors="coerce"),
        pd.to_datetime(b_snapshot.get("max_history_date_used"), errors="coerce"),
    ]
    valid_dates = [date for date in max_dates if pd.notna(date)]
    row: dict[str, Any] = {
        "fight_id": fight_row.get("fight_id"),
        "event_id": fight_row.get("event_id"),
        "event_name": fight_row.get("event_name", ""),
        "fight_date": fight_date,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "winner": fight_row.get("winner", ""),
        "fighter_a_win": target,
        "weight_class": weight_class,
        "scheduled_rounds": float(pd.to_numeric(fight_row.get("scheduled_rounds", 3), errors="coerce")),
        "main_event": float(pd.to_numeric(fight_row.get("main_event", 0), errors="coerce")),
        "title_fight": float(pd.to_numeric(fight_row.get("title_fight", 0), errors="coerce")),
        "catchweight": float(pd.to_numeric(fight_row.get("catchweight", 0), errors="coerce")),
        "missed_weight": float(pd.to_numeric(fight_row.get("missed_weight", 0), errors="coerce")),
        "short_notice_replacement": float(pd.to_numeric(fight_row.get("short_notice_replacement", 0), errors="coerce")),
        "event_location": fight_row.get("event_location", fight_row.get("location", "")),
        "max_history_date_used": max(valid_dates) if valid_dates else pd.NaT,
    }
    row.update(_prefix_snapshot("fighter_a", a_snapshot))
    row.update(_prefix_snapshot("fighter_b", b_snapshot))
    row.update(_difference_features(a_snapshot, b_snapshot))
    row.update(style)
    return row


def _fight_context(row: pd.Series) -> dict[str, Any]:
    return {
        "fight_id": row.get("fight_id"),
        "event_name": row.get("event_name", ""),
        "fight_date": str(row.get("fight_date", "")),
        "fighter_a": row.get("fighter_a", ""),
        "fighter_b": row.get("fighter_b", ""),
        "winner": row.get("winner", ""),
    }


def build_fight_dataset(
    fights: pd.DataFrame,
    fight_stats: pd.DataFrame | None = None,
    fighters: pd.DataFrame | None = None,
    scorecards: pd.DataFrame | None = None,
    two_way: bool = False,
    randomize_order: bool = False,
    random_state: int = 42,
    limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 250,
) -> pd.DataFrame:
    """Build one pre-fight matchup row per bout.

    Features for each row are computed strictly from fights with fight_date less
    than the row fight date. Fighter order can be doubled or randomized without
    changing target semantics.
    """

    total_fights_read = int(len(fights))
    frame = _prepare_fights(fights)
    if frame.empty:
        empty = pd.DataFrame()
        empty.attrs["build_report"] = BuildDatasetReport(total_fights_read=total_fights_read)
        return empty
    if limit is not None:
        frame = frame.head(limit).copy()
    total_fights = int(len(frame))
    context = FighterHistoryContext.from_frames(frame, fight_stats=fight_stats, fighters=fighters)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    report = BuildDatasetReport(total_fights_read=total_fights_read, total_fights_processed=total_fights)
    start = time.perf_counter()
    for processed, (_, fight_row) in enumerate(frame.iterrows(), start=1):
        orders = [(fight_row["fighter_a"], fight_row["fighter_b"])]
        if two_way:
            orders.append((fight_row["fighter_b"], fight_row["fighter_a"]))
        elif randomize_order and bool(rng.integers(0, 2)):
            orders = [(fight_row["fighter_b"], fight_row["fighter_a"])]
        for fighter_a, fighter_b in orders:
            _, skip_reason = _target_with_reason(fight_row, fighter_a, fighter_b)
            if skip_reason is not None:
                report.skipped_rows_count += 1
                report.skipped_reasons[skip_reason] = report.skipped_reasons.get(skip_reason, 0) + 1
                continue
            try:
                dataset_row = _row_for_order(
                    fight_row=fight_row,
                    fighter_a=fighter_a,
                    fighter_b=fighter_b,
                    all_fights=frame,
                    fight_stats=fight_stats,
                    fighters=fighters,
                    scorecards=scorecards,
                    context=context,
                )
            except Exception as exc:
                raise BuildDatasetError("Feature generation failed while processing a fight.", _fight_context(fight_row)) from exc
            if dataset_row is None:
                report.skipped_rows_count += 1
                report.skipped_reasons["unknown_target_skip"] = report.skipped_reasons.get("unknown_target_skip", 0) + 1
            else:
                rows.append(dataset_row)
        if progress_callback is not None and (processed % progress_interval == 0 or processed == total_fights):
            elapsed = time.perf_counter() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = (total_fights - processed) / rate if rate > 0 else 0.0
            progress_callback(
                BuildProgress(
                    processed_fights=processed,
                    total_fights=total_fights,
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=remaining,
                )
            )
    dataset = pd.DataFrame(rows)
    if dataset.empty:
        dataset.attrs["build_report"] = report
        return dataset

    elo = build_elo_features(frame)
    elo_columns = [
        "fight_id",
        "fighter_a",
        "fighter_b",
        "fighter_a_pre_fight_elo",
        "fighter_b_pre_fight_elo",
        "diff_pre_fight_elo",
        "fighter_a_pre_weight_class_elo",
        "fighter_b_pre_weight_class_elo",
        "diff_pre_weight_class_elo",
        "fighter_a_elo_expected_win_probability",
        "fighter_b_elo_expected_win_probability",
    ]
    if not elo.empty:
        direct = elo[elo_columns].copy()
        reverse = direct.rename(
            columns={
                "fighter_a": "fighter_b",
                "fighter_b": "fighter_a",
                "fighter_a_pre_fight_elo": "fighter_b_pre_fight_elo",
                "fighter_b_pre_fight_elo": "fighter_a_pre_fight_elo",
                "fighter_a_pre_weight_class_elo": "fighter_b_pre_weight_class_elo",
                "fighter_b_pre_weight_class_elo": "fighter_a_pre_weight_class_elo",
                "fighter_a_elo_expected_win_probability": "fighter_b_elo_expected_win_probability",
                "fighter_b_elo_expected_win_probability": "fighter_a_elo_expected_win_probability",
            }
        )
        reverse["diff_pre_fight_elo"] = -reverse["diff_pre_fight_elo"]
        reverse["diff_pre_weight_class_elo"] = -reverse["diff_pre_weight_class_elo"]
        elo_both = pd.concat([direct, reverse], ignore_index=True)
        dataset = dataset.merge(elo_both, on=["fight_id", "fighter_a", "fighter_b"], how="left")

    numeric_snapshot_columns = {
        f"{prefix}_{key}"
        for prefix in ["fighter_a", "fighter_b"]
        for key in SNAPSHOT_NUMERIC_DEFAULTS
    }
    numeric_columns = {
        *numeric_snapshot_columns,
        *[column for column in dataset.columns if column.startswith("diff_")],
        "scheduled_rounds",
        "main_event",
        "title_fight",
        "catchweight",
        "missed_weight",
        "short_notice_replacement",
        "fighter_a_pre_fight_elo",
        "fighter_b_pre_fight_elo",
        "fighter_a_pre_weight_class_elo",
        "fighter_b_pre_weight_class_elo",
        "fighter_a_elo_expected_win_probability",
        "fighter_b_elo_expected_win_probability",
    }
    for column in sorted(numeric_columns & set(dataset.columns)):
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    validate_training_frame(dataset)
    dataset = dataset.sort_values(["fight_date", "fight_id", "fighter_a"]).reset_index(drop=True)
    report.training_rows_written = int(len(dataset))
    dataset.attrs["build_report"] = report
    return dataset


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
        and not column.endswith("_date_used")
        and column != "winner"
    ]


def save_dataset(frame: pd.DataFrame, path: str | Path | None = None) -> Path:
    output = Path(path) if path else settings.processed_data_dir / "fight_dataset.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_to_write = frame.copy()
    frame_to_write.attrs = {}
    try:
        frame_to_write.to_parquet(output, index=False)
    except Exception:
        output = output.with_suffix(".csv")
        frame_to_write.to_csv(output, index=False)
    return output
