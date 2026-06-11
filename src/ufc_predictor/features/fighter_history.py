from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ufc_predictor.features.trendlines import compute_recent_trend_features
from ufc_predictor.features.validation import assert_history_is_before


SNAPSHOT_NUMERIC_DEFAULTS: dict[str, float] = {
    "age": 0.0,
    "height_in": 0.0,
    "weight_lb": 0.0,
    "reach_in": 0.0,
    "ufc_debut": 1.0,
    "total_ufc_fights_before": 0.0,
    "wins_before": 0.0,
    "losses_before": 0.0,
    "win_rate_before": 0.5,
    "strength_of_schedule": 1500.0,
    "average_opponent_elo": 1500.0,
    "best_win_elo": 1500.0,
    "worst_loss_elo": 1500.0,
    "last_1_win_rate": 0.5,
    "last_3_win_rate": 0.5,
    "last_5_win_rate": 0.5,
    "current_win_streak": 0.0,
    "current_loss_streak": 0.0,
    "days_since_last_fight": 999.0,
    "fights_past_12_months": 0.0,
    "fights_past_24_months": 0.0,
    "layoff_over_365": 0.0,
    "short_turnaround_under_60": 0.0,
    "age_decline_risk": 0.0,
    "weight_class_movement": 0.0,
    "sig_str_landed_per_min": 0.0,
    "sig_str_absorbed_per_min": 0.0,
    "striking_differential": 0.0,
    "sig_str_accuracy": 0.0,
    "sig_str_defense": 0.0,
    "knockdowns_per_15": 0.0,
    "head_strike_share": 0.0,
    "body_strike_share": 0.0,
    "leg_strike_share": 0.0,
    "damage_absorbed_trend_3": 0.0,
    "takedowns_landed_per_15": 0.0,
    "takedown_accuracy": 0.0,
    "takedown_defense": 0.0,
    "submission_attempts_per_15": 0.0,
    "control_time_per_15": 0.0,
    "control_time_allowed_per_15": 0.0,
    "get_up_score": 0.0,
    "wrestling_advantage_score": 0.0,
    "decision_fight_rate": 0.0,
    "split_decision_rate": 0.0,
    "close_decision_rate": 0.0,
    "average_rounds_won_decisions": 0.0,
    "judge_disagreement_score": 0.0,
    "ko_tko_win_rate": 0.0,
    "submission_win_rate": 0.0,
    "decision_win_rate": 0.0,
    "ko_tko_loss_rate": 0.0,
    "submission_loss_rate": 0.0,
    "finish_rate": 0.0,
    "been_finished": 0.0,
    "average_fight_duration": 0.0,
    "round_3_performance": 0.0,
    "championship_round_performance": 0.0,
    "five_round_experience": 0.0,
    "main_event_experience": 0.0,
    "title_fight_experience": 0.0,
    "home_region_fights": 0.0,
    "recent_striking_differential_trend": 0.0,
    "recent_takedown_differential_trend": 0.0,
    "recent_control_time_differential_trend": 0.0,
    "recent_damage_absorbed_trend": 0.0,
    "recent_fight_duration_trend": 0.0,
    "recent_opponent_elo_win_probability": 0.5,
}


def _normal(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(series: pd.Series, default: float = 0.0) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return default
    return float(numeric.mean())


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def parse_height_to_inches(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip().replace('"', "")
    if not text:
        return 0.0
    if "'" in text:
        feet, inches = text.split("'", 1)
        return _numeric(feet) * 12.0 + _numeric(inches)
    return _numeric(text)


def parse_time_to_seconds(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    text = str(value).strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return _numeric(minutes) * 60.0 + _numeric(seconds)
    return _numeric(text)


def infer_fight_duration_seconds(row: pd.Series) -> float:
    finish_round = max(int(_numeric(row.get("finish_round"), 1)), 1)
    elapsed_in_round = parse_time_to_seconds(row.get("finish_time", 0))
    return float((finish_round - 1) * 300 + elapsed_in_round)


def fighter_fight_history(fights: pd.DataFrame, fighter: str, as_of_date: object) -> pd.DataFrame:
    if fights.empty:
        return fights.copy()
    frame = fights.copy()
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce")
    fighter_name = _normal(fighter)
    mask = (
        ((frame["fighter_a"].map(_normal) == fighter_name) | (frame["fighter_b"].map(_normal) == fighter_name))
        & (frame["fight_date"] < pd.Timestamp(as_of_date))
    )
    history = frame.loc[mask].sort_values("fight_date").copy()
    assert_history_is_before(history, as_of_date)
    return history


def _fighter_bio(fighters: pd.DataFrame | None, fighter: str, as_of_date: object) -> dict[str, Any]:
    output: dict[str, Any] = {"stance": "Unknown", "height_in": 0.0, "weight_lb": 0.0, "reach_in": 0.0, "age": 0.0}
    if fighters is None or fighters.empty or "name" not in fighters.columns:
        return output
    rows = fighters[fighters["name"].map(_normal) == _normal(fighter)]
    if rows.empty:
        return output
    row = rows.iloc[-1]
    output["stance"] = _normal(row.get("stance")) or "Unknown"
    output["height_in"] = parse_height_to_inches(row.get("height_in", row.get("height", 0)))
    output["weight_lb"] = _numeric(row.get("weight_lb", row.get("weight", 0)))
    output["reach_in"] = parse_height_to_inches(row.get("reach_in", row.get("reach", 0)))
    dob = pd.to_datetime(row.get("date_of_birth", row.get("dob", None)), errors="coerce")
    if pd.notna(dob):
        output["age"] = max((pd.Timestamp(as_of_date) - dob).days / 365.25, 0.0)
    return output


def _results_from_history(history: pd.DataFrame, fighter: str) -> pd.Series:
    fighter_name = _normal(fighter)
    return history.apply(lambda row: 1.0 if _normal(row.get("winner")) == fighter_name else 0.0, axis=1)


def _current_streak(results: list[float], winning: bool) -> float:
    if not results:
        return 0.0
    wanted = 1.0 if winning else 0.0
    streak = 0
    for result in reversed(results):
        if result == wanted:
            streak += 1
        else:
            break
    return float(streak)


def _finish_flags(history: pd.DataFrame, fighter: str) -> dict[str, float]:
    if history.empty:
        return {}
    fighter_name = _normal(fighter)
    methods = history.get("method", pd.Series([""] * len(history))).fillna("").astype(str).str.lower()
    wins = history.get("winner", pd.Series([""] * len(history))).map(_normal) == fighter_name
    losses = ~wins
    fight_count = max(len(history), 1)
    win_count = max(int(wins.sum()), 1)
    loss_count = max(int(losses.sum()), 1)
    ko = methods.str.contains("ko|tko", regex=True)
    sub = methods.str.contains("sub")
    decision = methods.str.contains("decision")
    finished_loss = losses & (ko | sub)
    return {
        "ko_tko_win_rate": float((wins & ko).sum() / win_count),
        "submission_win_rate": float((wins & sub).sum() / win_count),
        "decision_win_rate": float((wins & decision).sum() / win_count),
        "ko_tko_loss_rate": float((losses & ko).sum() / loss_count),
        "submission_loss_rate": float((losses & sub).sum() / loss_count),
        "finish_rate": float(((wins & (ko | sub)).sum()) / fight_count),
        "been_finished": float(finished_loss.any()),
        "decision_fight_rate": float(decision.sum() / fight_count),
        "split_decision_rate": float(methods.str.contains("split").sum() / fight_count),
    }


def _stats_for_history(
    fight_stats: pd.DataFrame | None,
    history: pd.DataFrame,
    fighter: str,
) -> pd.DataFrame:
    if fight_stats is None or fight_stats.empty or history.empty:
        return pd.DataFrame()
    stats = fight_stats.copy()
    if "fight_id" in stats.columns and "fight_id" in history.columns:
        stats = stats[stats["fight_id"].isin(history["fight_id"])]
    if "fight_date" not in stats.columns and "fight_id" in stats.columns and "fight_id" in history.columns:
        stats = stats.merge(history[["fight_id", "fight_date", "fighter_a", "fighter_b"]], on="fight_id", how="left")
    if "fighter" not in stats.columns:
        return pd.DataFrame()
    return stats[stats["fighter"].map(_normal) == _normal(fighter)].sort_values("fight_date").copy()


def _opponent_stats_for_history(
    fight_stats: pd.DataFrame | None,
    history: pd.DataFrame,
    fighter: str,
) -> pd.DataFrame:
    if fight_stats is None or fight_stats.empty or history.empty:
        return pd.DataFrame()
    stats = fight_stats.copy()
    if "fight_id" in stats.columns and "fight_id" in history.columns:
        stats = stats[stats["fight_id"].isin(history["fight_id"])]
    if "fight_date" not in stats.columns and "fight_id" in stats.columns and "fight_id" in history.columns:
        stats = stats.merge(history[["fight_id", "fight_date", "fighter_a", "fighter_b"]], on="fight_id", how="left")
    if "fighter" not in stats.columns:
        return pd.DataFrame()
    return stats[stats["fighter"].map(_normal) != _normal(fighter)].copy()


def _rate_per_min(total: float, seconds: float) -> float:
    minutes = seconds / 60.0
    if minutes <= 0:
        return 0.0
    return float(total / minutes)


def _rate_per_15(total: float, seconds: float) -> float:
    minutes = seconds / 60.0
    if minutes <= 0:
        return 0.0
    return float(total * 15.0 / minutes)


def _stat_features(stats: pd.DataFrame, opponent_stats: pd.DataFrame, history: pd.DataFrame) -> dict[str, float]:
    if stats.empty:
        return {}
    durations = history.copy()
    if "fight_duration_seconds" not in durations.columns:
        durations["fight_duration_seconds"] = durations.apply(infer_fight_duration_seconds, axis=1)
    duration_by_id = (
        durations.set_index("fight_id")["fight_duration_seconds"].to_dict()
        if "fight_id" in durations.columns
        else {}
    )
    seconds = 0.0
    if "fight_id" in stats.columns:
        seconds = float(sum(duration_by_id.get(fight_id, 0.0) for fight_id in stats["fight_id"]))
    if seconds <= 0:
        seconds = float(len(stats) * 900)

    sig_landed = _sum_column(stats, "sig_str_landed")
    sig_attempted = _sum_column(stats, "sig_str_attempted")
    opp_sig_landed = _sum_column(opponent_stats, "sig_str_landed")
    opp_sig_attempted = _sum_column(opponent_stats, "sig_str_attempted")
    takedowns = _sum_column(stats, "takedowns_landed")
    takedown_attempts = _sum_column(stats, "takedowns_attempted")
    opp_takedowns = _sum_column(opponent_stats, "takedowns_landed")
    opp_takedown_attempts = _sum_column(opponent_stats, "takedowns_attempted")
    control = _sum_column(stats, "control_seconds")
    opp_control = _sum_column(opponent_stats, "control_seconds")

    landed_per_min = _rate_per_min(sig_landed, seconds)
    absorbed_per_min = _rate_per_min(opp_sig_landed, seconds)
    total_target_landed = sum(_sum_column(stats, column) for column in ["head_landed", "body_landed", "leg_landed"])
    if total_target_landed <= 0:
        total_target_landed = 1.0
    takedown_defense = 1.0
    if opp_takedown_attempts > 0:
        takedown_defense = 1.0 - (opp_takedowns / opp_takedown_attempts)

    recent_absorbed = opponent_stats.sort_values("fight_date").get("sig_str_landed", pd.Series(dtype=float)).tail(3)
    damage_trend = float(recent_absorbed.diff().mean()) if len(recent_absorbed) >= 2 else 0.0
    output = {
        "sig_str_landed_per_min": landed_per_min,
        "sig_str_absorbed_per_min": absorbed_per_min,
        "striking_differential": landed_per_min - absorbed_per_min,
        "sig_str_accuracy": float(sig_landed / sig_attempted) if sig_attempted > 0 else 0.0,
        "sig_str_defense": float(1.0 - (opp_sig_landed / opp_sig_attempted)) if opp_sig_attempted > 0 else 0.0,
        "knockdowns_per_15": _rate_per_15(_sum_column(stats, "knockdowns"), seconds),
        "head_strike_share": float(_sum_column(stats, "head_landed") / total_target_landed),
        "body_strike_share": float(_sum_column(stats, "body_landed") / total_target_landed),
        "leg_strike_share": float(_sum_column(stats, "leg_landed") / total_target_landed),
        "damage_absorbed_trend_3": damage_trend,
        "takedowns_landed_per_15": _rate_per_15(takedowns, seconds),
        "takedown_accuracy": float(takedowns / takedown_attempts) if takedown_attempts > 0 else 0.0,
        "takedown_defense": float(max(min(takedown_defense, 1.0), 0.0)),
        "submission_attempts_per_15": _rate_per_15(
            _sum_column(stats, "submission_attempts"), seconds
        ),
        "control_time_per_15": _rate_per_15(control, seconds),
        "control_time_allowed_per_15": _rate_per_15(opp_control, seconds),
        "get_up_score": float(1.0 / (1.0 + _rate_per_15(opp_control, seconds))),
    }
    output["wrestling_advantage_score"] = (
        output["takedowns_landed_per_15"] + output["submission_attempts_per_15"] + output["control_time_per_15"] / 60.0
        - output["control_time_allowed_per_15"] / 60.0
    )
    return output


def _scorecard_features(scorecards: pd.DataFrame | None, fighter: str, as_of_date: object) -> dict[str, float]:
    if scorecards is None or scorecards.empty:
        return {}
    cards = scorecards.copy()
    if "fight_date" not in cards.columns:
        return {}
    cards["fight_date"] = pd.to_datetime(cards["fight_date"], errors="coerce")
    fighter_name = _normal(fighter)
    if "fighter_a" not in cards.columns or "fighter_b" not in cards.columns:
        return {}
    cards = cards[
        (cards["fight_date"] < pd.Timestamp(as_of_date))
        & ((cards["fighter_a"].map(_normal) == fighter_name) | (cards["fighter_b"].map(_normal) == fighter_name))
    ].copy()
    assert_history_is_before(cards, as_of_date)
    if cards.empty:
        return {}
    is_a = cards["fighter_a"].map(_normal) == fighter_name
    totals_for = np.where(is_a, pd.to_numeric(cards["total_a"], errors="coerce"), pd.to_numeric(cards["total_b"], errors="coerce"))
    totals_against = np.where(is_a, pd.to_numeric(cards["total_b"], errors="coerce"), pd.to_numeric(cards["total_a"], errors="coerce"))
    margins = pd.Series(totals_for - totals_against).dropna()
    decision_type = cards.get("decision_type", pd.Series([""] * len(cards))).fillna("").str.lower()
    judge_counts = cards.groupby(["event", "fighter_a", "fighter_b"]).size()
    return {
        "close_decision_rate": float((margins.abs() <= 1).mean()) if not margins.empty else 0.0,
        "average_rounds_won_decisions": float((margins > 0).mean() * 3.0) if not margins.empty else 0.0,
        "judge_disagreement_score": float(judge_counts.std()) if len(judge_counts) > 1 else 0.0,
        "split_decision_rate": float(decision_type.str.contains("split").mean()),
    }


def _enriched_history_for_trends(
    history: pd.DataFrame,
    stats: pd.DataFrame,
    opponent_stats: pd.DataFrame,
) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    output = history.copy()
    output["fight_duration_seconds"] = output.apply(infer_fight_duration_seconds, axis=1)
    if stats.empty:
        return output
    by_id = stats.set_index("fight_id") if "fight_id" in stats.columns else pd.DataFrame()
    opp_by_id = opponent_stats.set_index("fight_id") if "fight_id" in opponent_stats.columns else pd.DataFrame()

    def stat_value(row: pd.Series, frame: pd.DataFrame, column: str) -> float:
        if frame.empty or column not in frame.columns:
            return 0.0
        fight_id = row.get("fight_id")
        if fight_id not in frame.index:
            return 0.0
        value = frame.loc[fight_id, column]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return _numeric(value)

    output["striking_differential"] = output.apply(
        lambda row: stat_value(row, by_id, "sig_str_landed") - stat_value(row, opp_by_id, "sig_str_landed"), axis=1
    )
    output["takedown_differential"] = output.apply(
        lambda row: stat_value(row, by_id, "takedowns_landed") - stat_value(row, opp_by_id, "takedowns_landed"), axis=1
    )
    output["control_time_differential"] = output.apply(
        lambda row: stat_value(row, by_id, "control_seconds") - stat_value(row, opp_by_id, "control_seconds"), axis=1
    )
    output["sig_str_absorbed"] = output.apply(lambda row: stat_value(row, opp_by_id, "sig_str_landed"), axis=1)
    return output


def compute_fighter_snapshot(
    fights: pd.DataFrame,
    fighter: str,
    as_of_date: object,
    fight_stats: pd.DataFrame | None = None,
    fighters: pd.DataFrame | None = None,
    scorecards: pd.DataFrame | None = None,
    weight_class: str | None = None,
) -> dict[str, Any]:
    """Compute one fighter's pre-fight feature snapshot using only prior fights."""

    snapshot: dict[str, Any] = dict(SNAPSHOT_NUMERIC_DEFAULTS)
    snapshot.update(_fighter_bio(fighters, fighter, as_of_date))
    snapshot["fighter"] = _normal(fighter)
    history = fighter_fight_history(fights, fighter, as_of_date)
    if history.empty:
        snapshot["max_history_date_used"] = pd.NaT
        return snapshot

    results = _results_from_history(history, fighter).tolist()
    fight_count = len(history)
    wins = float(sum(results))
    losses = float(fight_count - wins)
    last_date = pd.to_datetime(history["fight_date"], errors="coerce").max()
    as_timestamp = pd.Timestamp(as_of_date)
    days_since_last = (as_timestamp - last_date).days if pd.notna(last_date) else 999
    snapshot.update(
        {
            "ufc_debut": 0.0,
            "total_ufc_fights_before": float(fight_count),
            "wins_before": wins,
            "losses_before": losses,
            "win_rate_before": wins / fight_count if fight_count else 0.5,
            "last_1_win_rate": float(np.mean(results[-1:])) if results else 0.5,
            "last_3_win_rate": float(np.mean(results[-3:])) if results else 0.5,
            "last_5_win_rate": float(np.mean(results[-5:])) if results else 0.5,
            "current_win_streak": _current_streak(results, winning=True),
            "current_loss_streak": _current_streak(results, winning=False),
            "days_since_last_fight": float(days_since_last),
            "fights_past_12_months": float((history["fight_date"] >= as_timestamp - pd.Timedelta(days=365)).sum()),
            "fights_past_24_months": float((history["fight_date"] >= as_timestamp - pd.Timedelta(days=730)).sum()),
            "layoff_over_365": float(days_since_last > 365),
            "short_turnaround_under_60": float(days_since_last < 60),
            "age_decline_risk": float(snapshot.get("age", 0.0) >= 35 and days_since_last > 180),
            "average_fight_duration": float(history.apply(infer_fight_duration_seconds, axis=1).mean()),
            "five_round_experience": float((pd.to_numeric(history.get("scheduled_rounds", 0), errors="coerce") >= 5).sum()),
            "main_event_experience": _mean(history.get("main_event", pd.Series(dtype=float)), 0.0) * fight_count,
            "title_fight_experience": _mean(history.get("title_fight", pd.Series(dtype=float)), 0.0) * fight_count,
            "max_history_date_used": last_date,
        }
    )

    previous_weight_classes = history.get("weight_class", pd.Series(dtype=object)).dropna().astype(str)
    if weight_class and not previous_weight_classes.empty:
        snapshot["weight_class_movement"] = float(previous_weight_classes.iloc[-1] != str(weight_class))

    if "opponent_elo" in history.columns:
        opponent_elos = pd.to_numeric(history["opponent_elo"], errors="coerce").dropna()
        if not opponent_elos.empty:
            snapshot["strength_of_schedule"] = float(opponent_elos.mean())
            snapshot["average_opponent_elo"] = float(opponent_elos.mean())
            win_elos = opponent_elos[_results_from_history(history, fighter).astype(bool).values]
            loss_elos = opponent_elos[~_results_from_history(history, fighter).astype(bool).values]
            snapshot["best_win_elo"] = float(win_elos.max()) if not win_elos.empty else 1500.0
            snapshot["worst_loss_elo"] = float(loss_elos.min()) if not loss_elos.empty else 1500.0

    snapshot.update(_finish_flags(history, fighter))
    stats = _stats_for_history(fight_stats, history, fighter)
    opponent_stats = _opponent_stats_for_history(fight_stats, history, fighter)
    snapshot.update(_stat_features(stats, opponent_stats, history))
    trend_history = _enriched_history_for_trends(history, stats, opponent_stats)
    snapshot.update(compute_recent_trend_features(trend_history))
    snapshot.update(_scorecard_features(scorecards, fighter, as_of_date))

    if fight_count:
        snapshot["round_3_performance"] = snapshot["last_3_win_rate"]
        snapshot["championship_round_performance"] = snapshot["last_5_win_rate"] if snapshot["five_round_experience"] else 0.0
    for key, default in SNAPSHOT_NUMERIC_DEFAULTS.items():
        snapshot[key] = _numeric(snapshot.get(key), default)
    return snapshot
