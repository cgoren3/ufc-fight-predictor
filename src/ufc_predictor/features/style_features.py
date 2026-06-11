from __future__ import annotations

from typing import Any


def _value(snapshot: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = snapshot.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(min(value, high), low)


def stance_matchup(fighter_a_stance: Any, fighter_b_stance: Any) -> str:
    a = str(fighter_a_stance or "Unknown").strip().title()
    b = str(fighter_b_stance or "Unknown").strip().title()
    return f"{a}_vs_{b}"


def compute_style_matchup_features(
    fighter_a_snapshot: dict[str, Any],
    fighter_b_snapshot: dict[str, Any],
) -> dict[str, float | int | str]:
    """Create matchup interaction features from two pre-fight snapshots."""

    a_striking = _value(fighter_a_snapshot, "striking_differential")
    b_striking = _value(fighter_b_snapshot, "striking_differential")
    a_wrestling = _value(fighter_a_snapshot, "wrestling_advantage_score")
    b_wrestling = _value(fighter_b_snapshot, "wrestling_advantage_score")
    a_pressure = _value(fighter_a_snapshot, "sig_str_landed_per_min") - _value(
        fighter_a_snapshot, "average_fight_duration", 0.0
    ) / 900.0
    b_pressure = _value(fighter_b_snapshot, "sig_str_landed_per_min") - _value(
        fighter_b_snapshot, "average_fight_duration", 0.0
    ) / 900.0
    a_reach = _value(fighter_a_snapshot, "reach_in")
    b_reach = _value(fighter_b_snapshot, "reach_in")
    a_volume = _value(fighter_a_snapshot, "sig_str_landed_per_min")
    b_volume = _value(fighter_b_snapshot, "sig_str_landed_per_min")
    a_takedown_threat = _value(fighter_a_snapshot, "takedowns_landed_per_15") + _value(
        fighter_a_snapshot, "submission_attempts_per_15"
    )
    b_takedown_threat = _value(fighter_b_snapshot, "takedowns_landed_per_15") + _value(
        fighter_b_snapshot, "submission_attempts_per_15"
    )
    a_td_defense = _value(fighter_a_snapshot, "takedown_defense", 1.0)
    b_td_defense = _value(fighter_b_snapshot, "takedown_defense", 1.0)
    a_absorbed = _value(fighter_a_snapshot, "sig_str_absorbed_per_min")
    b_absorbed = _value(fighter_b_snapshot, "sig_str_absorbed_per_min")
    a_sub_threat = _value(fighter_a_snapshot, "submission_attempts_per_15")
    b_sub_threat = _value(fighter_b_snapshot, "submission_attempts_per_15")
    a_sub_vulnerability = _value(fighter_a_snapshot, "submission_loss_rate") + 0.5 * _value(
        fighter_a_snapshot, "been_finished"
    )
    b_sub_vulnerability = _value(fighter_b_snapshot, "submission_loss_rate") + 0.5 * _value(
        fighter_b_snapshot, "been_finished"
    )
    a_finish_rate = _value(fighter_a_snapshot, "finish_rate")
    b_finish_rate = _value(fighter_b_snapshot, "finish_rate")
    a_durability = 1.0 - _bounded(
        _value(fighter_a_snapshot, "ko_tko_loss_rate")
        + _value(fighter_a_snapshot, "submission_loss_rate")
        + 0.25 * _value(fighter_a_snapshot, "been_finished"),
        0.0,
        1.0,
    )
    b_durability = 1.0 - _bounded(
        _value(fighter_b_snapshot, "ko_tko_loss_rate")
        + _value(fighter_b_snapshot, "submission_loss_rate")
        + 0.25 * _value(fighter_b_snapshot, "been_finished"),
        0.0,
        1.0,
    )
    a_control = _value(fighter_a_snapshot, "control_time_per_15")
    b_control = _value(fighter_b_snapshot, "control_time_per_15")
    a_control_allowed = _value(fighter_a_snapshot, "control_time_allowed_per_15")
    b_control_allowed = _value(fighter_b_snapshot, "control_time_allowed_per_15")

    a_stance = str(fighter_a_snapshot.get("stance", "Unknown")).lower()
    b_stance = str(fighter_b_snapshot.get("stance", "Unknown")).lower()
    stance_edge = 0.0
    if a_stance != "unknown" and b_stance != "unknown" and a_stance != b_stance:
        stance_edge = 1.0 if "southpaw" in a_stance or "switch" in a_stance else -1.0 if "southpaw" in b_stance or "switch" in b_stance else 0.0
    a_opponent_adjusted_striking = a_volume - b_absorbed
    b_opponent_adjusted_striking = b_volume - a_absorbed
    a_opponent_adjusted_takedowns = _value(fighter_a_snapshot, "takedowns_landed_per_15") - (1.0 - b_td_defense)
    b_opponent_adjusted_takedowns = _value(fighter_b_snapshot, "takedowns_landed_per_15") - (1.0 - a_td_defense)
    a_opponent_adjusted_submissions = a_sub_threat - b_sub_vulnerability
    b_opponent_adjusted_submissions = b_sub_threat - a_sub_vulnerability
    a_opponent_adjusted_control = a_control - b_control_allowed
    b_opponent_adjusted_control = b_control - a_control_allowed
    a_opponent_adjusted_finishing = a_finish_rate - (1.0 - b_durability)
    b_opponent_adjusted_finishing = b_finish_rate - (1.0 - a_durability)
    return {
        "stance_matchup": stance_matchup(fighter_a_snapshot.get("stance"), fighter_b_snapshot.get("stance")),
        "southpaw_matchup_flag": int("southpaw" in {a_stance, b_stance}),
        "fighter_a_opponent_adjusted_striking": a_opponent_adjusted_striking,
        "fighter_b_opponent_adjusted_striking": b_opponent_adjusted_striking,
        "diff_opponent_adjusted_striking": a_opponent_adjusted_striking - b_opponent_adjusted_striking,
        "fighter_a_opponent_adjusted_takedowns": a_opponent_adjusted_takedowns,
        "fighter_b_opponent_adjusted_takedowns": b_opponent_adjusted_takedowns,
        "diff_opponent_adjusted_takedowns": a_opponent_adjusted_takedowns - b_opponent_adjusted_takedowns,
        "fighter_a_opponent_adjusted_submissions": a_opponent_adjusted_submissions,
        "fighter_b_opponent_adjusted_submissions": b_opponent_adjusted_submissions,
        "diff_opponent_adjusted_submissions": a_opponent_adjusted_submissions - b_opponent_adjusted_submissions,
        "fighter_a_opponent_adjusted_control": a_opponent_adjusted_control,
        "fighter_b_opponent_adjusted_control": b_opponent_adjusted_control,
        "diff_opponent_adjusted_control": a_opponent_adjusted_control - b_opponent_adjusted_control,
        "fighter_a_opponent_adjusted_finishing": a_opponent_adjusted_finishing,
        "fighter_b_opponent_adjusted_finishing": b_opponent_adjusted_finishing,
        "diff_opponent_adjusted_finishing": a_opponent_adjusted_finishing - b_opponent_adjusted_finishing,
        "diff_striker_vs_grappler_score": (a_striking - b_wrestling) - (b_striking - a_wrestling),
        "diff_pressure_vs_counter_score": a_pressure - b_pressure,
        "diff_reach_volume_interaction": (a_reach - b_reach) * (a_volume - b_volume),
        "diff_takedown_threat_vs_defense": (
            a_takedown_threat * _value(fighter_b_snapshot, "takedown_defense")
        )
        - (b_takedown_threat * _value(fighter_a_snapshot, "takedown_defense")),
        "diff_takedown_offense_vs_defense_clash": (
            _value(fighter_a_snapshot, "takedowns_landed_per_15") * (1.0 - b_td_defense)
        )
        - (_value(fighter_b_snapshot, "takedowns_landed_per_15") * (1.0 - a_td_defense)),
        "diff_strike_volume_vs_absorption_clash": (a_volume * (1.0 + b_absorbed)) - (b_volume * (1.0 + a_absorbed)),
        "diff_submission_threat_vs_vulnerability_clash": (a_sub_threat * (1.0 + b_sub_vulnerability))
        - (b_sub_threat * (1.0 + a_sub_vulnerability)),
        "diff_finishing_vs_durability_clash": (a_finish_rate * (1.0 - b_durability))
        - (b_finish_rate * (1.0 - a_durability)),
        "diff_reach_stance_clash": (a_reach - b_reach) + stance_edge,
        "diff_cardio_trend_by_round": _value(fighter_a_snapshot, "round_3_performance")
        - _value(fighter_b_snapshot, "round_3_performance"),
        "diff_five_round_experience": _value(fighter_a_snapshot, "five_round_experience")
        - _value(fighter_b_snapshot, "five_round_experience"),
    }
