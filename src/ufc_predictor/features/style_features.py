from __future__ import annotations

from typing import Any


def _value(snapshot: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = snapshot.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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

    a_stance = str(fighter_a_snapshot.get("stance", "Unknown")).lower()
    b_stance = str(fighter_b_snapshot.get("stance", "Unknown")).lower()
    return {
        "stance_matchup": stance_matchup(fighter_a_snapshot.get("stance"), fighter_b_snapshot.get("stance")),
        "southpaw_matchup_flag": int("southpaw" in {a_stance, b_stance}),
        "diff_striker_vs_grappler_score": (a_striking - b_wrestling) - (b_striking - a_wrestling),
        "diff_pressure_vs_counter_score": a_pressure - b_pressure,
        "diff_reach_volume_interaction": (a_reach - b_reach) * (a_volume - b_volume),
        "diff_takedown_threat_vs_defense": (
            a_takedown_threat * _value(fighter_b_snapshot, "takedown_defense")
        )
        - (b_takedown_threat * _value(fighter_a_snapshot, "takedown_defense")),
        "diff_cardio_trend_by_round": _value(fighter_a_snapshot, "round_3_performance")
        - _value(fighter_b_snapshot, "round_3_performance"),
        "diff_five_round_experience": _value(fighter_a_snapshot, "five_round_experience")
        - _value(fighter_b_snapshot, "five_round_experience"),
    }
