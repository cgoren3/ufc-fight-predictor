from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FEATURE_REASON_TEMPLATES: dict[str, str] = {
    "diff_pre_fight_elo": "{winner} has {value:+.1f} Elo advantage",
    "diff_pre_weight_class_elo": "{winner} has {value:+.1f} weight-class Elo advantage",
    "diff_win_rate_before": "{winner} has the stronger prior UFC win rate",
    "diff_total_ufc_fights_before": "{winner} has more UFC experience",
    "diff_last_3_win_rate": "{winner} has better recent form over the last 3 fights",
    "diff_current_win_streak": "{winner} has the stronger current win streak",
    "diff_days_since_last_fight": "{winner} has the timing edge from recent activity",
    "diff_striking_differential": "{winner} has stronger striking differential",
    "diff_sig_str_landed_per_min": "{winner} lands more significant strikes per minute",
    "diff_sig_str_absorbed_per_min": "{loser} absorbs more significant strikes per minute",
    "diff_sig_str_accuracy": "{winner} has better significant-strike accuracy",
    "diff_sig_str_defense": "{winner} has better significant-strike defense",
    "diff_takedowns_landed_per_15": "{winner} brings more takedown volume",
    "diff_takedown_accuracy": "{winner} has better takedown accuracy",
    "diff_takedown_defense": "{winner} has stronger takedown defense",
    "diff_submission_attempts_per_15": "{winner} threatens more submissions",
    "diff_control_time_per_15": "{winner} has more top/control time",
    "diff_wrestling_advantage_score": "{winner} grades better in the wrestling matchup",
    "diff_finish_rate": "{winner} has the higher historical finish rate",
    "diff_been_finished": "{loser} has the higher historical finished-loss flag",
    "diff_average_fight_duration": "{winner} has more demonstrated fight-duration experience",
    "diff_five_round_experience": "{winner} has more five-round experience",
    "diff_reach_advantage": "{winner} has reach advantage",
    "diff_reach_volume_interaction": "{winner}'s reach and volume interact better in this matchup",
    "diff_takedown_threat_vs_defense": "{winner}'s takedown threat matches well against {loser}'s defense",
    "diff_opponent_adjusted_striking": "{winner} has the better opponent-adjusted striking profile",
    "diff_opponent_adjusted_takedowns": "{winner} has the better opponent-adjusted takedown profile",
    "diff_opponent_adjusted_submissions": "{winner} has the better opponent-adjusted submission threat",
    "diff_opponent_adjusted_control": "{winner} has the better opponent-adjusted control profile",
    "diff_opponent_adjusted_finishing": "{winner} has the better opponent-adjusted finishing profile",
    "diff_takedown_offense_vs_defense_clash": "{winner}'s takedown offense maps well against {loser}'s defense",
    "diff_strike_volume_vs_absorption_clash": "{winner}'s strike volume maps well against {loser}'s absorption profile",
    "diff_submission_threat_vs_vulnerability_clash": "{winner}'s submission threat maps well against {loser}'s vulnerability",
    "diff_finishing_vs_durability_clash": "{winner}'s finishing profile maps well against {loser}'s durability",
    "diff_reach_stance_clash": "{winner} has a reach/stance matchup edge",
    "diff_cardio_trend_by_round": "{winner} has the better late-fight trend",
}


def _numeric(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def top_reasons_from_features(
    features: pd.Series | dict[str, Any],
    fighter_a: str,
    fighter_b: str,
    predicted_winner: str,
    max_reasons: int = 10,
) -> list[str]:
    row = pd.Series(features)
    direction = 1.0 if predicted_winner == fighter_a else -1.0
    winner = fighter_a if predicted_winner == fighter_a else fighter_b
    loser = fighter_b if predicted_winner == fighter_a else fighter_a
    scored: list[tuple[float, str]] = []
    for feature, template in FEATURE_REASON_TEMPLATES.items():
        if feature not in row:
            continue
        raw_value = _numeric(row[feature])
        oriented = raw_value * direction
        if abs(oriented) < 1e-6:
            continue
        if "loser" in template and oriented > 0:
            value_for_text = abs(raw_value)
        elif oriented <= 0:
            continue
        else:
            value_for_text = oriented
        scored.append((abs(oriented), template.format(winner=winner, loser=loser, value=value_for_text)))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return ["Model has limited historical information; probability is driven mostly by defaults and Elo priors."]
    return [reason for _, reason in scored[:max_reasons]]


def top_factors_by_side(
    features: pd.Series | dict[str, Any],
    fighter_a: str,
    fighter_b: str,
    max_reasons: int = 5,
) -> tuple[list[str], list[str]]:
    row = pd.Series(features)
    a_reasons: list[tuple[float, str]] = []
    b_reasons: list[tuple[float, str]] = []
    for feature, template in FEATURE_REASON_TEMPLATES.items():
        if feature not in row:
            continue
        raw_value = _numeric(row[feature])
        if abs(raw_value) < 1e-6:
            continue
        if raw_value > 0:
            a_reasons.append((abs(raw_value), template.format(winner=fighter_a, loser=fighter_b, value=raw_value)))
        else:
            b_reasons.append((abs(raw_value), template.format(winner=fighter_b, loser=fighter_a, value=abs(raw_value))))
    a_reasons.sort(key=lambda item: item[0], reverse=True)
    b_reasons.sort(key=lambda item: item[0], reverse=True)
    return [reason for _, reason in a_reasons[:max_reasons]], [reason for _, reason in b_reasons[:max_reasons]]


def uncertainty_factors_from_features(features: pd.Series | dict[str, Any]) -> list[str]:
    row = pd.Series(features)
    factors: list[str] = []
    if _numeric(row.get("fighter_a_total_ufc_fights_before")) < 3 or _numeric(row.get("fighter_b_total_ufc_fights_before")) < 3:
        factors.append("Limited historical sample for one or both fighters.")
    if _numeric(row.get("fighter_a_reach_in")) <= 0 or _numeric(row.get("fighter_b_reach_in")) <= 0:
        factors.append("Reach data is missing for one or both fighters.")
    if str(row.get("fighter_a_stance", "Unknown")) == "Unknown" or str(row.get("fighter_b_stance", "Unknown")) == "Unknown":
        factors.append("Stance data is missing or unknown for one or both fighters.")
    if abs(_numeric(row.get("diff_pre_fight_elo"))) < 25:
        factors.append("Pre-fight Elo ratings are close.")
    if not factors:
        factors.append("Uncertainty is mainly estimated from calibrated ensemble disagreement.")
    return factors


def missing_data_warnings_from_features(features: pd.Series | dict[str, Any]) -> list[str]:
    row = pd.Series(features)
    warnings: list[str] = []
    if _numeric(row.get("fighter_a_total_ufc_fights_before")) <= 0:
        warnings.append("Fighter A has no usable prior fight history in the dataset.")
    if _numeric(row.get("fighter_b_total_ufc_fights_before")) <= 0:
        warnings.append("Fighter B has no usable prior fight history in the dataset.")
    if _numeric(row.get("fighter_a_reach_in")) <= 0 or _numeric(row.get("fighter_b_reach_in")) <= 0:
        warnings.append("Reach data is missing for one or both fighters.")
    if str(row.get("event_location", "")).strip() == "":
        warnings.append("Event location is missing; no location-context features are available.")
    return warnings


def shap_explanations_if_available(model: Any, frame: pd.DataFrame, max_features: int = 10) -> list[dict[str, float]]:
    """Return SHAP values if shap is installed and compatible; otherwise return []."""

    try:
        import shap
    except Exception:
        return []
    try:  # pragma: no cover - optional dependency integration
        explainer = shap.Explainer(model)
        values = explainer(frame)
        mean_abs = np.abs(values.values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:max_features]
        return [{"feature": frame.columns[i], "mean_abs_shap": float(mean_abs[i])} for i in order]
    except Exception:
        return []
