from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.features.elo import EloSystem
from ufc_predictor.features.fighter_history import compute_fighter_snapshot
from ufc_predictor.features.style_features import compute_style_matchup_features
from ufc_predictor.models.explain import top_reasons_from_features


WARNING_TEXT = "Prediction is not guaranteed. Confidence is based on calibrated historical performance."


def confidence_tier(
    confidence_score: float,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
) -> str:
    low = settings.confidence_low_threshold if low_threshold is None else low_threshold
    high = settings.confidence_high_threshold if high_threshold is None else high_threshold
    if confidence_score < low:
        return "Low"
    if confidence_score < high:
        return "Medium"
    return "High"


def normalize_probability_pair(probability_a: float) -> tuple[float, float]:
    a = float(np.clip(probability_a, 0.0, 1.0))
    b = 1.0 - a
    total = a + b
    if total <= 0:
        return 0.5, 0.5
    return a / total, b / total


def format_prediction_output(
    fighter_a: str,
    fighter_b: str,
    fighter_a_win_probability: float,
    top_factors: list[str] | None = None,
) -> dict[str, Any]:
    prob_a, prob_b = normalize_probability_pair(fighter_a_win_probability)
    predicted_winner = fighter_a if prob_a >= prob_b else fighter_b
    confidence_score = max(prob_a, prob_b)
    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "predicted_winner": predicted_winner,
        "fighter_a_win_probability": round(prob_a, 4),
        "fighter_b_win_probability": round(prob_b, 4),
        "confidence_score": round(confidence_score, 4),
        "confidence_tier": confidence_tier(confidence_score),
        "top_factors_for_prediction": top_factors or [],
        "warning": WARNING_TEXT,
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


def _prefix(prefix: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in snapshot.items() if key not in {"fighter", "max_history_date_used"}}


def _diff(a_snapshot: dict[str, Any], b_snapshot: dict[str, Any]) -> dict[str, float]:
    from ufc_predictor.features.fighter_history import SNAPSHOT_NUMERIC_DEFAULTS

    return {
        f"diff_{key}": _numeric(a_snapshot.get(key)) - _numeric(b_snapshot.get(key))
        for key in SNAPSHOT_NUMERIC_DEFAULTS
    } | {"diff_reach_advantage": _numeric(a_snapshot.get("reach_in")) - _numeric(b_snapshot.get("reach_in"))}


def pre_fight_elo_snapshot(
    fights: pd.DataFrame,
    fighter_a: str,
    fighter_b: str,
    fight_date: object,
    weight_class: str = "",
) -> dict[str, float]:
    system = EloSystem()
    if not fights.empty:
        frame = fights.copy()
        frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce")
        past = frame[frame["fight_date"] < pd.Timestamp(fight_date)].sort_values("fight_date")
        for _, row in past.iterrows():
            system.update_fight(
                _normal(row.get("fighter_a")),
                _normal(row.get("fighter_b")),
                row.get("winner", ""),
                row.get("method", ""),
                row.get("weight_class", ""),
            )
    return system.snapshot(fighter_a, fighter_b, weight_class)


def build_prediction_features(
    fighter_a: str,
    fighter_b: str,
    fight_date: object,
    weight_class: str,
    scheduled_rounds: int,
    fights: pd.DataFrame | None = None,
    fight_stats: pd.DataFrame | None = None,
    fighters: pd.DataFrame | None = None,
    scorecards: pd.DataFrame | None = None,
    extra_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    fights_frame = fights.copy() if fights is not None else pd.DataFrame()
    fight_date_ts = pd.Timestamp(fight_date)
    a_snapshot = compute_fighter_snapshot(
        fights_frame,
        fighter_a,
        fight_date_ts,
        fight_stats=fight_stats,
        fighters=fighters,
        scorecards=scorecards,
        weight_class=weight_class,
    )
    b_snapshot = compute_fighter_snapshot(
        fights_frame,
        fighter_b,
        fight_date_ts,
        fight_stats=fight_stats,
        fighters=fighters,
        scorecards=scorecards,
        weight_class=weight_class,
    )
    row: dict[str, Any] = {
        "fight_date": fight_date_ts,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "weight_class": weight_class,
        "scheduled_rounds": float(scheduled_rounds),
        "main_event": 0.0,
        "title_fight": 0.0,
        "catchweight": 0.0,
        "missed_weight": 0.0,
        "short_notice_replacement": 0.0,
        "event_location": "",
    }
    row.update(extra_context or {})
    row.update(_prefix("fighter_a", a_snapshot))
    row.update(_prefix("fighter_b", b_snapshot))
    row.update(_diff(a_snapshot, b_snapshot))
    row.update(compute_style_matchup_features(a_snapshot, b_snapshot))
    row.update(pre_fight_elo_snapshot(fights_frame, fighter_a, fighter_b, fight_date_ts, weight_class))
    return pd.DataFrame([row])


def predict_fight(
    model: Any | str | Path | None,
    fighter_a: str,
    fighter_b: str,
    fight_date: object,
    weight_class: str,
    scheduled_rounds: int,
    fights: pd.DataFrame | None = None,
    fight_stats: pd.DataFrame | None = None,
    fighters: pd.DataFrame | None = None,
    scorecards: pd.DataFrame | None = None,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = build_prediction_features(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        fight_date=fight_date,
        weight_class=weight_class,
        scheduled_rounds=scheduled_rounds,
        fights=fights,
        fight_stats=fight_stats,
        fighters=fighters,
        scorecards=scorecards,
        extra_context=extra_context,
    )
    bundle = model
    if isinstance(model, (str, Path)) or model is None:
        try:
            from ufc_predictor.models.train import load_model_bundle

            bundle = load_model_bundle(model)
        except Exception:
            bundle = None
    if bundle is not None:
        try:
            prob_a = float(bundle.predict_proba(row)[:, 1][0])
        except Exception:
            prob_a = float(row.get("fighter_a_elo_expected_win_probability", pd.Series([0.5])).iloc[0])
    else:
        prob_a = float(row.get("fighter_a_elo_expected_win_probability", pd.Series([0.5])).iloc[0])
    predicted = fighter_a if prob_a >= 0.5 else fighter_b
    reasons = top_reasons_from_features(row.iloc[0], fighter_a, fighter_b, predicted_winner=predicted, max_reasons=10)
    return format_prediction_output(fighter_a, fighter_b, prob_a, reasons)


def prediction_to_json(prediction: dict[str, Any]) -> str:
    return json.dumps(prediction, indent=2, default=str)
