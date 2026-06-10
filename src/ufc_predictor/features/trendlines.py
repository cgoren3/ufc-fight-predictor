from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def exponential_weighted_average(values: Iterable[float], alpha: float = 0.60, default: float = 0.0) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    if not clean:
        return default
    estimate = clean[0]
    for value in clean[1:]:
        estimate = alpha * value + (1.0 - alpha) * estimate
    return float(estimate)


def rolling_average(values: Iterable[float], window: int, default: float = 0.0) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    if not clean:
        return default
    return float(np.mean(clean[-window:]))


def rolling_slope(values: Iterable[float], window: int = 3, default: float = 0.0) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    clean = clean[-window:]
    if len(clean) < 2:
        return default
    x = np.arange(len(clean), dtype=float)
    slope = np.polyfit(x, np.array(clean, dtype=float), 1)[0]
    return float(slope)


def compute_recent_trend_features(history: pd.DataFrame) -> dict[str, float]:
    """Compute rolling and exponentially weighted trends from prior fighter rows."""

    if history.empty:
        return {
            "recent_striking_differential_trend": 0.0,
            "recent_takedown_differential_trend": 0.0,
            "recent_control_time_differential_trend": 0.0,
            "recent_damage_absorbed_trend": 0.0,
            "recent_fight_duration_trend": 0.0,
            "recent_opponent_elo_win_probability": 0.5,
        }

    ordered = history.sort_values("fight_date")
    return {
        "recent_striking_differential_trend": exponential_weighted_average(
            ordered.get("striking_differential", pd.Series(dtype=float))
        ),
        "recent_takedown_differential_trend": exponential_weighted_average(
            ordered.get("takedown_differential", pd.Series(dtype=float))
        ),
        "recent_control_time_differential_trend": exponential_weighted_average(
            ordered.get("control_time_differential", pd.Series(dtype=float))
        ),
        "recent_damage_absorbed_trend": rolling_slope(
            ordered.get("sig_str_absorbed", pd.Series(dtype=float)), window=3
        ),
        "recent_fight_duration_trend": exponential_weighted_average(
            ordered.get("fight_duration_seconds", pd.Series(dtype=float))
        ),
        "recent_opponent_elo_win_probability": exponential_weighted_average(
            ordered.get("opponent_elo_expected_win_probability", pd.Series(dtype=float)), default=0.5
        ),
    }
