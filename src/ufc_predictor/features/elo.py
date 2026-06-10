from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    start_rating: float = 1500.0
    k_factor: float = 32.0
    finish_multiplier: float = 1.25
    split_decision_multiplier: float = 0.75
    decision_multiplier: float = 0.90
    draw_multiplier: float = 0.30


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def _normal(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _method_multiplier(method: Any, config: EloConfig) -> float:
    method_text = _normal(method).lower()
    if any(token in method_text for token in ["ko", "tko", "submission", "sub"]):
        return config.finish_multiplier
    if "split" in method_text or "majority" in method_text:
        return config.split_decision_multiplier
    if "decision" in method_text:
        return config.decision_multiplier
    if "draw" in method_text:
        return config.draw_multiplier
    return 1.0


def _actual_score(fighter: str, opponent: str, winner: Any) -> float | None:
    winner_name = _normal(winner)
    if not winner_name or winner_name.lower() in {"draw", "nc", "no contest"}:
        return None
    if winner_name == fighter:
        return 1.0
    if winner_name == opponent:
        return 0.0
    return None


@dataclass
class EloSystem:
    """Chronological UFC Elo system.

    Ratings are read before a fight and updated only after the result is known.
    Weight-class ratings are tracked independently alongside global ratings.
    """

    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)
    weight_class_ratings: dict[tuple[str, str], float] = field(default_factory=dict)

    def rating(self, fighter: str) -> float:
        return self.ratings.get(fighter, self.config.start_rating)

    def weight_rating(self, fighter: str, weight_class: Any) -> float:
        return self.weight_class_ratings.get((fighter, _normal(weight_class)), self.config.start_rating)

    def snapshot(self, fighter_a: str, fighter_b: str, weight_class: Any = "") -> dict[str, float]:
        a = self.rating(fighter_a)
        b = self.rating(fighter_b)
        aw = self.weight_rating(fighter_a, weight_class)
        bw = self.weight_rating(fighter_b, weight_class)
        return {
            "fighter_a_pre_fight_elo": a,
            "fighter_b_pre_fight_elo": b,
            "diff_pre_fight_elo": a - b,
            "fighter_a_pre_weight_class_elo": aw,
            "fighter_b_pre_weight_class_elo": bw,
            "diff_pre_weight_class_elo": aw - bw,
            "fighter_a_elo_expected_win_probability": expected_score(a, b),
            "fighter_b_elo_expected_win_probability": expected_score(b, a),
        }

    def update_fight(
        self,
        fighter_a: str,
        fighter_b: str,
        winner: Any,
        method: Any = "",
        weight_class: Any = "",
    ) -> None:
        score_a = _actual_score(fighter_a, fighter_b, winner)
        if score_a is None:
            return
        score_b = 1.0 - score_a
        multiplier = _method_multiplier(method, self.config)
        self._update_pair(fighter_a, fighter_b, score_a, score_b, multiplier)
        self._update_weight_pair(fighter_a, fighter_b, weight_class, score_a, score_b, multiplier)

    def _update_pair(self, fighter_a: str, fighter_b: str, score_a: float, score_b: float, multiplier: float) -> None:
        rating_a = self.rating(fighter_a)
        rating_b = self.rating(fighter_b)
        expected_a = expected_score(rating_a, rating_b)
        expected_b = expected_score(rating_b, rating_a)
        k = self.config.k_factor * multiplier
        self.ratings[fighter_a] = rating_a + k * (score_a - expected_a)
        self.ratings[fighter_b] = rating_b + k * (score_b - expected_b)

    def _update_weight_pair(
        self,
        fighter_a: str,
        fighter_b: str,
        weight_class: Any,
        score_a: float,
        score_b: float,
        multiplier: float,
    ) -> None:
        wc = _normal(weight_class)
        key_a = (fighter_a, wc)
        key_b = (fighter_b, wc)
        rating_a = self.weight_class_ratings.get(key_a, self.config.start_rating)
        rating_b = self.weight_class_ratings.get(key_b, self.config.start_rating)
        expected_a = expected_score(rating_a, rating_b)
        expected_b = expected_score(rating_b, rating_a)
        k = self.config.k_factor * multiplier
        self.weight_class_ratings[key_a] = rating_a + k * (score_a - expected_a)
        self.weight_class_ratings[key_b] = rating_b + k * (score_b - expected_b)


def build_elo_features(fights: pd.DataFrame, config: EloConfig | None = None) -> pd.DataFrame:
    """Build pre-fight Elo features in strict chronological order."""

    if fights.empty:
        return pd.DataFrame()
    required = {"fighter_a", "fighter_b", "fight_date"}
    missing = required - set(fights.columns)
    if missing:
        raise ValueError(f"Fights frame is missing required Elo columns: {sorted(missing)}")
    frame = fights.copy()
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce")
    sort_columns = ["fight_date"] + (["fight_id"] if "fight_id" in frame.columns else [])
    frame = frame.sort_values(sort_columns).reset_index(drop=True)
    system = EloSystem(config or EloConfig())
    rows: list[dict[str, Any]] = []
    for order_index, row in frame.iterrows():
        fighter_a = _normal(row["fighter_a"])
        fighter_b = _normal(row["fighter_b"])
        weight_class = row.get("weight_class", "")
        snapshot = system.snapshot(fighter_a, fighter_b, weight_class)
        snapshot.update(
            {
                "fight_id": row.get("fight_id", order_index),
                "fight_date": row["fight_date"],
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "elo_order": order_index,
            }
        )
        rows.append(snapshot)
        system.update_fight(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            winner=row.get("winner", ""),
            method=row.get("method", ""),
            weight_class=weight_class,
        )
    return pd.DataFrame(rows)
