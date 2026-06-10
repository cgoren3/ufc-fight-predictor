from __future__ import annotations

import pandas as pd

from ufc_predictor.features.build_fight_dataset import build_fight_dataset
from ufc_predictor.features.elo import build_elo_features
from ufc_predictor.features.fighter_history import compute_fighter_snapshot


def sample_fights() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fight_id": 1,
                "fight_date": "2020-01-01",
                "fighter_a": "Alpha",
                "fighter_b": "Bravo",
                "winner": "Alpha",
                "method": "KO/TKO",
                "weight_class": "Lightweight",
                "scheduled_rounds": 3,
                "finish_round": 1,
                "finish_time": "2:00",
            },
            {
                "fight_id": 2,
                "fight_date": "2020-02-01",
                "fighter_a": "Alpha",
                "fighter_b": "Charlie",
                "winner": "Charlie",
                "method": "Split Decision",
                "weight_class": "Lightweight",
                "scheduled_rounds": 3,
                "finish_round": 3,
                "finish_time": "5:00",
            },
            {
                "fight_id": 3,
                "fight_date": "2020-03-01",
                "fighter_a": "Alpha",
                "fighter_b": "Delta",
                "winner": "Alpha",
                "method": "Submission",
                "weight_class": "Welterweight",
                "scheduled_rounds": 3,
                "finish_round": 2,
                "finish_time": "1:30",
            },
        ]
    )


def test_no_future_fights_used_in_prefight_features() -> None:
    fights = sample_fights()
    snapshot = compute_fighter_snapshot(fights, "Alpha", "2020-02-01")

    assert snapshot["total_ufc_fights_before"] == 1
    assert pd.Timestamp(snapshot["max_history_date_used"]) < pd.Timestamp("2020-02-01")
    assert snapshot["wins_before"] == 1
    assert snapshot["losses_before"] == 0


def test_build_dataset_history_dates_are_strictly_before_fight_date() -> None:
    dataset = build_fight_dataset(sample_fights())

    used_dates = pd.to_datetime(dataset["max_history_date_used"], errors="coerce")
    fight_dates = pd.to_datetime(dataset["fight_date"], errors="coerce")
    assert ((used_dates.isna()) | (used_dates < fight_dates)).all()


def test_elo_updates_only_after_fights_occur() -> None:
    elo = build_elo_features(sample_fights())

    first = elo.loc[elo["fight_id"] == 1].iloc[0]
    second = elo.loc[elo["fight_id"] == 2].iloc[0]

    assert first["fighter_a_pre_fight_elo"] == 1500
    assert first["fighter_b_pre_fight_elo"] == 1500
    assert second["fighter_a_pre_fight_elo"] > 1500
    assert second["fighter_b_pre_fight_elo"] == 1500
