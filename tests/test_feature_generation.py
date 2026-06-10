from __future__ import annotations

import pandas as pd

from ufc_predictor.features.build_fight_dataset import build_fight_dataset


def _fights() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fight_id": 1,
                "fight_date": "2021-01-01",
                "fighter_a": "One",
                "fighter_b": "Two",
                "winner": "One",
                "method": "Decision",
                "weight_class": "Featherweight",
                "scheduled_rounds": 3,
                "finish_round": 3,
                "finish_time": "5:00",
            },
            {
                "fight_id": 2,
                "fight_date": "2021-05-01",
                "fighter_a": "Three",
                "fighter_b": "One",
                "winner": "Three",
                "method": "KO/TKO",
                "weight_class": "Featherweight",
                "scheduled_rounds": 3,
                "finish_round": 2,
                "finish_time": "1:00",
            },
        ]
    )


def test_build_fight_dataset_two_way_order_inverts_target_safely() -> None:
    dataset = build_fight_dataset(_fights(), two_way=True)
    first_fight = dataset[dataset["fight_id"] == 1].sort_values("fighter_a")

    assert len(first_fight) == 2
    targets = dict(zip(first_fight["fighter_a"], first_fight["fighter_a_win"]))
    assert targets["One"] == 1
    assert targets["Two"] == 0
    assert set(first_fight["diff_pre_fight_elo"]) == {0}


def test_build_fight_dataset_randomized_order_keeps_valid_targets() -> None:
    dataset = build_fight_dataset(_fights(), randomize_order=True, random_state=7)

    assert len(dataset) == 2
    for _, row in dataset.iterrows():
        assert row["fighter_a_win"] == int(row["winner"] == row["fighter_a"])
        assert row["fighter_a"] != row["fighter_b"]
