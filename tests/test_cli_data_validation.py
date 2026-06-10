from __future__ import annotations

import pytest

from ufc_predictor.data_io import InputDataError, read_required_csv
from ufc_predictor.features.build_fight_dataset import build_fight_dataset
from ufc_predictor.models.train import _split_features
from ufc_predictor.sample_data import load_sample_data


REQUIRED_FIGHT_COLUMNS = ["fighter_a", "fighter_b", "fight_date", "winner"]


def test_empty_fights_csv_has_clear_error(tmp_path) -> None:
    path = tmp_path / "fights.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(InputDataError, match="is empty"):
        read_required_csv(path, required_columns=REQUIRED_FIGHT_COLUMNS, label="fights CSV")


def test_missing_fights_csv_has_clear_error(tmp_path) -> None:
    path = tmp_path / "fights.csv"

    with pytest.raises(InputDataError, match="Missing fights CSV"):
        read_required_csv(path, required_columns=REQUIRED_FIGHT_COLUMNS, label="fights CSV")


def test_headers_only_fights_csv_has_clear_error(tmp_path) -> None:
    path = tmp_path / "fights.csv"
    path.write_text("fighter_a,fighter_b,fight_date,winner\n", encoding="utf-8")

    with pytest.raises(InputDataError, match="headers but no data rows"):
        read_required_csv(path, required_columns=REQUIRED_FIGHT_COLUMNS, label="fights CSV")


def test_valid_sample_dataset_builds() -> None:
    fights, fighters, fight_stats, scorecards = load_sample_data()

    dataset = build_fight_dataset(fights=fights, fighters=fighters, fight_stats=fight_stats, scorecards=scorecards)

    assert len(dataset) == len(fights)
    assert {"fighter_a", "fighter_b", "fighter_a_win", "diff_pre_fight_elo"} <= set(dataset.columns)
    assert set(dataset["fighter_a_win"]) == {0, 1}


def test_sample_dataset_categorical_columns_survive_csv_roundtrip(tmp_path) -> None:
    fights, fighters, fight_stats, scorecards = load_sample_data()
    dataset = build_fight_dataset(fights=fights, fighters=fighters, fight_stats=fight_stats, scorecards=scorecards)
    path = tmp_path / "dataset.csv"
    dataset.to_csv(path, index=False)
    roundtripped = __import__("pandas").read_csv(path)

    _, numeric, categorical = _split_features(roundtripped)

    assert "weight_class" in categorical
    assert "weight_class" not in numeric
