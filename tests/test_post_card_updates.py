from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ufc_predictor.card_update import apply_card_update, prepare_upcoming_card, update_after_card, validate_card_update
from ufc_predictor.cli import _ablation_groups, app
from ufc_predictor.features.build_fight_dataset import build_fight_dataset
from ufc_predictor.fighter_profiles import enrich_fighter_profiles


runner = CliRunner()


def _write_imports(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "fight_id": 1,
                "event_name": "Existing Event",
                "fight_date": "2024-01-01",
                "event_location": "Las Vegas, NV",
                "fighter_a": "Old A",
                "fighter_b": "Old B",
                "winner": "Old A",
                "weight_class": "Lightweight",
                "method": "Decision - Unanimous",
                "finish_round": 3,
                "finish_time": "5:00",
                "scheduled_rounds": 3,
                "main_event": 0,
                "source_url": "",
            }
        ]
    ).to_csv(path / "fights.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "Old A",
                "stance": "Orthodox",
                "height_in": 70,
                "weight_lb": 155,
                "reach_in": 72,
                "date_of_birth": "1990-01-01",
                "record": "",
                "source_url": "",
            },
            {
                "name": "Old B",
                "stance": "",
                "height_in": "",
                "weight_lb": 155,
                "reach_in": "",
                "date_of_birth": "",
                "record": "",
                "source_url": "",
            },
        ]
    ).to_csv(path / "fighters.csv", index=False)
    pd.DataFrame(columns=["fight_id", "fighter", "opponent", "knockdowns"]).to_csv(path / "fight_stats.csv", index=False)
    pd.DataFrame(columns=["event", "fight_date", "fighter_a", "fighter_b", "weight_class", "event_location", "main_event", "title_fight", "scheduled_rounds"]).to_csv(
        path / "fight_enrichment.csv",
        index=False,
    )


def _card_csv(path: Path, date: str = "2024-02-01") -> Path:
    frame = pd.DataFrame(
        [
            {
                "Fight Date": date,
                "Event": "UFC Test Card",
                "Location": "Boston, MA",
                "Red Fighter": "New A",
                "Blue Fighter": "New B",
                "Winner": "New A",
                "Division": "Welterweight",
                "Method": "Decision - Unanimous",
                "Round": 3,
                "Time": "5:00",
                "Is Main Event": 1,
                "Is Title Fight": 0,
                "Score Cards": "29-28, 29-28, 30-27",
                "Card Type": "Decision",
                "Red Fighter Moneyline Odds": -150,
                "Blue Fighter Moneyline Odds": 130,
            }
        ]
    )
    output = path / "card.csv"
    frame.to_csv(output, index=False)
    return output


def test_update_after_card_writes_staging_files_only(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    card = _card_csv(tmp_path)
    staging = tmp_path / "staging"

    report = update_after_card(str(card), source="espn", staging_dir=staging)

    assert (staging / "new_fights.csv").exists()
    assert (staging / "new_fight_stats.csv").exists()
    assert (staging / "new_scorecards.csv").exists()
    assert report.rows_written["new_fights.csv"] == 1
    existing = pd.read_csv(imports / "fights.csv")
    assert len(existing) == 1


def test_validate_card_update_catches_duplicates(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    staging = tmp_path / "staging"
    update_after_card(str(_card_csv(tmp_path, "2024-01-01")), source="espn", staging_dir=staging)
    fights = pd.read_csv(staging / "new_fights.csv")
    fights.loc[0, "fighter_a"] = "Old A"
    fights.loc[0, "fighter_b"] = "Old B"
    fights.to_csv(staging / "new_fights.csv", index=False)

    report = validate_card_update(staging_dir=staging, imports_dir=imports)

    assert not report.ok
    assert any("already present" in error for error in report.errors)


def test_validate_card_update_catches_future_fight_dates(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    staging = tmp_path / "staging"
    update_after_card(str(_card_csv(tmp_path, "2999-01-01")), source="espn", staging_dir=staging)

    report = validate_card_update(staging_dir=staging, imports_dir=imports)

    assert not report.ok
    assert any("future fight dates" in error for error in report.errors)


def test_apply_card_update_creates_backup_and_does_not_duplicate(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    raw = tmp_path / "raw"
    _write_imports(imports)
    staging = tmp_path / "staging"
    update_after_card(str(_card_csv(tmp_path)), source="espn", staging_dir=staging)
    validation = validate_card_update(staging_dir=staging, imports_dir=imports)
    assert validation.ok

    first = apply_card_update(staging_dir=staging, imports_dir=imports, raw_dir=raw, backup_root=tmp_path / "backups")
    assert first.rows_added["fights"] == 1
    assert first.backup_dir.exists()
    fights = pd.read_csv(imports / "fights.csv")
    assert len(fights) == 2

    duplicate_report = validate_card_update(staging_dir=staging, imports_dir=imports)
    assert not duplicate_report.ok
    assert any("already present" in error for error in duplicate_report.errors)


def test_fighter_profile_enrichment_improves_missing_reach_without_overwriting(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    sources = tmp_path / "sources"
    _write_imports(imports)
    sources.mkdir()
    pd.DataFrame(
        [
            {"name": "Old A", "reach": 99, "height": 75, "stance": "Southpaw"},
            {"name": "Old B", "reach": 71, "height": 69, "stance": "Switch"},
        ]
    ).to_csv(sources / "fighters.csv", index=False)

    report = enrich_fighter_profiles(
        source="local",
        fighters_path=imports / "fighters.csv",
        source_dir=sources,
        output_path=imports / "fighter_profile_enrichment.csv",
        report_path=tmp_path / "profile_report.json",
        apply=True,
    )

    fighters = pd.read_csv(imports / "fighters.csv")
    assert report.fields_updated["reach_in"] == 1
    assert fighters.loc[fighters["name"] == "Old A", "reach_in"].iloc[0] == 72
    assert fighters.loc[fighters["name"] == "Old B", "reach_in"].iloc[0] == 71


def test_reach_feature_creation_and_missing_indicators() -> None:
    fights = pd.DataFrame(
        [
            {
                "fight_id": 1,
                "event_name": "Test",
                "fight_date": "2024-01-01",
                "fighter_a": "Reach A",
                "fighter_b": "Reach B",
                "winner": "Reach A",
                "weight_class": "Lightweight",
                "scheduled_rounds": 3,
            }
        ]
    )
    fighters = pd.DataFrame(
        [
            {"name": "Reach A", "height_in": 70, "reach_in": 72, "stance": "Orthodox"},
            {"name": "Reach B", "height_in": 69, "reach_in": "", "stance": "Southpaw"},
        ]
    )

    dataset = build_fight_dataset(fights, fighters=fighters)

    row = dataset.iloc[0]
    assert row["fighter_a_reach"] == 72
    assert row["fighter_b_reach_missing"] == 1
    assert row["both_reach_missing"] == 0
    assert "height_reach_interaction" in dataset.columns


def test_physical_attributes_ablation_group_exists() -> None:
    groups = _ablation_groups()

    assert "physical_attributes" in groups
    assert any("reach" in pattern for pattern in groups["physical_attributes"])


def test_rebuild_after_update_stops_on_failed_validation(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate-card-update", "--staging-dir", str(tmp_path / "missing"), "--imports-dir", str(tmp_path / "imports")])

    assert result.exit_code == 1
    assert "No staged fights found" in result.output


def test_prepare_upcoming_card_creates_expected_template(tmp_path: Path) -> None:
    card = _card_csv(tmp_path)
    output = tmp_path / "upcoming.csv"

    frame, path = prepare_upcoming_card(str(card), output_path=output)

    assert path == output
    assert len(frame) == 1
    assert list(frame.columns) == [
        "fighter_a",
        "fighter_b",
        "date",
        "weight_class",
        "scheduled_rounds",
        "fighter_a_odds",
        "fighter_b_odds",
        "main_event",
        "title_fight",
    ]


def test_predict_card_works_with_manual_odds(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    raw = tmp_path / "raw"
    _write_imports(imports)
    card = pd.DataFrame(
        [
            {
                "fighter_a": "Old A",
                "fighter_b": "Old B",
                "date": "2025-01-01",
                "weight_class": "Lightweight",
                "scheduled_rounds": 3,
                "fighter_a_odds": -150,
                "fighter_b_odds": 130,
                "main_event": 0,
                "title_fight": 0,
            }
        ]
    )
    card_path = tmp_path / "card_predictions.csv"
    card.to_csv(card_path, index=False)
    for name in ["fights.csv", "fighters.csv", "fight_stats.csv"]:
        (raw / name).parent.mkdir(parents=True, exist_ok=True)
        (raw / name).write_text((imports / name).read_text(encoding="utf-8"), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "predict-card",
            "--file",
            str(card_path),
            "--model-mode",
            "market-aware",
            "--show-value-analysis",
            "--fights-csv",
            str(raw / "fights.csv"),
            "--fighters-csv",
            str(raw / "fighters.csv"),
            "--fight-stats-csv",
            str(raw / "fight_stats.csv"),
            "--scorecards-csv",
            str(raw / "missing_scorecards.csv"),
            "--odds-csv",
            str(raw / "missing_odds.csv"),
            "--model-path",
            str(tmp_path / "missing_model.pkl"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["predictions"][0]["odds_source"] == "manual_input"
    assert "value_analysis_note" in payload["predictions"][0]


def test_no_post_fight_leakage_into_event_enrichment(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    staging = tmp_path / "staging"
    update_after_card(str(_card_csv(tmp_path)), source="espn", staging_dir=staging)
    enrichment = pd.read_csv(staging / "new_event_enrichment.csv")
    enrichment["winner"] = "New A"
    enrichment.to_csv(staging / "new_event_enrichment.csv", index=False)

    report = validate_card_update(staging_dir=staging, imports_dir=imports)

    assert not report.ok
    assert any("post-fight result columns" in error for error in report.errors)
