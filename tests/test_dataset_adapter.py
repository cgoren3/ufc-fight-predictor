from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ufc_predictor.cli import app
from ufc_predictor.dataset_adapter import DatasetAdapterError, adapt_dataset
from ufc_predictor.import_validation import validate_import_directory


runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"
KAGGLE_FIXTURE = FIXTURES / "kaggle_ufc_dataset"
LONG_FORMAT_FIXTURE = FIXTURES / "kaggle_long_format"
IMPORT_FIXTURE = FIXTURES / "imports"


def test_dataset_columns_command_prints_source_columns() -> None:
    result = runner.invoke(app, ["dataset-columns", "--source", str(KAGGLE_FIXTURE)])

    assert result.exit_code == 0
    assert "kaggle_fights.csv" in result.output
    assert "R_fighter" in result.output
    assert "B_fighter" in result.output
    assert "R_SIG_STR." in result.output


def test_adapt_dataset_command_writes_import_schema(tmp_path) -> None:
    output_dir = tmp_path / "imports"

    result = runner.invoke(app, ["adapt-dataset", "--source", str(KAGGLE_FIXTURE), "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert "Adapted external UFC CSV dataset" in result.output
    fights = pd.read_csv(output_dir / "fights.csv")
    fighters = pd.read_csv(output_dir / "fighters.csv")
    fight_stats = pd.read_csv(output_dir / "fight_stats.csv")
    assert list(fights["fighter_a"]) == ["Red One", "Red Two"]
    assert list(fights["winner"]) == ["Red One", "Blue Two"]
    assert len(fighters) == 4
    assert len(fight_stats) == 4
    assert fight_stats.loc[fight_stats["fighter"] == "Red One", "sig_str_landed"].iloc[0] == 12
    assert validate_import_directory(output_dir).ok


def test_adapt_dataset_supports_long_format_fighter_performance_csv(tmp_path) -> None:
    output_dir = tmp_path / "imports"

    result = runner.invoke(app, ["adapt-dataset", "--source", str(LONG_FORMAT_FIXTURE), "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert "long-format fighter-performance CSV" in result.output
    fights = pd.read_csv(output_dir / "fights.csv")
    fighters = pd.read_csv(output_dir / "fighters.csv")
    fight_stats = pd.read_csv(output_dir / "fight_stats.csv")

    assert len(fights) == 2
    assert len(fight_stats) == 4
    assert len(fighters) == 4
    assert fights.loc[0, "fighter_a"] == "Alice Alpha"
    assert fights.loc[0, "fighter_b"] == "Beth Beta"
    assert fights.loc[0, "winner"] == "Alice Alpha"
    assert fights.loc[0, "weight_class"] == "Unknown"
    assert fights.loc[0, "scheduled_rounds"] == 3
    assert fights.loc[0, "main_event"] == 0

    alice_stats = fight_stats.loc[fight_stats["fighter"] == "Alice Alpha"].iloc[0]
    assert alice_stats["knockdowns"] == 1
    assert alice_stats["sig_str_landed"] == 12
    assert alice_stats["sig_str_attempted"] == 30
    assert alice_stats["takedowns_landed"] == 2
    assert alice_stats["submission_attempts"] == 1
    assert alice_stats["raw_str"] == "12 of 30"

    cara_stats = fight_stats.loc[fight_stats["fighter"] == "Cara Delta"].iloc[0]
    assert pd.isna(cara_stats["knockdowns"])
    assert cara_stats["raw_kd"] == "not available"
    assert validate_import_directory(output_dir).ok


def test_adapt_dataset_copies_already_normalized_schema(tmp_path) -> None:
    output_dir = tmp_path / "imports"

    result = adapt_dataset(IMPORT_FIXTURE, output_dir)

    assert result.copied_existing_schema is True
    assert set(result.files) >= {"fights", "fighters", "fight_stats"}
    assert len(pd.read_csv(output_dir / "fights.csv")) == 8


def test_adapt_dataset_refuses_ambiguous_missing_fight_mapping(tmp_path) -> None:
    source = tmp_path / "download"
    source.mkdir()
    (source / "bad.csv").write_text(
        "fighter,opponent,date,result\n"
        "A,B,2020-01-01,A\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetAdapterError, match="Could not find a fight-level CSV"):
        adapt_dataset(source, tmp_path / "imports")


def test_adapt_dataset_failure_warns_about_existing_import_files(tmp_path) -> None:
    source = tmp_path / "download"
    output_dir = tmp_path / "imports"
    source.mkdir()
    output_dir.mkdir()
    (source / "bad.csv").write_text("fighter,opponent,date,result\nA,B,2020-01-01,A\n", encoding="utf-8")
    (output_dir / "fights.csv").write_text("fighter_a,fighter_b,fight_date,winner\nOld A,Old B,2020-01-01,Old A\n", encoding="utf-8")

    result = runner.invoke(app, ["adapt-dataset", "--source", str(source), "--output-dir", str(output_dir)])

    assert result.exit_code == 1
    assert "Existing import files still exist" in result.output
    assert "fights.csv" in result.output
