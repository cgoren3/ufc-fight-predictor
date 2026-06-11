from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ufc_predictor.cli import app
from ufc_predictor.features.build_fight_dataset import BuildDatasetError, build_fight_dataset


runner = CliRunner()
FIXTURE_IMPORTS = Path(__file__).parent / "fixtures" / "imports"


def test_build_dataset_limit_option(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    result = runner.invoke(
        app,
        [
            "build-dataset",
            "--imports-dir",
            str(FIXTURE_IMPORTS),
            "--fights-csv",
            str(raw_dir / "fights.csv"),
            "--fight-stats-csv",
            str(raw_dir / "fight_stats.csv"),
            "--fighters-csv",
            str(raw_dir / "fighters.csv"),
            "--scorecards-csv",
            str(raw_dir / "scorecards.csv"),
            "--output",
            str(processed_dir / "fight_dataset.parquet"),
            "--limit",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "Total fights read: 8" in result.output
    assert "Total fights processed: 3" in result.output
    assert "Total training rows written: 3" in result.output


def test_build_dataset_verbose_option_prints_input_rows(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    result = runner.invoke(
        app,
        [
            "build-dataset",
            "--imports-dir",
            str(FIXTURE_IMPORTS),
            "--fights-csv",
            str(raw_dir / "fights.csv"),
            "--fight-stats-csv",
            str(raw_dir / "fight_stats.csv"),
            "--fighters-csv",
            str(raw_dir / "fighters.csv"),
            "--scorecards-csv",
            str(raw_dir / "scorecards.csv"),
            "--output",
            str(processed_dir / "fight_dataset.parquet"),
            "--limit",
            "3",
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert "Input rows: fights=8" in result.output
    assert "Limit: processing first 3 chronological fights." in result.output


def test_largeish_build_reports_progress() -> None:
    fights = pd.DataFrame(
        {
            "fight_id": range(1, 601),
            "fight_date": pd.date_range("2020-01-01", periods=600, freq="D").astype(str),
            "fighter_a": [f"Fighter {index % 40}" for index in range(600)],
            "fighter_b": [f"Opponent {index % 40}" for index in range(600)],
            "winner": [f"Fighter {index % 40}" for index in range(600)],
            "weight_class": ["Lightweight"] * 600,
            "scheduled_rounds": [3] * 600,
        }
    )
    progress = []

    dataset = build_fight_dataset(
        fights=fights,
        progress_callback=lambda item: progress.append(item.processed_fights),
        progress_interval=250,
    )

    assert len(dataset) == 600
    assert progress == [250, 500, 600]
    report = dataset.attrs["build_report"]
    assert report.total_fights_read == 600
    assert report.training_rows_written == 600


def test_build_dataset_failure_output_is_clear(monkeypatch, tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    def fail_build(*args, **kwargs):
        raise BuildDatasetError(
            "Feature generation failed while processing a fight.",
            {"fight_id": 123, "fighter_a": "A", "fighter_b": "B"},
        ) from ValueError("boom")

    build_module = importlib.import_module("ufc_predictor.features.build_fight_dataset")
    monkeypatch.setattr(build_module, "build_fight_dataset", fail_build)

    result = runner.invoke(
        app,
        [
            "build-dataset",
            "--imports-dir",
            str(FIXTURE_IMPORTS),
            "--fights-csv",
            str(raw_dir / "fights.csv"),
            "--fight-stats-csv",
            str(raw_dir / "fight_stats.csv"),
            "--fighters-csv",
            str(raw_dir / "fighters.csv"),
            "--scorecards-csv",
            str(raw_dir / "scorecards.csv"),
            "--output",
            str(processed_dir / "fight_dataset.parquet"),
        ],
    )

    assert result.exit_code == 1
    assert "Dataset build failed." in result.output
    assert "Exception type: ValueError" in result.output
    assert "Message: boom" in result.output
    assert "Current fight:" in result.output
