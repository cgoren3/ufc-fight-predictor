from __future__ import annotations

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from ufc_predictor.cli import app
from ufc_predictor.enrichment import import_enrichment_csv
from ufc_predictor.import_validation import validate_import_directory
from ufc_predictor.models.evaluate import rolling_backtest
from ufc_predictor.reporting import build_data_quality_coverage


runner = CliRunner()


def _write_minimal_imports(path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "fight_id": 1,
                "event_name": "UFC Test 1",
                "fight_date": "2020-01-01",
                "event_location": "",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "winner": "Alice Alpha",
                "method": "Decision",
                "finish_round": 3,
                "finish_time": "5:00",
                "weight_class": "Unknown",
                "scheduled_rounds": 3,
                "main_event": 0,
                "title_fight": 0,
            }
        ]
    ).to_csv(path / "fights.csv", index=False)
    pd.DataFrame([{"name": "Alice Alpha"}, {"name": "Beth Beta"}]).to_csv(path / "fighters.csv", index=False)
    pd.DataFrame(
        [
            {"fight_id": 1, "fighter": "Alice Alpha", "opponent": "Beth Beta"},
            {"fight_id": 1, "fighter": "Beth Beta", "opponent": "Alice Alpha"},
        ]
    ).to_csv(path / "fight_stats.csv", index=False)


def test_import_enrichment_merges_fight_context_fields(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    _write_minimal_imports(import_dir)
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Test 1",
                "fighter_a": "Beth Beta",
                "fighter_b": "Alice Alpha",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas, Nevada, USA",
                "main_event": 1,
                "title_fight": 1,
                "scheduled_rounds": 5,
            }
        ]
    ).to_csv(import_dir / "fight_enrichment.csv", index=False)

    report = import_enrichment_csv(
        enrichment_path=import_dir / "fight_enrichment.csv",
        fights_path=import_dir / "fights.csv",
    )

    fights = pd.read_csv(import_dir / "fights.csv")
    assert report.matched_rows == 1
    assert fights.loc[0, "weight_class"] == "Lightweight"
    assert fights.loc[0, "event_location"] == "Las Vegas, Nevada, USA"
    assert fights.loc[0, "main_event"] == 1
    assert fights.loc[0, "title_fight"] == 1
    assert fights.loc[0, "scheduled_rounds"] == 5
    assert validate_import_directory(import_dir).ok


def test_import_enrichment_command(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    _write_minimal_imports(import_dir)
    (import_dir / "fight_enrichment.csv").write_text(
        "fight_date,event,fighter_a,fighter_b,weight_class,event_location,main_event,title_fight,scheduled_rounds\n"
        "2020-01-01,UFC Test 1,Alice Alpha,Beth Beta,Welterweight,New York,0,0,3\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "import-enrichment",
            "--enrichment-path",
            str(import_dir / "fight_enrichment.csv"),
            "--fights-path",
            str(import_dir / "fights.csv"),
        ],
    )

    assert result.exit_code == 0
    assert "Matched enrichment rows: 1" in result.output
    assert "weight_class: 1" in result.output


def test_validate_imports_detects_unapplied_enrichment(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    _write_minimal_imports(import_dir)
    (import_dir / "fight_enrichment.csv").write_text(
        "fight_date,event,fighter_a,fighter_b,weight_class,event_location,main_event,title_fight,scheduled_rounds\n"
        "2020-01-01,UFC Test 1,Alice Alpha,Beth Beta,Lightweight,Las Vegas,0,0,3\n",
        encoding="utf-8",
    )

    result = validate_import_directory(import_dir)

    assert not result.ok
    assert any("Run `ufc-predict import-enrichment`" in error for error in result.errors)


def test_validate_imports_rejects_numeric_event_location(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    _write_minimal_imports(import_dir)
    fights = pd.read_csv(import_dir / "fights.csv")
    fights["event_location"] = 12345
    fights.to_csv(import_dir / "fights.csv", index=False)

    result = validate_import_directory(import_dir)

    assert not result.ok
    assert any("event_location must be categorical" in error for error in result.errors)


def test_report_data_quality_coverage_counts_enriched_fields_odds_and_scorecards(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    fights = pd.DataFrame(
        [
            {
                "fight_id": 1,
                "event_name": "UFC Test 1",
                "fight_date": "2020-01-01",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas",
                "main_event": 1,
            },
            {
                "fight_id": 2,
                "event_name": "UFC Test 1",
                "fight_date": "2020-01-01",
                "fighter_a": "Cara Delta",
                "fighter_b": "Dana Echo",
                "weight_class": "Unknown",
                "event_location": "",
                "main_event": 0,
            },
        ]
    )
    fights.to_csv(raw_dir / "fights.csv", index=False)
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "sportsbook": "Book",
                "fighter_a_odds": -150,
                "fighter_b_odds": 130,
                "timestamp": "2019-12-31T12:00:00",
                "fighter_a_no_vig_probability": 0.58,
                "fighter_b_no_vig_probability": 0.42,
            }
        ]
    ).to_csv(raw_dir / "odds.csv", index=False)
    pd.DataFrame(
        [
            {
                "event": "UFC Test 1",
                "fight_date": "2020-01-01",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "judge": "Judge One",
            }
        ]
    ).to_csv(raw_dir / "scorecards.csv", index=False)

    coverage = build_data_quality_coverage(raw_dir)

    assert coverage["known_weight_class_pct"] == 50.0
    assert coverage["known_event_location_pct"] == 50.0
    assert coverage["known_main_event_pct"] == 100.0
    assert coverage["odds_coverage_pct"] == 50.0
    assert coverage["scorecard_coverage_pct"] == 50.0


def test_rolling_backtest_keeps_enriched_metadata_for_breakdowns(monkeypatch) -> None:
    class DummyBundle:
        feature_columns = ["feature_x"]

        def predict_proba(self, frame):
            probabilities = np.where(frame["feature_x"].to_numpy() >= 0, 0.7, 0.3)
            return np.column_stack([1.0 - probabilities, probabilities])

    import ufc_predictor.models.train as train_module

    monkeypatch.setattr(train_module, "train_ensemble", lambda *args, **kwargs: DummyBundle())
    dataset = pd.DataFrame(
        [
            {"fight_id": 1, "fight_date": "2019-01-01", "fighter_a_win": 1, "feature_x": 1, "weight_class": "Lightweight", "main_event": 0},
            {"fight_id": 2, "fight_date": "2019-02-01", "fighter_a_win": 0, "feature_x": -1, "weight_class": "Welterweight", "main_event": 0},
            {"fight_id": 3, "fight_date": "2020-01-15", "fighter_a_win": 1, "feature_x": 1, "weight_class": "Lightweight", "main_event": 1},
            {"fight_id": 4, "fight_date": "2020-02-15", "fighter_a_win": 0, "feature_x": -1, "weight_class": "Welterweight", "main_event": 0},
            {"fight_id": 5, "fight_date": "2021-01-01", "fighter_a_win": 1, "feature_x": 1, "weight_class": "Lightweight", "main_event": 1},
        ]
    )

    metrics = rolling_backtest(dataset, min_train_fights=2, step="YS")

    assert metrics["performance_by_weight_class"]["Lightweight"]["count"] == 1.0
    assert metrics["performance_by_main_event"]["1"]["count"] == 1.0
