from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from ufc_predictor.cli import app
from ufc_predictor.features.leakage_audit import run_leakage_audit
from ufc_predictor.features.style_features import compute_style_matchup_features
from ufc_predictor.models.evaluate import evaluate_predictions
from ufc_predictor.models.predict import format_prediction_output
from ufc_predictor.models.train import TrainedModelBundle, save_model_bundle
from ufc_predictor.ingest.scorecards_loader import load_scorecards_csv
from ufc_predictor.odds import american_odds_to_implied_probability, attach_odds_features
from ufc_predictor.reporting import build_data_quality_coverage


runner = CliRunner()


def test_style_features_include_opponent_adjusted_and_clash_values() -> None:
    a = {
        "sig_str_landed_per_min": 5,
        "sig_str_absorbed_per_min": 2,
        "takedowns_landed_per_15": 3,
        "takedown_defense": 0.8,
        "submission_attempts_per_15": 1,
        "submission_loss_rate": 0.1,
        "control_time_per_15": 120,
        "control_time_allowed_per_15": 30,
        "finish_rate": 0.6,
        "ko_tko_loss_rate": 0.1,
        "been_finished": 0,
        "reach_in": 72,
        "stance": "Southpaw",
    }
    b = {
        "sig_str_landed_per_min": 3,
        "sig_str_absorbed_per_min": 4,
        "takedowns_landed_per_15": 1,
        "takedown_defense": 0.5,
        "submission_attempts_per_15": 0.2,
        "submission_loss_rate": 0.3,
        "control_time_per_15": 20,
        "control_time_allowed_per_15": 80,
        "finish_rate": 0.2,
        "ko_tko_loss_rate": 0.2,
        "been_finished": 1,
        "reach_in": 70,
        "stance": "Orthodox",
    }

    features = compute_style_matchup_features(a, b)

    assert "diff_opponent_adjusted_striking" in features
    assert "diff_takedown_offense_vs_defense_clash" in features
    assert "diff_strike_volume_vs_absorption_clash" in features
    assert features["diff_opponent_adjusted_takedowns"] > 0


def test_prediction_output_includes_uncertainty_and_explanation_sections() -> None:
    output = format_prediction_output(
        "A",
        "B",
        0.62,
        ["A has an edge"],
        uncertainty_range=[0.56, 0.68],
        top_factors_for_fighter_a=["A factor"],
        top_factors_for_fighter_b=["B factor"],
        biggest_uncertainty_factors=["Small sample"],
        missing_data_warnings=["Missing reach"],
    )

    assert output["uncertainty_range"] == [0.56, 0.68]
    assert output["top_factors_favoring_fighter_a"] == ["A factor"]
    assert output["top_factors_favoring_fighter_b"] == ["B factor"]
    assert output["biggest_uncertainty_factors"] == ["Small sample"]
    assert output["missing_data_warnings"] == ["Missing reach"]


def test_leakage_audit_passes_for_prior_history_counts() -> None:
    raw = pd.DataFrame(
        [
            {"fight_id": 1, "fight_date": "2020-01-01", "fighter_a": "A", "fighter_b": "B", "winner": "A"},
            {"fight_id": 2, "fight_date": "2020-02-01", "fighter_a": "A", "fighter_b": "C", "winner": "C"},
        ]
    )
    dataset = pd.DataFrame(
        [
            {
                "fight_id": 2,
                "fight_date": "2020-02-01",
                "fighter_a": "A",
                "fighter_b": "C",
                "fighter_a_total_ufc_fights_before": 1,
                "fighter_b_total_ufc_fights_before": 0,
                "max_history_date_used": "2020-01-01",
            }
        ]
    )

    result = run_leakage_audit(dataset, raw, sample_size=1)

    assert result.passed
    assert result.passed_rows == 1


def test_leakage_audit_command_writes_report(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    fights_path = tmp_path / "fights.csv"
    output_path = tmp_path / "leakage_audit.json"
    pd.DataFrame(
        [
            {
                "fight_id": 1,
                "fight_date": "2020-01-01",
                "fighter_a": "A",
                "fighter_b": "B",
                "fighter_a_win": 1,
                "fighter_a_total_ufc_fights_before": 0,
                "fighter_b_total_ufc_fights_before": 0,
                "max_history_date_used": "",
            }
        ]
    ).to_csv(dataset_path, index=False)
    pd.DataFrame(
        [{"fight_id": 1, "fight_date": "2020-01-01", "fighter_a": "A", "fighter_b": "B", "winner": "A"}]
    ).to_csv(fights_path, index=False)

    result = runner.invoke(
        app,
        [
            "leakage-audit",
            "--sample-size",
            "1",
            "--dataset-path",
            str(dataset_path),
            "--fights-csv",
            str(fights_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Violations: 0" in result.output


def test_odds_import_and_attach_features(tmp_path) -> None:
    odds_path = tmp_path / "odds.csv"
    output_path = tmp_path / "raw_odds.csv"
    odds_path.write_text(
        "fight_date,fighter_a,fighter_b,sportsbook,fighter_a_odds,fighter_b_odds,timestamp\n"
        "2020-01-01,A,B,Book,-150,130,2019-12-31T12:00:00\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["import-odds", "--import-path", str(odds_path), "--output-path", str(output_path)])

    assert result.exit_code == 0
    assert american_odds_to_implied_probability(-150) == 0.6
    odds = pd.read_csv(output_path)
    dataset = pd.DataFrame([{"fight_date": "2020-01-01", "fighter_a": "A", "fighter_b": "B", "fighter_a_win": 1}])
    attached = attach_odds_features(dataset, odds)
    assert "market_fighter_a_implied_probability" in attached.columns
    assert attached.loc[0, "market_fighter_a_implied_probability"] > 0.5
    assert "source_file" not in attached.columns


def test_extract_odds_from_espn_source_and_import_coverage(tmp_path) -> None:
    source_dir = tmp_path / "enrichment_sources" / "espn_ufc_events"
    source_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Event Id": 10, "Event Name": "UFC Test", "Event Date": "2020-01-01T00:00Z"},
        ]
    ).to_csv(source_dir / "ufc_events.csv", index=False)
    pd.DataFrame(
        [
            {
                "Event Id": 10,
                "Fight Date": "2020-01-01T00:00Z",
                "Red Fighter": "Alice Alpha",
                "Blue Fighter": "Beth Beta",
                "Red Fighter Moneyline Odds": -150,
                "Blue Fighter Moneyline Odds": 130,
                "Winner": "Alice Alpha",
                "Division": "Lightweight",
                "Is Main Event": True,
                "Is Title Fight": False,
                "Score Cards": "29-28, 29-28, 30-27",
                "Card Type": "Main Card",
            },
            {
                "Event Id": 10,
                "Fight Date": "2020-01-01T00:00Z",
                "Red Fighter": "Cara Delta",
                "Blue Fighter": "Dana Echo",
                "Red Fighter Moneyline Odds": "",
                "Blue Fighter Moneyline Odds": "",
            },
        ]
    ).to_csv(source_dir / "ufc_fights.csv", index=False)
    imports = tmp_path / "raw" / "imports"
    imports.mkdir(parents=True)
    output_path = imports / "odds.csv"

    result = runner.invoke(app, ["extract-odds", "--source-dir", str(tmp_path / "enrichment_sources"), "--output-path", str(output_path)])

    assert result.exit_code == 0
    odds = pd.read_csv(output_path)
    assert len(odds) == 1
    assert odds.loc[0, "event"] == "UFC Test"
    assert odds.loc[0, "fighter_a_odds"] == -150
    assert odds.loc[0, "sportsbook"] == "espn_source"
    assert "recommend" not in result.output.lower()

    raw_dir = tmp_path / "raw"
    pd.DataFrame(
        [
            {"fight_date": "2020-01-01", "fighter_a": "Alice Alpha", "fighter_b": "Beth Beta"},
            {"fight_date": "2020-01-01", "fighter_a": "Cara Delta", "fighter_b": "Dana Echo"},
        ]
    ).to_csv(raw_dir / "fights.csv", index=False)
    imported = raw_dir / "odds.csv"
    result = runner.invoke(app, ["import-odds", "--import-path", str(output_path), "--output-path", str(imported)])

    assert result.exit_code == 0
    assert "Odds coverage: 1/2 (50.0%)" in result.output
    assert "recommend" not in result.output.lower()
    assert american_odds_to_implied_probability(130) == 100 / 230


def test_extract_scorecards_from_espn_source_and_load_messy_raw_strings(tmp_path) -> None:
    source_dir = tmp_path / "enrichment_sources" / "espn_ufc_events"
    source_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"Event Id": 10, "Event Name": "UFC Test", "Event Date": "2020-01-01T00:00Z"}]
    ).to_csv(source_dir / "ufc_events.csv", index=False)
    pd.DataFrame(
        [
            {
                "Event Id": 10,
                "Fight Date": "2020-01-01T00:00Z",
                "Red Fighter": "Alice Alpha",
                "Blue Fighter": "Beth Beta",
                "Winner": "Alice Alpha",
                "Score Cards": "Judge A 29-28 | Judge B 30-27 | messy text",
                "Card Type": "Main Card",
            }
        ]
    ).to_csv(source_dir / "ufc_fights.csv", index=False)
    output_path = tmp_path / "raw" / "imports" / "scorecards.csv"

    result = runner.invoke(
        app,
        ["extract-scorecards", "--source-dir", str(tmp_path / "enrichment_sources"), "--output-path", str(output_path)],
    )

    assert result.exit_code == 0
    extracted = pd.read_csv(output_path)
    assert len(extracted) == 1
    assert extracted.loc[0, "raw_scorecards"].startswith("Judge A")
    loaded = load_scorecards_csv(output_path)
    assert len(loaded) == 1
    assert "raw_scorecards" in loaded.columns
    assert loaded.loc[0, "judge"] == "raw_scorecards"


def test_report_coverage_includes_odds_and_scorecard_sources(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    imports = raw_dir / "imports"
    imports.mkdir(parents=True)
    pd.DataFrame(
        [{"fight_date": "2020-01-01", "fighter_a": "Alice Alpha", "fighter_b": "Beth Beta", "weight_class": "Lightweight"}]
    ).to_csv(raw_dir / "fights.csv", index=False)
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "sportsbook": "espn_source",
                "fighter_a_odds": -150,
                "fighter_b_odds": 130,
                "timestamp": "",
                "fighter_a_no_vig_probability": 0.58,
                "fighter_b_no_vig_probability": 0.42,
            }
        ]
    ).to_csv(imports / "odds.csv", index=False)
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Test",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "raw_scorecards": "29-28",
            }
        ]
    ).to_csv(imports / "scorecards.csv", index=False)

    coverage = build_data_quality_coverage(raw_dir)

    assert coverage["odds_matched_fights"] == 1
    assert coverage["scorecard_matched_fights"] == 1
    assert coverage["odds_source"].endswith("imports\\odds.csv") or coverage["odds_source"].endswith("imports/odds.csv")
    assert coverage["scorecards_source"].endswith("imports\\scorecards.csv") or coverage["scorecards_source"].endswith("imports/scorecards.csv")


def test_model_vs_market_uses_analysis_labels_without_recommendations() -> None:
    metrics = evaluate_predictions(
        [1, 0],
        [0.62, 0.45],
        metadata=pd.DataFrame(
            {
                "market_fighter_a_implied_probability": [0.55, 0.5],
                "closing_odds_favorite_is_a": pd.Series([True, pd.NA], dtype="boolean"),
            }
        ),
    )

    market = metrics["model_vs_market"]
    assert "mean_model_probability" in market
    assert "mean_market_implied_probability" in market
    assert "mean_difference_vs_market" in market
    text = " ".join(str(value) for value in market.values()).lower()
    assert "recommend" not in text


def test_model_card_is_saved(tmp_path) -> None:
    bundle = TrainedModelBundle(
        model_version="test",
        estimators=[],
        feature_columns=["x"],
        numeric_features=["x"],
        categorical_features=[],
        metrics={"accuracy": 0.5, "performance_by_confidence_tier": {"High": {"accuracy": 0.75}}},
        model_card={"training_date": "today", "accuracy": 0.5},
    )

    save_model_bundle(bundle, model_dir=tmp_path)

    assert (tmp_path / "model_card.json").exists()
    assert json.loads((tmp_path / "model_card.json").read_text(encoding="utf-8"))["accuracy"] == 0.5


def test_report_command_prints_summary(monkeypatch, tmp_path) -> None:
    reporting = importlib.import_module("ufc_predictor.reporting")

    def fake_build():
        return {
            "dataset_size": {"fights": 10, "date_range": {"start": "2020-01-01", "end": "2020-02-01"}},
            "train_metrics": {"accuracy": 0.6},
            "backtest_metrics": {"accuracy": 0.61, "expected_calibration_error": 0.03},
            "known_missing_data_issues": ["No scorecards"],
        }

    monkeypatch.setattr(reporting, "build_performance_report", fake_build)

    result = runner.invoke(app, ["report", "--output", str(tmp_path / "report.json")])

    assert result.exit_code == 0
    assert "Performance report" in result.output
    assert "Dataset fights: 10" in result.output
