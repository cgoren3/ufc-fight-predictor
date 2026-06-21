from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from ufc_predictor.cli import app
from ufc_predictor.features.build_fight_dataset import build_fight_dataset
from ufc_predictor.models.evaluate import rolling_backtest, tune_market_blend_weight
from ufc_predictor.models.predict import format_prediction_output, value_label_for_edge
from ufc_predictor.models.train import apply_recency_weighting
from ufc_predictor.odds import american_odds_to_implied_probability, attach_odds_features


runner = CliRunner()


def test_odds_matching_handles_reversed_fighter_order() -> None:
    dataset = pd.DataFrame(
        [{"fight_date": "2024-01-01", "fighter_a": "Beth Beta", "fighter_b": "Alice Alpha", "fighter_a_win": 0}]
    )
    odds = pd.DataFrame(
        [
            {
                "fight_date": "2024-01-01",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "fighter_a_odds": -150,
                "fighter_b_odds": 130,
                "fighter_a_no_vig_probability": 0.58,
                "fighter_b_no_vig_probability": 0.42,
                "sportsbook": "espn_source",
                "timestamp": "",
            }
        ]
    )

    attached = attach_odds_features(dataset, odds)

    assert attached.loc[0, "market_fighter_a_implied_probability"] == 0.42
    assert attached.loc[0, "market_fighter_b_implied_probability"] == 0.58


def test_odds_matching_handles_normalized_names_and_accents() -> None:
    dataset = pd.DataFrame(
        [{"fight_date": "2024-01-01", "fighter_a": "Jiri Prochazka", "fighter_b": "Mark Madsen", "fighter_a_win": 1}]
    )
    odds = pd.DataFrame(
        [
            {
                "fight_date": "2024-01-01T03:00Z",
                "fighter_a": "Jiří Procházka",
                "fighter_b": "Mark O. Madsen",
                "fighter_a_odds": -120,
                "fighter_b_odds": 100,
                "sportsbook": "espn_source",
                "timestamp": "",
            }
        ]
    )

    attached = attach_odds_features(dataset, odds)

    assert pd.notna(attached.loc[0, "market_fighter_a_implied_probability"])
    assert american_odds_to_implied_probability(-120) is not None


def test_market_blend_weight_uses_supplied_training_window_only() -> None:
    past_weight = tune_market_blend_weight([1, 0], [0.51, 0.49], [0.9, 0.1])["weight"]
    leaked_weight = tune_market_blend_weight([1, 0, 0], [0.51, 0.49, 0.01], [0.9, 0.1, 0.99])["weight"]

    assert past_weight == 0.0
    assert leaked_weight != past_weight


def test_market_aware_rolling_backtest_records_past_blend_rows(monkeypatch) -> None:
    train_module = importlib.import_module("ufc_predictor.models.train")

    class FakeBundle:
        feature_columns = ["model_signal"]

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            probs = frame["model_signal"].astype(float).to_numpy()
            return np.column_stack([1.0 - probs, probs])

    monkeypatch.setattr(train_module, "train_ensemble", lambda *args, **kwargs: FakeBundle())
    dataset = pd.DataFrame(
        [
            {"fight_date": "2019-01-01", "fighter_a_win": 1, "model_signal": 0.55, "market_fighter_a_implied_probability": 0.9},
            {"fight_date": "2019-06-01", "fighter_a_win": 0, "model_signal": 0.45, "market_fighter_a_implied_probability": 0.1},
            {"fight_date": "2020-01-01", "fighter_a_win": 1, "model_signal": 0.01, "market_fighter_a_implied_probability": 0.9},
            {"fight_date": "2020-06-01", "fighter_a_win": 0, "model_signal": 0.99, "market_fighter_a_implied_probability": 0.1},
            {"fight_date": "2021-06-01", "fighter_a_win": 1, "model_signal": 0.4, "market_fighter_a_implied_probability": 0.8},
            {"fight_date": "2022-06-01", "fighter_a_win": 0, "model_signal": 0.4, "market_fighter_a_implied_probability": np.nan},
        ]
    )

    metrics = rolling_backtest(dataset, min_train_fights=2, model_mode="market-aware", min_blend_odds_rows=2)
    predictions = pd.DataFrame(metrics["predictions"])
    changed = predictions[predictions["fight_date"].dt.year == 2021].iloc[0]

    assert int(changed["blend_rows_used"]) == 2
    assert changed["final_probability_used"] != changed["pure_model_probability"]
    assert metrics["market_aware_probability_change"]["rows_changed"] > 0


def test_market_aware_uses_pure_model_when_test_odds_do_not_exist(monkeypatch) -> None:
    train_module = importlib.import_module("ufc_predictor.models.train")

    class FakeBundle:
        feature_columns = ["model_signal"]

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            probs = frame["model_signal"].astype(float).to_numpy()
            return np.column_stack([1.0 - probs, probs])

    monkeypatch.setattr(train_module, "train_ensemble", lambda *args, **kwargs: FakeBundle())
    dataset = pd.DataFrame(
        [
            {"fight_date": "2019-01-01", "fighter_a_win": 1, "model_signal": 0.55, "market_fighter_a_implied_probability": 0.9},
            {"fight_date": "2019-06-01", "fighter_a_win": 0, "model_signal": 0.45, "market_fighter_a_implied_probability": 0.1},
            {"fight_date": "2020-01-01", "fighter_a_win": 1, "model_signal": 0.01, "market_fighter_a_implied_probability": 0.9},
            {"fight_date": "2020-06-01", "fighter_a_win": 0, "model_signal": 0.99, "market_fighter_a_implied_probability": 0.1},
            {"fight_date": "2021-06-01", "fighter_a_win": 1, "model_signal": 0.6, "market_fighter_a_implied_probability": np.nan},
            {"fight_date": "2022-06-01", "fighter_a_win": 0, "model_signal": 0.4, "market_fighter_a_implied_probability": np.nan},
        ]
    )

    metrics = rolling_backtest(dataset, min_train_fights=2, model_mode="market-aware", min_blend_odds_rows=2)
    predictions = pd.DataFrame(metrics["predictions"])
    prediction = predictions[predictions["fight_date"].dt.year == 2021].iloc[0]

    assert prediction["final_probability_used"] == prediction["pure_model_probability"]


def test_scorecard_history_features_use_only_prior_fights() -> None:
    fights = pd.DataFrame(
        [
            {
                "fight_id": 1,
                "fight_date": "2020-01-01",
                "fighter_a": "Alice",
                "fighter_b": "Beth",
                "winner": "Alice",
                "method": "Decision - Unanimous",
            },
            {
                "fight_id": 2,
                "fight_date": "2021-01-01",
                "fighter_a": "Alice",
                "fighter_b": "Cara",
                "winner": "Alice",
                "method": "Decision - Unanimous",
            },
        ]
    )
    scorecards = pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "Past",
                "fighter_a": "Alice",
                "fighter_b": "Beth",
                "total_a": 29,
                "total_b": 28,
                "decision_type": "unanimous",
                "winner": "Alice",
            },
            {
                "fight_date": "2022-01-01",
                "event": "Future",
                "fighter_a": "Alice",
                "fighter_b": "Cara",
                "total_a": 30,
                "total_b": 27,
                "decision_type": "split",
                "winner": "Alice",
            },
        ]
    )

    dataset = build_fight_dataset(fights, scorecards=scorecards)
    target = dataset[dataset["fight_id"] == 2].iloc[0]

    assert target["fighter_a_prior_average_scorecard_margin"] == 1
    assert target["fighter_a_prior_split_decision_rate"] == 0


def test_compare_model_modes_command_output(monkeypatch, tmp_path) -> None:
    evaluate_module = importlib.import_module("ufc_predictor.models.evaluate")

    def fake_rolling(*args, model_mode="pure", **kwargs):
        probability = 0.6 if model_mode == "pure" else 0.65
        return {
            "accuracy": probability,
            "log_loss": 0.6,
            "brier_score": 0.2,
            "expected_calibration_error": 0.03,
            "model_mode": model_mode,
            "predictions": [
                {
                    "target": 1,
                    "prob": probability,
                    "final_probability_used": probability,
                    "market_implied_probability": 0.55,
                    "fight_date": "2020-01-01",
                }
            ],
        }

    monkeypatch.setattr(evaluate_module, "rolling_backtest", fake_rolling)
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame([{"fight_date": "2020-01-01", "fighter_a_win": 1}]).to_csv(dataset_path, index=False)
    output = tmp_path / "comparison.json"

    result = runner.invoke(app, ["compare-model-modes", "--dataset-path", str(dataset_path), "--output", str(output)])

    assert result.exit_code == 0
    assert "Model mode comparison" in result.output
    assert "odds-covered pure accuracy" in result.output
    assert json.loads(output.read_text(encoding="utf-8"))["market_aware_model"]["accuracy"] == 0.65


def test_ablation_report_command_output(monkeypatch, tmp_path) -> None:
    train_module = importlib.import_module("ufc_predictor.models.train")

    class FakeBundle:
        metrics = {"accuracy": 0.6, "log_loss": 0.7, "expected_calibration_error": 0.02}

    monkeypatch.setattr(train_module, "train_ensemble", lambda *args, **kwargs: FakeBundle())
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "fighter_a": "A",
                "fighter_b": "B",
                "fighter_a_win": 1,
                "diff_pre_fight_elo": 10,
            }
        ]
    ).to_csv(dataset_path, index=False)
    output = tmp_path / "ablation.json"

    result = runner.invoke(app, ["ablation-report", "--dataset-path", str(dataset_path), "--output", str(output)])

    assert result.exit_code == 0
    assert "Ablation report" in result.output
    assert "elo_opponent_adjusted_features" in json.loads(output.read_text(encoding="utf-8"))


def test_value_analysis_labels_and_no_reckless_language() -> None:
    output = format_prediction_output(
        "A",
        "B",
        0.62,
        pure_model_probability=0.62,
        market_implied_probability=0.55,
        blended_probability=0.58,
        final_probability_used=0.58,
        edge_vs_market=0.07,
        value_label=value_label_for_edge(0.07),
        show_value_analysis=True,
    )
    text = json.dumps(output).lower()

    assert output["value_label"].startswith("Medium potential edge")
    assert "this is not a guarantee" in text
    for forbidden in ["lock", "guaranteed", "must bet", "stake size", "stake-sizing"]:
        assert forbidden not in text


def test_apply_recency_weighting_duplicates_recent_training_rows() -> None:
    frame = pd.DataFrame(
        [
            {"fight_date": "2010-01-01", "fighter_a_win": 1},
            {"fight_date": "2020-01-01", "fighter_a_win": 0},
            {"fight_date": "2024-01-01", "fighter_a_win": 1},
        ]
    )

    weighted, summary = apply_recency_weighting(frame)

    assert summary["enabled"]
    assert len(weighted) > len(frame)
    assert summary["max_weight"] == 3


def test_predict_cli_accepts_manual_odds(monkeypatch, tmp_path) -> None:
    predict_module = importlib.import_module("ufc_predictor.models.predict")
    seen: dict = {}

    def fake_predict_fight(**kwargs):
        seen.update(kwargs)
        return {
            "fighter_a": kwargs["fighter_a"],
            "fighter_b": kwargs["fighter_b"],
            "market_implied_probability": 0.6923,
            "odds_source": "manual_input",
            "model_mode": "market_aware",
        }

    monkeypatch.setattr(predict_module, "predict_fight", fake_predict_fight)
    result = runner.invoke(
        app,
        [
            "predict",
            "--fighter-a",
            "A",
            "--fighter-b",
            "B",
            "--date",
            "2026-01-01",
            "--weight-class",
            "Lightweight",
            "--scheduled-rounds",
            "5",
            "--model-mode",
            "market-aware",
            "--show-value-analysis",
            "--fighter-a-odds",
            "-250",
            "--fighter-b-odds",
            "200",
            "--model-path",
            str(tmp_path / "missing.pkl"),
            "--fights-csv",
            str(tmp_path / "missing_fights.csv"),
            "--fight-stats-csv",
            str(tmp_path / "missing_stats.csv"),
            "--fighters-csv",
            str(tmp_path / "missing_fighters.csv"),
            "--scorecards-csv",
            str(tmp_path / "missing_scorecards.csv"),
            "--odds-csv",
            str(tmp_path / "missing_odds.csv"),
        ],
    )

    assert result.exit_code == 0
    assert seen["odds"].loc[0, "source_file"] == "manual_input"
    assert round(float(seen["odds"].loc[0, "fighter_a_no_vig_probability"]), 4) == 0.6818


def test_extract_odds_summarizes_invalid_rows_without_warning_spam(tmp_path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    pd.DataFrame(
        [
            {
                "Fight Date": "2024-01-01",
                "Red Fighter": "A",
                "Blue Fighter": "B",
                "Red Fighter Moneyline Odds": "bad",
                "Blue Fighter Moneyline Odds": 150,
            },
            {
                "Fight Date": "2024-01-02",
                "Red Fighter": "C",
                "Blue Fighter": "D",
                "Red Fighter Moneyline Odds": "",
                "Blue Fighter Moneyline Odds": "",
            },
        ]
    ).to_csv(source / "ufc_fights.csv", index=False)

    result = runner.invoke(app, ["extract-odds", "--source-dir", str(source), "--output-path", str(tmp_path / "odds.csv")])

    assert result.exit_code == 1
    assert "Invalid odds rows skipped: 1" in result.output
    assert "Blank odds rows skipped: 1" in result.output
    assert "Skipped invalid American odds" not in result.output
