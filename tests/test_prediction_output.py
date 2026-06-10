from __future__ import annotations

import pandas as pd

from ufc_predictor.models.predict import confidence_tier, format_prediction_output, predict_fight


class BiasedModel:
    def predict_proba(self, frame):
        return __import__("numpy").array([[0.01, 0.99]])


def test_prediction_probabilities_sum_to_one() -> None:
    prediction = format_prediction_output("Fighter A", "Fighter B", 0.63, ["A useful factor"])

    assert prediction["fighter_a_win_probability"] + prediction["fighter_b_win_probability"] == 1.0
    assert prediction["predicted_winner"] == "Fighter A"


def test_confidence_tier_thresholds() -> None:
    assert confidence_tier(0.56) == "Low"
    assert confidence_tier(0.57) == "Medium"
    assert confidence_tier(0.64) == "Medium"
    assert confidence_tier(0.65) == "High"


def test_missing_fighter_data_does_not_crash_prediction() -> None:
    prediction = predict_fight(
        model=None,
        fighter_a="Unknown A",
        fighter_b="Unknown B",
        fight_date="2026-10-01",
        weight_class="Lightweight",
        scheduled_rounds=3,
        fights=pd.DataFrame(),
        fight_stats=pd.DataFrame(),
        fighters=pd.DataFrame(),
        scorecards=pd.DataFrame(),
    )

    assert prediction["fighter_a_win_probability"] == 0.5
    assert prediction["fighter_b_win_probability"] == 0.5
    assert prediction["predicted_winner"] == "Unknown / Toss-up"
    assert prediction["confidence_score"] == 0.5
    assert prediction["confidence_tier"] == "Low"
    assert prediction["top_factors_for_prediction"]


def test_no_history_for_both_fighters_uses_neutral_fallback_even_with_model() -> None:
    unrelated_fights = pd.DataFrame(
        [
            {
                "fight_id": 1,
                "fight_date": "2024-01-01",
                "fighter_a": "Known A",
                "fighter_b": "Known B",
                "winner": "Known A",
                "weight_class": "Lightweight",
                "method": "Decision",
            }
        ]
    )

    prediction = predict_fight(
        model=BiasedModel(),
        fighter_a="Unknown A",
        fighter_b="Unknown B",
        fight_date="2026-10-01",
        weight_class="Lightweight",
        scheduled_rounds=3,
        fights=unrelated_fights,
        fight_stats=pd.DataFrame(),
        fighters=pd.DataFrame(),
        scorecards=pd.DataFrame(),
    )

    assert prediction["fighter_a_win_probability"] == 0.5
    assert prediction["fighter_b_win_probability"] == 0.5
    assert prediction["predicted_winner"] == "Unknown / Toss-up"
    assert prediction["top_factors_for_prediction"] == ["No historical data available; using neutral fallback."]
