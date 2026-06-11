from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ufc_predictor.models import train as train_module
from ufc_predictor.models.train import feature_selection_summary, train_ensemble


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fight_date": pd.date_range("2020-01-01", periods=8, freq="D"),
            "fighter_a": [f"A{i}" for i in range(8)],
            "fighter_b": [f"B{i}" for i in range(8)],
            "fighter_a_win": [0, 1, 0, 1, 0, 1, 0, 1],
            "event_location": [np.nan] * 8,
            "all_null_numeric": [np.nan] * 8,
            "numeric_signal": [0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6],
            "weight_class": ["Lightweight"] * 8,
        }
    )


def test_all_null_event_location_is_dropped() -> None:
    summary = feature_selection_summary(_training_frame())

    assert "event_location" in summary.dropped_all_null_features
    assert "event_location" not in summary.numeric_features
    assert "event_location" not in summary.categorical_features


def test_all_null_numeric_column_is_dropped() -> None:
    summary = feature_selection_summary(_training_frame())

    assert "all_null_numeric" in summary.dropped_all_null_features
    assert "all_null_numeric" not in summary.numeric_features


def test_event_location_with_values_is_categorical() -> None:
    frame = _training_frame()
    frame["event_location"] = ["Las Vegas", "New York", "Las Vegas", "London", "Paris", "Tokyo", "Miami", "Boston"]

    summary = feature_selection_summary(frame)

    assert "event_location" not in summary.dropped_all_null_features
    assert "event_location" in summary.categorical_features
    assert "event_location" not in summary.numeric_features


def test_training_drops_all_null_features_without_imputer_warning(monkeypatch) -> None:
    sk = train_module._imports()

    def one_estimator(random_state=42):
        return [("logistic_regression", sk["LogisticRegression"](max_iter=1000), True)]

    monkeypatch.setattr(train_module, "_candidate_estimators", one_estimator)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = train_ensemble(_training_frame(), save=False, test_fraction=0.25)

    messages = [str(item.message) for item in caught]
    assert not any("Skipping features without any observed values" in message for message in messages)
    assert bundle.dropped_all_null_features == ["event_location", "all_null_numeric"]
    assert "numeric_signal" in bundle.numeric_features
