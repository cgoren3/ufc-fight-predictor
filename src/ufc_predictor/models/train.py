from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ufc_predictor.config import settings
from ufc_predictor.data_sources import read_source_metadata
from ufc_predictor.features.build_fight_dataset import feature_columns
from ufc_predictor.models.calibrate import calibrate_estimator
from ufc_predictor.models.evaluate import baseline_metrics, evaluate_predictions


@dataclass
class FeatureSelectionSummary:
    total_features_before_cleaning: int
    dropped_all_null_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_features_before_cleaning": self.total_features_before_cleaning,
            "dropped_all_null_features": self.dropped_all_null_features,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "numeric_feature_count": len(self.numeric_features),
            "categorical_feature_count": len(self.categorical_features),
        }


@dataclass
class TrainedModelBundle:
    model_version: str
    estimators: list[Any]
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_all_null_features: list[str] = field(default_factory=list)
    feature_summary: dict[str, Any] = field(default_factory=dict)
    model_card: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def _align(self, frame: pd.DataFrame) -> pd.DataFrame:
        aligned = frame.copy()
        for column in self.feature_columns:
            if column not in aligned.columns:
                aligned[column] = np.nan
        return aligned[self.feature_columns]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        aligned = self._align(frame)
        if not self.estimators:
            raise RuntimeError("Model bundle has no estimators.")
        probabilities = []
        for estimator in self.estimators:
            proba = estimator.predict_proba(aligned)
            probabilities.append(proba)
        average = np.mean(probabilities, axis=0)
        row_sums = average.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return average / row_sums

    def member_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        aligned = self._align(frame)
        if not self.estimators:
            return np.empty((0, len(aligned)))
        probabilities = []
        for estimator in self.estimators:
            probabilities.append(estimator.predict_proba(aligned)[:, 1])
        return np.asarray(probabilities, dtype=float)

    def uncertainty_range(self, frame: pd.DataFrame) -> list[list[float]]:
        member_probs = self.member_probabilities(frame)
        if member_probs.size == 0:
            mean = self.predict_proba(frame)[:, 1]
            return [[round(float(value), 4), round(float(value), 4)] for value in mean]
        lower = np.nanmin(member_probs, axis=0)
        upper = np.nanmax(member_probs, axis=0)
        return [[round(float(max(low, 0.0)), 4), round(float(min(high, 1.0)), 4)] for low, high in zip(lower, upper)]


def _imports():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:  # pragma: no cover - dependency path
        raise RuntimeError("scikit-learn is required for training. Install project dependencies.") from exc
    return {
        "ColumnTransformer": ColumnTransformer,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


ALWAYS_CATEGORICAL_FEATURES = {"event_location"}


def _is_all_null_feature(series: pd.Series) -> bool:
    if series.empty:
        return True
    if is_numeric_dtype(series):
        return bool(series.isna().all())
    text = series.dropna().astype(str).str.strip()
    return text.empty or bool(text.eq("").all())


def feature_selection_summary(frame: pd.DataFrame) -> FeatureSelectionSummary:
    columns = feature_columns(frame)
    dropped = [column for column in columns if _is_all_null_feature(frame[column])]
    kept = [column for column in columns if column not in dropped]
    numeric = [
        column
        for column in kept
        if is_numeric_dtype(frame[column]) and column not in ALWAYS_CATEGORICAL_FEATURES
    ]
    categorical = [column for column in kept if column not in numeric]
    return FeatureSelectionSummary(
        total_features_before_cleaning=len(columns),
        dropped_all_null_features=dropped,
        numeric_features=numeric,
        categorical_features=categorical,
    )


def _split_features(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    summary = feature_selection_summary(frame)
    columns = [*summary.numeric_features, *summary.categorical_features]
    numeric = summary.numeric_features
    categorical = summary.categorical_features
    return columns, numeric, categorical


def _preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool):
    sk = _imports()
    numeric_steps = [("imputer", sk["SimpleImputer"](strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", sk["StandardScaler"]()))
    categorical_pipeline = sk["Pipeline"](
        steps=[
            ("imputer", sk["SimpleImputer"](strategy="most_frequent")),
            ("onehot", sk["OneHotEncoder"](handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return sk["ColumnTransformer"](
        transformers=[
            ("numeric", sk["Pipeline"](numeric_steps), numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _pipeline(estimator: Any, numeric: list[str], categorical: list[str], scale_numeric: bool):
    sk = _imports()
    return sk["Pipeline"]([("preprocess", _preprocessor(numeric, categorical, scale_numeric)), ("model", estimator)])


def _candidate_estimators(random_state: int = 42) -> list[tuple[str, Any, bool]]:
    sk = _imports()
    candidates: list[tuple[str, Any, bool]] = [
        ("logistic_regression", sk["LogisticRegression"](max_iter=2000, class_weight="balanced"), True),
        (
            "random_forest",
            sk["RandomForestClassifier"](n_estimators=300, min_samples_leaf=3, random_state=random_state, n_jobs=-1),
            False,
        ),
    ]
    try:
        from xgboost import XGBClassifier

        candidates.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=random_state,
                ),
                False,
            )
        )
    except Exception:
        candidates.append(
            (
                "hist_gradient_boosting",
                sk["HistGradientBoostingClassifier"](learning_rate=0.04, max_iter=250, random_state=random_state),
                False,
            )
        )
    return candidates


def chronological_split(
    dataset: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = dataset.sort_values("fight_date").reset_index(drop=True)
    if len(frame) < 5:
        return frame, frame.iloc[0:0].copy()
    split_at = max(int(len(frame) * (1.0 - test_fraction)), 1)
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()


def train_ensemble(
    dataset: pd.DataFrame,
    model_dir: str | Path | None = None,
    calibration_method: str = "sigmoid",
    test_fraction: float = 0.20,
    random_state: int = 42,
    save: bool = True,
) -> TrainedModelBundle:
    if dataset.empty:
        raise ValueError("Cannot train on an empty dataset.")
    train_frame, test_frame = chronological_split(dataset, test_fraction=test_fraction)
    if train_frame["fighter_a_win"].nunique() < 2:
        raise ValueError("Training data must contain wins and losses for fighter_a.")
    selection = feature_selection_summary(train_frame)
    columns = [*selection.numeric_features, *selection.categorical_features]
    numeric = selection.numeric_features
    categorical = selection.categorical_features
    if not columns:
        raise ValueError("No usable model features remain after dropping all-null columns.")
    x_train = train_frame[columns]
    y_train = train_frame["fighter_a_win"].astype(int)
    estimators = []
    for name, estimator, scale_numeric in _candidate_estimators(random_state=random_state):
        pipeline = _pipeline(estimator, numeric, categorical, scale_numeric=scale_numeric)
        cv = min(3, int(y_train.value_counts().min()))
        fitted = calibrate_estimator(pipeline, x_train, y_train, method=calibration_method, cv=max(cv, 2)) if cv >= 2 else pipeline.fit(x_train, y_train)
        fitted.model_name = name  # type: ignore[attr-defined]
        estimators.append(fitted)
    bundle = TrainedModelBundle(
        model_version=f"ufc_predictor_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        estimators=estimators,
        feature_columns=columns,
        numeric_features=numeric,
        categorical_features=categorical,
        dropped_all_null_features=selection.dropped_all_null_features,
        feature_summary=selection.as_dict(),
    )
    metrics: dict[str, Any] = {"baselines": baseline_metrics(dataset)}
    metrics["feature_summary"] = selection.as_dict()
    if not test_frame.empty and test_frame["fighter_a_win"].nunique() >= 1:
        probabilities = bundle.predict_proba(test_frame[columns])[:, 1]
        metrics.update(evaluate_predictions(test_frame["fighter_a_win"], probabilities, metadata=test_frame))
    bundle.metrics = metrics
    bundle.model_card = build_model_card(bundle, dataset)
    if save:
        save_model_bundle(bundle, model_dir=model_dir)
    return bundle


def build_model_card(bundle: TrainedModelBundle, dataset: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(dataset.get("fight_date", pd.Series(dtype=object)), errors="coerce").dropna()
    high_confidence = bundle.metrics.get("performance_by_confidence_tier", {}).get("High", {})
    known_weaknesses = [
        "Historical MMA data can be incomplete or inconsistently sourced.",
        "Late-breaking injuries, camps, weight misses, and opponent changes are not reliably modeled unless imported manually.",
        "Predictions are calibrated historical probabilities, not guarantees and not betting advice.",
    ]
    if bundle.dropped_all_null_features:
        known_weaknesses.append(
            "Some features were dropped because they were completely missing: "
            + ", ".join(bundle.dropped_all_null_features[:10])
        )
    if "market_fighter_a_implied_probability" not in dataset.columns:
        known_weaknesses.append("No betting odds were attached for market comparison in this training run.")
    return {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "data_source": read_source_metadata(settings.raw_data_dir).get("source", "unknown"),
        "date_range": {
            "start": dates.min().date().isoformat() if not dates.empty else None,
            "end": dates.max().date().isoformat() if not dates.empty else None,
        },
        "number_of_training_rows": int(len(dataset)),
        "feature_count": int(len(bundle.feature_columns)),
        "dropped_features": bundle.dropped_all_null_features,
        "model_type": [getattr(estimator, "model_name", type(estimator).__name__) for estimator in bundle.estimators],
        "accuracy": bundle.metrics.get("accuracy"),
        "log_loss": bundle.metrics.get("log_loss"),
        "brier_score": bundle.metrics.get("brier_score"),
        "calibration_error": bundle.metrics.get("expected_calibration_error"),
        "high_confidence_accuracy": high_confidence.get("accuracy"),
        "known_weaknesses": known_weaknesses,
    }


def save_model_bundle(bundle: TrainedModelBundle, model_dir: str | Path | None = None) -> Path:
    output_dir = Path(model_dir) if model_dir else settings.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ufc_predictor_model.pkl"
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    metadata = {
        "model_version": bundle.model_version,
        "created_at": bundle.created_at,
        "feature_columns": bundle.feature_columns,
        "numeric_features": bundle.numeric_features,
        "categorical_features": bundle.categorical_features,
        "dropped_all_null_features": bundle.dropped_all_null_features,
        "feature_summary": bundle.feature_summary,
        "model_card": bundle.model_card,
        "metrics": bundle.metrics,
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (output_dir / "model_card.json").write_text(json.dumps(bundle.model_card, indent=2, default=str), encoding="utf-8")
    return path


def load_model_bundle(path: str | Path | None = None) -> TrainedModelBundle:
    model_path = Path(path) if path else settings.model_dir / "ufc_predictor_model.pkl"
    with model_path.open("rb") as handle:
        return pickle.load(handle)
