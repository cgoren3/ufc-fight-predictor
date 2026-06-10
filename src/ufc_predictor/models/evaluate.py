from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.models.calibrate import calibration_curve_data, expected_calibration_error
from ufc_predictor.models.predict import confidence_tier


def _clip_probs(y_prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)


def evaluate_predictions(
    y_true: np.ndarray | pd.Series | list[float],
    y_prob: np.ndarray | pd.Series | list[float],
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = _clip_probs(np.asarray(y_prob, dtype=float))
    y_pred = (y_prob_array >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "accuracy": float((y_pred == y_true_array).mean()) if len(y_true_array) else float("nan"),
        "log_loss": float(-(y_true_array * np.log(y_prob_array) + (1 - y_true_array) * np.log(1 - y_prob_array)).mean())
        if len(y_true_array)
        else float("nan"),
        "brier_score": float(np.mean((y_prob_array - y_true_array) ** 2)) if len(y_true_array) else float("nan"),
        "expected_calibration_error": expected_calibration_error(y_true_array, y_prob_array),
        "calibration_curve": calibration_curve_data(y_true_array, y_prob_array).to_dict(orient="records"),
    }
    try:
        from sklearn.metrics import roc_auc_score

        metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_prob_array)) if len(set(y_true_array)) > 1 else float("nan")
    except Exception:
        metrics["roc_auc"] = float("nan")

    confidence_scores = np.maximum(y_prob_array, 1.0 - y_prob_array)
    tiers = [confidence_tier(score) for score in confidence_scores]
    metrics["performance_by_confidence_tier"] = _group_performance(y_true_array, y_pred, tiers)
    if metadata is not None and not metadata.empty:
        metrics["performance_by_year"] = _metadata_group(metadata, y_true_array, y_pred, "fight_year")
        metrics["performance_by_weight_class"] = _metadata_group(metadata, y_true_array, y_pred, "weight_class")
        metrics["performance_by_main_event"] = _metadata_group(metadata, y_true_array, y_pred, "main_event")
        if "sex" in metadata.columns:
            metrics["performance_by_men_women"] = _metadata_group(metadata, y_true_array, y_pred, "sex")
        if "closing_odds_favorite_is_a" in metadata.columns:
            metrics["performance_on_underdogs"] = _underdog_performance(metadata, y_true_array, y_pred)
    return metrics


def _group_performance(y_true: np.ndarray, y_pred: np.ndarray, groups: list[Any]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})
    output: dict[str, dict[str, float]] = {}
    for group, rows in frame.groupby("group"):
        output[str(group)] = {"count": float(len(rows)), "accuracy": float((rows["y_true"] == rows["y_pred"]).mean())}
    return output


def _metadata_group(metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, column: str) -> dict[str, dict[str, float]]:
    meta = metadata.copy()
    if column == "fight_year" and "fight_date" in meta.columns:
        meta[column] = pd.to_datetime(meta["fight_date"], errors="coerce").dt.year
    if column not in meta.columns:
        return {}
    return _group_performance(y_true, y_pred, meta[column].fillna("Unknown").tolist())


def _underdog_performance(metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    favorite_is_a = metadata["closing_odds_favorite_is_a"].astype(bool).to_numpy()
    model_picked_a = y_pred.astype(bool)
    underdog_mask = favorite_is_a != model_picked_a
    if not underdog_mask.any():
        return {"count": 0.0, "accuracy": float("nan")}
    return {
        "count": float(underdog_mask.sum()),
        "accuracy": float((y_true[underdog_mask] == y_pred[underdog_mask]).mean()),
    }


def baseline_metrics(dataset: pd.DataFrame) -> dict[str, Any]:
    if dataset.empty:
        return {}
    target = dataset["fighter_a_win"].astype(int).to_numpy()
    baselines: dict[str, np.ndarray] = {}
    if {"fighter_a_wins_before", "fighter_a_losses_before", "fighter_b_wins_before", "fighter_b_losses_before"} <= set(
        dataset.columns
    ):
        record_a_total = (dataset["fighter_a_wins_before"] + dataset["fighter_a_losses_before"]).replace(0, np.nan)
        record_b_total = (dataset["fighter_b_wins_before"] + dataset["fighter_b_losses_before"]).replace(0, np.nan)
        record_a = dataset["fighter_a_wins_before"] / record_a_total
        record_b = dataset["fighter_b_wins_before"] / record_b_total
        baselines["pick_better_record"] = (record_a.fillna(0.5) >= record_b.fillna(0.5)).astype(int).to_numpy()
    if "diff_pre_fight_elo" in dataset.columns:
        baselines["pick_higher_elo"] = (dataset["diff_pre_fight_elo"].fillna(0) >= 0).astype(int).to_numpy()
    if "closing_odds_favorite_is_a" in dataset.columns:
        baselines["pick_betting_favorite"] = dataset["closing_odds_favorite_is_a"].astype(int).to_numpy()
    return {
        name: {"accuracy": float((prediction == target).mean()), "count": float(len(target))}
        for name, prediction in baselines.items()
    }


def save_backtest_result(metrics: dict[str, Any], path: str | Path | None = None) -> Path:
    output = Path(path) if path else settings.processed_data_dir / "backtest_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(metrics, default=str))
    serializable["created_at"] = datetime.utcnow().isoformat()
    output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output


def rolling_backtest(
    dataset: pd.DataFrame,
    min_train_fights: int = 50,
    step: str = "MS",
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train on fights before each period and predict the next period."""

    if dataset.empty:
        return {}
    from ufc_predictor.models.train import train_ensemble

    frame = dataset.sort_values("fight_date").copy()
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce")
    periods = pd.date_range(frame["fight_date"].min(), frame["fight_date"].max(), freq=step)
    rows = []
    for start, end in zip(periods[:-1], periods[1:]):
        train = frame[frame["fight_date"] < start]
        test = frame[(frame["fight_date"] >= start) & (frame["fight_date"] < end)]
        if len(train) < min_train_fights or test.empty or train["fighter_a_win"].nunique() < 2:
            continue
        bundle = train_ensemble(train, model_dir=model_dir, save=False)
        probs = bundle.predict_proba(test[bundle.feature_columns])[:, 1]
        for (_, item), prob in zip(test.iterrows(), probs):
            rows.append({"fight_id": item.get("fight_id"), "fight_date": item["fight_date"], "target": item["fighter_a_win"], "prob": prob})
    if not rows:
        return {"message": "Not enough chronological data for rolling backtest.", "predictions": []}
    predictions = pd.DataFrame(rows)
    metrics = evaluate_predictions(predictions["target"], predictions["prob"], metadata=predictions)
    metrics["predictions"] = predictions.to_dict(orient="records")
    return metrics
