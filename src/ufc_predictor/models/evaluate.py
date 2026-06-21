from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.models.calibrate import calibration_curve_data, expected_calibration_error
from ufc_predictor.models.predict import confidence_tier


BACKTEST_METADATA_COLUMNS = [
    "weight_class",
    "main_event",
    "title_fight",
    "scheduled_rounds",
    "event_location",
    "sex",
    "market_fighter_a_implied_probability",
    "market_fighter_b_implied_probability",
    "closing_odds_favorite_is_a",
]


def _clip_probs(y_prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)


def _accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(((_clip_probs(y_prob) >= 0.5).astype(int) == y_true.astype(int)).mean())


def _log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    prob = _clip_probs(y_prob)
    target = y_true.astype(int)
    return float(-(target * np.log(prob) + (1 - target) * np.log(1 - prob)).mean())


def tune_market_blend_weight(
    y_true: np.ndarray | pd.Series | list[float],
    model_probability: np.ndarray | pd.Series | list[float],
    market_probability: np.ndarray | pd.Series | list[float],
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Tune blend weight on a past-only training window.

    The returned weight is the model share in:
    w * model_probability + (1 - w) * market_probability.
    """

    y = np.asarray(y_true, dtype=int)
    model = np.asarray(model_probability, dtype=float)
    market = np.asarray(market_probability, dtype=float)
    valid = ~np.isnan(market)
    if not valid.any():
        return {"weight": 1.0, "rows_used": 0, "best_log_loss": float("nan")}
    weights = np.linspace(0.0, 1.0, 21) if grid is None else grid
    best_weight = 1.0
    best_loss = float("inf")
    for weight in weights:
        blended = _clip_probs(weight * model[valid] + (1.0 - weight) * market[valid])
        target = y[valid]
        loss = float(-(target * np.log(blended) + (1 - target) * np.log(1 - blended)).mean())
        if loss < best_loss:
            best_loss = loss
            best_weight = float(weight)
    return {"weight": best_weight, "rows_used": int(valid.sum()), "best_log_loss": best_loss}


def apply_market_blend(
    model_probability: np.ndarray | pd.Series | list[float],
    market_probability: np.ndarray | pd.Series | list[float],
    weight: float,
) -> np.ndarray:
    model = np.asarray(model_probability, dtype=float)
    market = np.asarray(market_probability, dtype=float)
    blended = model.copy()
    valid = ~np.isnan(market)
    blended[valid] = weight * model[valid] + (1.0 - weight) * market[valid]
    return _clip_probs(blended)


def _fold_market_diagnostics(
    *,
    period_index: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_rows: int,
    test_rows: int,
    train_market: pd.Series,
    test_market: pd.Series,
    target: pd.Series,
    pure_probs: np.ndarray,
    blended_probs: np.ndarray,
    blend: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    market = pd.to_numeric(test_market, errors="coerce")
    valid = market.notna().to_numpy()
    target_array = target.astype(int).to_numpy()
    pure = np.asarray(pure_probs, dtype=float)
    blended = np.asarray(blended_probs, dtype=float)
    market_values = market.to_numpy(dtype=float)
    probability_diff = np.abs(blended - pure)
    diagnostics: dict[str, Any] = {
        "period_index": period_index,
        "fold_start": pd.Timestamp(start).date().isoformat(),
        "fold_end": pd.Timestamp(end).date().isoformat(),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "training_rows_with_market_odds": int(pd.to_numeric(train_market, errors="coerce").notna().sum()),
        "test_rows_with_market_odds": int(valid.sum()),
        "learned_blend_weight": float(blend.get("weight", 1.0)),
        "blend_rows_used": int(blend.get("rows_used", 0)),
        "blend_validation_rows_used": int(blend.get("validation_rows_used", blend.get("rows_used", 0))),
        "blend_validation_method": str(blend.get("validation_method", "")),
        "blend_reason": reason,
        "rows_where_market_aware_changed_probability": int((probability_diff > 1e-12).sum()),
        "average_absolute_probability_change": float(probability_diff.mean()) if len(probability_diff) else float("nan"),
    }
    if valid.any():
        diagnostics.update(
            {
                "pure_accuracy_on_odds_covered_test_rows": _accuracy(target_array[valid], pure[valid]),
                "market_only_accuracy_on_odds_covered_test_rows": _accuracy(target_array[valid], market_values[valid]),
                "blended_accuracy_on_odds_covered_test_rows": _accuracy(target_array[valid], blended[valid]),
                "pure_log_loss_on_odds_covered_test_rows": _log_loss(target_array[valid], pure[valid]),
                "market_log_loss_on_odds_covered_test_rows": _log_loss(target_array[valid], market_values[valid]),
                "blended_log_loss_on_odds_covered_test_rows": _log_loss(target_array[valid], blended[valid]),
            }
        )
    else:
        diagnostics.update(
            {
                "pure_accuracy_on_odds_covered_test_rows": float("nan"),
                "market_only_accuracy_on_odds_covered_test_rows": float("nan"),
                "blended_accuracy_on_odds_covered_test_rows": float("nan"),
                "pure_log_loss_on_odds_covered_test_rows": float("nan"),
                "market_log_loss_on_odds_covered_test_rows": float("nan"),
                "blended_log_loss_on_odds_covered_test_rows": float("nan"),
            }
        )
    return diagnostics


def _historical_market_validation_predictions(
    train: pd.DataFrame,
    train_ensemble_func: Callable[..., Any],
    min_train_fights: int,
    model_dir: str | Path | None = None,
) -> tuple[pd.Series, np.ndarray, pd.Series]:
    market = pd.to_numeric(train.get("market_fighter_a_implied_probability", pd.Series([np.nan] * len(train))), errors="coerce")
    valid_market = market.notna()
    if not valid_market.any():
        return pd.Series(dtype=int), np.asarray([], dtype=float), pd.Series(dtype=float)
    frame = train.copy()
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce")
    market_rows = frame.loc[valid_market].copy()
    years = sorted(market_rows["fight_date"].dt.year.dropna().astype(int).unique().tolist())
    targets: list[pd.Series] = []
    probabilities: list[np.ndarray] = []
    markets: list[pd.Series] = []
    for year in years:
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year + 1, month=1, day=1)
        nested_train = frame[frame["fight_date"] < start]
        validation = frame[(frame["fight_date"] >= start) & (frame["fight_date"] < end) & market.notna()]
        if len(nested_train) < min_train_fights or validation.empty or nested_train["fighter_a_win"].nunique() < 2:
            continue
        nested_bundle = train_ensemble_func(nested_train, model_dir=model_dir, save=False, test_fraction=0.0)
        probabilities.append(nested_bundle.predict_proba(validation[nested_bundle.feature_columns])[:, 1])
        targets.append(validation["fighter_a_win"].astype(int))
        markets.append(pd.to_numeric(validation["market_fighter_a_implied_probability"], errors="coerce"))
    if not probabilities:
        return pd.Series(dtype=int), np.asarray([], dtype=float), pd.Series(dtype=float)
    return (
        pd.concat(targets, ignore_index=True),
        np.concatenate(probabilities),
        pd.concat(markets, ignore_index=True),
    )


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
        metrics["performance_by_modern_era"] = _modern_era_performance(metadata, y_true_array, y_pred)
        if "sex" in metadata.columns:
            metrics["performance_by_men_women"] = _metadata_group(metadata, y_true_array, y_pred, "sex")
        if "closing_odds_favorite_is_a" in metadata.columns:
            metrics["performance_on_underdogs"] = _underdog_performance(metadata, y_true_array, y_pred)
        if "market_fighter_a_implied_probability" in metadata.columns:
            metrics["model_vs_market"] = _model_vs_market(metadata, y_true_array, y_prob_array)
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
    favorite = metadata["closing_odds_favorite_is_a"].astype("boolean")
    valid = favorite.notna().to_numpy()
    if not valid.any():
        return {"count": 0.0, "accuracy": float("nan")}
    favorite_is_a = favorite.loc[valid].astype(bool).to_numpy()
    model_picked_a = y_pred[valid].astype(bool)
    target = y_true[valid]
    underdog_mask = favorite_is_a != model_picked_a
    if not underdog_mask.any():
        return {"count": 0.0, "accuracy": float("nan")}
    return {
        "count": float(underdog_mask.sum()),
        "accuracy": float((target[underdog_mask] == y_pred[valid][underdog_mask]).mean()),
    }


def _modern_era_performance(metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    if "fight_date" not in metadata.columns:
        return {}
    dates = pd.to_datetime(metadata["fight_date"], errors="coerce")
    output: dict[str, dict[str, float]] = {
        "all_years": {"count": float(len(y_true)), "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan")}
    }
    for start_year in [2015, 2020, 2022]:
        mask = dates.dt.year.ge(start_year).fillna(False).to_numpy()
        if not mask.any():
            output[f"{start_year}+"] = {"count": 0.0, "accuracy": float("nan")}
        else:
            output[f"{start_year}+"] = {
                "count": float(mask.sum()),
                "accuracy": float((y_true[mask] == y_pred[mask]).mean()),
            }
    return output


def _model_vs_market(metadata: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | str]:
    market = pd.to_numeric(metadata["market_fighter_a_implied_probability"], errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(market)
    if not valid.any():
        return {"count": 0.0, "note": "No matched odds rows available for model-vs-market analysis."}
    model_prob = y_prob[valid]
    market_prob = _clip_probs(market[valid])
    target = y_true[valid]
    model_pred = (model_prob >= 0.5).astype(int)
    market_pred = (market_prob >= 0.5).astype(int)
    return {
        "count": float(valid.sum()),
        "mean_model_probability": float(np.mean(model_prob)),
        "mean_market_implied_probability": float(np.mean(market_prob)),
        "mean_difference_vs_market": float(np.mean(model_prob - market_prob)),
        "mean_absolute_probability_delta": float(np.mean(np.abs(model_prob - market_prob))),
        "model_accuracy": float((model_pred == target).mean()),
        "market_accuracy": float((market_pred == target).mean()),
        "model_log_loss": float(-(target * np.log(_clip_probs(model_prob)) + (1 - target) * np.log(1 - _clip_probs(model_prob))).mean()),
        "market_log_loss": float(-(target * np.log(market_prob) + (1 - target) * np.log(1 - market_prob)).mean()),
        "note": "Model-vs-market comparison is analytical only.",
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
        favorite = dataset["closing_odds_favorite_is_a"]
        if favorite.notna().any():
            baselines["pick_betting_favorite"] = favorite.fillna(False).astype(bool).astype(int).to_numpy()
    return {
        name: {"accuracy": float((prediction == target).mean()), "count": float(len(target))}
        for name, prediction in baselines.items()
    }


def save_backtest_result(metrics: dict[str, Any], path: str | Path | None = None) -> Path:
    output = Path(path) if path else settings.processed_data_dir / "backtest_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(metrics, default=str))
    serializable["created_at"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output


def rolling_backtest(
    dataset: pd.DataFrame,
    min_train_fights: int = 50,
    step: str = "YS",
    model_dir: str | Path | None = None,
    model_mode: str = "pure",
    min_blend_odds_rows: int = 25,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train on fights before each period and predict the next period."""

    if dataset.empty:
        return {}
    mode = model_mode.strip().lower().replace("_", "-")
    if mode not in {"pure", "market-aware"}:
        raise ValueError("--model-mode must be either 'pure' or 'market-aware'.")
    from ufc_predictor.models.train import train_ensemble

    frame = dataset.sort_values("fight_date").copy()
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce")
    periods = pd.date_range(frame["fight_date"].min(), frame["fight_date"].max(), freq=step)
    period_pairs = list(zip(periods[:-1], periods[1:]))
    rows = []
    fold_diagnostics: list[dict[str, Any]] = []
    skipped_periods = 0
    for index, (start, end) in enumerate(period_pairs, start=1):
        train = frame[frame["fight_date"] < start]
        test = frame[(frame["fight_date"] >= start) & (frame["fight_date"] < end)]
        if len(train) < min_train_fights or test.empty or train["fighter_a_win"].nunique() < 2:
            skipped_periods += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "period_index": index,
                        "total_periods": len(period_pairs),
                        "start": start,
                        "end": end,
                        "status": "skipped",
                        "train_rows": len(train),
                        "test_rows": len(test),
                    }
                )
            continue
        if progress_callback is not None:
            progress_callback(
                {
                    "period_index": index,
                    "total_periods": len(period_pairs),
                    "start": start,
                    "end": end,
                    "status": "training",
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
        )
        bundle = train_ensemble(train, model_dir=model_dir, save=False, test_fraction=0.0)
        pure_probs = bundle.predict_proba(test[bundle.feature_columns])[:, 1]
        train_market = pd.to_numeric(train.get("market_fighter_a_implied_probability", pd.Series([np.nan] * len(train))), errors="coerce")
        test_market = pd.to_numeric(test.get("market_fighter_a_implied_probability", pd.Series([np.nan] * len(test))), errors="coerce")
        blend = {"weight": 1.0, "rows_used": 0, "best_log_loss": float("nan")}
        final_probs = pure_probs
        blend_reason = "Pure model mode."
        if mode == "market-aware":
            valid_train_market = train_market.notna()
            train_market_rows = int(valid_train_market.sum())
            if train_market_rows < min_blend_odds_rows:
                blend_reason = f"Not enough historical odds rows before fold: {train_market_rows} < {min_blend_odds_rows}."
            else:
                validation_target, validation_probs, validation_market = _historical_market_validation_predictions(
                    train,
                    train_ensemble_func=train_ensemble,
                    min_train_fights=min_train_fights,
                    model_dir=model_dir,
                )
                if len(validation_target) < min_blend_odds_rows:
                    blend_reason = (
                        "Not enough past-only validation odds rows before fold: "
                        f"{len(validation_target)} < {min_blend_odds_rows}."
                    )
                else:
                    blend = tune_market_blend_weight(
                        validation_target,
                        validation_probs,
                        validation_market,
                    )
                    blend["validation_rows_used"] = int(len(validation_target))
                    blend["validation_method"] = "nested_past_year_validation"
                    blend_reason = (
                        "Learned blend from nested past-only validation odds rows."
                        if float(blend.get("weight", 1.0)) < 1.0
                        else "Nested validation selected pure model weight for this fold."
                    )
            final_probs = apply_market_blend(pure_probs, test_market, blend["weight"])
        fold_diagnostics.append(
            _fold_market_diagnostics(
                period_index=index,
                start=start,
                end=end,
                train_rows=len(train),
                test_rows=len(test),
                train_market=train_market,
                test_market=test_market,
                target=test["fighter_a_win"],
                pure_probs=pure_probs,
                blended_probs=final_probs,
                blend=blend,
                reason=blend_reason,
            )
        )
        for (_, item), pure_prob, final_prob, market_prob in zip(test.iterrows(), pure_probs, final_probs, test_market):
            row = {
                "fight_id": item.get("fight_id"),
                "fight_date": item["fight_date"],
                "target": item["fighter_a_win"],
                "prob": final_prob,
                "pure_model_probability": pure_prob,
                "final_probability_used": final_prob,
                "model_mode": mode,
                "blend_weight": blend.get("weight", 1.0),
                "blend_rows_used": blend.get("rows_used", 0),
            }
            for column in BACKTEST_METADATA_COLUMNS:
                if column in item.index:
                    row[column] = item.get(column)
            row["model_probability"] = pure_prob
            if not pd.isna(market_prob):
                row["market_implied_probability"] = float(market_prob)
                row["difference_vs_market"] = float(pure_prob - market_prob)
                if mode == "market-aware":
                    row["blended_probability"] = float(final_prob)
            rows.append(row)
        if progress_callback is not None:
            progress_callback(
                {
                    "period_index": index,
                    "total_periods": len(period_pairs),
                    "start": start,
                    "end": end,
                    "status": "predicted",
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
            )
    if not rows:
        return {"message": "Not enough chronological data for rolling backtest.", "predictions": []}
    predictions = pd.DataFrame(rows)
    metrics = evaluate_predictions(predictions["target"], predictions["prob"], metadata=predictions)
    metrics["predictions"] = predictions.to_dict(orient="records")
    metrics["model_mode"] = mode
    metrics["market_aware_fold_diagnostics"] = fold_diagnostics
    metrics["market_aware_probability_change"] = {
        "rows_changed": int(
            (np.abs(predictions["final_probability_used"].astype(float) - predictions["pure_model_probability"].astype(float)) > 1e-12).sum()
        ),
        "average_absolute_difference": float(
            np.abs(predictions["final_probability_used"].astype(float) - predictions["pure_model_probability"].astype(float)).mean()
        ),
    }
    metrics["backtest_summary"] = {
        "step": step,
        "periods": len(period_pairs),
        "skipped_periods": skipped_periods,
        "prediction_rows": len(predictions),
    }
    return metrics
