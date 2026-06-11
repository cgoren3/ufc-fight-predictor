from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.data_io import read_optional_csv
from ufc_predictor.data_sources import summarize_raw_data
from ufc_predictor.odds import attach_odds_features


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON at {path}"}


def _name_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _date_key(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return ""
    return date.date().isoformat()


def _pair_key(row: pd.Series | dict[str, Any]) -> str:
    return "|".join(sorted([_name_key(row.get("fighter_a")), _name_key(row.get("fighter_b"))]))


def _fight_key(row: pd.Series | dict[str, Any]) -> str:
    return "||".join([_date_key(row.get("fight_date")), _pair_key(row)])


def _percent(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator) * 100.0, 2)


def _known_text_mask(series: pd.Series, unknown_values: set[str] | None = None) -> pd.Series:
    unknown = unknown_values or {"", "unknown", "unk", "n/a", "na", "none", "nan"}
    text = series.fillna("").astype(str).str.strip()
    return (text != "") & ~text.str.lower().isin(unknown)


def _known_main_event_mask(fights: pd.DataFrame) -> pd.Series:
    if fights.empty or "main_event" not in fights.columns:
        return pd.Series([False] * len(fights), index=fights.index)
    flags = pd.to_numeric(fights["main_event"], errors="coerce").fillna(0)
    if "event_name" in fights.columns and "fight_date" in fights.columns:
        grouped = pd.DataFrame(
            {
                "fight_date": pd.to_datetime(fights["fight_date"], errors="coerce").dt.date.astype("string"),
                "event_name": fights["event_name"].fillna("").astype(str).str.strip().str.lower(),
                "has_main_event": flags.eq(1),
            },
            index=fights.index,
        )
        event_has_main = grouped.groupby(["fight_date", "event_name"])["has_main_event"].transform("any")
        return event_has_main.fillna(False).astype(bool)
    return flags.eq(1)


def _odds_coverage(fights: pd.DataFrame, odds: pd.DataFrame | None) -> tuple[int, float]:
    if fights.empty or odds is None or odds.empty:
        return 0, 0.0
    attached = attach_odds_features(fights[["fight_date", "fighter_a", "fighter_b"]].copy(), odds)
    if "market_fighter_a_implied_probability" not in attached.columns:
        return 0, 0.0
    count = int(pd.to_numeric(attached["market_fighter_a_implied_probability"], errors="coerce").notna().sum())
    return count, _percent(count, len(fights))


def _scorecard_coverage(fights: pd.DataFrame, scorecards: pd.DataFrame | None) -> tuple[int, float]:
    if fights.empty or scorecards is None or scorecards.empty:
        return 0, 0.0
    if not {"fight_date", "fighter_a", "fighter_b"} <= set(scorecards.columns):
        return 0, 0.0
    fight_keys = fights.apply(_fight_key, axis=1)
    scorecard_keys = set(scorecards.apply(_fight_key, axis=1).tolist())
    count = int(fight_keys.isin(scorecard_keys).sum())
    return count, _percent(count, len(fights))


def build_data_quality_coverage(raw_dir: str | Path | None = None) -> dict[str, Any]:
    raw = Path(raw_dir) if raw_dir else settings.raw_data_dir
    fights = read_optional_csv(raw / "fights.csv", label="fights CSV")
    odds = read_optional_csv(raw / "odds.csv", label="odds CSV")
    scorecards = read_optional_csv(raw / "scorecards.csv", label="scorecards CSV")
    if fights is None or fights.empty:
        return {
            "fights": 0,
            "known_weight_class_pct": 0.0,
            "known_event_location_pct": 0.0,
            "known_main_event_pct": 0.0,
            "odds_coverage_pct": 0.0,
            "scorecard_coverage_pct": 0.0,
        }

    total = len(fights)
    known_weight = int(_known_text_mask(fights.get("weight_class", pd.Series(dtype=object))).sum())
    known_location = int(_known_text_mask(fights.get("event_location", pd.Series(dtype=object))).sum())
    known_main = int(_known_main_event_mask(fights).sum())
    odds_count, odds_pct = _odds_coverage(fights, odds)
    scorecard_count, scorecard_pct = _scorecard_coverage(fights, scorecards)
    return {
        "fights": int(total),
        "known_weight_class_count": known_weight,
        "known_weight_class_pct": _percent(known_weight, total),
        "known_event_location_count": known_location,
        "known_event_location_pct": _percent(known_location, total),
        "known_main_event_count": known_main,
        "known_main_event_pct": _percent(known_main, total),
        "odds_matched_fights": odds_count,
        "odds_coverage_pct": odds_pct,
        "scorecard_matched_fights": scorecard_count,
        "scorecard_coverage_pct": scorecard_pct,
    }


def build_performance_report(
    model_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
) -> dict[str, Any]:
    models = Path(model_dir) if model_dir else settings.model_dir
    processed = Path(processed_dir) if processed_dir else settings.processed_data_dir
    raw = Path(raw_dir) if raw_dir else settings.raw_data_dir
    metadata = _read_json(models / "model_metadata.json")
    model_card = _read_json(models / "model_card.json")
    backtest = _read_json(processed / "backtest_results.json")
    summary = summarize_raw_data(raw_dir=raw)
    coverage = build_data_quality_coverage(raw)
    train_metrics = metadata.get("metrics", {})
    known_missing = []
    feature_summary = metadata.get("feature_summary", {})
    dropped = feature_summary.get("dropped_all_null_features", [])
    if dropped:
        known_missing.append(f"All-null features dropped during training: {', '.join(dropped)}")
    if summary.get("scorecards_row_count", 0) == 0:
        known_missing.append("No scorecard rows are currently loaded.")
    if not (raw / "odds.csv").exists():
        known_missing.append("No odds.csv is currently loaded for market comparison.")
    if coverage.get("known_weight_class_pct", 0.0) < 50.0:
        known_missing.append("Most fights are missing known weight_class values.")
    if coverage.get("known_event_location_pct", 0.0) < 50.0:
        known_missing.append("Most fights are missing known event_location values.")
    if coverage.get("known_main_event_pct", 0.0) < 50.0:
        known_missing.append("Most events do not have a known main_event flag.")
    return {
        "dataset_size": {
            "fights": summary.get("fights_row_count", 0),
            "fighters": summary.get("fighters_row_count", 0),
            "fight_stats": summary.get("fight_stats_row_count", 0),
            "scorecards": summary.get("scorecards_row_count", 0),
            "unique_fighters": summary.get("unique_fighters", 0),
            "date_range": summary.get("date_range", {}),
            "data_source": summary.get("data_source", "unknown"),
        },
        "train_metrics": {
            "accuracy": train_metrics.get("accuracy"),
            "log_loss": train_metrics.get("log_loss"),
            "brier_score": train_metrics.get("brier_score"),
            "expected_calibration_error": train_metrics.get("expected_calibration_error"),
            "confidence_tier_performance": train_metrics.get("performance_by_confidence_tier", {}),
        },
        "backtest_metrics": {
            "accuracy": backtest.get("accuracy"),
            "log_loss": backtest.get("log_loss"),
            "brier_score": backtest.get("brier_score"),
            "expected_calibration_error": backtest.get("expected_calibration_error"),
            "confidence_tier_performance": backtest.get("performance_by_confidence_tier", {}),
            "yearly_performance": backtest.get("performance_by_year", {}),
            "performance_by_weight_class": backtest.get("performance_by_weight_class", {}),
            "performance_by_main_event": backtest.get("performance_by_main_event", {}),
            "model_vs_market": backtest.get("model_vs_market", {}),
            "summary": backtest.get("backtest_summary", {}),
        },
        "data_quality_coverage": coverage,
        "calibration": {
            "train_curve": train_metrics.get("calibration_curve", []),
            "backtest_curve": backtest.get("calibration_curve", []),
        },
        "model_card": model_card,
        "known_missing_data_issues": known_missing,
    }


def save_performance_report(report: dict[str, Any], path: str | Path | None = None) -> Path:
    output = Path(path) if path else settings.processed_data_dir / "performance_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return output
