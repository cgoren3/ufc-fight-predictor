from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ufc_predictor.config import settings
from ufc_predictor.data_sources import summarize_raw_data


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON at {path}"}


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
            "model_vs_market": backtest.get("model_vs_market", {}),
            "summary": backtest.get("backtest_summary", {}),
        },
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
