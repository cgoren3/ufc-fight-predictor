from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.config import settings


@dataclass
class LeakageAuditResult:
    sampled_rows: int
    passed_rows: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def passed(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "sampled_rows": self.sampled_rows,
            "passed_rows": self.passed_rows,
            "passed": self.passed,
            "violations": self.violations,
            "warnings": self.warnings,
        }


def _normal(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _fighter_history(fights: pd.DataFrame, fighter: str, fight_date: pd.Timestamp) -> pd.DataFrame:
    if fights.empty:
        return fights.copy()
    fighter_name = _normal(fighter)
    frame = fights.copy()
    frame["fight_date"] = pd.to_datetime(frame["fight_date"], errors="coerce")
    mask = (
        (frame["fight_date"] < fight_date)
        & ((frame["fighter_a"].map(_normal) == fighter_name) | (frame["fighter_b"].map(_normal) == fighter_name))
    )
    return frame.loc[mask].copy()


def _history_count_mismatch(row: pd.Series, fights: pd.DataFrame, prefix: str) -> dict[str, Any] | None:
    fighter = _normal(row.get(prefix))
    fight_date = pd.Timestamp(row["fight_date"])
    expected_column = f"{prefix}_total_ufc_fights_before"
    if expected_column not in row:
        return None
    expected = float(pd.to_numeric(pd.Series([row.get(expected_column)]), errors="coerce").fillna(0.0).iloc[0])
    actual = float(len(_fighter_history(fights, fighter, fight_date)))
    if abs(expected - actual) > 1e-6:
        return {
            "type": "history_count_mismatch",
            "fighter": fighter,
            "expected_feature_value": expected,
            "actual_prior_fights": actual,
        }
    return None


def run_leakage_audit(
    dataset: pd.DataFrame,
    raw_fights: pd.DataFrame,
    sample_size: int = 100,
    random_state: int = 42,
) -> LeakageAuditResult:
    if dataset.empty:
        return LeakageAuditResult(sampled_rows=0, warnings=["Dataset is empty; no rows audited."])
    sample_count = min(sample_size, len(dataset))
    sampled = dataset.sample(n=sample_count, random_state=random_state) if sample_count < len(dataset) else dataset.copy()
    raw = raw_fights.copy()
    if not raw.empty and "fight_date" in raw.columns:
        raw["fight_date"] = pd.to_datetime(raw["fight_date"], errors="coerce")

    result = LeakageAuditResult(sampled_rows=int(sample_count))
    for _, row in sampled.iterrows():
        fight_date = pd.Timestamp(row["fight_date"])
        row_context = {
            "fight_id": row.get("fight_id"),
            "fight_date": str(row.get("fight_date")),
            "fighter_a": row.get("fighter_a"),
            "fighter_b": row.get("fighter_b"),
        }
        row_violations: list[dict[str, Any]] = []
        max_history = pd.to_datetime(row.get("max_history_date_used"), errors="coerce")
        if pd.notna(max_history) and max_history >= fight_date:
            row_violations.append(
                {
                    "type": "max_history_date_not_before_fight",
                    "max_history_date_used": str(max_history),
                    "fight_date": str(fight_date),
                }
            )
        for prefix in ["fighter_a", "fighter_b"]:
            mismatch = _history_count_mismatch(row, raw, prefix)
            if mismatch is not None:
                row_violations.append(mismatch)
        if row_violations:
            result.violations.append({"row": row_context, "violations": row_violations})
        else:
            result.passed_rows += 1
    return result


def save_leakage_audit_report(result: LeakageAuditResult, path: str | Path | None = None) -> Path:
    output = Path(path) if path else settings.processed_data_dir / "leakage_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    return output
