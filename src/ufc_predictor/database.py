from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.config import Settings, settings


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS fighters (
        fighter_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        stance TEXT,
        height_in REAL,
        weight_lb REAL,
        reach_in REAL,
        date_of_birth TEXT,
        country TEXT,
        state TEXT,
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        event_date TEXT NOT NULL,
        location TEXT,
        altitude_ft REAL,
        source_url TEXT UNIQUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fights (
        fight_id INTEGER PRIMARY KEY,
        event_id INTEGER,
        event_name TEXT,
        fight_date TEXT NOT NULL,
        fighter_a TEXT NOT NULL,
        fighter_b TEXT NOT NULL,
        winner TEXT,
        method TEXT,
        finish_round INTEGER,
        finish_time TEXT,
        weight_class TEXT,
        scheduled_rounds INTEGER,
        main_event INTEGER DEFAULT 0,
        title_fight INTEGER DEFAULT 0,
        catchweight INTEGER DEFAULT 0,
        missed_weight INTEGER DEFAULT 0,
        short_notice_replacement INTEGER DEFAULT 0,
        source_url TEXT UNIQUE,
        FOREIGN KEY(event_id) REFERENCES events(event_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fight_stats (
        fight_stat_id INTEGER PRIMARY KEY,
        fight_id INTEGER,
        fighter TEXT NOT NULL,
        opponent TEXT NOT NULL,
        knockdowns REAL,
        sig_str_landed REAL,
        sig_str_attempted REAL,
        total_str_landed REAL,
        total_str_attempted REAL,
        takedowns_landed REAL,
        takedowns_attempted REAL,
        submission_attempts REAL,
        reversals REAL,
        control_seconds REAL,
        head_landed REAL,
        body_landed REAL,
        leg_landed REAL,
        FOREIGN KEY(fight_id) REFERENCES fights(fight_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fighter_fight_history (
        history_id INTEGER PRIMARY KEY,
        fight_id INTEGER,
        fighter TEXT NOT NULL,
        fight_date TEXT NOT NULL,
        pre_fight_elo REAL,
        pre_weight_class_elo REAL,
        total_ufc_fights_before REAL,
        wins_before REAL,
        losses_before REAL,
        generated_at TEXT,
        FOREIGN KEY(fight_id) REFERENCES fights(fight_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scorecards (
        scorecard_id INTEGER PRIMARY KEY,
        event TEXT,
        fight_date TEXT,
        fighter_a TEXT,
        fighter_b TEXT,
        judge TEXT,
        round_1_a REAL,
        round_1_b REAL,
        round_2_a REAL,
        round_2_b REAL,
        round_3_a REAL,
        round_3_b REAL,
        round_4_a REAL,
        round_4_b REAL,
        round_5_a REAL,
        round_5_b REAL,
        total_a REAL,
        total_b REAL,
        decision_type TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_predictions (
        prediction_id INTEGER PRIMARY KEY,
        model_version TEXT,
        created_at TEXT,
        fight_date TEXT,
        fighter_a TEXT,
        fighter_b TEXT,
        predicted_winner TEXT,
        fighter_a_win_probability REAL,
        fighter_b_win_probability REAL,
        confidence_score REAL,
        confidence_tier TEXT,
        top_factors_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_results (
        backtest_id INTEGER PRIMARY KEY,
        model_version TEXT,
        created_at TEXT,
        train_start_date TEXT,
        train_end_date TEXT,
        test_start_date TEXT,
        test_end_date TEXT,
        accuracy REAL,
        log_loss REAL,
        brier_score REAL,
        roc_auc REAL,
        expected_calibration_error REAL,
        metrics_json TEXT
    )
    """,
]


def get_connection(
    db_path: str | Path | None = None,
    engine: str | None = None,
    cfg: Settings = settings,
) -> Any:
    """Open a SQLite or DuckDB connection."""

    selected_engine = (engine or cfg.database_engine).lower()
    selected_path = Path(db_path) if db_path is not None else cfg.database_path
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_engine == "duckdb":
        try:
            import duckdb
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("duckdb is not installed. Install project dependencies or use SQLite.") from exc
        return duckdb.connect(str(selected_path))
    if selected_engine != "sqlite":
        raise ValueError(f"Unsupported database engine: {selected_engine}")
    return sqlite3.connect(selected_path)


def initialize_database(
    db_path: str | Path | None = None,
    engine: str | None = None,
    cfg: Settings = settings,
) -> Any:
    """Create all normalized project tables and return an open connection."""

    conn = get_connection(db_path=db_path, engine=engine, cfg=cfg)
    for statement in SCHEMA_STATEMENTS:
        conn.execute(statement)
    try:
        conn.commit()
    except AttributeError:
        pass
    return conn


def write_dataframe(
    frame: pd.DataFrame,
    table_name: str,
    db_path: str | Path | None = None,
    engine: str | None = None,
    if_exists: str = "append",
) -> None:
    conn = initialize_database(db_path=db_path, engine=engine)
    try:
        frame.to_sql(table_name, conn, if_exists=if_exists, index=False)
    finally:
        conn.close()


def read_table(
    table_name: str,
    db_path: str | Path | None = None,
    engine: str | None = None,
) -> pd.DataFrame:
    conn = initialize_database(db_path=db_path, engine=engine)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
