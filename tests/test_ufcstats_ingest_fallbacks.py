from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ufc_predictor.cli import app
from ufc_predictor.ingest.ufcstats_scraper import FetchDiagnostics, UFCStatsScraper, UFCStatsScraperError


runner = CliRunner()
FIXTURE_IMPORTS = Path(__file__).parent / "fixtures" / "imports"


class FailingSession:
    def get(self, url, timeout):
        raise TimeoutError("network blocked")


def test_failed_live_fetch_has_diagnostics(tmp_path) -> None:
    scraper = UFCStatsScraper(cache_dir=tmp_path, retry_count=1, delay_seconds=0, backoff_base_seconds=0)
    scraper.session = FailingSession()

    diagnostics = scraper.fetch_diagnostics("https://www.ufcstats.com/statistics/events/completed?page=all")

    assert diagnostics.requested_url.endswith("completed?page=all")
    assert diagnostics.exception_type == "TimeoutError"
    assert diagnostics.cached_data_used is False
    assert diagnostics.attempts == 1


def test_strict_ingest_failure_is_clean_runtime_error(monkeypatch) -> None:
    diagnostics = FetchDiagnostics(
        requested_url="https://www.ufcstats.com/statistics/events/completed?page=all",
        status_code=None,
        exception_type="TimeoutError",
        exception_message="network blocked",
        body_preview="",
        cached_data_used=False,
        attempts=1,
    )

    def fail_run_to_csv(self, **kwargs):
        raise UFCStatsScraperError("Failed to fetch test URL.", diagnostics=diagnostics)

    monkeypatch.setattr(UFCStatsScraper, "run_to_csv", fail_run_to_csv)

    result = runner.invoke(app, ["ingest-ufcstats", "--no-sample-on-failure"])

    assert result.exit_code == 1
    assert "UFCStats ingestion failed" in result.output
    assert "Requested URL:" in result.output
    assert "Invalid value" not in result.output


def test_check_ufcstats_command_prints_diagnostics(monkeypatch) -> None:
    def fake_check():
        return FetchDiagnostics(
            requested_url="https://www.ufcstats.com/statistics/events/completed?page=all",
            status_code=200,
            body_preview="<html>ok</html>",
            cached_data_used=False,
            attempts=1,
        )

    monkeypatch.setattr("ufc_predictor.ingest.ufcstats_scraper.check_completed_events_page", fake_check)

    result = runner.invoke(app, ["check-ufcstats"])

    assert result.exit_code == 0
    assert "HTTP status code: 200" in result.output
    assert "<html>ok</html>" in result.output


def test_html_fallback_discovers_event_links_and_writes_fights(tmp_path, monkeypatch) -> None:
    html_path = tmp_path / "ufcstats_completed_events.html"
    html_path.write_text(
        '<html><body><a href="https://www.ufcstats.com/event-details/test-event">Sample Event</a></body></html>',
        encoding="utf-8",
    )
    output_dir = tmp_path / "raw"

    def fake_scrape_event(self, event_url):
        return {
            "event": {
                "name": "Sample UFCStats Event",
                "event_date": "January 1, 2024",
                "location": "Las Vegas, Nevada, USA",
                "source_url": event_url,
                "raw_details": "Date: January 1, 2024 | Location: Las Vegas, Nevada, USA",
            },
            "fights": [
                {
                    "event_name": "Sample UFCStats Event",
                    "fight_date": "2024-01-01",
                    "event_location": "Las Vegas, Nevada, USA",
                    "fighter_a": "Fighter A",
                    "fighter_b": "Fighter B",
                    "winner": "Fighter A",
                    "weight_class": "Lightweight",
                    "method": "Decision",
                    "finish_round": 3,
                    "finish_time": "5:00",
                    "scheduled_rounds": 3,
                    "source_url": "https://www.ufcstats.com/fight-details/test-fight",
                }
            ],
        }

    monkeypatch.setattr(UFCStatsScraper, "scrape_event", fake_scrape_event)

    result = runner.invoke(app, ["ingest-ufcstats", "--from-html", str(html_path), "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    fights = pd.read_csv(output_dir / "fights.csv")
    assert len(fights) == 1
    assert fights.loc[0, "fighter_a"] == "Fighter A"


def test_csv_import_fallback_writes_real_imports(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    output_dir = tmp_path / "raw"
    import_dir.mkdir()
    (import_dir / "fights.csv").write_text(
        "fight_id,fight_date,fighter_a,fighter_b,winner,weight_class,method\n"
        "1,2024-01-01,Imported A,Imported B,Imported B,Welterweight,KO/TKO\n",
        encoding="utf-8",
    )
    (import_dir / "fighters.csv").write_text(
        "name,stance,height_in,weight_lb,reach_in,date_of_birth\n"
        "Imported A,Orthodox,70,170,72,1990-01-01\n"
        "Imported B,Southpaw,71,170,73,1991-01-01\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["ingest-ufcstats", "--from-csv-imports", str(import_dir), "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert "Imported real raw CSV files" in result.output
    fights = pd.read_csv(output_dir / "fights.csv")
    assert len(fights) == 1
    assert fights.loc[0, "fighter_b"] == "Imported B"


def test_import_csv_command_works(tmp_path) -> None:
    output_dir = tmp_path / "raw"

    result = runner.invoke(app, ["import-csv", "--import-dir", str(FIXTURE_IMPORTS), "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert "Imported real raw CSV files" in result.output
    assert (output_dir / "source_metadata.json").exists()
    fights = pd.read_csv(output_dir / "fights.csv")
    fight_stats = pd.read_csv(output_dir / "fight_stats.csv")
    assert len(fights) == 8
    assert len(fight_stats) == 16


def test_data_summary_detects_imported_data(tmp_path) -> None:
    output_dir = tmp_path / "raw"
    import_result = runner.invoke(app, ["import-csv", "--import-dir", str(FIXTURE_IMPORTS), "--output-dir", str(output_dir)])
    assert import_result.exit_code == 0

    result = runner.invoke(app, ["data-summary", "--raw-dir", str(output_dir), "--scorecards-csv", str(output_dir / "scorecards.csv")])

    assert result.exit_code == 0
    assert "Data source: csv import" in result.output
    assert "Fights rows: 8" in result.output
    assert "Fight stats rows: 16" in result.output
    assert "Unique fighters: 8" in result.output


def test_build_dataset_uses_imported_data_and_warns_on_tiny_dataset(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    result = runner.invoke(
        app,
        [
            "build-dataset",
            "--imports-dir",
            str(FIXTURE_IMPORTS),
            "--fights-csv",
            str(raw_dir / "fights.csv"),
            "--fight-stats-csv",
            str(raw_dir / "fight_stats.csv"),
            "--fighters-csv",
            str(raw_dir / "fighters.csv"),
            "--scorecards-csv",
            str(raw_dir / "scorecards.csv"),
            "--output",
            str(processed_dir / "fight_dataset.parquet"),
        ],
    )

    assert result.exit_code == 0
    assert "Using real CSV imports" in result.output
    assert "Wrote 8 training rows" in result.output
    assert "Warning: only 8 training rows" in result.output
    assert pd.read_csv(raw_dir / "fights.csv").loc[0, "event_name"] == "Offline Import FC 1"
