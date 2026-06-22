from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ufc_predictor.card_update import (
    apply_card_update,
    create_card_update_template,
    import_card_csv,
    prepare_upcoming_card,
    update_after_card,
    validate_card_update,
)
from ufc_predictor.cli import _ablation_groups, app
from ufc_predictor.features.build_fight_dataset import build_fight_dataset
from ufc_predictor.fighter_profiles import (
    apply_fighter_profile_enrichment,
    enrich_fighter_profiles,
    import_fighter_profile_csv,
    validate_fighter_profile_enrichment,
)
from ufc_predictor.ingest.ufcstats_scraper import UFCStatsScraper


runner = CliRunner()


def _write_imports(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "fight_id": 1,
                "event_name": "Existing Event",
                "fight_date": "2024-01-01",
                "event_location": "Las Vegas, NV",
                "fighter_a": "Old A",
                "fighter_b": "Old B",
                "winner": "Old A",
                "weight_class": "Lightweight",
                "method": "Decision - Unanimous",
                "finish_round": 3,
                "finish_time": "5:00",
                "scheduled_rounds": 3,
                "main_event": 0,
                "source_url": "",
            }
        ]
    ).to_csv(path / "fights.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": "Old A",
                "stance": "Orthodox",
                "height_in": 70,
                "weight_lb": 155,
                "reach_in": 72,
                "date_of_birth": "1990-01-01",
                "record": "",
                "source_url": "",
            },
            {
                "name": "Old B",
                "stance": "",
                "height_in": "",
                "weight_lb": 155,
                "reach_in": "",
                "date_of_birth": "",
                "record": "",
                "source_url": "",
            },
        ]
    ).to_csv(path / "fighters.csv", index=False)
    pd.DataFrame(columns=["fight_id", "fighter", "opponent", "knockdowns"]).to_csv(path / "fight_stats.csv", index=False)
    pd.DataFrame(columns=["event", "fight_date", "fighter_a", "fighter_b", "weight_class", "event_location", "main_event", "title_fight", "scheduled_rounds"]).to_csv(
        path / "fight_enrichment.csv",
        index=False,
    )


def _card_csv(path: Path, date: str = "2024-02-01") -> Path:
    frame = pd.DataFrame(
        [
            {
                "Fight Date": date,
                "Event": "UFC Test Card",
                "Location": "Boston, MA",
                "Red Fighter": "New A",
                "Blue Fighter": "New B",
                "Winner": "New A",
                "Division": "Welterweight",
                "Method": "Decision - Unanimous",
                "Round": 3,
                "Time": "5:00",
                "Is Main Event": 1,
                "Is Title Fight": 0,
                "Score Cards": "29-28, 29-28, 30-27",
                "Card Type": "Decision",
                "Red Fighter Moneyline Odds": -150,
                "Blue Fighter Moneyline Odds": 130,
            }
        ]
    )
    output = path / "card.csv"
    frame.to_csv(output, index=False)
    return output


def _manual_completed_card_csv(path: Path, date: str = "2024-02-01", winner: str = "New A") -> Path:
    frame = pd.DataFrame(
        [
            {
                "event_name": "Manual UFC Test Card",
                "event_date": date,
                "fighter_a": "New A",
                "fighter_b": "New B",
                "winner": winner,
                "method": "Decision - Unanimous",
                "round": 3,
                "time": "5:00",
                "weight_class": "Welterweight",
                "scheduled_rounds": 3,
                "main_event": 1,
                "title_fight": 0,
                "fighter_a_kd": 1,
                "fighter_b_kd": 0,
                "fighter_a_sig_str_landed": 42,
                "fighter_a_sig_str_attempted": 91,
                "fighter_b_sig_str_landed": 31,
                "fighter_b_sig_str_attempted": 88,
                "fighter_a_total_str_landed": 60,
                "fighter_a_total_str_attempted": 112,
                "fighter_b_total_str_landed": 45,
                "fighter_b_total_str_attempted": 105,
                "fighter_a_td_landed": 2,
                "fighter_a_td_attempted": 5,
                "fighter_b_td_landed": 0,
                "fighter_b_td_attempted": 2,
                "fighter_a_sub_attempts": 1,
                "fighter_b_sub_attempts": 0,
                "fighter_a_control_time": "3:12",
                "fighter_b_control_time": "0:15",
                "fighter_a_odds": -150,
                "fighter_b_odds": 130,
            }
        ]
    )
    output = path / "manual_completed_card.csv"
    frame.to_csv(output, index=False)
    return output


def _ufcstats_event_html() -> str:
    return """
    <html>
      <head><title>UFCStats - UFC Test</title></head>
      <body>
        <span class="b-content__title-highlight">UFC Test: Alpha vs Beta</span>
        <li class="b-list__box-list-item">Date: February 01, 2024</li>
        <li class="b-list__box-list-item">Location: Boston, Massachusetts, USA</li>
        <table class="b-fight-details__table">
          <tr class="b-fight-details__table-row b-fight-details__table-row__hover" data-link="http://ufcstats.com/fight-details/abc">
            <td><p>win</p><p>loss</p></td>
            <td>
              <p><a href="http://ufcstats.com/fighter-details/a1">Alpha Fighter</a></p>
              <p><a href="http://ufcstats.com/fighter-details/b1">Beta Fighter</a></p>
            </td>
            <td><p>1</p><p>0</p></td>
            <td><p>12 of 30</p><p>8 of 25</p></td>
            <td><p>2 of 4</p><p>0 of 1</p></td>
            <td><p>1</p><p>0</p></td>
            <td>Lightweight</td>
            <td>Decision - Unanimous</td>
            <td>3</td>
            <td>5:00</td>
          </tr>
        </table>
      </body>
    </html>
    """


def _write_ufcstats_html(path: Path) -> Path:
    output = path / "ufcstats_event.html"
    output.write_text(_ufcstats_event_html(), encoding="utf-8")
    return output


def test_update_after_card_writes_staging_files_only(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    card = _card_csv(tmp_path)
    staging = tmp_path / "staging"

    report = update_after_card(str(card), source="espn", staging_dir=staging)

    assert (staging / "new_fights.csv").exists()
    assert (staging / "new_fight_stats.csv").exists()
    assert (staging / "new_scorecards.csv").exists()
    assert report.rows_written["new_fights.csv"] == 1
    existing = pd.read_csv(imports / "fights.csv")
    assert len(existing) == 1


def test_create_card_update_template(tmp_path: Path) -> None:
    output = tmp_path / "manual_completed_card.csv"

    frame, path = create_card_update_template("UFC Example", "2025-01-01", output)

    assert path == output
    assert output.exists()
    assert list(frame.columns)[0:4] == ["event_name", "event_date", "fighter_a", "fighter_b"]
    saved = pd.read_csv(output)
    assert saved.loc[0, "event_name"] == "UFC Example"
    assert saved.loc[0, "event_date"] == "2025-01-01"


def test_import_card_csv_writes_staging_files_and_manual_odds(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    manual = _manual_completed_card_csv(tmp_path)

    report = import_card_csv(manual, staging_dir=staging)

    fights = pd.read_csv(staging / "new_fights.csv")
    stats = pd.read_csv(staging / "new_fight_stats.csv")
    odds = pd.read_csv(staging / "new_odds.csv")
    assert report.rows_written["new_fights.csv"] == 1
    assert report.rows_written["new_fight_stats.csv"] == 2
    assert report.rows_written["new_odds.csv"] == 1
    assert fights.loc[0, "event_name"] == "Manual UFC Test Card"
    assert len(stats) == 2
    assert odds.loc[0, "sportsbook"] == "manual_card_csv"


def test_import_card_csv_rejects_missing_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad_card.csv"
    pd.DataFrame([{"event_name": "Bad"}]).to_csv(bad, index=False)

    result = runner.invoke(app, ["import-card-csv", "--file", str(bad), "--staging-dir", str(tmp_path / "staging")])

    assert result.exit_code == 1
    assert "missing required columns" in result.output


def test_update_after_card_stages_nonzero_rows_from_ufcstats_fixture_html(tmp_path: Path) -> None:
    html = _write_ufcstats_html(tmp_path)
    staging = tmp_path / "staging"

    report = update_after_card(str(html), source="ufcstats", event_html=html, staging_dir=staging)

    fights = pd.read_csv(staging / "new_fights.csv")
    stats = pd.read_csv(staging / "new_fight_stats.csv")
    fighters = pd.read_csv(staging / "new_fighters.csv")
    assert report.rows_written["new_fights.csv"] == 1
    assert fights.loc[0, "event_name"] == "UFC Test: Alpha vs Beta"
    assert fights.loc[0, "fighter_a"] == "Alpha Fighter"
    assert fights.loc[0, "fighter_b"] == "Beta Fighter"
    assert fights.loc[0, "winner"] == "Alpha Fighter"
    assert fights.loc[0, "method"] == "Decision - Unanimous"
    assert str(fights.loc[0, "finish_round"]) == "3"
    assert fights.loc[0, "finish_time"] == "5:00"
    assert fights.loc[0, "weight_class"] == "Lightweight"
    assert fights.loc[0, "source_url"] == "http://ufcstats.com/fight-details/abc"
    assert len(stats) == 2
    assert len(fighters) == 2


def test_zero_parsed_ufcstats_page_has_clear_diagnostics(tmp_path: Path) -> None:
    html = tmp_path / "challenge.html"
    html.write_text("<html><title>Loading...</title><body>Checking your browser...</body></html>", encoding="utf-8")

    report = update_after_card(str(html), source="ufcstats", event_html=html, staging_dir=tmp_path / "staging")

    assert report.rows_written["new_fights.csv"] == 0
    assert report.status == "blocked_by_browser_challenge"
    assert report.diagnostics["browser_challenge_detected"] is True
    assert "challenge" in report.diagnostics["parse_reason"].lower()


def test_browser_challenge_cli_exits_gracefully_and_recommends_manual_csv(tmp_path: Path) -> None:
    html = tmp_path / "challenge.html"
    html.write_text("<html><title>Loading...</title><body>Checking your browser... This site requires JavaScript.</body></html>", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "update-after-card",
            "--event-html",
            str(html),
            "--source",
            "ufcstats",
            "--staging-dir",
            str(tmp_path / "staging"),
        ],
    )

    assert result.exit_code == 1
    assert "blocked_by_browser_challenge" in result.output
    assert "create-card-update-template" in result.output
    report = json.loads((tmp_path / "staging" / "update_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked_by_browser_challenge"


def test_raw_html_save_and_load_flow(monkeypatch, tmp_path: Path) -> None:
    from ufc_predictor import card_update

    class FakeAdapter(card_update.UFCStatsAdapter):
        def fetch_event_html(self, event_url: str):
            html = _ufcstats_event_html()
            return html, self._diagnostics(event_url, html, 200, event_url, "network")

    monkeypatch.setattr(card_update, "UFCStatsAdapter", FakeAdapter)
    staging = tmp_path / "staging"

    fetched = update_after_card("http://ufcstats.com/event-details/test", source="ufcstats", staging_dir=staging, save_raw_html=True)
    loaded = update_after_card(str(staging / "raw_event_page.html"), source="ufcstats", event_html=staging / "raw_event_page.html", staging_dir=tmp_path / "staging2")

    assert fetched.rows_written["new_fights.csv"] == 1
    assert (staging / "raw_event_page.html").exists()
    assert loaded.rows_written["new_fights.csv"] == 1


def test_validate_card_update_catches_duplicates(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    staging = tmp_path / "staging"
    update_after_card(str(_card_csv(tmp_path, "2024-01-01")), source="espn", staging_dir=staging)
    fights = pd.read_csv(staging / "new_fights.csv")
    fights.loc[0, "fighter_a"] = "Old A"
    fights.loc[0, "fighter_b"] = "Old B"
    fights.to_csv(staging / "new_fights.csv", index=False)

    report = validate_card_update(staging_dir=staging, imports_dir=imports)

    assert not report.ok
    assert any("already present" in error for error in report.errors)


def test_validate_card_update_catches_bad_manual_winner(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    staging = tmp_path / "staging"
    import_card_csv(_manual_completed_card_csv(tmp_path, winner="Someone Else"), staging_dir=staging)

    report = validate_card_update(staging_dir=staging, imports_dir=imports)

    assert not report.ok
    assert any("Winner is not one of the two fighters" in error for error in report.errors)


def test_validate_card_update_catches_invalid_manual_odds(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    staging = tmp_path / "staging"
    manual = _manual_completed_card_csv(tmp_path)
    frame = pd.read_csv(manual)
    frame.loc[0, "fighter_a_odds"] = 0
    frame.to_csv(manual, index=False)
    import_card_csv(manual, staging_dir=staging)

    report = validate_card_update(staging_dir=staging, imports_dir=imports)

    assert not report.ok
    assert any("invalid American odds" in error for error in report.errors)


def test_existing_event_still_stages_before_validation_catches_duplicate(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    html = _write_ufcstats_html(tmp_path)
    staging = tmp_path / "staging"
    update_after_card(str(html), source="ufcstats", event_html=html, staging_dir=staging)
    fights = pd.read_csv(staging / "new_fights.csv")
    fights.loc[0, "fight_date"] = "2024-01-01"
    fights.loc[0, "fighter_a"] = "Old A"
    fights.loc[0, "fighter_b"] = "Old B"
    fights.to_csv(staging / "new_fights.csv", index=False)

    validation = validate_card_update(staging_dir=staging, imports_dir=imports)

    assert len(pd.read_csv(staging / "new_fights.csv")) == 1
    assert not validation.ok
    assert any("already present" in error for error in validation.errors)


def test_validate_card_update_catches_future_fight_dates(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    staging = tmp_path / "staging"
    update_after_card(str(_card_csv(tmp_path, "2999-01-01")), source="espn", staging_dir=staging)

    report = validate_card_update(staging_dir=staging, imports_dir=imports)

    assert not report.ok
    assert any("future fight dates" in error for error in report.errors)


def test_validate_card_update_catches_future_manual_fight_dates(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    staging = tmp_path / "staging"
    import_card_csv(_manual_completed_card_csv(tmp_path, "2999-01-01"), staging_dir=staging)

    report = validate_card_update(staging_dir=staging, imports_dir=imports)

    assert not report.ok
    assert any("future fight dates" in error for error in report.errors)


def test_apply_card_update_creates_backup_and_does_not_duplicate(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    raw = tmp_path / "raw"
    _write_imports(imports)
    staging = tmp_path / "staging"
    update_after_card(str(_card_csv(tmp_path)), source="espn", staging_dir=staging)
    validation = validate_card_update(staging_dir=staging, imports_dir=imports)
    assert validation.ok

    first = apply_card_update(staging_dir=staging, imports_dir=imports, raw_dir=raw, backup_root=tmp_path / "backups")
    assert first.rows_added["fights"] == 1
    assert first.backup_dir.exists()
    fights = pd.read_csv(imports / "fights.csv")
    assert len(fights) == 2

    duplicate_report = validate_card_update(staging_dir=staging, imports_dir=imports)
    assert not duplicate_report.ok
    assert any("already present" in error for error in duplicate_report.errors)


def test_apply_card_update_adds_manual_rows_without_duplication(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    raw = tmp_path / "raw"
    _write_imports(imports)
    staging = tmp_path / "staging"
    import_card_csv(_manual_completed_card_csv(tmp_path), staging_dir=staging)
    validation = validate_card_update(staging_dir=staging, imports_dir=imports)
    assert validation.ok

    report = apply_card_update(staging_dir=staging, imports_dir=imports, raw_dir=raw, backup_root=tmp_path / "backups")

    fights = pd.read_csv(imports / "fights.csv")
    stats = pd.read_csv(imports / "fight_stats.csv")
    odds = pd.read_csv(imports / "odds.csv")
    assert report.backup_dir.exists()
    assert report.rows_added["fights"] == 1
    assert report.rows_added["fight_stats"] == 2
    assert report.rows_added["odds"] == 1
    assert len(fights) == 2
    assert len(stats) == 2
    assert len(odds) == 1


def test_fighter_profile_enrichment_improves_missing_reach_without_overwriting(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    sources = tmp_path / "sources"
    _write_imports(imports)
    sources.mkdir()
    pd.DataFrame(
        [
            {"name": "Old A", "reach": 99, "height": 75, "stance": "Southpaw"},
            {"name": "Old B", "reach": 71, "height": 69, "stance": "Switch"},
        ]
    ).to_csv(sources / "fighters.csv", index=False)

    proposal = enrich_fighter_profiles(
        source="local",
        fighters_path=imports / "fighters.csv",
        source_dir=sources,
        output_path=imports / "fighter_profile_enrichment.csv",
        report_path=tmp_path / "profile_report.json",
    )
    validation = validate_fighter_profile_enrichment(
        fighters_path=imports / "fighters.csv",
        enrichment_path=imports / "fighter_profile_enrichment.csv",
        report_path=tmp_path / "validation.json",
    )
    report = apply_fighter_profile_enrichment(
        fighters_path=imports / "fighters.csv",
        enrichment_path=imports / "fighter_profile_enrichment.csv",
        backup_root=tmp_path / "backups",
    )

    fighters = pd.read_csv(imports / "fighters.csv")
    assert proposal.coverage["enrichment_rows_with_reach"] == 2
    assert validation.matched_fighters == 2
    assert report.fields_updated["reach_in"] == 1
    assert report.backup_dir.exists()
    assert fighters.loc[fighters["name"] == "Old A", "reach_in"].iloc[0] == 72
    assert fighters.loc[fighters["name"] == "Old B", "reach_in"].iloc[0] == 71


def test_fighter_profile_enrichment_does_not_overwrite_unless_requested(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    sources = tmp_path / "sources"
    _write_imports(imports)
    sources.mkdir()
    pd.DataFrame([{"Full Name": "Old A", "Reach": 99, "Ht.": 6.00, "Stance": "Southpaw"}]).to_csv(sources / "fighters.csv", index=False)
    enrich_fighter_profiles(source="local", fighters_path=imports / "fighters.csv", source_dir=sources, output_path=imports / "fighter_profile_enrichment.csv")

    apply_fighter_profile_enrichment(fighters_path=imports / "fighters.csv", enrichment_path=imports / "fighter_profile_enrichment.csv", backup_root=tmp_path / "backups")
    fighters = pd.read_csv(imports / "fighters.csv")
    assert fighters.loc[fighters["name"] == "Old A", "reach_in"].iloc[0] == 72

    apply_fighter_profile_enrichment(
        fighters_path=imports / "fighters.csv",
        enrichment_path=imports / "fighter_profile_enrichment.csv",
        backup_root=tmp_path / "backups2",
        overwrite=True,
    )
    fighters = pd.read_csv(imports / "fighters.csv")
    assert fighters.loc[fighters["name"] == "Old A", "reach_in"].iloc[0] == 99


def test_normalized_name_matching_for_profile_enrichment(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    sources = tmp_path / "sources"
    _write_imports(imports)
    fighters = pd.read_csv(imports / "fighters.csv")
    fighters.loc[fighters["name"] == "Old B", "name"] = "Jose Aldo Jr."
    fighters.to_csv(imports / "fighters.csv", index=False)
    sources.mkdir()
    pd.DataFrame([{"Full Name": "José Aldo", "Reach": 70, "Ht.": 5.70, "Stance": "Orthodox"}]).to_csv(sources / "fighters.csv", index=False)

    enrich_fighter_profiles(source="local", fighters_path=imports / "fighters.csv", source_dir=sources, output_path=imports / "fighter_profile_enrichment.csv")
    report = apply_fighter_profile_enrichment(fighters_path=imports / "fighters.csv", enrichment_path=imports / "fighter_profile_enrichment.csv", backup_root=tmp_path / "backups")

    fighters = pd.read_csv(imports / "fighters.csv")
    assert report.fields_updated["reach_in"] == 1
    assert fighters.loc[fighters["name"] == "Jose Aldo Jr.", "reach_in"].iloc[0] == 70


def test_manual_fighter_profile_csv_import(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    manual = tmp_path / "manual_fighter_profiles.csv"
    pd.DataFrame(
        [
            {
                "fighter_name": "Old B",
                "height": "5'9",
                "weight": 155,
                "reach": 71,
                "stance": "Switch",
                "dob": "1991-02-03",
                "weight_class": "Lightweight",
                "source": "manual",
            }
        ]
    ).to_csv(manual, index=False)

    frame, path = import_fighter_profile_csv(manual, output_path=imports / "fighter_profile_enrichment.csv", append=False)
    report = apply_fighter_profile_enrichment(fighters_path=imports / "fighters.csv", enrichment_path=path, backup_root=tmp_path / "backups")

    assert len(frame) == 1
    assert report.fields_updated["reach_in"] == 1


def test_ufcstats_fighter_page_parser_fixture(monkeypatch) -> None:
    html = """
    <html><body>
      <span class="b-content__title-highlight">Fixture Fighter</span>
      <span class="b-content__title-record">Record: 10-1-0</span>
      <li class="b-list__box-list-item">Height: 5' 11"</li>
      <li class="b-list__box-list-item">Weight: 155 lbs.</li>
      <li class="b-list__box-list-item">Reach: 72"</li>
      <li class="b-list__box-list-item">STANCE: Southpaw</li>
      <li class="b-list__box-list-item">DOB: Jan 01, 1990</li>
    </body></html>
    """
    scraper = UFCStatsScraper()
    monkeypatch.setattr(scraper, "fetch", lambda url: html)

    profile = scraper.scrape_fighter("http://ufcstats.com/fighter-details/test")

    assert profile["name"] == "Fixture Fighter"
    assert profile["height_in"] == "5' 11"
    assert profile["weight_lb"] == "155"
    assert profile["reach_in"] == "72"
    assert profile["stance"] == "Southpaw"


def test_reach_feature_creation_and_missing_indicators() -> None:
    fights = pd.DataFrame(
        [
            {
                "fight_id": 1,
                "event_name": "Test",
                "fight_date": "2024-01-01",
                "fighter_a": "Reach A",
                "fighter_b": "Reach B",
                "winner": "Reach A",
                "weight_class": "Lightweight",
                "scheduled_rounds": 3,
            }
        ]
    )
    fighters = pd.DataFrame(
        [
            {"name": "Reach A", "height_in": 70, "reach_in": 72, "stance": "Orthodox"},
            {"name": "Reach B", "height_in": 69, "reach_in": "", "stance": "Southpaw"},
        ]
    )

    dataset = build_fight_dataset(fights, fighters=fighters)

    row = dataset.iloc[0]
    assert row["fighter_a_reach"] == 72
    assert row["fighter_b_reach_missing"] == 1
    assert row["both_reach_missing"] == 0
    assert "height_reach_interaction" in dataset.columns


def test_physical_attributes_ablation_group_exists() -> None:
    groups = _ablation_groups()

    assert "physical_attributes" in groups
    assert any("reach" in pattern for pattern in groups["physical_attributes"])


def test_rebuild_after_update_stops_on_failed_validation(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate-card-update", "--staging-dir", str(tmp_path / "missing"), "--imports-dir", str(tmp_path / "imports")])

    assert result.exit_code == 1
    assert "No staged fights found" in result.output


def test_prepare_upcoming_card_creates_expected_template(tmp_path: Path) -> None:
    card = _card_csv(tmp_path)
    output = tmp_path / "upcoming.csv"

    frame, path = prepare_upcoming_card(str(card), output_path=output)

    assert path == output
    assert len(frame) == 1
    assert list(frame.columns) == [
        "fighter_a",
        "fighter_b",
        "date",
        "weight_class",
        "scheduled_rounds",
        "fighter_a_odds",
        "fighter_b_odds",
        "main_event",
        "title_fight",
    ]


def test_predict_card_works_with_manual_odds(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    raw = tmp_path / "raw"
    _write_imports(imports)
    card = pd.DataFrame(
        [
            {
                "fighter_a": "Old A",
                "fighter_b": "Old B",
                "date": "2025-01-01",
                "weight_class": "Lightweight",
                "scheduled_rounds": 3,
                "fighter_a_odds": -150,
                "fighter_b_odds": 130,
                "main_event": 0,
                "title_fight": 0,
            }
        ]
    )
    card_path = tmp_path / "card_predictions.csv"
    card.to_csv(card_path, index=False)
    for name in ["fights.csv", "fighters.csv", "fight_stats.csv"]:
        (raw / name).parent.mkdir(parents=True, exist_ok=True)
        (raw / name).write_text((imports / name).read_text(encoding="utf-8"), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "predict-card",
            "--file",
            str(card_path),
            "--model-mode",
            "market-aware",
            "--show-value-analysis",
            "--fights-csv",
            str(raw / "fights.csv"),
            "--fighters-csv",
            str(raw / "fighters.csv"),
            "--fight-stats-csv",
            str(raw / "fight_stats.csv"),
            "--scorecards-csv",
            str(raw / "missing_scorecards.csv"),
            "--odds-csv",
            str(raw / "missing_odds.csv"),
            "--model-path",
            str(tmp_path / "missing_model.pkl"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["predictions"][0]["odds_source"] == "manual_input"
    assert "value_analysis_note" in payload["predictions"][0]


def test_no_post_fight_leakage_into_event_enrichment(tmp_path: Path) -> None:
    imports = tmp_path / "imports"
    _write_imports(imports)
    staging = tmp_path / "staging"
    update_after_card(str(_card_csv(tmp_path)), source="espn", staging_dir=staging)
    enrichment = pd.read_csv(staging / "new_event_enrichment.csv")
    enrichment["winner"] = "New A"
    enrichment.to_csv(staging / "new_event_enrichment.csv", index=False)

    report = validate_card_update(staging_dir=staging, imports_dir=imports)

    assert not report.ok
    assert any("post-fight result columns" in error for error in report.errors)
