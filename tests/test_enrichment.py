from __future__ import annotations

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from ufc_predictor.cli import app
from ufc_predictor.enrichment import auto_enrich, import_enrichment_csv, merge_enrichment_into_fights
from ufc_predictor.import_validation import validate_import_directory
from ufc_predictor.models.evaluate import rolling_backtest
from ufc_predictor.reporting import build_data_quality_coverage


runner = CliRunner()


def _write_minimal_imports(path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "fight_id": 1,
                "event_name": "UFC Test 1",
                "fight_date": "2020-01-01",
                "event_location": "",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "winner": "Alice Alpha",
                "method": "Decision",
                "finish_round": 3,
                "finish_time": "5:00",
                "weight_class": "Unknown",
                "scheduled_rounds": 3,
                "main_event": 0,
                "title_fight": 0,
            }
        ]
    ).to_csv(path / "fights.csv", index=False)
    pd.DataFrame([{"name": "Alice Alpha"}, {"name": "Beth Beta"}]).to_csv(path / "fighters.csv", index=False)
    pd.DataFrame(
        [
            {"fight_id": 1, "fighter": "Alice Alpha", "opponent": "Beth Beta"},
            {"fight_id": 1, "fighter": "Beth Beta", "opponent": "Alice Alpha"},
        ]
    ).to_csv(path / "fight_stats.csv", index=False)


def test_import_enrichment_merges_fight_context_fields(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    _write_minimal_imports(import_dir)
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Test 1",
                "fighter_a": "Beth Beta",
                "fighter_b": "Alice Alpha",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas, Nevada, USA",
                "main_event": 1,
                "title_fight": 1,
                "scheduled_rounds": 5,
            }
        ]
    ).to_csv(import_dir / "fight_enrichment.csv", index=False)

    report = import_enrichment_csv(
        enrichment_path=import_dir / "fight_enrichment.csv",
        fights_path=import_dir / "fights.csv",
    )

    fights = pd.read_csv(import_dir / "fights.csv")
    assert report.matched_rows == 1
    assert fights.loc[0, "weight_class"] == "Lightweight"
    assert fights.loc[0, "event_location"] == "Las Vegas, Nevada, USA"
    assert fights.loc[0, "main_event"] == 1
    assert fights.loc[0, "title_fight"] == 1
    assert fights.loc[0, "scheduled_rounds"] == 5
    assert validate_import_directory(import_dir).ok


def test_import_enrichment_command(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    _write_minimal_imports(import_dir)
    (import_dir / "fight_enrichment.csv").write_text(
        "fight_date,event,fighter_a,fighter_b,weight_class,event_location,main_event,title_fight,scheduled_rounds\n"
        "2020-01-01,UFC Test 1,Alice Alpha,Beth Beta,Welterweight,New York,0,0,3\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "import-enrichment",
            "--enrichment-path",
            str(import_dir / "fight_enrichment.csv"),
            "--fights-path",
            str(import_dir / "fights.csv"),
        ],
    )

    assert result.exit_code == 0
    assert "Enrichment file read:" in result.output
    assert "Fights file read:" in result.output
    assert "Fights file written:" in result.output
    assert "Fields updated:" in result.output
    assert "Enrichment format: fight-level" in result.output
    assert "Matched enrichment rows: 1" in result.output
    assert "weight_class: 1" in result.output


def test_import_enrichment_handles_string_destination_columns_and_integer_flags(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    import_dir.mkdir(parents=True)
    fights = pd.DataFrame(
        {
            "fight_id": [1],
            "event_name": ["UFC Test 1"],
            "fight_date": ["2020-01-01"],
            "event_location": pd.Series([""], dtype="string"),
            "fighter_a": ["Alice Alpha"],
            "fighter_b": ["Beth Beta"],
            "winner": ["Alice Alpha"],
            "weight_class": pd.Series(["Unknown"], dtype="string"),
            "scheduled_rounds": pd.Series([""], dtype="string"),
            "main_event": pd.Series([""], dtype="string"),
            "title_fight": pd.Series([""], dtype="string"),
        }
    )
    enrichment = pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Test 1",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas",
                "main_event": 0,
                "title_fight": 1,
                "scheduled_rounds": 5,
            }
        ]
    )
    fights.to_csv(import_dir / "fights.csv", index=False)
    enrichment.to_csv(import_dir / "fight_enrichment.csv", index=False)

    report = import_enrichment_csv(import_dir / "fight_enrichment.csv", import_dir / "fights.csv")

    output = pd.read_csv(import_dir / "fights.csv")
    assert report.matched_rows == 1
    assert output.loc[0, "main_event"] == 0
    assert output.loc[0, "title_fight"] == 1
    assert output.loc[0, "scheduled_rounds"] == 5
    assert output.loc[0, "weight_class"] == "Lightweight"
    assert output.loc[0, "event_location"] == "Las Vegas"


def test_merge_enrichment_coerces_arrow_string_like_columns_before_assignment(tmp_path) -> None:
    fights = pd.DataFrame(
        {
            "event_name": ["UFC Test 1"],
            "fight_date": ["2020-01-01"],
            "fighter_a": ["Alice Alpha"],
            "fighter_b": ["Beth Beta"],
            "winner": ["Alice Alpha"],
            "weight_class": pd.Series(["Unknown"], dtype="string"),
            "event_location": pd.Series([""], dtype="string"),
            "main_event": pd.Series([""], dtype="string"),
            "title_fight": pd.Series([""], dtype="string"),
            "scheduled_rounds": pd.Series([""], dtype="string"),
        }
    )
    enrichment = pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Test 1",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas",
                "main_event": 0,
                "title_fight": 1,
                "scheduled_rounds": 5,
            }
        ]
    )

    merged, report = merge_enrichment_into_fights(
        fights=fights,
        enrichment=enrichment,
        fights_path=tmp_path / "fights.csv",
        enrichment_path=tmp_path / "fight_enrichment.csv",
        output_path=tmp_path / "fights.csv",
    )

    assert report.matched_rows == 1
    assert merged.loc[0, "main_event"] == 0
    assert merged.loc[0, "title_fight"] == 1
    assert merged.loc[0, "scheduled_rounds"] == 5


def test_build_enrichment_template_command(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    output = tmp_path / "fight_enrichment_template.csv"
    _write_minimal_imports(import_dir)

    result = runner.invoke(
        app,
        [
            "build-enrichment-template",
            "--fights-path",
            str(import_dir / "fights.csv"),
            "--output-path",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote enrichment template with 1 fight rows" in result.output
    template = pd.read_csv(output)
    assert list(template.columns) == [
        "fight_date",
        "event",
        "fighter_a",
        "fighter_b",
        "weight_class",
        "event_location",
        "main_event",
        "title_fight",
        "scheduled_rounds",
    ]
    assert template.loc[0, "event"] == "UFC Test 1"
    assert template.loc[0, "weight_class"] == "Unknown"
    assert pd.isna(template.loc[0, "main_event"])


def test_enrichment_summary_command(tmp_path) -> None:
    path = tmp_path / "fight_enrichment_template.csv"
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Test 1",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas",
                "main_event": 1,
                "title_fight": 0,
                "scheduled_rounds": 5,
            },
            {
                "fight_date": "2020-01-01",
                "event": "UFC Test 1",
                "fighter_a": "Cara Delta",
                "fighter_b": "Dana Echo",
                "weight_class": "Unknown",
                "event_location": "",
                "main_event": 0,
                "title_fight": 0,
                "scheduled_rounds": 3,
            },
        ]
    ).to_csv(path, index=False)

    result = runner.invoke(app, ["enrichment-summary", "--path", str(path)])

    assert result.exit_code == 0
    assert "Total fights: 2" in result.output
    assert "Known weight_class: 1/2 (50.0%)" in result.output
    assert "Known event_location: 1/2 (50.0%)" in result.output
    assert "Known main_event: 2/2 (100.0%)" in result.output
    assert "Known scheduled_rounds: 2/2 (100.0%)" in result.output


def test_auto_enrich_creates_file_and_infers_main_event_and_title(tmp_path) -> None:
    template = tmp_path / "fight_enrichment_template.csv"
    output = tmp_path / "fight_enrichment.csv"
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Fight Night: Alpha vs. Beta",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Unknown",
                "event_location": "",
                "main_event": "",
                "title_fight": "",
                "scheduled_rounds": 5,
            },
            {
                "fight_date": "2020-01-01",
                "event": "UFC Fight Night: Alpha vs. Beta",
                "fighter_a": "Cara Delta",
                "fighter_b": "Dana Echo",
                "weight_class": "Unknown",
                "event_location": "",
                "main_event": "",
                "title_fight": "",
                "scheduled_rounds": 3,
            },
        ]
    ).to_csv(template, index=False)

    result = runner.invoke(app, ["auto-enrich", "--template-path", str(template), "--output-path", str(output)])

    assert result.exit_code == 0
    assert output.exists()
    enriched = pd.read_csv(output)
    assert enriched.loc[0, "main_event"] == 1
    assert enriched.loc[1, "main_event"] == 0
    assert enriched.loc[0, "title_fight"] == 1
    assert enriched.loc[1, "title_fight"] == 0
    assert enriched.loc[0, "weight_class"] == "Unknown"
    assert pd.isna(enriched.loc[0, "event_location"])


def test_auto_enrich_merges_external_sources(tmp_path) -> None:
    template = tmp_path / "fight_enrichment_template.csv"
    output = tmp_path / "fight_enrichment.csv"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Fight Night: Alpha vs. Beta",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Unknown",
                "event_location": "",
                "main_event": "",
                "title_fight": "",
                "scheduled_rounds": 5,
            }
        ]
    ).to_csv(template, index=False)
    pd.DataFrame(
        [
            {
                "event": "Different Event Name",
                "date": "2020-01-01",
                "location": "Las Vegas, Nevada, USA",
                "weight_class": "Lightweight",
                "bout": "Alice Alpha vs Beth Beta",
                "main_event": 1,
            }
        ]
    ).to_csv(source_dir / "external.csv", index=False)

    frame, report = auto_enrich(template_path=template, output_path=output, source_dir=source_dir)

    assert output.exists()
    assert report.external_sources[0].matched_rows == 1
    assert report.external_sources[0].unmatched_rows == 0
    assert frame.loc[0, "weight_class"] == "Lightweight"
    assert frame.loc[0, "event_location"] == "Las Vegas, Nevada, USA"
    assert frame.loc[0, "main_event"] == 1


def test_auto_enrich_discovers_nested_sources_and_joins_event_id_metadata(tmp_path) -> None:
    template = tmp_path / "fight_enrichment_template.csv"
    output = tmp_path / "fight_enrichment.csv"
    source_dir = tmp_path / "sources"
    nested = source_dir / "ufc_2025_dataset" / "data"
    nested.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Test: Alpha vs. Beta",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Unknown",
                "event_location": "",
                "main_event": "",
                "title_fight": "",
                "scheduled_rounds": "",
            }
        ]
    ).to_csv(template, index=False)
    pd.DataFrame(
        [
            {
                "Event_Id": "E1",
                "Name": "UFC Test: Alpha vs. Beta",
                "Date": "2020-01-01",
                "Location": "Las Vegas, Nevada, USA",
            }
        ]
    ).to_csv(nested / "Events.csv", index=False)
    pd.DataFrame(
        [
            {
                "Event_Id": "E1",
                "Fighter_1": "Alice Alpha",
                "Fighter_2": "Beth Beta",
                "Weight_Class": "Lightweight",
                "Time Format": "5 Rnd (5-5-5-5-5)",
            }
        ]
    ).to_csv(nested / "Fights.csv", index=False)

    frame, report = auto_enrich(template_path=template, output_path=output, source_dir=source_dir)

    assert output.exists()
    assert frame.loc[0, "weight_class"] == "Lightweight"
    assert frame.loc[0, "event_location"] == "Las Vegas, Nevada, USA"
    assert frame.loc[0, "scheduled_rounds"] == 5
    fight_source = next(source for source in report.external_sources if source.path.name == "Fights.csv")
    assert fight_source.matched_rows == 1
    assert "Weight_Class" in fight_source.columns
    assert "weight_class" in fight_source.usable_fields


def test_enrichment_sources_command_recursively_detects_source_types(tmp_path) -> None:
    source_dir = tmp_path / "enrichment_sources"
    nested = source_dir / "ufc_2025_dataset" / "data"
    nested.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Event_Id": "E1",
                "Name": "UFC Test: Alpha vs. Beta",
                "Date": "2020-01-01",
                "Location": "Las Vegas, Nevada, USA",
            }
        ]
    ).to_csv(nested / "Events.csv", index=False)
    pd.DataFrame(
        [
            {
                "Event_Id": "E1",
                "Fighter_1": "Alice Alpha",
                "Fighter_2": "Beth Beta",
                "WeightClass": "Lightweight",
                "Max Rounds": 5,
                "Title Bout": 1,
            }
        ]
    ).to_csv(nested / "Fights.csv", index=False)

    result = runner.invoke(app, ["enrichment-sources", "--source-dir", str(source_dir)])

    assert result.exit_code == 0
    assert "Searched enrichment directory:" in result.output
    assert "Directory exists: True" in result.output
    assert "Source files found: 2" in result.output
    assert "Events.csv:" in result.output
    assert "guessed dataset type: event-level enrichment" in result.output
    assert "Fights.csv:" in result.output
    assert "guessed dataset type: fight-level enrichment" in result.output
    assert "usable fields: weight_class, title_fight, scheduled_rounds" in result.output


def test_auto_enrich_does_not_overwrite_known_values_with_blanks(tmp_path) -> None:
    template = tmp_path / "fight_enrichment_template.csv"
    output = tmp_path / "fight_enrichment.csv"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Fight Night: Alpha vs. Beta",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas",
                "main_event": "",
                "title_fight": "",
                "scheduled_rounds": 5,
            }
        ]
    ).to_csv(template, index=False)
    pd.DataFrame(
        [
            {
                "event": "UFC Fight Night: Alpha vs. Beta",
                "date": "2020-01-01",
                "location": "",
                "weight_class": "",
                "bout": "Alice Alpha vs Beth Beta",
                "main_event": "",
            }
        ]
    ).to_csv(source_dir / "external.csv", index=False)

    frame, _ = auto_enrich(template_path=template, output_path=output, source_dir=source_dir)

    assert frame.loc[0, "weight_class"] == "Lightweight"
    assert frame.loc[0, "event_location"] == "Las Vegas"


def test_auto_enrich_does_not_overwrite_known_values_with_unknown(tmp_path) -> None:
    template = tmp_path / "fight_enrichment_template.csv"
    output = tmp_path / "fight_enrichment.csv"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Fight Night: Alpha vs. Beta",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas",
                "main_event": 1,
                "title_fight": 1,
                "scheduled_rounds": 5,
            }
        ]
    ).to_csv(template, index=False)
    pd.DataFrame(
        [
            {
                "event": "UFC Fight Night: Alpha vs. Beta",
                "date": "2020-01-01",
                "location": "",
                "weight_class": "Unknown",
                "bout": "Alice Alpha vs Beth Beta",
            }
        ]
    ).to_csv(source_dir / "external.csv", index=False)

    frame, _ = auto_enrich(template_path=template, output_path=output, source_dir=source_dir)

    assert frame.loc[0, "weight_class"] == "Lightweight"
    assert frame.loc[0, "event_location"] == "Las Vegas"


def test_auto_enrich_uses_conservative_matching(tmp_path) -> None:
    template = tmp_path / "fight_enrichment_template.csv"
    output = tmp_path / "fight_enrichment.csv"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Fight Night: Alpha vs. Beta",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Unknown",
                "event_location": "",
                "main_event": "",
                "title_fight": "",
                "scheduled_rounds": 3,
            }
        ]
    ).to_csv(template, index=False)
    pd.DataFrame(
        [
            {
                "event": "Different Event",
                "date": "2020-02-01",
                "location": "Las Vegas",
                "division": "Lightweight",
                "bout": "Alice Alpha vs Beth Beta",
            }
        ]
    ).to_csv(source_dir / "external.csv", index=False)

    frame, report = auto_enrich(template_path=template, output_path=output, source_dir=source_dir)

    assert frame.loc[0, "weight_class"] == "Unknown"
    assert pd.isna(frame.loc[0, "event_location"]) or frame.loc[0, "event_location"] == ""
    assert report.external_sources[0].matched_rows == 0
    assert report.external_sources[0].unmatched_rows == 1


def test_auto_enrich_verbose_prints_source_summary(tmp_path) -> None:
    template = tmp_path / "fight_enrichment_template.csv"
    output = tmp_path / "fight_enrichment.csv"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Fight Night: Alpha vs. Beta",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Unknown",
                "event_location": "",
                "main_event": "",
                "title_fight": "",
                "scheduled_rounds": 3,
            }
        ]
    ).to_csv(template, index=False)
    pd.DataFrame(
        [
            {
                "event": "UFC Fight Night: Alpha vs. Beta",
                "date": "2020-01-01",
                "location": "Las Vegas",
                "division": "Lightweight",
                "bout": "Alice Alpha vs Beth Beta",
                "is_main_event": True,
                "is_title_fight": False,
            }
        ]
    ).to_csv(source_dir / "external.csv", index=False)

    result = runner.invoke(
        app,
        [
            "auto-enrich",
            "--template-path",
            str(template),
            "--output-path",
            str(output),
            "--source-dir",
            str(source_dir),
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert "Searched enrichment directory:" in result.output
    assert "Directory exists: True" in result.output
    assert "Source files found: 1" in result.output
    assert "Sources loaded: 1" in result.output
    assert "dataset type: fight-level enrichment" in result.output
    assert "columns: event, date, location, division, bout, is_main_event, is_title_fight" in result.output
    assert "usable fields:" in result.output
    assert "fields filled:" in result.output
    assert "Total matched rows: 1" in result.output
    assert "Total unmatched rows: 0" in result.output
    assert "Final enrichment coverage:" in result.output


def test_enrichment_summary_file_alias_works_on_fight_enrichment_csv(tmp_path) -> None:
    path = tmp_path / "fight_enrichment.csv"
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "event": "UFC Fight Night: Alpha vs. Beta",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas",
                "main_event": 1,
                "title_fight": 1,
                "scheduled_rounds": 5,
            }
        ]
    ).to_csv(path, index=False)

    result = runner.invoke(app, ["enrichment-summary", "--file", str(path)])

    assert result.exit_code == 0
    assert "Total fights: 1" in result.output
    assert "Known weight_class: 1/1 (100.0%)" in result.output


def test_import_enrichment_missing_file_error_points_to_template_workflow(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    _write_minimal_imports(import_dir)

    result = runner.invoke(
        app,
        [
            "import-enrichment",
            "--enrichment-path",
            str(import_dir / "fight_enrichment.csv"),
            "--fights-path",
            str(import_dir / "fights.csv"),
        ],
    )

    assert result.exit_code == 1
    assert "Run ufc-predict build-enrichment-template" in result.output
    assert "save it as data/raw/imports/fight_enrichment.csv" in result.output


def test_import_enrichment_supports_event_level_location_data(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    _write_minimal_imports(import_dir)
    pd.DataFrame(
        [
            {
                "event": "UFC Test 1",
                "event_date": "2020-01-01",
                "weight_class": "",
                "event_location": "Austin, Texas, USA",
                "main_event": "",
                "title_fight": "",
                "scheduled_rounds": "",
            }
        ]
    ).to_csv(import_dir / "fight_enrichment.csv", index=False)

    result = runner.invoke(
        app,
        [
            "import-enrichment",
            "--enrichment-path",
            str(import_dir / "fight_enrichment.csv"),
            "--fights-path",
            str(import_dir / "fights.csv"),
        ],
    )

    fights = pd.read_csv(import_dir / "fights.csv")
    assert result.exit_code == 0
    assert "Enrichment format: event-level" in result.output
    assert fights.loc[0, "event_location"] == "Austin, Texas, USA"


def test_validate_imports_detects_unapplied_enrichment(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    _write_minimal_imports(import_dir)
    (import_dir / "fight_enrichment.csv").write_text(
        "fight_date,event,fighter_a,fighter_b,weight_class,event_location,main_event,title_fight,scheduled_rounds\n"
        "2020-01-01,UFC Test 1,Alice Alpha,Beth Beta,Lightweight,Las Vegas,0,0,3\n",
        encoding="utf-8",
    )

    result = validate_import_directory(import_dir)

    assert not result.ok
    assert any("Run `ufc-predict import-enrichment`" in error for error in result.errors)


def test_validate_imports_rejects_numeric_event_location(tmp_path) -> None:
    import_dir = tmp_path / "imports"
    _write_minimal_imports(import_dir)
    fights = pd.read_csv(import_dir / "fights.csv")
    fights["event_location"] = 12345
    fights.to_csv(import_dir / "fights.csv", index=False)

    result = validate_import_directory(import_dir)

    assert not result.ok
    assert any("event_location must be categorical" in error for error in result.errors)


def test_report_data_quality_coverage_counts_enriched_fields_odds_and_scorecards(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    fights = pd.DataFrame(
        [
            {
                "fight_id": 1,
                "event_name": "UFC Test 1",
                "fight_date": "2020-01-01",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas",
                "main_event": 1,
            },
            {
                "fight_id": 2,
                "event_name": "UFC Test 1",
                "fight_date": "2020-01-01",
                "fighter_a": "Cara Delta",
                "fighter_b": "Dana Echo",
                "weight_class": "Unknown",
                "event_location": "",
                "main_event": 0,
            },
        ]
    )
    fights.to_csv(raw_dir / "fights.csv", index=False)
    pd.DataFrame(
        [
            {
                "fight_date": "2020-01-01",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "sportsbook": "Book",
                "fighter_a_odds": -150,
                "fighter_b_odds": 130,
                "timestamp": "2019-12-31T12:00:00",
                "fighter_a_no_vig_probability": 0.58,
                "fighter_b_no_vig_probability": 0.42,
            }
        ]
    ).to_csv(raw_dir / "odds.csv", index=False)
    pd.DataFrame(
        [
            {
                "event": "UFC Test 1",
                "fight_date": "2020-01-01",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "judge": "Judge One",
            }
        ]
    ).to_csv(raw_dir / "scorecards.csv", index=False)

    coverage = build_data_quality_coverage(raw_dir)

    assert coverage["coverage_source"].endswith("fights.csv")
    assert coverage["known_weight_class_pct"] == 50.0
    assert coverage["known_event_location_pct"] == 50.0
    assert coverage["known_main_event_pct"] == 100.0
    assert "known_title_fight_count" in coverage
    assert coverage["odds_coverage_pct"] == 50.0
    assert coverage["scorecard_coverage_pct"] == 50.0


def test_report_data_quality_coverage_prefers_enriched_imports(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    imports_dir = raw_dir / "imports"
    imports_dir.mkdir(parents=True)
    stale = pd.DataFrame(
        [
            {
                "fight_id": 1,
                "event_name": "UFC Test 1",
                "fight_date": "2020-01-01",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Unknown",
                "event_location": "",
                "main_event": "",
                "title_fight": "",
            }
        ]
    )
    stale.to_csv(raw_dir / "fights.csv", index=False)
    enriched = pd.DataFrame(
        [
            {
                "fight_id": 1,
                "event_name": "UFC Test 1",
                "fight_date": "2020-01-01",
                "fighter_a": "Alice Alpha",
                "fighter_b": "Beth Beta",
                "weight_class": "Lightweight",
                "event_location": "Las Vegas",
                "main_event": 1,
                "title_fight": 1,
            }
        ]
    )
    enriched.to_csv(imports_dir / "fights.csv", index=False)

    coverage = build_data_quality_coverage(raw_dir)

    assert coverage["coverage_source"].endswith("imports\\fights.csv") or coverage["coverage_source"].endswith("imports/fights.csv")
    assert coverage["known_weight_class_pct"] == 100.0
    assert coverage["known_event_location_pct"] == 100.0
    assert coverage["known_main_event_pct"] == 100.0
    assert coverage["known_title_fight_pct"] == 100.0


def test_rolling_backtest_keeps_enriched_metadata_for_breakdowns(monkeypatch) -> None:
    class DummyBundle:
        feature_columns = ["feature_x"]

        def predict_proba(self, frame):
            probabilities = np.where(frame["feature_x"].to_numpy() >= 0, 0.7, 0.3)
            return np.column_stack([1.0 - probabilities, probabilities])

    import ufc_predictor.models.train as train_module

    monkeypatch.setattr(train_module, "train_ensemble", lambda *args, **kwargs: DummyBundle())
    dataset = pd.DataFrame(
        [
            {"fight_id": 1, "fight_date": "2019-01-01", "fighter_a_win": 1, "feature_x": 1, "weight_class": "Lightweight", "main_event": 0},
            {"fight_id": 2, "fight_date": "2019-02-01", "fighter_a_win": 0, "feature_x": -1, "weight_class": "Welterweight", "main_event": 0},
            {"fight_id": 3, "fight_date": "2020-01-15", "fighter_a_win": 1, "feature_x": 1, "weight_class": "Lightweight", "main_event": 1},
            {"fight_id": 4, "fight_date": "2020-02-15", "fighter_a_win": 0, "feature_x": -1, "weight_class": "Welterweight", "main_event": 0},
            {"fight_id": 5, "fight_date": "2021-01-01", "fighter_a_win": 1, "feature_x": 1, "weight_class": "Lightweight", "main_event": 1},
        ]
    )

    metrics = rolling_backtest(dataset, min_train_fights=2, step="YS")

    assert metrics["performance_by_weight_class"]["Lightweight"]["count"] == 1.0
    assert metrics["performance_by_main_event"]["1"]["count"] == 1.0
