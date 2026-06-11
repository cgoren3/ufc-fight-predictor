from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.data_sources import SOURCE_SAMPLE, imports_dir_has_fights, read_source_metadata, summarize_raw_data
from ufc_predictor.data_io import InputDataError, read_optional_csv, read_required_csv
from ufc_predictor.dataset_adapter import DatasetAdapterError
from ufc_predictor.import_validation import MIN_MEANINGFUL_FIGHTS, MIN_RELIABLE_BACKTEST_FIGHTS

try:
    import typer
    from rich.console import Console
except Exception:  # pragma: no cover - CLI dependency path
    typer = None
    Console = None


if typer is None:  # pragma: no cover - import guard for environments without CLI deps
    class _TyperShim:
        class BadParameter(ValueError):
            pass

        class Exit(SystemExit):
            pass

        @staticmethod
        def Option(default=None, *args, **kwargs):
            return default

        @staticmethod
        def Argument(default=None, *args, **kwargs):
            return default

    class _MissingTyperApp:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("typer and rich are required for the CLI. Install project dependencies.")

        def command(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    typer = _TyperShim()
    app = _MissingTyperApp()
    console = None
else:
    app = typer.Typer(help="Leakage-safe UFC fight prediction CLI.")
    console = Console()


REQUIRED_FIGHT_COLUMNS = ["fighter_a", "fighter_b", "fight_date", "winner"]
TINY_DATASET_WARNING_THRESHOLD = 500


def _print(message: str) -> None:
    if console is not None:
        console.print(message)
    else:  # pragma: no cover - only used without CLI deps
        print(message)


def _print_json(data: dict) -> None:
    if console is not None:
        console.print_json(data=data)
    else:  # pragma: no cover - only used without CLI deps
        print(data)


def _cli_error(message: str) -> None:
    _print(f"[red]{message}[/red]" if console is not None else message)
    raise typer.BadParameter(message)


def _runtime_error(message: str, code: int = 1) -> None:
    _print(f"[red]{message}[/red]" if console is not None else message)
    raise typer.Exit(code)


def _print_fetch_diagnostics(diagnostics) -> None:
    _print(diagnostics.format())


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_fallback = path.with_suffix(".csv")
            if csv_fallback.exists():
                return read_required_csv(csv_fallback, label="processed fight dataset")
            raise
    return read_required_csv(path, label="processed fight dataset")


def _default_dataset_path() -> Path:
    parquet = settings.processed_data_dir / "fight_dataset.parquet"
    csv = settings.processed_data_dir / "fight_dataset.csv"
    return parquet if parquet.exists() else csv


@app.command("init-db")
def init_db() -> None:
    from ufc_predictor.database import initialize_database

    settings.ensure_directories()
    initialize_database()
    _print(f"Initialized database at {settings.database_path}")


@app.command("ingest-ufcstats")
def ingest_ufcstats(
    max_events: Optional[int] = typer.Option(None, help="Limit event pages for a small/resumable run."),
    output_dir: Path = typer.Option(settings.raw_data_dir, help="Where raw CSV files should be written."),
    include_details: bool = typer.Option(False, help="Also parse fight detail and fighter profile pages."),
    ignore_resume: bool = typer.Option(False, help="Ignore the local UFCStats resume file."),
    from_html: Optional[Path] = typer.Option(
        None,
        help="Use a manually downloaded UFCStats completed-events HTML page to discover event links.",
    ),
    from_csv_imports: Optional[Path] = typer.Option(
        None,
        help="Import real raw CSV files from a directory such as data/raw/imports.",
    ),
    sample_on_failure: bool = typer.Option(
        True,
        "--sample-on-failure/--no-sample-on-failure",
        help="Write bundled sample data if live UFCStats scraping fails.",
    ),
) -> None:
    from ufc_predictor.ingest.ufcstats_scraper import UFCStatsScraper, UFCStatsScraperError, import_raw_csvs
    from ufc_predictor.sample_data import write_sample_data

    settings.ensure_directories()
    if from_csv_imports is not None:
        try:
            paths = import_raw_csvs(from_csv_imports, output_dir=output_dir)
        except UFCStatsScraperError as exc:
            _runtime_error(f"CSV import failed:\n{exc}")
        _print("Imported real raw CSV files:")
        for name, path in paths.items():
            _print(f"- {name}: {path}")
        return

    try:
        paths = UFCStatsScraper().run_to_csv(
            output_dir=output_dir,
            max_events=max_events,
            include_details=include_details,
            ignore_resume=ignore_resume,
            from_html=from_html,
        )
        _print("Wrote UFCStats CSV files:" if from_html is None else "Wrote UFCStats CSV files from manual HTML link discovery:")
    except UFCStatsScraperError as exc:
        if not sample_on_failure:
            if exc.diagnostics is not None:
                _print_fetch_diagnostics(exc.diagnostics)
            _runtime_error(f"UFCStats ingestion failed:\n{exc}")
        _print(f"[yellow]UFCStats ingestion did not produce live data: {exc}[/yellow]")
        if exc.diagnostics is not None:
            _print_fetch_diagnostics(exc.diagnostics)
        _print(
            "[yellow]Writing bundled sample/dev data instead so the MVP pipeline can run. "
            "Use --no-sample-on-failure for strict live scraping. This data is sample data only.[/yellow]"
        )
        paths = write_sample_data(output_dir)
    for name, path in paths.items():
        _print(f"- {name}: {path}")


@app.command("import-csv")
def import_csv(
    import_dir: Path = typer.Option(settings.raw_data_dir / "imports", help="Directory containing real raw CSV imports."),
    output_dir: Path = typer.Option(settings.raw_data_dir, help="Where normalized raw CSV files should be written."),
) -> None:
    from ufc_predictor.ingest.ufcstats_scraper import UFCStatsScraperError, import_raw_csvs

    try:
        paths = import_raw_csvs(import_dir, output_dir=output_dir)
    except UFCStatsScraperError as exc:
        _runtime_error(f"CSV import failed:\n{exc}")
    _print("Imported real raw CSV files:")
    for name, path in paths.items():
        _print(f"- {name}: {path}")


@app.command("dataset-columns")
def dataset_columns(
    source: Path = typer.Option(..., "--source", help="Folder containing downloaded Kaggle/third-party UFC CSV files."),
) -> None:
    from ufc_predictor.dataset_adapter import list_csv_columns

    try:
        infos = list_csv_columns(source)
    except DatasetAdapterError as exc:
        _runtime_error(str(exc))
    _print(f"CSV columns under {source}:")
    for info in infos:
        _print(f"- {info.path}:")
        _print("  " + ", ".join(info.columns))


@app.command("adapt-dataset")
def adapt_dataset(
    source: Path = typer.Option(..., "--source", help="Folder containing downloaded Kaggle/third-party UFC CSV files."),
    output_dir: Path = typer.Option(settings.raw_data_dir / "imports", "--output-dir", help="Where adapted import CSVs should be written."),
) -> None:
    from ufc_predictor.dataset_adapter import adapt_dataset as adapt_external_dataset

    try:
        result = adapt_external_dataset(source, output_dir)
    except DatasetAdapterError as exc:
        _runtime_error(f"Dataset adaptation failed:\n{exc}")

    if result.copied_existing_schema:
        _print("Source already matched the import schema; copied CSV files directly.")
    else:
        _print("Adapted external UFC CSV dataset into project import schema.")
    _print(f"Output directory: {result.output_dir}")
    for name, path in result.files.items():
        source_file = result.source_files.get(name)
        suffix = f" from {source_file}" if source_file else ""
        _print(f"- {name}: {path}{suffix}")
    for warning in result.warnings:
        _print(f"[yellow]Warning: {warning}[/yellow]")
    _print("Next steps: ufc-predict validate-imports && ufc-predict import-csv && ufc-predict data-summary")


@app.command("validate-imports")
def validate_imports(
    import_dir: Path = typer.Option(settings.raw_data_dir / "imports", help="Directory containing real raw CSV imports."),
) -> None:
    from ufc_predictor.import_validation import validate_import_directory

    result = validate_import_directory(import_dir)
    _print(f"Import directory: {result.import_dir}")
    for name, count in result.counts.items():
        _print(f"{name} rows: {count}")
    _print(f"Date range: {result.date_range['start'] or 'n/a'} to {result.date_range['end'] or 'n/a'}")
    for warning in result.warnings:
        _print(f"[yellow]Warning: {warning}[/yellow]")
    for error in result.errors:
        _print(f"[red]Error: {error}[/red]")
    if not result.ok:
        raise typer.Exit(1)
    _print("Import validation passed.")


@app.command("check-ufcstats")
def check_ufcstats() -> None:
    from ufc_predictor.ingest.ufcstats_scraper import check_completed_events_page

    diagnostics = check_completed_events_page()
    _print_fetch_diagnostics(diagnostics)
    if diagnostics.exception_type is not None:
        raise typer.Exit(1)


@app.command("data-summary")
def data_summary(
    raw_dir: Path = typer.Option(settings.raw_data_dir, help="Raw data directory to summarize."),
    scorecards_csv: Path = typer.Option(settings.raw_data_dir / "scorecards.csv", help="Optional scorecards CSV path."),
) -> None:
    summary = summarize_raw_data(raw_dir=raw_dir, scorecards_path=scorecards_csv)
    date_range = summary["date_range"]
    _print(f"Data source: {summary['data_source']}")
    _print(f"Fights rows: {summary['fights_row_count']}")
    _print(f"Fighters rows: {summary['fighters_row_count']}")
    _print(f"Fight stats rows: {summary['fight_stats_row_count']}")
    _print(f"Scorecards rows: {summary['scorecards_row_count']}")
    _print(f"Unique fighters: {summary['unique_fighters']}")
    _print(f"Date range: {date_range['start'] or 'n/a'} to {date_range['end'] or 'n/a'}")
    fights_count = summary["fights_row_count"]
    if fights_count < MIN_MEANINGFUL_FIGHTS:
        _print("[yellow]Warning: Too small for meaningful model training[/yellow]")
    if fights_count < MIN_RELIABLE_BACKTEST_FIGHTS:
        _print("[yellow]Warning: Backtest reliability will be limited[/yellow]")


@app.command("load-scorecards")
def load_scorecards(
    csv_path: Path = typer.Argument(..., help="Manual official UFC scorecard CSV."),
) -> None:
    from ufc_predictor.ingest.scorecards_loader import import_scorecards

    frame = import_scorecards(csv_path)
    _print(f"Imported {len(frame)} scorecard rows.")


@app.command("build-dataset")
def build_dataset(
    fights_csv: Path = typer.Option(settings.raw_data_dir / "fights.csv", help="Raw fights CSV."),
    fight_stats_csv: Path = typer.Option(settings.raw_data_dir / "fight_stats.csv", help="Optional fight stats CSV."),
    fighters_csv: Path = typer.Option(settings.raw_data_dir / "fighters.csv", help="Optional fighters CSV."),
    scorecards_csv: Path = typer.Option(settings.raw_data_dir / "scorecards.csv", help="Optional scorecards CSV."),
    output: Path = typer.Option(settings.processed_data_dir / "fight_dataset.parquet", help="Output dataset path."),
    two_way: bool = typer.Option(False, help="Create both fighter orderings for each fight."),
    randomize_order: bool = typer.Option(False, help="Randomize fighter order once per fight."),
    use_sample_data: bool = typer.Option(False, help="Build from the bundled sample/dev dataset."),
    imports_dir: Path = typer.Option(settings.raw_data_dir / "imports", help="Prefer real CSV imports from this directory when present."),
) -> None:
    from ufc_predictor.features.build_fight_dataset import build_fight_dataset, save_dataset
    from ufc_predictor.ingest.ufcstats_scraper import UFCStatsScraperError, import_raw_csvs
    from ufc_predictor.sample_data import load_sample_data, write_sample_data

    raw_output_dir = fights_csv.parent
    if use_sample_data:
        fights, fighters, fight_stats, scorecards = load_sample_data()
        write_sample_data(raw_output_dir)
        _print("Using sample/dev data for dataset build.")
    else:
        if imports_dir_has_fights(imports_dir):
            try:
                import_raw_csvs(imports_dir, output_dir=raw_output_dir)
            except UFCStatsScraperError as exc:
                _runtime_error(f"CSV import failed before dataset build:\n{exc}")
            _print(f"Using real CSV imports from {imports_dir}.")
            fights_csv = raw_output_dir / "fights.csv"
            fight_stats_csv = raw_output_dir / "fight_stats.csv"
            fighters_csv = raw_output_dir / "fighters.csv"
            scorecards_csv = raw_output_dir / "scorecards.csv"
        source_metadata = read_source_metadata(raw_output_dir)
        if source_metadata.get("source") == SOURCE_SAMPLE:
            _runtime_error(
                "Refusing to build from sample data without --use-sample-data. "
                "Run `ufc-predict import-csv`, `ufc-predict ingest-ufcstats`, or pass --use-sample-data for development."
            )
        _print(f"Using {source_metadata.get('source', 'unknown')} data for dataset build.")
        try:
            fights = read_required_csv(fights_csv, required_columns=REQUIRED_FIGHT_COLUMNS, label="fights CSV")
        except InputDataError as exc:
            _runtime_error(str(exc))
        fight_stats = read_optional_csv(fight_stats_csv, label="fight stats CSV")
        fighters = read_optional_csv(fighters_csv, label="fighters CSV")
        scorecards = read_optional_csv(scorecards_csv, label="scorecards CSV")
    dataset = build_fight_dataset(
        fights=fights,
        fight_stats=fight_stats,
        fighters=fighters,
        scorecards=scorecards,
        two_way=two_way,
        randomize_order=randomize_order,
    )
    path = save_dataset(dataset, output)
    _print(f"Wrote {len(dataset)} training rows to {path}")
    if len(dataset) < TINY_DATASET_WARNING_THRESHOLD:
        _print(
            f"[yellow]Warning: only {len(dataset)} training rows were built. "
            f"Model results will be unstable below {TINY_DATASET_WARNING_THRESHOLD} rows.[/yellow]"
        )


@app.command("train")
def train(
    dataset_path: Path = typer.Option(_default_dataset_path(), help="Processed fight dataset."),
    model_dir: Path = typer.Option(settings.model_dir, help="Model output directory."),
) -> None:
    from ufc_predictor.models.train import save_model_bundle, train_ensemble

    dataset = _read_dataset(dataset_path)
    bundle = train_ensemble(dataset, model_dir=model_dir, save=False)
    model_path = save_model_bundle(bundle, model_dir=model_dir)
    _print(f"Saved model to {model_path}")
    _print_json(bundle.metrics)


@app.command("backtest")
def backtest(
    dataset_path: Path = typer.Option(_default_dataset_path(), help="Processed fight dataset."),
    output: Path = typer.Option(settings.processed_data_dir / "backtest_results.json", help="Backtest JSON output."),
    min_train_fights: int = typer.Option(50, help="Minimum training rows before a rolling prediction."),
) -> None:
    from ufc_predictor.models.evaluate import rolling_backtest, save_backtest_result

    dataset = _read_dataset(dataset_path)
    metrics = rolling_backtest(dataset, min_train_fights=min_train_fights)
    path = save_backtest_result(metrics, output)
    _print(f"Wrote backtest metrics to {path}")
    _print_json(metrics)


@app.command("predict")
def predict(
    fighter_a: str = typer.Option(..., "--fighter-a"),
    fighter_b: str = typer.Option(..., "--fighter-b"),
    date: str = typer.Option(..., "--date"),
    weight_class: str = typer.Option(..., "--weight-class"),
    scheduled_rounds: int = typer.Option(3, "--scheduled-rounds"),
    model_path: Path = typer.Option(settings.model_dir / "ufc_predictor_model.pkl", help="Trained model path."),
    fights_csv: Path = typer.Option(settings.raw_data_dir / "fights.csv", help="Historical fights CSV."),
    fight_stats_csv: Path = typer.Option(settings.raw_data_dir / "fight_stats.csv", help="Optional fight stats CSV."),
    fighters_csv: Path = typer.Option(settings.raw_data_dir / "fighters.csv", help="Optional fighters CSV."),
    scorecards_csv: Path = typer.Option(settings.external_data_dir / "scorecards.csv", help="Optional scorecards CSV."),
) -> None:
    from ufc_predictor.models.predict import predict_fight

    fights = read_optional_csv(fights_csv, label="fights CSV")
    if fights is None:
        fights = pd.DataFrame()
    fight_stats = read_optional_csv(fight_stats_csv, label="fight stats CSV")
    fighters = read_optional_csv(fighters_csv, label="fighters CSV")
    scorecards = read_optional_csv(scorecards_csv, label="scorecards CSV")
    model = model_path if model_path.exists() else None
    prediction = predict_fight(
        model=model,
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        fight_date=date,
        weight_class=weight_class,
        scheduled_rounds=scheduled_rounds,
        fights=fights,
        fight_stats=fight_stats,
        fighters=fighters,
        scorecards=scorecards,
    )
    _print_json(prediction)


if __name__ == "__main__":  # pragma: no cover
    app()
