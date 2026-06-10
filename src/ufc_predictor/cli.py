from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ufc_predictor.config import settings

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


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_fallback = path.with_suffix(".csv")
            if csv_fallback.exists():
                return pd.read_csv(csv_fallback)
            raise
    return pd.read_csv(path)


def _default_dataset_path() -> Path:
    parquet = settings.processed_data_dir / "fight_dataset.parquet"
    csv = settings.processed_data_dir / "fight_dataset.csv"
    return parquet if parquet.exists() else csv


@app.command("init-db")
def init_db() -> None:
    from ufc_predictor.database import initialize_database

    settings.ensure_directories()
    initialize_database()
    console.print(f"Initialized database at {settings.database_path}")


@app.command("ingest-ufcstats")
def ingest_ufcstats(
    max_events: Optional[int] = typer.Option(None, help="Limit event pages for a small/resumable run."),
    output_dir: Path = typer.Option(settings.raw_data_dir, help="Where raw CSV files should be written."),
    include_details: bool = typer.Option(False, help="Also parse fight detail and fighter profile pages."),
) -> None:
    from ufc_predictor.ingest.ufcstats_scraper import UFCStatsScraper

    settings.ensure_directories()
    paths = UFCStatsScraper().run_to_csv(output_dir=output_dir, max_events=max_events, include_details=include_details)
    console.print("Wrote UFCStats CSV files:")
    for name, path in paths.items():
        console.print(f"- {name}: {path}")


@app.command("load-scorecards")
def load_scorecards(
    csv_path: Path = typer.Argument(..., help="Manual official UFC scorecard CSV."),
) -> None:
    from ufc_predictor.ingest.scorecards_loader import import_scorecards

    frame = import_scorecards(csv_path)
    console.print(f"Imported {len(frame)} scorecard rows.")


@app.command("build-dataset")
def build_dataset(
    fights_csv: Path = typer.Option(settings.raw_data_dir / "fights.csv", help="Raw fights CSV."),
    fight_stats_csv: Path = typer.Option(settings.raw_data_dir / "fight_stats.csv", help="Optional fight stats CSV."),
    fighters_csv: Path = typer.Option(settings.raw_data_dir / "fighters.csv", help="Optional fighters CSV."),
    scorecards_csv: Path = typer.Option(settings.external_data_dir / "scorecards.csv", help="Optional scorecards CSV."),
    output: Path = typer.Option(settings.processed_data_dir / "fight_dataset.parquet", help="Output dataset path."),
    two_way: bool = typer.Option(False, help="Create both fighter orderings for each fight."),
    randomize_order: bool = typer.Option(False, help="Randomize fighter order once per fight."),
) -> None:
    from ufc_predictor.features.build_fight_dataset import build_fight_dataset, save_dataset

    if not fights_csv.exists():
        raise typer.BadParameter(f"Could not find fights CSV at {fights_csv}")
    fights = pd.read_csv(fights_csv)
    fight_stats = _read_optional_csv(fight_stats_csv)
    fighters = _read_optional_csv(fighters_csv)
    scorecards = _read_optional_csv(scorecards_csv)
    dataset = build_fight_dataset(
        fights=fights,
        fight_stats=fight_stats,
        fighters=fighters,
        scorecards=scorecards,
        two_way=two_way,
        randomize_order=randomize_order,
    )
    path = save_dataset(dataset, output)
    console.print(f"Wrote {len(dataset)} training rows to {path}")


@app.command("train")
def train(
    dataset_path: Path = typer.Option(_default_dataset_path(), help="Processed fight dataset."),
    model_dir: Path = typer.Option(settings.model_dir, help="Model output directory."),
) -> None:
    from ufc_predictor.models.train import save_model_bundle, train_ensemble

    dataset = _read_dataset(dataset_path)
    bundle = train_ensemble(dataset, model_dir=model_dir, save=False)
    model_path = save_model_bundle(bundle, model_dir=model_dir)
    console.print(f"Saved model to {model_path}")
    console.print_json(data=bundle.metrics)


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
    console.print(f"Wrote backtest metrics to {path}")
    console.print_json(data=metrics)


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

    fights = _read_optional_csv(fights_csv)
    if fights is None:
        fights = pd.DataFrame()
    fight_stats = _read_optional_csv(fight_stats_csv)
    fighters = _read_optional_csv(fighters_csv)
    scorecards = _read_optional_csv(scorecards_csv)
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
    console.print_json(data=prediction)


if __name__ == "__main__":  # pragma: no cover
    app()
