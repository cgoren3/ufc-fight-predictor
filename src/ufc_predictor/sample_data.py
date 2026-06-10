from __future__ import annotations

from pathlib import Path

import pandas as pd

from ufc_predictor.config import settings


SAMPLE_DATA_DIR = settings.project_root / "data" / "sample"


def load_sample_fights() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_DIR / "fights.csv")


def load_sample_fighters() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_DIR / "fighters.csv")


def load_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, None, None]:
    return load_sample_fights(), load_sample_fighters(), None, None


def write_sample_data(output_dir: str | Path | None = None) -> dict[str, Path]:
    output = Path(output_dir) if output_dir else settings.raw_data_dir
    output.mkdir(parents=True, exist_ok=True)
    fights = load_sample_fights()
    fighters = load_sample_fighters()
    paths = {
        "fights": output / "fights.csv",
        "fighters": output / "fighters.csv",
    }
    fights.to_csv(paths["fights"], index=False)
    fighters.to_csv(paths["fighters"], index=False)
    return paths
