from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - exercised when optional deps are absent
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _path_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables and .env."""

    project_root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    external_data_dir: Path
    database_engine: str
    database_path: Path
    model_dir: Path
    cache_dir: Path
    user_agent: str
    request_timeout_seconds: int
    scrape_delay_seconds: float
    retry_count: int
    confidence_low_threshold: float
    confidence_high_threshold: float
    sportsdataio_api_key: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        data_dir = _path_env("UFC_PREDICTOR_DATA_DIR", "data")
        return cls(
            project_root=PROJECT_ROOT,
            data_dir=data_dir,
            raw_data_dir=_path_env("UFC_PREDICTOR_RAW_DATA_DIR", "data/raw"),
            processed_data_dir=_path_env("UFC_PREDICTOR_PROCESSED_DATA_DIR", "data/processed"),
            external_data_dir=_path_env("UFC_PREDICTOR_EXTERNAL_DATA_DIR", "data/external"),
            database_engine=os.getenv("UFC_PREDICTOR_DB_ENGINE", "sqlite").lower(),
            database_path=_path_env("UFC_PREDICTOR_DB_PATH", "data/processed/ufc_predictor.sqlite"),
            model_dir=_path_env("UFC_PREDICTOR_MODEL_DIR", "models"),
            cache_dir=_path_env("UFC_PREDICTOR_CACHE_DIR", "data/raw/cache"),
            user_agent=os.getenv(
                "UFC_PREDICTOR_USER_AGENT",
                "ufc-fight-predictor/0.1 respectful research scraper",
            ),
            request_timeout_seconds=_int_env("UFC_PREDICTOR_REQUEST_TIMEOUT_SECONDS", 30),
            scrape_delay_seconds=_float_env("UFC_PREDICTOR_SCRAPE_DELAY_SECONDS", 2.0),
            retry_count=_int_env("UFC_PREDICTOR_RETRY_COUNT", 3),
            confidence_low_threshold=_float_env("UFC_PREDICTOR_CONFIDENCE_LOW_THRESHOLD", 0.57),
            confidence_high_threshold=_float_env("UFC_PREDICTOR_CONFIDENCE_HIGH_THRESHOLD", 0.65),
            sportsdataio_api_key=os.getenv("SPORTS_DATA_IO_API_KEY") or None,
        )

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.cache_dir,
            self.model_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()
