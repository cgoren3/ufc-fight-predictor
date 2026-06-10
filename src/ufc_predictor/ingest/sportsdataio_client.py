from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ufc_predictor.config import settings


@dataclass
class SportsDataIOClient:
    """Small optional client wrapper.

    The project runs without an API key. Methods that need the remote API return
    empty data when no key is configured so local training and tests do not fail.
    """

    api_key: str | None = settings.sportsdataio_api_key
    base_url: str = "https://api.sportsdata.io/v3/mma"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            return []
        try:
            import requests
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("requests is required for SportsDataIO calls.") from exc
        query = dict(params or {})
        query["key"] = self.api_key
        response = requests.get(f"{self.base_url}/{path.lstrip('/')}", params=query, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_odds_stub(self, season: str | int | None = None) -> list[dict[str, Any]]:
        """Return betting odds when configured, otherwise an empty list."""

        if not self.api_key:
            return []
        path = f"odds/json/GameOddsBySeason/{season}" if season else "odds/json/GameOdds"
        data = self._get(path)
        return data if isinstance(data, list) else []
