from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from io import StringIO
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.data_io import InputDataError, read_optional_csv
from ufc_predictor.ingest.ufcstats_scraper import FIGHT_COLUMNS, FIGHT_STAT_COLUMNS, FIGHTER_COLUMNS, UFCStatsScraper


STAGING_COLUMNS: dict[str, list[str]] = {
    "new_fights.csv": FIGHT_COLUMNS,
    "new_fight_stats.csv": FIGHT_STAT_COLUMNS,
    "new_scorecards.csv": [
        "event",
        "fight_date",
        "fighter_a",
        "fighter_b",
        "judge",
        "round_1_a",
        "round_1_b",
        "round_2_a",
        "round_2_b",
        "round_3_a",
        "round_3_b",
        "round_4_a",
        "round_4_b",
        "round_5_a",
        "round_5_b",
        "total_a",
        "total_b",
        "decision_type",
        "winner",
        "card_type",
        "raw_scorecards",
        "source_file",
    ],
    "new_fighters.csv": FIGHTER_COLUMNS + ["nickname", "weight_class"],
    "new_event_enrichment.csv": [
        "fight_date",
        "event",
        "fighter_a",
        "fighter_b",
        "weight_class",
        "event_location",
        "main_event",
        "title_fight",
        "scheduled_rounds",
    ],
    "new_odds.csv": [
        "fight_date",
        "event",
        "fighter_a",
        "fighter_b",
        "sportsbook",
        "fighter_a_odds",
        "fighter_b_odds",
        "timestamp",
        "source_file",
    ],
}


@dataclass
class CardUpdateReport:
    event_url: str
    source: str
    staging_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    rows_written: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_url": self.event_url,
            "source": self.source,
            "staging_dir": str(self.staging_dir),
            "files": {name: str(path) for name, path in self.files.items()},
            "rows_written": self.rows_written,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class ValidationReport:
    ok: bool
    staging_dir: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "staging_dir": str(self.staging_dir),
            "errors": self.errors,
            "warnings": self.warnings,
            "counts": self.counts,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class ApplyReport:
    backup_dir: Path
    rows_added: dict[str, int] = field(default_factory=dict)
    fields_updated: dict[str, int] = field(default_factory=dict)
    files_written: dict[str, Path] = field(default_factory=dict)
    skipped_duplicates: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backup_dir": str(self.backup_dir),
            "rows_added": self.rows_added,
            "fields_updated": self.fields_updated,
            "files_written": {name: str(path) for name, path in self.files_written.items()},
            "skipped_duplicates": self.skipped_duplicates,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _safe_preview(value: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    return text[:limit]


def _normal(value: Any) -> str:
    return _clean(value).lower()


def _date_key(value: Any) -> str:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return ""
    return pd.Timestamp(date).date().isoformat()


def _pair_key(fighter_a: Any, fighter_b: Any) -> str:
    return "|".join(sorted([_normal(fighter_a), _normal(fighter_b)]))


def fight_identity(row: pd.Series | dict[str, Any]) -> tuple[str, str, str]:
    event = row.get("event_name", row.get("event", ""))
    return (_date_key(row.get("fight_date")), _normal(event), _pair_key(row.get("fighter_a"), row.get("fighter_b")))


def fight_date_pair_identity(row: pd.Series | dict[str, Any]) -> tuple[str, str]:
    return (_date_key(row.get("fight_date")), _pair_key(row.get("fighter_a"), row.get("fighter_b")))


def _is_empty(value: Any) -> bool:
    text = _clean(value).lower()
    return text in {"", "nan", "none", "null", "unknown", "<na>"}


def _truthy_flag(value: Any) -> int | pd.NA:
    if _is_empty(value):
        return pd.NA
    text = _clean(value).lower()
    if text in {"1", "true", "yes", "y", "main", "title"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return pd.NA
    return 1 if float(numeric) > 0 else 0


def _first_present(row: pd.Series | dict[str, Any], aliases: list[str]) -> Any:
    for alias in aliases:
        if alias in row and not _is_empty(row.get(alias)):
            return row.get(alias)
    return ""


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    import re

    output = frame.copy()
    output.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())).strip("_")
        for column in output.columns
    ]
    return output


def _read_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[[*columns, *[column for column in frame.columns if column not in columns]]]


def _write_staging_frame(staging_dir: Path, filename: str, rows: list[dict[str, Any]] | pd.DataFrame) -> Path:
    columns = STAGING_COLUMNS[filename]
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[[*columns, *[column for column in frame.columns if column not in columns]]]
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / filename
    frame.to_csv(path, index=False)
    return path


def _soup_from_html(html: str) -> Any:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("beautifulsoup4 is required for UFCStats parsing.") from exc
    return BeautifulSoup(html, "html.parser")


def _cell_values(cell: Any) -> list[str]:
    values = [_clean(item.get_text(" ", strip=True)) for item in cell.select("p")]
    values = [value for value in values if value]
    if len(values) >= 2:
        return values[:2]
    anchors = [_clean(item.get_text(" ", strip=True)) for item in cell.select("a")]
    anchors = [value for value in anchors if value]
    if len(anchors) >= 2:
        return anchors[:2]
    text = _clean(cell.get_text(" ", strip=True))
    if not text:
        return ["", ""]
    return [text, ""]


def _parse_landed_attempted(value: Any) -> tuple[float | None, float | None]:
    text = _clean(value).lower()
    if not text:
        return None, None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:of|/|-)\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None, None
    return float(numeric), None


def _parse_numeric(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _parse_seconds(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    if ":" in text:
        left, right = text.split(":", 1)
        try:
            return float(left) * 60.0 + float(right)
        except ValueError:
            return None
    return _parse_numeric(text)


class UFCStatsAdapter:
    """Source adapter for UFCStats event, fight-stat, and profile pages."""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    def fetch_event_html(self, event_url: str) -> tuple[str, dict[str, Any]]:
        path_text = event_url.replace("file://", "")
        path = Path(path_text)
        if path.exists():
            html = path.read_text(encoding="utf-8")
            return html, self._diagnostics(event_url, html, status_code=None, final_url=str(path), source="local_file")
        try:
            import requests
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("requests is required for UFCStats fetching.") from exc
        diagnostics: dict[str, Any] = {"source_attempted": "ufcstats", "url_requested": event_url}
        try:
            response = requests.get(
                event_url,
                headers=self.headers,
                timeout=settings.request_timeout_seconds,
                allow_redirects=True,
            )
            html = response.text or ""
            diagnostics.update(
                self._diagnostics(
                    event_url,
                    html,
                    status_code=response.status_code,
                    final_url=response.url,
                    source="network",
                )
            )
            diagnostics["exception_type"] = None
            diagnostics["exception_message"] = None
            if response.status_code >= 400:
                diagnostics["parse_reason"] = f"HTTP {response.status_code} response."
            return html, diagnostics
        except Exception as exc:
            diagnostics.update(
                {
                    "http_status_code": None,
                    "final_url": "",
                    "response_content_length": 0,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "ufcstats_table_marker_found": False,
                    "fight_row_marker_found": False,
                    "page_title": "",
                    "event_name_detected": "",
                    "parse_reason": "Network fetch failed.",
                    "body_preview": "",
                }
            )
            return "", diagnostics

    def _diagnostics(
        self,
        requested_url: str,
        html: str,
        status_code: int | None,
        final_url: str,
        source: str,
    ) -> dict[str, Any]:
        soup = _soup_from_html(html) if html else None
        title = _clean(soup.title.get_text(" ", strip=True)) if soup is not None and soup.title is not None else ""
        event_name = ""
        if soup is not None:
            event_node = soup.select_one(".b-content__title-highlight")
            event_name = _clean(event_node.get_text(" ", strip=True)) if event_node is not None else ""
        challenge = "Checking your browser" in html or "requires JavaScript" in html
        return {
            "source_attempted": "ufcstats",
            "fetch_source": source,
            "url_requested": requested_url,
            "http_status_code": status_code,
            "final_url": final_url,
            "response_content_length": len(html or ""),
            "ufcstats_table_marker_found": "b-fight-details__table" in html,
            "fight_row_marker_found": "b-fight-details__table-row" in html or "data-link" in html,
            "browser_challenge_detected": challenge,
            "page_title": title,
            "event_name_detected": event_name,
            "body_preview": _safe_preview(soup.get_text(" ", strip=True) if soup is not None else html),
        }

    def parse_event_html(
        self,
        html: str,
        event_url: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
        diagnostics = dict(diagnostics or self._diagnostics(event_url, html, None, event_url, "html"))
        warnings: list[str] = []
        if not html.strip():
            diagnostics["parse_reason"] = "Empty HTML response."
            return [], [], [], [], warnings, diagnostics
        soup = _soup_from_html(html)
        event_node = soup.select_one(".b-content__title-highlight")
        event_name = diagnostics.get("event_name_detected") or (_clean(event_node.get_text(" ", strip=True)) if event_node is not None else "")
        details = [_clean(item.get_text(" ", strip=True)) for item in soup.select(".b-list__box-list-item")]
        fight_date = ""
        event_location = ""
        for item in details:
            lower = item.lower()
            if lower.startswith("date:"):
                fight_date = _clean(item.split(":", 1)[1])
            elif lower.startswith("location:"):
                event_location = _clean(item.split(":", 1)[1])
        fight_rows = soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover")
        if not fight_rows:
            fight_rows = [row for row in soup.select("tr[data-link]") if row.select("td")]
        fights: list[dict[str, Any]] = []
        stats_rows: list[dict[str, Any]] = []
        fighters: dict[str, dict[str, Any]] = {}
        enrichment: list[dict[str, Any]] = []
        for index, tr in enumerate(fight_rows):
            cells = tr.select("td")
            if len(cells) < 2:
                continue
            statuses = [value.lower() for value in _cell_values(cells[0])]
            names = _cell_values(cells[1])
            names = [name for name in names[:2] if name]
            if len(names) < 2:
                continue
            fight_url = tr.get("data-link") or ""
            if fight_url.startswith("/"):
                fight_url = "http://ufcstats.com" + fight_url
            winner = ""
            for status_index, status in enumerate(statuses[:2]):
                if "win" in status and status_index < len(names):
                    winner = names[status_index]
                    break
            weight_class = _clean(cells[6].get_text(" ", strip=True)) if len(cells) > 6 else "Unknown"
            method = _clean(cells[7].get_text(" ", strip=True)) if len(cells) > 7 else ""
            finish_round = _clean(cells[8].get_text(" ", strip=True)) if len(cells) > 8 else ""
            finish_time = _clean(cells[9].get_text(" ", strip=True)) if len(cells) > 9 else ""
            scheduled_rounds = 5 if ("title" in method.lower() or index == 0 and "vs" in event_name.lower()) else 3
            fight_id = f"staged_{index}"
            fight = {
                "fight_id": fight_id,
                "event_name": event_name,
                "fight_date": fight_date,
                "event_location": event_location,
                "fighter_a": names[0],
                "fighter_b": names[1],
                "winner": winner,
                "weight_class": weight_class or "Unknown",
                "method": method,
                "finish_round": finish_round,
                "finish_time": finish_time,
                "scheduled_rounds": scheduled_rounds,
                "main_event": 1 if index == 0 else 0,
                "source_url": fight_url,
                "title_fight": 1 if scheduled_rounds == 5 else 0,
            }
            fights.append(fight)
            enrichment.append(
                {
                    "fight_date": fight_date,
                    "event": event_name,
                    "fighter_a": names[0],
                    "fighter_b": names[1],
                    "weight_class": weight_class or "Unknown",
                    "event_location": event_location,
                    "main_event": fight["main_event"],
                    "title_fight": fight["title_fight"],
                    "scheduled_rounds": scheduled_rounds,
                }
            )
            fighter_links = cells[1].select('a[href*="/fighter-details/"]')
            for fighter_index, fighter_name in enumerate(names[:2]):
                fighter_url = fighter_links[fighter_index].get("href") if fighter_index < len(fighter_links) else ""
                fighters.setdefault(
                    _normal(fighter_name),
                    {
                        "name": fighter_name,
                        "stance": "",
                        "height_in": "",
                        "weight_lb": "",
                        "reach_in": "",
                        "date_of_birth": "",
                        "record": "",
                        "source_url": fighter_url or "",
                    },
                )
            if len(cells) > 5:
                kd_values = _cell_values(cells[2]) if len(cells) > 2 else ["", ""]
                str_values = _cell_values(cells[3]) if len(cells) > 3 else ["", ""]
                td_values = _cell_values(cells[4]) if len(cells) > 4 else ["", ""]
                sub_values = _cell_values(cells[5]) if len(cells) > 5 else ["", ""]
                for fighter_index, fighter_name in enumerate(names[:2]):
                    opponent = names[1 - fighter_index]
                    sig_landed, sig_attempted = _parse_landed_attempted(str_values[fighter_index] if fighter_index < len(str_values) else "")
                    td_landed, td_attempted = _parse_landed_attempted(td_values[fighter_index] if fighter_index < len(td_values) else "")
                    stats_rows.append(
                        {
                            "fight_id": fight_id,
                            "source_url": fight_url,
                            "fighter": fighter_name,
                            "opponent": opponent,
                            "knockdowns": _parse_numeric(kd_values[fighter_index] if fighter_index < len(kd_values) else ""),
                            "sig_str_landed": sig_landed,
                            "sig_str_attempted": sig_attempted,
                            "takedowns_landed": td_landed,
                            "takedowns_attempted": td_attempted,
                            "submission_attempts": _parse_numeric(sub_values[fighter_index] if fighter_index < len(sub_values) else ""),
                        }
                    )
        if fights:
            diagnostics["parse_reason"] = "Parsed UFCStats event table."
        elif diagnostics.get("browser_challenge_detected"):
            diagnostics["parse_reason"] = "UFCStats returned a browser/JavaScript challenge page instead of an event table."
        elif not diagnostics.get("ufcstats_table_marker_found"):
            diagnostics["parse_reason"] = "No UFCStats fight table markers were found."
        else:
            diagnostics["parse_reason"] = "UFCStats table markers were present, but no valid fight rows were parsed."
        diagnostics["parsed_fights"] = len(fights)
        diagnostics["parsed_fight_stats"] = len(stats_rows)
        diagnostics["parsed_fighters"] = len(fighters)
        return fights, stats_rows, list(fighters.values()), enrichment, warnings, diagnostics


class ManualCsvAdapter:
    """Adapter for local CSV/HTML tables used as a safe manual fallback."""

    def parse(self, source_path_or_url: str) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        frame = _read_generic_source(source_path_or_url)
        fights, stats, fighters, scorecards, enrichment, odds = _generic_rows_from_frame(frame, source_path_or_url)
        diagnostics = {
            "source_attempted": "manual",
            "url_requested": source_path_or_url,
            "rows_read": int(len(frame)),
            "parsed_fights": int(len(fights)),
            "parse_reason": "Parsed manual source rows." if fights else "No manual rows could be mapped to fight schema.",
        }
        return fights, stats, fighters, scorecards, enrichment, odds, diagnostics


class OddsApiAdapter:
    """Optional odds adapter placeholder; active only when credentials are configured."""

    def is_configured(self) -> bool:
        return bool(getattr(settings, "odds_api_key", None))


class BestFightOddsAdapter:
    """Optional isolated odds adapter placeholder.

    It is deliberately non-required so failures cannot break the local/free update workflow.
    """

    def is_configured(self) -> bool:
        return False


class SportsDataIOAdapter:
    """Optional paid-source adapter placeholder; active only with SportsDataIO credentials."""

    def is_configured(self) -> bool:
        return bool(settings.sportsdataio_api_key)


def _load_url_or_file(event_url: str) -> str:
    path_text = event_url.replace("file://", "")
    path = Path(path_text)
    if path.exists():
        return path.read_text(encoding="utf-8")
    try:
        import requests
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("requests is required for remote card updates.") from exc
    headers = {
        "User-Agent": settings.user_agent,
        "Accept": "text/html,application/xhtml+xml,text/csv,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    response = requests.get(event_url, headers=headers, timeout=settings.request_timeout_seconds)
    response.raise_for_status()
    return response.text


def _read_generic_source(event_url: str) -> pd.DataFrame:
    path_text = event_url.replace("file://", "")
    path = Path(path_text)
    if path.exists() and path.suffix.lower() == ".csv":
        return _canonical_columns(pd.read_csv(path))
    text = _load_url_or_file(event_url)
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return _canonical_columns(pd.read_json(StringIO(text)))
    if path.suffix.lower() == ".csv" or "," in text.splitlines()[0]:
        try:
            return _canonical_columns(pd.read_csv(StringIO(text)))
        except Exception:
            pass
    try:
        tables = pd.read_html(StringIO(text))
    except Exception:
        tables = []
    if not tables:
        return pd.DataFrame()
    return _canonical_columns(max(tables, key=len))


def _generic_rows_from_frame(frame: pd.DataFrame, source_file: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fights: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    fighters: dict[str, dict[str, Any]] = {}
    scorecards: list[dict[str, Any]] = []
    enrichment: list[dict[str, Any]] = []
    odds: list[dict[str, Any]] = []
    if frame.empty:
        return fights, stats, list(fighters.values()), scorecards, enrichment, odds

    for index, row in frame.iterrows():
        fighter_a = _clean(_first_present(row, ["fighter_a", "red_fighter", "r_fighter", "fight_fighter", "fighter"]))
        fighter_b = _clean(_first_present(row, ["fighter_b", "blue_fighter", "b_fighter", "opponent"]))
        bout = _first_present(row, ["bout", "matchup"])
        if (not fighter_a or not fighter_b) and bout and " vs " in str(bout).lower():
            left, right = str(bout).replace(" VS ", " vs ").replace(" Vs ", " vs ").split(" vs ", 1)
            fighter_a, fighter_b = _clean(left), _clean(right)
        if not fighter_a or not fighter_b:
            continue
        fight_date = _date_key(_first_present(row, ["fight_date", "event_date", "date"]))
        event = _clean(_first_present(row, ["event", "event_name", "event_title"]))
        weight_class = _clean(_first_present(row, ["weight_class", "division", "bout_weight", "weightclass"])) or "Unknown"
        method = _clean(_first_present(row, ["method", "result_method"]))
        finish_round = _first_present(row, ["finish_round", "round"])
        finish_time = _first_present(row, ["finish_time", "time"])
        winner = _clean(_first_present(row, ["winner"]))
        if winner.lower() in {"red", "r", "fighter_a"}:
            winner = fighter_a
        elif winner.lower() in {"blue", "b", "fighter_b"}:
            winner = fighter_b
        scheduled_rounds = _first_present(row, ["scheduled_rounds", "rounds", "max_rounds"])
        if _is_empty(scheduled_rounds):
            scheduled_rounds = 5 if _truthy_flag(_first_present(row, ["is_title_fight", "title_fight", "title_bout", "championship"])) == 1 else 3
        main_event = _truthy_flag(_first_present(row, ["main_event", "is_main_event"]))
        if "bout_order" in row and pd.to_numeric(pd.Series([row.get("bout_order")]), errors="coerce").iloc[0] == 1:
            main_event = 1
        title_fight = _truthy_flag(_first_present(row, ["title_fight", "is_title_fight", "championship", "belt", "title_bout"]))
        if pd.isna(title_fight):
            title_fight = 1 if pd.to_numeric(pd.Series([scheduled_rounds]), errors="coerce").fillna(0).iloc[0] >= 5 else 0
        fight_id = _clean(_first_present(row, ["fight_id"])) or f"staged_{index}"
        fight_row = {
            "fight_id": fight_id,
            "event_name": event,
            "fight_date": fight_date,
            "event_location": _clean(_first_present(row, ["event_location", "location", "venue"])),
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "winner": winner,
            "weight_class": weight_class,
            "method": method,
            "finish_round": finish_round,
            "finish_time": finish_time,
            "scheduled_rounds": scheduled_rounds,
            "main_event": 0 if pd.isna(main_event) else main_event,
            "source_url": source_file,
            "title_fight": title_fight,
        }
        fights.append(fight_row)
        enrichment.append(
            {
                "fight_date": fight_date,
                "event": event,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "weight_class": weight_class,
                "event_location": fight_row["event_location"],
                "main_event": fight_row["main_event"],
                "title_fight": title_fight,
                "scheduled_rounds": scheduled_rounds,
            }
        )
        for fighter, opponent, prefix in [(fighter_a, fighter_b, "red"), (fighter_b, fighter_a, "blue")]:
            fighters.setdefault(
                _normal(fighter),
                {
                    "name": fighter,
                    "stance": "",
                    "height_in": "",
                    "weight_lb": "",
                    "reach_in": "",
                    "date_of_birth": "",
                    "record": "",
                    "source_url": source_file,
                },
            )
            kd = _first_present(row, [f"{prefix}_kd", f"{prefix}_knockdowns"])
            if not _is_empty(kd):
                stats.append(
                    {
                        "fight_id": fight_id,
                        "source_url": source_file,
                        "fighter": fighter,
                        "opponent": opponent,
                        "knockdowns": kd,
                        "sig_str_landed": _first_present(row, [f"{prefix}_sig_str_landed", f"{prefix}_significant_strikes_landed"]),
                        "sig_str_attempted": _first_present(row, [f"{prefix}_sig_str_attempted", f"{prefix}_significant_strikes_attempted"]),
                        "total_str_landed": _first_present(row, [f"{prefix}_total_str_landed"]),
                        "total_str_attempted": _first_present(row, [f"{prefix}_total_str_attempted"]),
                        "takedowns_landed": _first_present(row, [f"{prefix}_td_landed", f"{prefix}_takedowns_landed"]),
                        "takedowns_attempted": _first_present(row, [f"{prefix}_td_attempted", f"{prefix}_takedowns_attempted"]),
                        "submission_attempts": _first_present(row, [f"{prefix}_sub", f"{prefix}_submission_attempts"]),
                        "reversals": _first_present(row, [f"{prefix}_reversals"]),
                        "control_seconds": _first_present(row, [f"{prefix}_control_seconds", f"{prefix}_control_time"]),
                    }
                )
        raw_scorecards = _clean(_first_present(row, ["score_cards", "scorecards", "raw_scorecards"]))
        if raw_scorecards:
            scorecards.append(
                {
                    "event": event,
                    "fight_date": fight_date,
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "judge": "raw_scorecards",
                    "decision_type": _clean(_first_present(row, ["card_type", "decision_type"])),
                    "winner": winner,
                    "card_type": _clean(_first_present(row, ["card_type"])),
                    "raw_scorecards": raw_scorecards,
                    "source_file": source_file,
                }
            )
        red_odds = _first_present(row, ["red_fighter_moneyline_odds", "fighter_a_odds"])
        blue_odds = _first_present(row, ["blue_fighter_moneyline_odds", "fighter_b_odds"])
        if not _is_empty(red_odds) and not _is_empty(blue_odds):
            odds.append(
                {
                    "fight_date": fight_date,
                    "event": event,
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "sportsbook": _clean(_first_present(row, ["sportsbook"])) or "card_update_source",
                    "fighter_a_odds": red_odds,
                    "fighter_b_odds": blue_odds,
                    "timestamp": "",
                    "source_file": source_file,
                }
            )
    return fights, stats, list(fighters.values()), scorecards, enrichment, odds


def _ufcstats_rows(
    event_url: str,
    event_html: str | Path | None = None,
    staging_dir: Path | None = None,
    save_raw_html: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
]:
    adapter = UFCStatsAdapter()
    html_source = str(event_html or event_url)
    if event_html is not None:
        path = Path(event_html)
        if not path.exists():
            raise InputDataError(f"UFCStats event HTML file not found: {path}")
        html = path.read_text(encoding="utf-8")
        diagnostics = adapter._diagnostics(html_source, html, status_code=None, final_url=str(path), source="event_html")
    else:
        html, diagnostics = adapter.fetch_event_html(event_url)
    if save_raw_html and staging_dir is not None:
        staging_dir.mkdir(parents=True, exist_ok=True)
        raw_path = staging_dir / "raw_event_page.html"
        raw_path.write_text(html, encoding="utf-8")
        diagnostics["raw_html_saved_to"] = str(raw_path)
    fights, stats_rows, fighter_rows, enrichment, warnings, diagnostics = adapter.parse_event_html(
        html,
        html_source,
        diagnostics=diagnostics,
    )
    return fights, stats_rows, fighter_rows, [], enrichment, [], warnings, diagnostics


def update_after_card(
    event_url: str,
    source: str = "auto",
    staging_dir: str | Path | None = None,
    event_html: str | Path | None = None,
    save_raw_html: bool = False,
) -> CardUpdateReport:
    staging = Path(staging_dir) if staging_dir else settings.raw_data_dir / "staging"
    normalized_source = source.strip().lower()
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {}
    if normalized_source == "auto":
        normalized_source = "ufcstats" if (event_html is not None or "ufcstats.com" in event_url.lower()) else "manual"
    if normalized_source == "ufcstats":
        fights, stats, fighters, scorecards, enrichment, odds, warnings, diagnostics = _ufcstats_rows(
            event_url=event_url,
            event_html=event_html,
            staging_dir=staging,
            save_raw_html=save_raw_html,
        )
    elif normalized_source in {"espn", "manual"}:
        fights, stats, fighters, scorecards, enrichment, odds, diagnostics = ManualCsvAdapter().parse(event_url)
        diagnostics["source_attempted"] = normalized_source
    else:
        raise InputDataError("--source must be one of: ufcstats, espn, manual, auto.")

    if not fights:
        warnings.append(diagnostics.get("parse_reason", "Parsing produced zero fights."))
    report = CardUpdateReport(
        event_url=event_url,
        source=normalized_source,
        staging_dir=staging,
        warnings=warnings,
        diagnostics=diagnostics,
    )
    payloads = {
        "new_fights.csv": fights,
        "new_fight_stats.csv": stats,
        "new_scorecards.csv": scorecards,
        "new_fighters.csv": fighters,
        "new_event_enrichment.csv": enrichment,
        "new_odds.csv": odds,
    }
    for filename, rows in payloads.items():
        path = _write_staging_frame(staging, filename, rows)
        report.files[filename] = path
        report.rows_written[filename] = int(len(rows))
    report_path = staging / "update_report.json"
    report.files["update_report.json"] = report_path
    report_path.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return report


def _existing_imports(imports_dir: Path) -> dict[str, pd.DataFrame]:
    fights = read_optional_csv(imports_dir / "fights.csv", label="existing fights CSV")
    fighters = read_optional_csv(imports_dir / "fighters.csv", label="existing fighters CSV")
    fight_stats = read_optional_csv(imports_dir / "fight_stats.csv", label="existing fight stats CSV")
    scorecards = read_optional_csv(imports_dir / "scorecards.csv", label="existing scorecards CSV")
    return {
        "fights": fights if fights is not None else pd.DataFrame(),
        "fighters": fighters if fighters is not None else pd.DataFrame(),
        "fight_stats": fight_stats if fight_stats is not None else pd.DataFrame(),
        "scorecards": scorecards if scorecards is not None else pd.DataFrame(),
    }


def validate_card_update(
    staging_dir: str | Path | None = None,
    imports_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> ValidationReport:
    staging = Path(staging_dir) if staging_dir else settings.raw_data_dir / "staging"
    imports = Path(imports_dir) if imports_dir else settings.raw_data_dir / "imports"
    output = Path(output_path) if output_path else staging / "validation_report.json"
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    fights = _read_frame(staging / "new_fights.csv", STAGING_COLUMNS["new_fights.csv"])
    stats = _read_frame(staging / "new_fight_stats.csv", STAGING_COLUMNS["new_fight_stats.csv"])
    scorecards = _read_frame(staging / "new_scorecards.csv", STAGING_COLUMNS["new_scorecards.csv"])
    fighters = _read_frame(staging / "new_fighters.csv", STAGING_COLUMNS["new_fighters.csv"])
    enrichment = _read_frame(staging / "new_event_enrichment.csv", STAGING_COLUMNS["new_event_enrichment.csv"])
    counts.update(
        {
            "new_fights": int(len(fights)),
            "new_fight_stats": int(len(stats)),
            "new_scorecards": int(len(scorecards)),
            "new_fighters": int(len(fighters)),
            "new_event_enrichment": int(len(enrichment)),
        }
    )
    if fights.empty:
        errors.append("No staged fights found. Run update-after-card before validation.")
    required = ["fighter_a", "fighter_b", "fight_date", "winner"]
    for column in required:
        if column not in fights.columns:
            errors.append(f"new_fights.csv is missing required column: {column}")
        elif fights[column].map(_is_empty).any():
            bad_count = int(fights[column].map(_is_empty).sum())
            if column == "winner":
                non_decisive = fights.get("method", pd.Series([""] * len(fights))).astype(str).str.lower().str.contains("draw|no contest|nc", regex=True)
                bad_count = int((fights[column].map(_is_empty) & ~non_decisive).sum())
            if bad_count:
                errors.append(f"new_fights.csv has {bad_count} rows with missing {column}.")
    fight_dates = pd.to_datetime(fights.get("fight_date", pd.Series(dtype=object)), errors="coerce")
    if fight_dates.isna().any():
        errors.append(f"new_fights.csv has {int(fight_dates.isna().sum())} unparseable fight_date values.")
    today = pd.Timestamp.now().normalize()
    if fight_dates.gt(today).any():
        errors.append(f"new_fights.csv has {int(fight_dates.gt(today).sum())} future fight dates.")
    seen: set[tuple[str, str, str]] = set()
    seen_date_pairs: set[tuple[str, str]] = set()
    duplicate_staged = 0
    for _, row in fights.iterrows():
        key = fight_identity(row)
        date_pair_key = fight_date_pair_identity(row)
        if key in seen or date_pair_key in seen_date_pairs:
            duplicate_staged += 1
        seen.add(key)
        seen_date_pairs.add(date_pair_key)
        winner = _clean(row.get("winner"))
        method = _clean(row.get("method")).lower()
        allowed_non_decisive = {"draw", "nc", "no contest"}
        if winner.lower() not in allowed_non_decisive and winner not in {_clean(row.get("fighter_a")), _clean(row.get("fighter_b"))}:
            if not ("draw" in method or "no contest" in method or method == "nc"):
                errors.append(f"Winner is not one of the two fighters: {winner} ({row.get('fighter_a')} vs {row.get('fighter_b')}).")
    if duplicate_staged:
        errors.append(f"Staging contains {duplicate_staged} duplicate event/date/fighter pairs.")
    if "fight_id" in fights.columns:
        non_empty_ids = fights["fight_id"].dropna().astype(str).str.strip()
        duplicate_ids = int(non_empty_ids[non_empty_ids.ne("")].duplicated().sum())
        if duplicate_ids:
            errors.append(f"Staging contains {duplicate_ids} duplicate fight_id values.")
    existing = _existing_imports(imports)
    existing_fights = existing["fights"]
    if not existing_fights.empty:
        existing_keys = {fight_identity(row) for _, row in existing_fights.iterrows()}
        existing_date_pairs = {fight_date_pair_identity(row) for _, row in existing_fights.iterrows()}
        duplicate_existing = sum(
            1
            for _, row in fights.iterrows()
            if fight_identity(row) in existing_keys or fight_date_pair_identity(row) in existing_date_pairs
        )
        if duplicate_existing:
            errors.append(f"Staging contains {duplicate_existing} fights already present in imports.")
    numeric_columns = [
        "knockdowns",
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "takedowns_landed",
        "takedowns_attempted",
        "submission_attempts",
        "reversals",
        "control_seconds",
    ]
    for column in numeric_columns:
        if column in stats.columns:
            values = stats[column]
            mask = ~values.map(_is_empty)
            invalid = pd.to_numeric(values[mask], errors="coerce").isna()
            if invalid.any():
                errors.append(f"new_fight_stats.csv has {int(invalid.sum())} non-numeric values in {column}.")
    if not scorecards.empty and not fights.empty:
        fight_methods = {fight_identity(row): _clean(row.get("method")).lower() for _, row in fights.iterrows()}
        for _, row in scorecards.iterrows():
            key = fight_identity(row)
            method = fight_methods.get(key, "")
            if method and "decision" not in method:
                errors.append("Scorecards are staged for a non-decision fight.")
                break
            if not method:
                warnings.append("A staged scorecard could not be matched to a staged fight.")
    leakage_columns = {"winner", "method", "finish_round", "finish_time", "result"}
    leaked = sorted(leakage_columns & set(enrichment.columns))
    if leaked:
        errors.append("new_event_enrichment.csv contains post-fight result columns: " + ", ".join(leaked))
    for _, row in fighters.iterrows():
        name = _clean(row.get("name"))
        if not name:
            warnings.append("A staged fighter row has a blank name.")
        for column in ["reach_in", "height_in", "stance", "date_of_birth"]:
            if column in fighters.columns and _is_empty(row.get(column)):
                warnings.append(f"Missing fighter profile field for {name or 'unknown fighter'}: {column}")
    if (staging / "new_odds.csv").exists():
        odds = _read_frame(staging / "new_odds.csv", STAGING_COLUMNS["new_odds.csv"])
        if odds.empty:
            warnings.append("No staged odds were found for this card.")
    else:
        warnings.append("No staged odds file was found.")
    if scorecards.empty:
        warnings.append("No staged scorecards were found.")
    report = ValidationReport(ok=not errors, staging_dir=staging, errors=errors, warnings=warnings, counts=counts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return report


def _copy_backup(paths: list[Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            destination = backup_dir / path.name
            if destination.exists():
                destination = backup_dir / f"{path.parent.name}_{path.name}"
            shutil.copy2(path, destination)


def _next_fight_ids(existing: pd.DataFrame, count: int) -> list[int]:
    if existing.empty or "fight_id" not in existing.columns:
        start = 0
    else:
        numeric = pd.to_numeric(existing["fight_id"], errors="coerce").dropna()
        start = int(numeric.max()) + 1 if not numeric.empty else len(existing)
    return list(range(start, start + count))


def _merge_fighters(existing: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int, dict[str, int]]:
    if existing.empty:
        existing = pd.DataFrame(columns=STAGING_COLUMNS["new_fighters.csv"])
    output = existing.copy()
    for column in STAGING_COLUMNS["new_fighters.csv"]:
        if column not in output.columns:
            output[column] = pd.NA
        if column not in new.columns:
            new[column] = pd.NA
    added = 0
    updated: dict[str, int] = {}
    name_to_index = {_normal(row.get("name")): index for index, row in output.iterrows() if _normal(row.get("name"))}
    for _, row in new.iterrows():
        key = _normal(row.get("name"))
        if not key:
            continue
        if key not in name_to_index:
            output = pd.concat([output, pd.DataFrame([row])], ignore_index=True)
            name_to_index[key] = output.index[-1]
            added += 1
            continue
        target_index = name_to_index[key]
        for column in ["stance", "height_in", "weight_lb", "reach_in", "date_of_birth", "record", "source_url", "nickname", "weight_class"]:
            new_value = row.get(column)
            if column in output.columns and not _is_empty(new_value) and _is_empty(output.at[target_index, column]):
                output.at[target_index, column] = new_value
                updated[column] = updated.get(column, 0) + 1
    return output, added, updated


def _append_dedup(existing: pd.DataFrame, new: pd.DataFrame, key_columns: list[str]) -> tuple[pd.DataFrame, int, int]:
    if new.empty:
        return existing.copy(), 0, 0
    if existing.empty:
        return new.copy(), int(len(new)), 0
    output = existing.copy()
    for column in key_columns:
        if column not in output.columns:
            output[column] = pd.NA
        if column not in new.columns:
            new[column] = pd.NA
    existing_keys = set(output[key_columns].astype(str).agg("|".join, axis=1))
    rows = []
    skipped = 0
    for _, row in new.iterrows():
        key = "|".join(str(row.get(column, "")) for column in key_columns)
        if key in existing_keys:
            skipped += 1
            continue
        existing_keys.add(key)
        rows.append(row)
    if rows:
        output = pd.concat([output, pd.DataFrame(rows)], ignore_index=True)
    return output, len(rows), skipped


def _merge_enrichment(existing: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int, dict[str, int]]:
    if existing.empty:
        return new.copy(), int(len(new)), {}
    output = existing.copy()
    updated: dict[str, int] = {}
    added_rows: list[pd.Series] = []
    index_by_key = {fight_identity(row): index for index, row in output.iterrows()}
    for _, row in new.iterrows():
        key = fight_identity(row)
        if key not in index_by_key:
            added_rows.append(row)
            continue
        target = index_by_key[key]
        for field in ["weight_class", "event_location", "main_event", "title_fight", "scheduled_rounds"]:
            if field not in output.columns:
                output[field] = pd.NA
            value = row.get(field)
            if not _is_empty(value) and _is_empty(output.at[target, field]):
                output.at[target, field] = value
                updated[field] = updated.get(field, 0) + 1
    if added_rows:
        output = pd.concat([output, pd.DataFrame(added_rows)], ignore_index=True)
    return output, len(added_rows), updated


def apply_card_update(
    staging_dir: str | Path | None = None,
    imports_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    backup_root: str | Path | None = None,
) -> ApplyReport:
    staging = Path(staging_dir) if staging_dir else settings.raw_data_dir / "staging"
    imports = Path(imports_dir) if imports_dir else settings.raw_data_dir / "imports"
    raw = Path(raw_dir) if raw_dir else settings.raw_data_dir
    raw.mkdir(parents=True, exist_ok=True)
    validation_path = staging / "validation_report.json"
    if not validation_path.exists():
        validation = validate_card_update(staging, imports)
    else:
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        validation = ValidationReport(ok=bool(payload.get("ok")), staging_dir=staging, errors=payload.get("errors", []), warnings=payload.get("warnings", []), counts=payload.get("counts", {}))
    if not validation.ok:
        raise InputDataError("Card update validation failed; refusing to merge staging data.")

    backup_base = Path(backup_root) if backup_root else raw / "backups"
    backup_dir = backup_base / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    targets = [
        imports / "fights.csv",
        imports / "fight_stats.csv",
        imports / "fighters.csv",
        imports / "scorecards.csv",
        imports / "fight_enrichment.csv",
        imports / "odds.csv",
        raw / "scorecards.csv",
        raw / "odds.csv",
    ]
    _copy_backup(targets, backup_dir)
    report = ApplyReport(backup_dir=backup_dir)
    imports.mkdir(parents=True, exist_ok=True)

    existing_fights = read_optional_csv(imports / "fights.csv", label="existing fights CSV")
    existing_fights = existing_fights if existing_fights is not None else pd.DataFrame(columns=FIGHT_COLUMNS)
    new_fights = _read_frame(staging / "new_fights.csv", STAGING_COLUMNS["new_fights.csv"])
    existing_keys = {fight_identity(row) for _, row in existing_fights.iterrows()}
    id_mapping: dict[Any, Any] = {}
    rows_to_add: list[pd.Series] = []
    next_ids = _next_fight_ids(existing_fights, len(new_fights))
    skipped_fights = 0
    for generated_id, (_, row) in zip(next_ids, new_fights.iterrows()):
        key = fight_identity(row)
        if key in existing_keys:
            skipped_fights += 1
            continue
        old_id = row.get("fight_id")
        row = row.copy()
        row["fight_id"] = generated_id
        id_mapping[old_id] = generated_id
        rows_to_add.append(row)
        existing_keys.add(key)
    merged_fights = pd.concat([existing_fights, pd.DataFrame(rows_to_add)], ignore_index=True) if rows_to_add else existing_fights.copy()
    for column in FIGHT_COLUMNS:
        if column not in merged_fights.columns:
            merged_fights[column] = pd.NA
    merged_fights.to_csv(imports / "fights.csv", index=False)
    report.rows_added["fights"] = len(rows_to_add)
    report.skipped_duplicates["fights"] = skipped_fights
    report.files_written["fights"] = imports / "fights.csv"

    existing_fighters = read_optional_csv(imports / "fighters.csv", label="existing fighters CSV")
    existing_fighters = existing_fighters if existing_fighters is not None else pd.DataFrame(columns=FIGHTER_COLUMNS)
    new_fighters = _read_frame(staging / "new_fighters.csv", STAGING_COLUMNS["new_fighters.csv"])
    merged_fighters, added_fighters, fighter_updates = _merge_fighters(existing_fighters, new_fighters)
    merged_fighters.to_csv(imports / "fighters.csv", index=False)
    report.rows_added["fighters"] = added_fighters
    for field, count in fighter_updates.items():
        report.fields_updated[f"fighters.{field}"] = count
    report.files_written["fighters"] = imports / "fighters.csv"

    existing_stats = read_optional_csv(imports / "fight_stats.csv", label="existing fight stats CSV")
    existing_stats = existing_stats if existing_stats is not None else pd.DataFrame(columns=FIGHT_STAT_COLUMNS)
    new_stats = _read_frame(staging / "new_fight_stats.csv", STAGING_COLUMNS["new_fight_stats.csv"])
    if not new_stats.empty and id_mapping:
        new_stats = new_stats[new_stats["fight_id"].isin(id_mapping)].copy()
        new_stats["fight_id"] = new_stats["fight_id"].map(id_mapping)
    merged_stats, added_stats, skipped_stats = _append_dedup(existing_stats, new_stats, ["fight_id", "fighter"])
    merged_stats.to_csv(imports / "fight_stats.csv", index=False)
    report.rows_added["fight_stats"] = added_stats
    report.skipped_duplicates["fight_stats"] = skipped_stats
    report.files_written["fight_stats"] = imports / "fight_stats.csv"

    existing_scorecards = read_optional_csv(imports / "scorecards.csv", label="existing scorecards CSV")
    existing_scorecards = existing_scorecards if existing_scorecards is not None else pd.DataFrame()
    new_scorecards = _read_frame(staging / "new_scorecards.csv", STAGING_COLUMNS["new_scorecards.csv"])
    merged_scorecards, added_scorecards, skipped_scorecards = _append_dedup(
        existing_scorecards,
        new_scorecards,
        ["fight_date", "fighter_a", "fighter_b", "judge", "raw_scorecards"],
    )
    merged_scorecards.to_csv(imports / "scorecards.csv", index=False)
    merged_scorecards.to_csv(raw / "scorecards.csv", index=False)
    report.rows_added["scorecards"] = added_scorecards
    report.skipped_duplicates["scorecards"] = skipped_scorecards
    report.files_written["scorecards"] = imports / "scorecards.csv"

    existing_enrichment = read_optional_csv(imports / "fight_enrichment.csv", label="existing fight enrichment CSV")
    existing_enrichment = existing_enrichment if existing_enrichment is not None else pd.DataFrame()
    new_enrichment = _read_frame(staging / "new_event_enrichment.csv", STAGING_COLUMNS["new_event_enrichment.csv"])
    merged_enrichment, added_enrichment, enrichment_updates = _merge_enrichment(existing_enrichment, new_enrichment)
    merged_enrichment.to_csv(imports / "fight_enrichment.csv", index=False)
    report.rows_added["fight_enrichment"] = added_enrichment
    for field, count in enrichment_updates.items():
        report.fields_updated[f"fight_enrichment.{field}"] = count
    report.files_written["fight_enrichment"] = imports / "fight_enrichment.csv"

    new_odds = _read_frame(staging / "new_odds.csv", STAGING_COLUMNS["new_odds.csv"])
    if not new_odds.empty:
        for odds_target in [imports / "odds.csv", raw / "odds.csv"]:
            existing_odds = read_optional_csv(odds_target, label="existing odds CSV")
            existing_odds = existing_odds if existing_odds is not None else pd.DataFrame()
            merged_odds, added_odds, skipped_odds = _append_dedup(
                existing_odds,
                new_odds,
                ["fight_date", "fighter_a", "fighter_b", "sportsbook", "timestamp"],
            )
            merged_odds.to_csv(odds_target, index=False)
            report.files_written[str(odds_target)] = odds_target
        report.rows_added["odds"] = added_odds
        report.skipped_duplicates["odds"] = skipped_odds

    apply_report_path = staging / "apply_report.json"
    apply_report_path.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    report.files_written["apply_report"] = apply_report_path
    return report


def run_rebuild_after_update() -> list[dict[str, Any]]:
    steps = [
        ["validate-card-update"],
        ["apply-card-update"],
        ["enrich-fighter-profiles", "--source", "auto"],
        ["validate-fighter-profile-enrichment"],
        ["apply-fighter-profile-enrichment"],
        ["build-dataset", "--verbose"],
        ["train"],
        ["backtest", "--model-mode", "pure"],
        ["backtest", "--model-mode", "market-aware"],
        ["compare-model-modes"],
        ["leakage-audit", "--sample-size", "100"],
        ["report"],
    ]
    results: list[dict[str, Any]] = []
    for step in steps:
        command = [sys.executable, "-m", "ufc_predictor.cli", *step]
        completed = subprocess.run(command, cwd=settings.project_root, text=True, capture_output=True)
        results.append(
            {
                "step": " ".join(step),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Failed step: {' '.join(step)}\n{completed.stdout}\n{completed.stderr}")
    return results


def prepare_upcoming_card(
    event_url: str,
    output_path: str | Path | None = None,
    source: str = "auto",
) -> tuple[pd.DataFrame, Path]:
    output = Path(output_path) if output_path else settings.raw_data_dir / "staging" / "upcoming_card_predictions.csv"
    source_name = source.strip().lower()
    if source_name == "auto":
        source_name = "ufcstats" if "ufcstats.com" in event_url.lower() else "manual"
    if source_name == "ufcstats":
        fights, _, _, _, _, _, _, diagnostics = _ufcstats_rows(event_url=event_url)
        if not fights:
            raise InputDataError(f"UFCStats upcoming-card parser produced zero fights: {diagnostics.get('parse_reason')}")
        rows = [
            {
                "fighter_a": fight.get("fighter_a", ""),
                "fighter_b": fight.get("fighter_b", ""),
                "date": _date_key(fight.get("fight_date", "")),
                "weight_class": fight.get("weight_class", "Unknown"),
                "scheduled_rounds": fight.get("scheduled_rounds", 3),
                "fighter_a_odds": "",
                "fighter_b_odds": "",
                "main_event": 1 if index == 0 else 0,
                "title_fight": 1 if pd.to_numeric(pd.Series([fight.get("scheduled_rounds")]), errors="coerce").fillna(0).iloc[0] >= 5 else 0,
            }
            for index, fight in enumerate(fights)
        ]
    else:
        frame = _read_generic_source(event_url)
        fights, _, _, _, enrichment, odds = _generic_rows_from_frame(frame, event_url)
        odds_by_key = {fight_identity(row): row for row in odds}
        enrichment_by_key = {fight_identity(row): row for row in enrichment}
        rows = []
        for fight in fights:
            key = fight_identity(fight)
            enr = enrichment_by_key.get(key, {})
            odd = odds_by_key.get(key, {})
            rows.append(
                {
                    "fighter_a": fight.get("fighter_a", ""),
                    "fighter_b": fight.get("fighter_b", ""),
                    "date": _date_key(fight.get("fight_date", "")),
                    "weight_class": enr.get("weight_class", fight.get("weight_class", "Unknown")),
                    "scheduled_rounds": enr.get("scheduled_rounds", fight.get("scheduled_rounds", 3)),
                    "fighter_a_odds": odd.get("fighter_a_odds", ""),
                    "fighter_b_odds": odd.get("fighter_b_odds", ""),
                    "main_event": enr.get("main_event", fight.get("main_event", 0)),
                    "title_fight": enr.get("title_fight", fight.get("title_fight", 0)),
                }
            )
    columns = [
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
    frame = pd.DataFrame(rows, columns=columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame, output
