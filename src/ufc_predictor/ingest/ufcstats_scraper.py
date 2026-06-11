from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.config import settings
from ufc_predictor.data_sources import SOURCE_CSV_IMPORT, SOURCE_LIVE_SCRAPE, SOURCE_MANUAL_HTML, write_source_metadata
from ufc_predictor.data_io import InputDataError, inspect_csv


class UFCStatsScraperError(RuntimeError):
    """Raised when UFCStats scraping cannot produce a usable raw dataset."""

    def __init__(self, message: str, diagnostics: "FetchDiagnostics | None" = None):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass
class FetchDiagnostics:
    requested_url: str
    status_code: int | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    body_preview: str = ""
    cached_data_used: bool = False
    attempts: int = 0

    def format(self) -> str:
        lines = [
            f"Requested URL: {self.requested_url}",
            f"HTTP status code: {self.status_code if self.status_code is not None else 'n/a'}",
            f"Exception type: {self.exception_type or 'n/a'}",
            f"Exception message: {self.exception_message or 'n/a'}",
            f"Cached data used: {self.cached_data_used}",
            f"Attempts: {self.attempts}",
            "Body preview:",
            self.body_preview[:500] if self.body_preview else "n/a",
        ]
        return "\n".join(lines)


EVENT_COLUMNS = ["name", "event_date", "location", "source_url", "raw_details"]
FIGHT_COLUMNS = [
    "fight_id",
    "event_name",
    "fight_date",
    "event_location",
    "fighter_a",
    "fighter_b",
    "winner",
    "weight_class",
    "method",
    "finish_round",
    "finish_time",
    "scheduled_rounds",
    "main_event",
    "source_url",
]
FIGHT_STAT_COLUMNS = [
    "fight_id",
    "source_url",
    "fighter",
    "opponent",
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
    "head_landed",
    "body_landed",
    "leg_landed",
]
FIGHTER_COLUMNS = ["name", "stance", "height_in", "weight_lb", "reach_in", "date_of_birth", "record", "source_url"]
DEFAULT_COMPLETED_EVENTS_URL = "https://www.ufcstats.com/statistics/events/completed?page=all"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def _slug(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return digest


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def _parse_landed_attempted(value: str) -> tuple[float | None, float | None]:
    text = _clean_text(value).lower()
    if "of" not in text:
        return None, None
    left, right = text.split("of", 1)
    try:
        return float(left.strip()), float(right.strip())
    except ValueError:
        return None, None


def _parse_seconds(value: str) -> float | None:
    text = _clean_text(value)
    if not text or text == "--":
        return None
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        try:
            return float(minutes) * 60.0 + float(seconds)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _pair_values(cell: Any) -> list[str]:
    values = [_clean_text(item) for item in cell.select("p")]
    values = [value for value in values if value]
    if len(values) >= 2:
        return values[:2]
    text = _clean_text(cell)
    return [text, text] if text else ["", ""]


def _soup_from_html(html: str) -> Any:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("beautifulsoup4 is required for scraping.") from exc
    return BeautifulSoup(html, "html.parser")


def _event_links_from_soup(soup: Any) -> list[str]:
    links: list[str] = []
    for anchor in soup.select('a[href*="/event-details/"]'):
        href = anchor.get("href")
        if href and "/event-details/" in href:
            if href.startswith("/"):
                href = f"https://www.ufcstats.com{href}"
            links.append(href)
    return list(dict.fromkeys(links))


@dataclass
class UFCStatsScraper:
    """Respectful UFCStats scraper with cache, retries, rate limiting, and resume support."""

    cache_dir: Path = settings.cache_dir
    user_agent: str = settings.user_agent
    delay_seconds: float = settings.scrape_delay_seconds
    retry_count: int = settings.retry_count
    timeout_seconds: int = settings.request_timeout_seconds
    backoff_base_seconds: float = 1.0
    base_url: str = "https://www.ufcstats.com"
    session: Any = field(default=None, init=False)
    _last_request_at: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_session(self) -> Any:
        if self.session is None:
            try:
                import requests
            except Exception as exc:  # pragma: no cover - depends on environment
                raise RuntimeError("requests is required for scraping. Install project dependencies.") from exc
            self.session = requests.Session()
            headers = DEFAULT_HEADERS.copy()
            if self.user_agent and self.user_agent != settings.user_agent:
                headers["User-Agent"] = self.user_agent
            self.session.headers.update(headers)
        return self.session

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{_slug(url)}.html"

    def fetch(self, url: str, force: bool = False) -> str:
        """Fetch a URL using local cache unless force=True."""

        cache_path = self._cache_path(url)
        if cache_path.exists() and not force:
            cached = cache_path.read_text(encoding="utf-8")
            if cached.strip():
                return cached
            cache_path.unlink(missing_ok=True)
        diagnostics = self.fetch_diagnostics(url, force=force)
        if diagnostics.exception_type is None and cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        raise UFCStatsScraperError(
            f"Failed to fetch {url}. Check network access, UFCStats availability, "
            "and the local scraper cache under data/raw/cache.",
            diagnostics=diagnostics,
        )

    def fetch_diagnostics(self, url: str, force: bool = True) -> FetchDiagnostics:
        """Probe a URL and return diagnostics without hiding HTTP/network details."""

        cache_path = self._cache_path(url)
        if cache_path.exists() and not force:
            cached = cache_path.read_text(encoding="utf-8")
            if cached.strip():
                return FetchDiagnostics(
                    requested_url=url,
                    status_code=None,
                    body_preview=cached[:500],
                    cached_data_used=True,
                    attempts=0,
                )
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        session = self._ensure_session()
        diagnostics = FetchDiagnostics(requested_url=url)
        attempts = max(self.retry_count, 1)
        for attempt in range(attempts):
            diagnostics.attempts = attempt + 1
            try:
                response = session.get(url, timeout=self.timeout_seconds)
                diagnostics.status_code = response.status_code
                diagnostics.body_preview = response.text[:500] if response.text else ""
                response.raise_for_status()
                self._last_request_at = time.monotonic()
                cache_path.write_text(response.text, encoding="utf-8")
                diagnostics.cached_data_used = False
                return diagnostics
            except Exception as exc:  # pragma: no cover - network dependent
                diagnostics.exception_type = type(exc).__name__
                diagnostics.exception_message = str(exc)
                if hasattr(exc, "response") and exc.response is not None:
                    diagnostics.status_code = getattr(exc.response, "status_code", diagnostics.status_code)
                    diagnostics.body_preview = getattr(exc.response, "text", "")[:500]
                if attempt < attempts - 1:
                    time.sleep(min(self.backoff_base_seconds * (2**attempt), 10.0))
        return diagnostics

    def soup(self, url: str) -> Any:
        return _soup_from_html(self.fetch(url))

    def scrape_event_links(self, completed_only: bool = True) -> list[str]:
        suffix = "statistics/events/completed?page=all" if completed_only else "statistics/events/search?page=all"
        soup = self.soup(f"{self.base_url}/{suffix}")
        links = _event_links_from_soup(soup)
        if not links:
            raise UFCStatsScraperError(
                "No UFCStats event links were found on the completed-events page. "
                "The site layout may have changed, the response may be blocked, or "
                "a stale/empty cache file may be present under data/raw/cache."
            )
        return links

    def scrape_event_links_from_html(self, html_path: str | Path) -> list[str]:
        path = Path(html_path)
        if not path.exists():
            raise UFCStatsScraperError(f"Manual UFCStats HTML file not found: {path}")
        html = path.read_text(encoding="utf-8")
        links = _event_links_from_soup(_soup_from_html(html))
        if not links:
            raise UFCStatsScraperError(
                f"No UFCStats event links were found in manual HTML file: {path}. "
                "Make sure it is the completed-events page HTML from UFCStats."
            )
        return links

    def scrape_event(self, event_url: str) -> dict[str, Any]:
        soup = self.soup(event_url)
        title = _clean_text(soup.select_one(".b-content__title-highlight"))
        details = [_clean_text(item) for item in soup.select(".b-list__box-list-item")]
        detail_text = " | ".join(details)
        event_date = ""
        location = ""
        for item in details:
            lower = item.lower()
            if lower.startswith("date:"):
                event_date = item.split(":", 1)[1].strip()
            if lower.startswith("location:"):
                location = item.split(":", 1)[1].strip()

        fights = []
        for row in soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover"):
            fight_link = row.get("data-link") or ""
            cells = row.select("td")
            if len(cells) < 10:
                continue
            cell_text = [_clean_text(cell) for cell in cells]
            statuses = [value.lower() for value in _pair_values(cells[0])]
            fighters = _pair_values(cells[1])
            winner = ""
            if "win" in statuses:
                winner_index = statuses.index("win")
                if winner_index < len(fighters):
                    winner = fighters[winner_index]
            fights.append(
                {
                    "event_name": title,
                    "fight_date": event_date,
                    "event_location": location,
                    "fighter_a": fighters[0] if fighters else "",
                    "fighter_b": fighters[1] if len(fighters) > 1 else "",
                    "winner": winner,
                    "weight_class": cell_text[6] if len(cell_text) > 6 else "",
                    "method": cell_text[7] if len(cell_text) > 7 else "",
                    "finish_round": cell_text[8] if len(cell_text) > 8 else "",
                    "finish_time": cell_text[9] if len(cell_text) > 9 else "",
                    "scheduled_rounds": 5 if "title" in title.lower() else 3,
                    "source_url": fight_link,
                }
            )
        return {
            "event": {
                "name": title,
                "event_date": event_date,
                "location": location,
                "source_url": event_url,
                "raw_details": detail_text,
            },
            "fights": fights,
        }

    def scrape_fight(self, fight_url: str) -> dict[str, Any]:
        soup = self.soup(fight_url)
        fight: dict[str, Any] = {"source_url": fight_url}
        for item in soup.select(".b-fight-details__fight-title, .b-list__box-list-item"):
            text = _clean_text(item)
            if ":" in text:
                key, value = text.split(":", 1)
                fight[key.strip().lower().replace(" ", "_")] = value.strip()
        totals = []
        for row in soup.select("tbody.b-fight-details__table-body tr"):
            cells = [_clean_text(cell) for cell in row.select("td")]
            if cells:
                totals.append(cells)
        fight["raw_stat_rows_json"] = json.dumps(totals)
        return fight

    def scrape_fighter(self, fighter_url: str) -> dict[str, Any]:
        """Parse a UFCStats fighter profile page."""

        soup = self.soup(fighter_url)
        profile: dict[str, Any] = {"source_url": fighter_url}
        profile["name"] = _clean_text(soup.select_one(".b-content__title-highlight"))
        profile["record"] = _clean_text(soup.select_one(".b-content__title-record")).replace("Record:", "").strip()
        for item in soup.select(".b-list__box-list-item"):
            text = _clean_text(item)
            if ":" not in text:
                continue
            key, value = text.split(":", 1)
            key = key.lower().strip().replace(" ", "_").replace(".", "")
            profile[key] = value.strip()
        return {
            "name": profile.get("name", ""),
            "stance": profile.get("stance", ""),
            "height_in": profile.get("height", ""),
            "weight_lb": profile.get("weight", "").replace("lbs.", "").strip(),
            "reach_in": profile.get("reach", "").replace('"', "").strip(),
            "date_of_birth": profile.get("dob", ""),
            "record": profile.get("record", ""),
            "source_url": fighter_url,
        }

    def scrape_fight_stats(self, fight_url: str) -> list[dict[str, Any]]:
        """Parse per-fighter fight statistics from a UFCStats fight page."""

        soup = self.soup(fight_url)
        rows: list[dict[str, Any]] = []
        tables = soup.select("table.b-fight-details__table")
        if not tables:
            return rows

        totals_by_fighter: dict[str, dict[str, Any]] = {}
        for table in tables:
            header_text = _clean_text(table.find_previous("p") or "").lower()
            for tr in table.select("tbody tr"):
                cells = tr.select("td")
                if len(cells) < 2:
                    continue
                fighter_names = _pair_values(cells[0])
                if not fighter_names[0]:
                    continue
                cell_values = [_pair_values(cell) for cell in cells]
                for index, fighter_name in enumerate(fighter_names[:2]):
                    row = totals_by_fighter.setdefault(
                        fighter_name,
                        {
                            "source_url": fight_url,
                            "fighter": fighter_name,
                            "opponent": fighter_names[1 - index] if len(fighter_names) > 1 else "",
                        },
                    )
                    if "sig. str. by target" in header_text or len(cells) >= 9:
                        head_landed, _ = _parse_landed_attempted(cell_values[6][index] if len(cell_values) > 6 else "")
                        body_landed, _ = _parse_landed_attempted(cell_values[7][index] if len(cell_values) > 7 else "")
                        leg_landed, _ = _parse_landed_attempted(cell_values[8][index] if len(cell_values) > 8 else "")
                        row["head_landed"] = head_landed
                        row["body_landed"] = body_landed
                        row["leg_landed"] = leg_landed
                    else:
                        kd = cell_values[1][index] if len(cell_values) > 1 else ""
                        sig = cell_values[2][index] if len(cell_values) > 2 else ""
                        total = cell_values[4][index] if len(cell_values) > 4 else ""
                        td = cell_values[5][index] if len(cell_values) > 5 else ""
                        sig_landed, sig_attempted = _parse_landed_attempted(sig)
                        total_landed, total_attempted = _parse_landed_attempted(total)
                        td_landed, td_attempted = _parse_landed_attempted(td)
                        row.update(
                            {
                                "knockdowns": kd,
                                "sig_str_landed": sig_landed,
                                "sig_str_attempted": sig_attempted,
                                "total_str_landed": total_landed,
                                "total_str_attempted": total_attempted,
                                "takedowns_landed": td_landed,
                                "takedowns_attempted": td_attempted,
                                "submission_attempts": cell_values[7][index] if len(cell_values) > 7 else "",
                                "reversals": cell_values[8][index] if len(cell_values) > 8 else "",
                                "control_seconds": _parse_seconds(cell_values[9][index] if len(cell_values) > 9 else ""),
                            }
                        )
        rows.extend(totals_by_fighter.values())
        return rows

    def scrape_events(
        self,
        max_events: int | None = None,
        resume_file: str | Path | None = None,
        ignore_resume: bool = False,
        event_links: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scrape event and fight-card rows.

        This method caches every page and records completed event URLs in a resume
        file, making interrupted runs safe to restart.
        """

        resume_path = Path(resume_file) if resume_file else self.cache_dir / "ufcstats_resume.json"
        completed: set[str] = set()
        if resume_path.exists() and not ignore_resume:
            completed = set(json.loads(resume_path.read_text(encoding="utf-8")))

        event_rows: list[dict[str, Any]] = []
        fight_rows: list[dict[str, Any]] = []
        links = event_links if event_links is not None else self.scrape_event_links()
        if max_events is not None:
            links = links[:max_events]
        for link in links:
            if link in completed:
                continue
            payload = self.scrape_event(link)
            if not payload["fights"]:
                continue
            event_rows.append(payload["event"])
            fight_rows.extend(payload["fights"])
            completed.add(link)
            resume_path.write_text(json.dumps(sorted(completed), indent=2), encoding="utf-8")
        events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
        fights = pd.DataFrame(fight_rows)
        if fights.empty:
            raise UFCStatsScraperError(
                "UFCStats scraping completed but produced zero fight rows. The event-page "
                "parser may need updating, all requested events may be marked complete in "
                "the resume file, or cached pages may be stale. Check data/raw/cache or "
                "rerun with --ignore-resume."
            )
        return events, fights

    def run_to_csv(
        self,
        output_dir: str | Path | None = None,
        max_events: int | None = None,
        include_details: bool = False,
        ignore_resume: bool = False,
        from_html: str | Path | None = None,
    ) -> dict[str, Path]:
        output = Path(output_dir) if output_dir else settings.raw_data_dir
        output.mkdir(parents=True, exist_ok=True)
        event_links = self.scrape_event_links_from_html(from_html) if from_html else None
        events, fights = self.scrape_events(
            max_events=max_events,
            ignore_resume=ignore_resume or from_html is not None,
            event_links=event_links,
        )
        if not fights.empty and "fight_id" not in fights.columns:
            fights = fights.reset_index(drop=True)
            fights["fight_id"] = fights.index
        for column in FIGHT_COLUMNS:
            if column not in fights.columns:
                fights[column] = ""
        fights = fights[FIGHT_COLUMNS]
        if fights.empty:
            raise UFCStatsScraperError("UFCStats scraping produced no fights; refusing to write an empty fights.csv.")
        if fights[["fighter_a", "fighter_b", "fight_date"]].isna().any().any() or (
            fights[["fighter_a", "fighter_b", "fight_date"]].astype(str).apply(lambda column: column.str.strip()).eq("").any().any()
        ):
            raise UFCStatsScraperError(
                "UFCStats scraping produced fight rows with missing fighter names or dates; "
                "refusing to write an invalid fights.csv."
            )
        fight_stats_rows: list[dict[str, Any]] = []
        fighter_rows: dict[str, dict[str, Any]] = {}
        if include_details and not fights.empty and "source_url" in fights.columns:
            for _, row in fights.dropna(subset=["source_url"]).iterrows():
                fight_url = row["source_url"]
                stats_rows = self.scrape_fight_stats(fight_url)
                for stats_row in stats_rows:
                    stats_row["fight_id"] = row.get("fight_id")
                    fight_stats_rows.append(stats_row)
                soup = self.soup(fight_url)
                for anchor in soup.select('a[href*="/fighter-details/"]'):
                    href = anchor.get("href")
                    if href and href not in fighter_rows:
                        fighter_rows[href] = self.scrape_fighter(href)
        paths = {
            "events": output / "events.csv",
            "fights": output / "fights.csv",
        }
        events.reindex(columns=EVENT_COLUMNS).to_csv(paths["events"], index=False)
        fights.to_csv(paths["fights"], index=False)
        try:
            inspect_csv(paths["fights"], required_columns=["fighter_a", "fighter_b", "fight_date"], label="UFCStats fights CSV")
        except InputDataError as exc:
            raise UFCStatsScraperError(str(exc)) from exc
        if include_details:
            paths["fight_stats"] = output / "fight_stats.csv"
            paths["fighters"] = output / "fighters.csv"
            pd.DataFrame(fight_stats_rows, columns=FIGHT_STAT_COLUMNS).to_csv(paths["fight_stats"], index=False)
            pd.DataFrame(fighter_rows.values(), columns=FIGHTER_COLUMNS).to_csv(paths["fighters"], index=False)
        write_source_metadata(
            SOURCE_MANUAL_HTML if from_html else SOURCE_LIVE_SCRAPE,
            raw_dir=output,
            details={"from_html": str(from_html) if from_html else None, "include_details": include_details},
        )
        return paths


def check_completed_events_page() -> FetchDiagnostics:
    return UFCStatsScraper().fetch_diagnostics(DEFAULT_COMPLETED_EVENTS_URL, force=True)


def import_raw_csvs(
    import_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    source = Path(import_dir)
    output = Path(output_dir) if output_dir else settings.raw_data_dir
    if not source.exists():
        raise UFCStatsScraperError(f"CSV import directory not found: {source}")
    output.mkdir(parents=True, exist_ok=True)
    required_fights = source / "fights.csv"
    try:
        inspect_csv(required_fights, required_columns=["fighter_a", "fighter_b", "fight_date"], label="imported fights CSV")
    except InputDataError as exc:
        raise UFCStatsScraperError(str(exc)) from exc
    paths: dict[str, Path] = {}
    if source.resolve() != output.resolve():
        for filename in ["fights.csv", "fighters.csv", "fight_stats.csv", "events.csv", "scorecards.csv"]:
            (output / filename).unlink(missing_ok=True)
    for name, columns in [
        ("fights", FIGHT_COLUMNS),
        ("fighters", FIGHTER_COLUMNS),
        ("fight_stats", FIGHT_STAT_COLUMNS),
        ("events", EVENT_COLUMNS),
        ("scorecards", None),
    ]:
        source_file = source / f"{name}.csv"
        if not source_file.exists():
            continue
        frame = pd.read_csv(source_file)
        if name == "fights":
            for column in FIGHT_COLUMNS:
                if column not in frame.columns:
                    frame[column] = ""
            extra_columns = [column for column in frame.columns if column not in FIGHT_COLUMNS]
            frame = frame[[*FIGHT_COLUMNS, *extra_columns]]
        elif columns is None:
            frame = frame.copy()
        elif name == "fight_stats":
            for column in columns:
                if column not in frame.columns:
                    frame[column] = ""
            extra_columns = [column for column in frame.columns if column not in columns]
            frame = frame[[*columns, *extra_columns]]
        else:
            frame = frame.reindex(columns=columns)
        destination = output / f"{name}.csv"
        frame.to_csv(destination, index=False)
        paths[name] = destination
    if "fights" not in paths:
        raise UFCStatsScraperError(f"CSV import did not write fights.csv from {source}")
    write_source_metadata(SOURCE_CSV_IMPORT, raw_dir=output, details={"import_dir": str(source)})
    return paths
