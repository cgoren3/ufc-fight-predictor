from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ufc_predictor.config import settings


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


@dataclass
class UFCStatsScraper:
    """Respectful UFCStats scraper with cache, retries, rate limiting, and resume support."""

    cache_dir: Path = settings.cache_dir
    user_agent: str = settings.user_agent
    delay_seconds: float = settings.scrape_delay_seconds
    retry_count: int = settings.retry_count
    timeout_seconds: int = settings.request_timeout_seconds
    base_url: str = "http://ufcstats.com"
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
            self.session.headers.update({"User-Agent": self.user_agent})
        return self.session

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{_slug(url)}.html"

    def fetch(self, url: str, force: bool = False) -> str:
        """Fetch a URL using local cache unless force=True."""

        cache_path = self._cache_path(url)
        if cache_path.exists() and not force:
            return cache_path.read_text(encoding="utf-8")
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        session = self._ensure_session()
        last_error: Exception | None = None
        for attempt in range(self.retry_count):
            try:
                response = session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                self._last_request_at = time.monotonic()
                cache_path.write_text(response.text, encoding="utf-8")
                return response.text
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"Failed to fetch {url}") from last_error

    def soup(self, url: str) -> Any:
        try:
            from bs4 import BeautifulSoup
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("beautifulsoup4 is required for scraping.") from exc
        return BeautifulSoup(self.fetch(url), "html.parser")

    def scrape_event_links(self, completed_only: bool = True) -> list[str]:
        suffix = "statistics/events/completed?page=all" if completed_only else "statistics/events/search?page=all"
        soup = self.soup(f"{self.base_url}/{suffix}")
        links: list[str] = []
        for anchor in soup.select("a.b-link.b-link_style_black"):
            href = anchor.get("href")
            if href and "/event-details/" in href:
                links.append(href)
        return list(dict.fromkeys(links))

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
            cells = [_clean_text(cell) for cell in row.select("td")]
            if len(cells) < 10:
                continue
            fighters = [name.strip() for name in cells[1].split("  ") if name.strip()]
            winners = row.select(".b-flag__text")
            winner = ""
            if winners:
                winner = _clean_text(winners[0])
            fights.append(
                {
                    "event_name": title,
                    "fight_date": event_date,
                    "event_location": location,
                    "fighter_a": fighters[0] if fighters else "",
                    "fighter_b": fighters[1] if len(fighters) > 1 else "",
                    "winner": winner,
                    "weight_class": cells[6] if len(cells) > 6 else "",
                    "method": cells[7] if len(cells) > 7 else "",
                    "finish_round": cells[8] if len(cells) > 8 else "",
                    "finish_time": cells[9] if len(cells) > 9 else "",
                    "scheduled_rounds": cells[10] if len(cells) > 10 else "",
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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scrape event and fight-card rows.

        This method caches every page and records completed event URLs in a resume
        file, making interrupted runs safe to restart.
        """

        resume_path = Path(resume_file) if resume_file else self.cache_dir / "ufcstats_resume.json"
        completed: set[str] = set()
        if resume_path.exists():
            completed = set(json.loads(resume_path.read_text(encoding="utf-8")))

        event_rows: list[dict[str, Any]] = []
        fight_rows: list[dict[str, Any]] = []
        links = self.scrape_event_links()
        if max_events is not None:
            links = links[:max_events]
        for link in links:
            if link in completed:
                continue
            payload = self.scrape_event(link)
            event_rows.append(payload["event"])
            fight_rows.extend(payload["fights"])
            completed.add(link)
            resume_path.write_text(json.dumps(sorted(completed), indent=2), encoding="utf-8")
        return pd.DataFrame(event_rows), pd.DataFrame(fight_rows)

    def run_to_csv(
        self,
        output_dir: str | Path | None = None,
        max_events: int | None = None,
        include_details: bool = False,
    ) -> dict[str, Path]:
        output = Path(output_dir) if output_dir else settings.raw_data_dir
        output.mkdir(parents=True, exist_ok=True)
        events, fights = self.scrape_events(max_events=max_events)
        if not fights.empty and "fight_id" not in fights.columns:
            fights = fights.reset_index(drop=True)
            fights["fight_id"] = fights.index
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
        events.to_csv(paths["events"], index=False)
        fights.to_csv(paths["fights"], index=False)
        if include_details:
            paths["fight_stats"] = output / "fight_stats.csv"
            paths["fighters"] = output / "fighters.csv"
            pd.DataFrame(fight_stats_rows).to_csv(paths["fight_stats"], index=False)
            pd.DataFrame(fighter_rows.values()).to_csv(paths["fighters"], index=False)
        return paths
