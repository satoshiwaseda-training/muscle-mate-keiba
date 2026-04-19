"""Shared helpers for weekend-batch automation scripts.

Intentionally stdlib-only. Every function is side-effect-isolated so scripts
can be partially re-run for recovery without corrupting state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

# Windows consoles default to cp932 which cannot encode several chars we use
# (em-dash, many kanji when stdout is not a TTY, etc). Force UTF-8 with a
# replacement fallback so logging NEVER crashes the batch.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ── Paths ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUTOLOG_DIR = DATA_DIR / "autolog"
WEEKLY_REPORT_DIR = DATA_DIR / "weekly_reports"


def ensure_project_on_path() -> None:
    """Make the project root importable (scraper, live_pipeline, ...).

    Must be called BEFORE any `import scraper` / `import live_pipeline`
    in the tool scripts, since they live in a subdirectory.
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


# ── Weekday gating ────────────────────────────────────

SAT = 5
SUN = 6
MON = 0

WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]


def is_weekend(d: dt.date) -> bool:
    return d.weekday() in (SAT, SUN)


def previous_race_day(d: dt.date) -> dt.date | None:
    """For Sun → Sat, for Mon → Sun, else None."""
    wd = d.weekday()
    if wd == SUN:
        return d - dt.timedelta(days=1)
    if wd == MON:
        return d - dt.timedelta(days=1)
    return None


def last_weekend(d: dt.date) -> tuple[dt.date, dt.date]:
    """Return (last_saturday, last_sunday) for the weekly report.

    If today is Monday → the Sat/Sun that just ended.
    If today is Sunday → yesterday's Sat and today (partial).
    If today is Saturday → the Sat before and the Sun before (fully complete).
    If a weekday → the most recent completed Sat/Sun.
    """
    wd = d.weekday()
    if wd == MON:          # Mon → Sat, Sun (yesterday and day before)
        sun = d - dt.timedelta(days=1)
        sat = d - dt.timedelta(days=2)
    elif wd == SUN:        # Sun → Sat, Sun (yesterday and today)
        sun = d
        sat = d - dt.timedelta(days=1)
    elif wd == SAT:        # Sat → previous Sat, Sun
        sun = d - dt.timedelta(days=6)
        sat = d - dt.timedelta(days=7)
    else:                  # Tue-Fri → most recent completed weekend
        days_since_sun = (wd - SUN) % 7
        sun = d - dt.timedelta(days=days_since_sun)
        sat = sun - dt.timedelta(days=1)
    return sat, sun


# ── Logging ───────────────────────────────────────────

def _log_path(d: dt.date) -> Path:
    AUTOLOG_DIR.mkdir(parents=True, exist_ok=True)
    return AUTOLOG_DIR / f"{d.isoformat()}.log"


def _latest_log_path() -> Path:
    AUTOLOG_DIR.mkdir(parents=True, exist_ok=True)
    return AUTOLOG_DIR / "latest.log"


def log(msg: str, *, level: str = "INFO", date: dt.date | None = None) -> None:
    """Append a timestamped line to today's log and to latest.log, and echo
    to stdout. Never raises — logging must not take down the batch."""
    ts = dt.datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        d = date or dt.date.today()
        with _log_path(d).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        with _latest_log_path().open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[{ts}] [WARN] log-write failed: {e}", flush=True)


def banner(msg: str) -> None:
    bar = "=" * max(40, len(msg) + 4)
    print(bar, flush=True)
    print(f"  {msg}", flush=True)
    print(bar, flush=True)


# ── CLI arg parsing (shared flags) ────────────────────

def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore the weekday guard and run anyway.",
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="Override the target date (YYYY-MM-DD). "
             "For weekend_autolog: the race date to predict. "
             "For attach_results: the race date to attach results for. "
             "For weekly_report: any date inside the target week.",
    )
    return p


def parse_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        raise SystemExit(f"[FATAL] --date must be YYYY-MM-DD, got: {s}")
