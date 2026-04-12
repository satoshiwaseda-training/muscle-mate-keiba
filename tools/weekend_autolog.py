"""Weekend prediction autolog.

Runs on Saturday / Sunday only. Fetches the day's race list from netkeiba,
calls live_pipeline.predict_live(auto_log=True) on every race, and writes
each prediction to data/live_predictions.json via prediction_log.

Weekday guard:
  - Mon-Fri → exit with a logged skip message.
  - --force overrides the guard (for manual recovery).

Usage:
  python tools/weekend_autolog.py                    # normal weekend run
  python tools/weekend_autolog.py --force            # force on a weekday
  python tools/weekend_autolog.py --date 2026-04-11  # predict a specific date
"""

from __future__ import annotations

import datetime as dt
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._autolog_utils import (
    banner, build_parser, ensure_project_on_path, is_weekend,
    log, parse_date, WEEKDAY_JP,
)

ensure_project_on_path()

# Lazy imports after path is fixed
import scraper  # noqa: E402
import live_pipeline as lp  # noqa: E402
import prediction_log as plog  # noqa: E402


def main() -> int:
    parser = build_parser("Weekend prediction autolog (Sat/Sun only).")
    args = parser.parse_args()

    today = dt.date.today()
    race_date = parse_date(args.date) or today
    wd = today.weekday()
    race_wd = race_date.weekday()

    banner(
        f"weekend_autolog: today={today.isoformat()}({WEEKDAY_JP[wd]}) "
        f"target={race_date.isoformat()}({WEEKDAY_JP[race_wd]})"
    )

    # Weekday guard — applies to "today", not to --date
    if not is_weekend(today) and not args.force:
        log(
            f"skip: today is {WEEKDAY_JP[wd]}曜日 (non-weekend). "
            f"Use --force to override.",
            level="SKIP",
        )
        return 0

    # Also guard against predicting a non-weekend race_date (netkeiba returns
    # nothing) unless explicitly forced.
    if not is_weekend(race_date) and not args.force:
        log(
            f"skip: target date {race_date.isoformat()} is "
            f"{WEEKDAY_JP[race_wd]}曜日. JRA does not race on weekdays. "
            f"Use --force to override.",
            level="SKIP",
        )
        return 0

    started = time.time()
    log(f"fetching race list for {race_date.isoformat()} ...")

    try:
        races = scraper.fetch_race_list(race_date) or []
    except Exception as e:
        log(f"FATAL: fetch_race_list failed: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return 2

    if not races:
        log(
            f"no races returned for {race_date.isoformat()}. "
            f"Possible causes: off-season, scraper block, network issue.",
            level="WARN",
        )
        return 1

    log(f"found {len(races)} races. starting predictions.")

    # ── Summary counters (logged at end) ──
    success_count = 0         # persisted, stage=final (trustworthy odds)
    not_published_count = 0   # persisted, stage=early due to not-published API
    unknown_schema_count = 0  # api_schema_version startswith "unknown-"
    skipped_count = 0         # empty race_id / exception / other skip
    loose_count = 0           # total loose bets aggregated

    for i, race in enumerate(races, 1):
        rid = race.get("race_id", "") or ""
        rname = race.get("race_name", "") or ""
        venue = race.get("venue", "") or ""
        if not rid:
            log(f"[{i}/{len(races)}] skip: empty race_id ({rname})", level="WARN")
            skipped_count += 1
            continue

        try:
            t0 = time.time()
            result = lp.predict_live(
                race_id=rid,
                venue=venue,
                race_name=rname,
                race_date=race_date.isoformat(),
                progress_cb=None,
                auto_log=True,  # persists via prediction_log.store_prediction
            )
            elapsed = time.time() - t0
            nloose = len(result.get("loose_bets") or [])
            loose_count += nloose

            stage = result.get("prediction_stage") or "final"
            api_meta = result.get("odds_api_meta") or {}
            schema = str(api_meta.get("api_schema_version") or "")
            is_unknown_schema = schema.startswith("unknown")

            if stage == "final":
                success_count += 1
            else:
                not_published_count += 1
            if is_unknown_schema:
                unknown_schema_count += 1

            log(
                f"[{i}/{len(races)}] ok  rid={rid} stage={stage} "
                f"schema={schema or '-'} '{rname}' loose={nloose} "
                f"({elapsed:.1f}s)"
            )
        except Exception as e:
            skipped_count += 1
            log(
                f"[{i}/{len(races)}] FAIL rid={rid} '{rname}': {e}",
                level="ERROR",
            )
            log(traceback.format_exc(), level="ERROR")
            continue  # one race failure must NOT kill the batch

    # Post-run sanity: count what's actually in live_predictions.json for this race_date
    persisted_today = [
        e for e in plog.list_predictions(only_live=True)
        if e.get("race_date") == race_date.isoformat()
    ]
    persisted_n = len(persisted_today)
    elapsed_total = time.time() - started

    # ── Mandatory summary block (fixed field names for easy grep/monitoring) ──
    log("── weekend_autolog summary ──")
    log(f"  race_date             : {race_date.isoformat()}")
    log(f"  success_count         : {success_count}")
    log(f"  not_published_count   : {not_published_count}")
    log(f"  unknown_schema_count  : {unknown_schema_count}")
    log(f"  attached_count        : 0")  # N/A — attach runs in a separate tool
    log(f"  skipped_count         : {skipped_count}")
    log(f"  total_races           : {len(races)}")
    log(f"  persisted_in_file     : {persisted_n}")
    log(f"  loose_bets_total      : {loose_count}")
    log(f"  elapsed_seconds       : {elapsed_total:.1f}")

    banner(
        f"weekend_autolog finished: "
        f"success={success_count} not_pub={not_published_count} "
        f"unknown_schema={unknown_schema_count} skipped={skipped_count} "
        f"persisted={persisted_n} ({elapsed_total:.1f}s)"
    )

    # Hard failure alerts — these conditions mean the run was useless
    if persisted_n == 0:
        log(
            "CRITICAL: zero predictions persisted after run. "
            "Check live_predictions.json and prediction_log.store_prediction.",
            level="ERROR",
        )
        return 3
    if skipped_count == len(races):
        log("CRITICAL: every race failed.", level="ERROR")
        return 4
    if unknown_schema_count > 0:
        log(
            f"WARN: {unknown_schema_count} race(s) returned unknown schema. "
            f"netkeiba API may have changed — inspect odds_api_meta in "
            f"live_predictions.json before trusting these predictions.",
            level="WARN",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
