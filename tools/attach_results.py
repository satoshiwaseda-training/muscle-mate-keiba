"""Result-attachment tool for weekend batch operation.

Runs on Sunday / Monday. Finds predictions whose race_date is yesterday
(Sunday → Saturday, Monday → Sunday) and calls scraper.fetch_result_netkeiba
to attach the actual finishing order and payouts.

Idempotent: races already carrying a result are skipped, so the script can
be safely re-run on partial failures.

Usage:
  python tools/attach_results.py                     # normal Sun/Mon run
  python tools/attach_results.py --force             # force any weekday
  python tools/attach_results.py --date 2026-04-11   # attach a specific date
"""

from __future__ import annotations

import datetime as dt
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._autolog_utils import (
    banner, build_parser, ensure_project_on_path, log, parse_date,
    previous_race_day, WEEKDAY_JP,
)

ensure_project_on_path()

import scraper  # noqa: E402
import prediction_log as plog  # noqa: E402


def main() -> int:
    parser = build_parser("Attach yesterday's results to logged predictions.")
    args = parser.parse_args()

    today = dt.date.today()
    wd = today.weekday()

    # Determine target race date
    if args.date:
        target = parse_date(args.date)
    else:
        target = previous_race_day(today)

    banner(
        f"attach_results: today={today.isoformat()}({WEEKDAY_JP[wd]}) "
        f"target={target.isoformat() if target else 'NONE'}"
    )

    if target is None and not args.force:
        log(
            f"skip: today is {WEEKDAY_JP[wd]}曜日. "
            f"attach_results runs on 日 (Sun→Sat) / 月 (Mon→Sun) only. "
            f"Use --force with --date for manual recovery.",
            level="SKIP",
        )
        return 0

    if target is None and args.force:
        log(
            "FATAL: --force without --date is ambiguous on a weekday. "
            "Please supply --date YYYY-MM-DD.",
            level="ERROR",
        )
        return 2

    # Gather candidate predictions for the target race date
    all_preds = plog.list_predictions(only_live=True)
    candidates = [e for e in all_preds if e.get("race_date") == target.isoformat()]

    if not candidates:
        try:
            target_races = scraper.fetch_race_list(target) or []
        except Exception:
            target_races = []
        if target_races:
            missing = ", ".join(
                f"{r.get('race_name', '?')}({r.get('race_id', '')})"
                for r in target_races
            )
            log(
                f"target G1/G2 races existed but had no saved predictions: {missing}",
                level="ERROR",
            )
        log(
            f"WARN: zero predictions found for race_date={target.isoformat()}. "
            f"Either weekend_autolog didn't run, or predictions were never "
            f"persisted. Check data/autolog/{target.isoformat()}.log",
            level="WARN",
        )
        return 1

    already_attached = [e for e in candidates if e.get("result")]
    needs_attach = [e for e in candidates if not e.get("result")]

    log(
        f"candidates={len(candidates)} "
        f"already_attached={len(already_attached)} "
        f"needs_attach={len(needs_attach)}"
    )

    if not needs_attach:
        log(
            f"all {len(candidates)} races already have results attached. "
            f"Nothing to do.",
            level="OK",
        )
        return 0

    started = time.time()
    attached = 0      # success_count / attached_count
    missing = 0       # not_published_count (result not yet available on netkeiba)
    errored = 0       # exception or attach_result returned False
    skipped = 0       # empty race_id / pre-existing result

    for i, e in enumerate(needs_attach, 1):
        rid = e.get("race_id", "")
        rname = e.get("race_name", "")
        if not rid:
            log(f"[{i}/{len(needs_attach)}] skip: empty race_id ({rname})",
                level="WARN")
            skipped += 1
            continue

        try:
            res = scraper.fetch_result_netkeiba(rid)
        except Exception as ex:
            log(
                f"[{i}/{len(needs_attach)}] FAIL rid={rid} '{rname}': "
                f"fetch_result_netkeiba raised {ex}",
                level="ERROR",
            )
            log(traceback.format_exc(), level="ERROR")
            errored += 1
            continue

        if not res or not (res.get("finishing_order") or []):
            log(
                f"[{i}/{len(needs_attach)}] missing rid={rid} '{rname}': "
                f"result not yet published (will retry next run)",
                level="WARN",
            )
            missing += 1
            continue

        try:
            ok = plog.attach_result(rid, res)
        except Exception as ex:
            log(
                f"[{i}/{len(needs_attach)}] FAIL rid={rid} '{rname}': "
                f"attach_result raised {ex}",
                level="ERROR",
            )
            errored += 1
            continue

        if ok:
            attached += 1
            log(f"[{i}/{len(needs_attach)}] attached rid={rid} '{rname}'")
        else:
            errored += 1
            log(
                f"[{i}/{len(needs_attach)}] FAIL rid={rid}: "
                f"attach_result returned False (not in log?)",
                level="ERROR",
            )

    # Compute attach rate against the full candidate set
    total = len(candidates)
    total_with_result = len(already_attached) + attached
    attach_rate = (total_with_result / total * 100.0) if total else 0.0
    elapsed = time.time() - started

    # Races that already had results are counted as skipped (no-op).
    skipped += len(already_attached)

    # ── Mandatory summary block (fixed field names, matches weekend_autolog) ──
    log("── attach_results summary ──")
    log(f"  target_date           : {target.isoformat()}")
    log(f"  success_count         : {attached}")
    log(f"  not_published_count   : {missing}")
    log(f"  unknown_schema_count  : 0")  # N/A for result page fetch
    log(f"  attached_count        : {attached}")
    log(f"  skipped_count         : {skipped}")
    log(f"  errored_count         : {errored}")
    log(f"  total_candidates      : {total}")
    log(f"  already_attached      : {len(already_attached)}")
    log(f"  attach_rate_pct       : {attach_rate:.1f}")
    log(f"  elapsed_seconds       : {elapsed:.1f}")

    banner(
        f"attach_results finished: "
        f"success={attached} not_pub={missing} errored={errored} "
        f"skipped={skipped} attach_rate={attach_rate:.1f}% "
        f"({elapsed:.1f}s)"
    )

    # Hard warning if attach rate is low
    if attach_rate < 50.0:
        log(
            f"CRITICAL: attach_rate below 50%. "
            f"Either results aren't published yet (re-run later today), "
            f"or scraper.fetch_result_netkeiba is broken.",
            level="ERROR",
        )
        return 3

    if missing > 0:
        log(
            f"{missing} races missing results. Schedule a re-run for later today.",
            level="WARN",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
