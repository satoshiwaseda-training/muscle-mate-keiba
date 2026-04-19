"""Expand backtest_races.json with additional races from results.json.

For each race_id in results.json that isn't already in backtest_races.json:
  1. Fetch shutuba entries → horse_ids
  2. Fetch horse_detail for each horse (uses cache if fresh)
  3. Add to backtest_races.json with metadata

After this, run build_snapshot.py --all to rebuild snapshots.

USAGE:
  python tools/expand_backtest_set.py
  python tools/expand_backtest_set.py --limit 100      # cap additions
  python tools/expand_backtest_set.py --skip-horse-fetch  # skip heavy fetches
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Max races to add (0 = all)")
    parser.add_argument("--skip-horse-fetch", action="store_true",
                        help="Skip horse detail fetches (rely on existing cache)")
    args = parser.parse_args()

    import scraper

    with open(PROJECT_ROOT / "data" / "results.json", "r", encoding="utf-8") as f:
        res = json.load(f)
    with open(PROJECT_ROOT / "data" / "backtest_cache" / "backtest_races.json",
              "r", encoding="utf-8") as f:
        bt = json.load(f)

    # Normalize keys, find extras
    existing = set(bt.keys())
    extras: list[tuple[str, dict]] = []
    for k, v in res.items():
        rid = k[3:] if k.startswith("bt_") else k
        if rid not in existing:
            extras.append((rid, v))

    # Sort by date (newer first — more likely to have fresh data)
    def _date_key(item):
        rid, v = item
        ts = v.get("timestamp", "")
        return ts

    extras.sort(key=_date_key, reverse=True)
    if args.limit > 0:
        extras = extras[:args.limit]

    print(f"[info] Adding {len(extras)} races")

    ok = 0
    fail = 0
    added = {}
    for i, (rid, r) in enumerate(extras, 1):
        try:
            # Parse race_date from timestamp like "2025-04-05T12:34:56"
            ts = r.get("timestamp", "")
            race_date = ""
            if ts and len(ts) >= 10:
                race_date = ts[:10]

            race_name = r.get("race_name", "")

            # Determine grade from race_name
            grade = ""
            for g in ("G1", "G2", "G3"):
                if f"({g})" in race_name:
                    grade = g
                    break

            # Fetch shutuba entries
            print(f"  [{i}/{len(extras)}] {rid} {race_name} ({race_date})...")
            entries = scraper.fetch_entries_netkeiba(rid) or []
            if not entries:
                print(f"    FAIL: no entries")
                fail += 1
                continue

            num_horses = len(entries)

            # Fetch horse details for each horse (uses cache)
            if not args.skip_horse_fetch:
                for e in entries:
                    hid = e.get("horse_id", "")
                    if hid:
                        try:
                            scraper._cached_horse_detail(hid)
                        except Exception:
                            pass  # non-fatal

            # Cache the enriched entries for build_snapshot to pick up
            scraper._cache_save("enrich_race", rid, entries)

            # Add to backtest metadata
            added[rid] = {
                "race_name": race_name,
                "race_date": race_date,
                "grade":     grade,
                "num_horses": num_horses,
            }
            ok += 1
            print(f"    OK grade={grade} horses={num_horses}")
        except Exception as e:
            print(f"    FAIL: {e}")
            traceback.print_exc()
            fail += 1

    # Merge and save
    bt.update(added)
    with open(PROJECT_ROOT / "data" / "backtest_cache" / "backtest_races.json",
              "w", encoding="utf-8") as f:
        json.dump(bt, f, ensure_ascii=False, indent=2)

    print(f"\n[summary] ok={ok} fail={fail} total_races_now={len(bt)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
