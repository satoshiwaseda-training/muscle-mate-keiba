"""Refresh stale horse cache entries (those missing damsire field).

Scans all snapshots to find horse_ids, then re-fetches any cached entry
that lacks the new schema (damsire/breeder). Safe to re-run.

USAGE:
  python tools/refresh_horse_cache.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scraper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = PROJECT_ROOT / "data" / "snapshot"
CACHE_DIR = PROJECT_ROOT / "data" / "scraper_cache" / "horse"


def collect_horse_ids() -> set:
    ids = set()
    for p in SNAP_DIR.glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
        for hid in s.get("horses", {}).keys():
            if hid:
                ids.add(hid)
    return ids


def is_stale(hid: str) -> bool:
    p = CACHE_DIR / f"{hid}.json"
    if not p.exists():
        return True
    try:
        with open(p, "r", encoding="utf-8") as f:
            c = json.load(f)
    except Exception:
        return True
    # Missing damsire key or empty sire = stale
    if c.get("damsire") is None:
        return True
    return False


def main():
    ids = collect_horse_ids()
    stale_ids = sorted(h for h in ids if is_stale(h))
    print(f"[info] {len(ids)} unique horses, {len(stale_ids)} stale")
    if not stale_ids:
        print("[info] Nothing to refresh.")
        return 0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    for i, hid in enumerate(stale_ids, 1):
        try:
            detail = scraper.fetch_horse_detail(hid) or {}
            if detail:
                p = CACHE_DIR / f"{hid}.json"
                tmp = p.with_name(p.name + ".tmp")
                with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(detail, f, ensure_ascii=False)
                os.replace(tmp, p)
                sire = detail.get("sire", "")
                damsire = detail.get("damsire", "")
                breeder = detail.get("breeder", "")
                print(f"  [{i}/{len(stale_ids)}] OK {hid} "
                      f"sire={sire} damsire={damsire} breeder={breeder}")
                ok += 1
            else:
                print(f"  [{i}/{len(stale_ids)}] EMPTY {hid}")
                fail += 1
        except Exception as e:
            print(f"  [{i}/{len(stale_ids)}] FAIL {hid}: {e}")
            fail += 1

    print(f"\n[summary] ok={ok} fail={fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
