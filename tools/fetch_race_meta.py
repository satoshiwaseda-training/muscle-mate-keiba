"""Fetch race meta (surface, distance, venue) for each snapshot.

Caches to data/race_meta/{race_id}.json so segment analysis can read them.

USAGE:
  python tools/fetch_race_meta.py
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
META_DIR = PROJECT_ROOT / "data" / "race_meta"


def main() -> int:
    META_DIR.mkdir(parents=True, exist_ok=True)

    snap_files = sorted(SNAP_DIR.glob("*.json"))
    ok = 0
    fail = 0
    skipped = 0
    for i, p in enumerate(snap_files, 1):
        rid = p.stem
        out_p = META_DIR / f"{rid}.json"
        if out_p.exists():
            skipped += 1
            continue
        try:
            info = scraper.fetch_race_info_netkeiba(rid) or {}
            surface_str = info.get("surface", "")
            # Parse "芝2400m" → surface + distance
            import re
            surf = "unknown"
            dist = 0
            m = re.search(r"(\u82dd|\u30c0)", surface_str)
            if m:
                surf = "turf" if m.group(1) == "\u82dd" else "dirt"
            m2 = re.search(r"(\d+)m", surface_str)
            if m2:
                dist = int(m2.group(1))
            meta = {
                "race_id":         rid,
                "surface_raw":     surface_str,
                "surface":         surf,
                "distance":        dist,
                "weather":         info.get("weather", ""),
                "track_condition": info.get("track_condition", ""),
            }
            with open(out_p, "w", encoding="utf-8", newline="\n") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"  [{i}/{len(snap_files)}] OK {rid} {surf} {dist}m")
            ok += 1
        except Exception as e:
            print(f"  [{i}/{len(snap_files)}] FAIL {rid}: {e}")
            fail += 1

    print(f"\n[summary] ok={ok} skipped={skipped} fail={fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
